"""Stage 1 eye detection and cropping using fine-tuned YOLOv8."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.vision import DetectorConfig

BoxXYXY = tuple[float, float, float, float]


def _clamp_and_pad_box(
    box: BoxXYXY,
    frame_h: int,
    frame_w: int,
    padding_ratio: float = 0.0,
) -> BoxXYXY | None:
    """Optionally pad a box around its center, clamp to frame bounds, and filter degenerate boxes."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if padding_ratio > 0.0:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        pad_w = 0.5 * padding_ratio * bw
        pad_h = 0.5 * padding_ratio * bh
        x1 -= pad_w
        x2 += pad_w
        y1 -= pad_h
        y2 += pad_h

    x1 = max(0.0, min(float(frame_w), x1))
    x2 = max(0.0, min(float(frame_w), x2))
    y1 = max(0.0, min(float(frame_h), y1))
    y2 = max(0.0, min(float(frame_h), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    if int(math.ceil(x2)) <= int(math.floor(x1)) or int(math.ceil(y2)) <= int(math.floor(y1)):
        return None
    return (x1, y1, x2, y2)


def select_top_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    eye_class_id: int,
    face_class_id: int,
    frame_h: int,
    frame_w: int,
    box_padding_ratio: float = 0.0,
) -> tuple[list[BoxXYXY], float]:
    """Filter valid boxes and select top-2 eye boxes and top-1 face center-x by confidence."""
    eye_candidates: list[tuple[float, BoxXYXY]] = []
    face_candidates: list[tuple[float, BoxXYXY]] = []

    for box_arr, score_val, cls_val in zip(boxes_xyxy, scores, class_ids):
        cls_id = int(cls_val)
        score = float(score_val)
        if cls_id == eye_class_id:
            clamped = _clamp_and_pad_box(
                (float(box_arr[0]), float(box_arr[1]), float(box_arr[2]), float(box_arr[3])),
                frame_h=frame_h,
                frame_w=frame_w,
                padding_ratio=box_padding_ratio,
            )
            if clamped is not None:
                eye_candidates.append((score, clamped))
        elif cls_id == face_class_id:
            clamped = _clamp_and_pad_box(
                (float(box_arr[0]), float(box_arr[1]), float(box_arr[2]), float(box_arr[3])),
                frame_h=frame_h,
                frame_w=frame_w,
                padding_ratio=0.0,
            )
            if clamped is not None:
                face_candidates.append((score, clamped))

    face_candidates.sort(key=lambda item: item[0], reverse=True)
    if face_candidates:
        fx1, _, fx2, _ = face_candidates[0][1]
        ref_cx = 0.5 * (fx1 + fx2)
    else:
        ref_cx = 0.5 * float(frame_w)

    eye_candidates.sort(key=lambda item: item[0], reverse=True)
    top_eyes = [box for _, box in eye_candidates[:2]]
    return top_eyes, ref_cx


def assign_eye_slots(
    eye_boxes: list[BoxXYXY],
    ref_cx: float,
) -> tuple[tuple[BoxXYXY | None, BoxXYXY | None], tuple[float, float]]:
    """Assign 0, 1, or 2 detected eye boxes into (image-left slot 0, image-right slot 1)."""
    if len(eye_boxes) == 0:
        return (None, None), (0.0, 0.0)

    if len(eye_boxes) == 1:
        box = eye_boxes[0]
        cx = 0.5 * (box[0] + box[2])
        if cx < ref_cx:
            return (box, None), (1.0, 0.0)
        return (None, box), (0.0, 1.0)

    sorted_eyes = sorted(eye_boxes[:2], key=lambda b: 0.5 * (b[0] + b[2]))
    return (sorted_eyes[0], sorted_eyes[1]), (1.0, 1.0)


def _extract_boxes_from_result(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize Ultralytics Results or mock prediction output into numpy arrays."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.detach().cpu().numpy()
    if isinstance(conf, torch.Tensor):
        conf = conf.detach().cpu().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.detach().cpu().numpy()

    return (
        np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
        np.asarray(conf, dtype=np.float32).reshape(-1),
        np.asarray(cls, dtype=np.int64).reshape(-1),
    )


class YOLOv8EyeCropper(nn.Module):
    """Detect and crop left/right eyes from raw video frames with missing-eye masking.

    Input:
        frames: (B, T, H, W, 3) uint8 [0, 255] or float32 [0, 1]
    Output:
        eye_crops: (B, T, 2, 3, H_eye, W_eye) float32 RGB (0.0 where eye is missing)
        eye_mask:  (B, T, 2) float32 in {0.0, 1.0} for [image-left, image-right]
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        detector: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else DetectorConfig()

        if detector is not None:
            det_obj = detector
        else:
            from ultralytics import YOLO

            weights_path = Path(self.config.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"YOLOv8 weights not found at: {weights_path}")
            det_obj = YOLO(str(weights_path))

        # Bypass nn.Module.__setattr__ because ultralytics.YOLO subclasses nn.Module
        # and overrides .train() to launch full YOLO training (coco8.yaml, 100 epochs)
        # whenever parent.eval() / parent.train(False) recurses over child modules!
        object.__setattr__(self, "_detector", det_obj)
        inner_model = getattr(det_obj, "model", None)
        if isinstance(inner_model, nn.Module):
            inner_model.eval()
            for param in inner_model.parameters():
                param.requires_grad = False

        if self.config.normalize_mean is not None and self.config.normalize_std is not None:
            # (3,) -> (1, 1, 3, 1, 1) for broadcasting against (N, 2, 3, H_eye, W_eye)
            mean = torch.tensor(self.config.normalize_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
            std = torch.tensor(self.config.normalize_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
            self.register_buffer("norm_mean", mean)
            self.register_buffer("norm_std", std)
        else:
            self.register_buffer("norm_mean", None)
            self.register_buffer("norm_std", None)

    def _frames_to_bgr_numpy(self, flat_frames: torch.Tensor) -> list[np.ndarray]:
        """Convert (N, H, W, 3) tensor to a list of uint8 BGR numpy images for YOLO."""
        arr = flat_frames.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            max_val = float(arr.max()) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        if self.config.input_color_format.upper() == "RGB":
            # (N, H, W, 3) RGB -> (N, H, W, 3) BGR
            arr = arr[..., ::-1]

        return [np.ascontiguousarray(img) for img in arr]

    def _frames_to_rgb_float(self, flat_frames: torch.Tensor) -> torch.Tensor:
        """Convert (N, H, W, 3) tensor to (N, 3, H, W) float32 RGB in [0, 1]."""
        if flat_frames.dtype == torch.uint8:
            rgb = flat_frames.to(dtype=torch.float32) / 255.0
        else:
            rgb = flat_frames.to(dtype=torch.float32)
            if rgb.numel() > 0 and float(rgb.max().item()) > 1.0:
                rgb = rgb / 255.0

        if self.config.input_color_format.upper() == "BGR":
            # (N, H, W, 3) BGR -> (N, H, W, 3) RGB
            rgb = rgb.flip(dims=(-1,))

        # (N, H, W, 3) -> (N, 3, H, W)
        return rgb.permute(0, 3, 1, 2).contiguous()

    def _crop_and_resize(self, frame_chw: torch.Tensor, box: BoxXYXY) -> torch.Tensor:
        """Crop a single (3, H, W) frame by XYXY box and resize to (3, H_eye, W_eye)."""
        _, frame_h, frame_w = frame_chw.shape
        x1_f, y1_f, x2_f, y2_f = box

        x1 = max(0, min(frame_w - 1, int(math.floor(x1_f))))
        y1 = max(0, min(frame_h - 1, int(math.floor(y1_f))))
        x2 = max(x1 + 1, min(frame_w, int(math.ceil(x2_f))))
        y2 = max(y1 + 1, min(frame_h, int(math.ceil(y2_f))))

        # (3, H, W) -> (1, 3, crop_h, crop_w)
        patch = frame_chw[:, y1:y2, x1:x2].unsqueeze(0)
        # (1, 3, crop_h, crop_w) -> (3, H_eye, W_eye)
        resized = F.interpolate(
            patch,
            size=self.config.eye_crop_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized

    def _predict_batch(self, images_bgr: list[np.ndarray]) -> list[Any]:
        """Run YOLO detection over a batch of BGR uint8 images."""
        if hasattr(self._detector, "predict"):
            kwargs: dict[str, Any] = {
                "source": images_bgr,
                "imgsz": self.config.imgsz,
                "conf": self.config.conf_thresh,
                "iou": self.config.iou_thresh,
                "verbose": False,
            }
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            return list(self._detector.predict(**kwargs))
        return list(self._detector(images_bgr))

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect eyes on raw video frames and return ordered eye crops and visibility mask."""
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected frames of shape (B, T, H, W, 3), got {tuple(frames.shape)}"
            )

        b, t, frame_h, frame_w, c = frames.shape
        h_eye, w_eye = self.config.eye_crop_size
        device = frames.device

        # (B, T, H, W, 3) -> (B * T, H, W, 3)
        flat_frames = frames.reshape(b * t, frame_h, frame_w, c)
        images_bgr = self._frames_to_bgr_numpy(flat_frames)
        # (B * T, H, W, 3) -> (B * T, 3, H, W)
        flat_rgb = self._frames_to_rgb_float(flat_frames)

        results = self._predict_batch(images_bgr)
        if len(results) != b * t:
            raise RuntimeError(
                f"Detector returned {len(results)} results for {b * t} frames"
            )

        # (B * T, 2, 3, H_eye, W_eye) and (B * T, 2)
        flat_crops = torch.zeros(
            (b * t, 2, 3, h_eye, w_eye), dtype=torch.float32, device=device
        )
        flat_mask = torch.zeros((b * t, 2), dtype=torch.float32, device=device)

        for idx, res in enumerate(results):
            xyxy, conf, cls = _extract_boxes_from_result(res)
            eye_boxes, ref_cx = select_top_detections(
                boxes_xyxy=xyxy,
                scores=conf,
                class_ids=cls,
                eye_class_id=self.config.eye_class_id,
                face_class_id=self.config.face_class_id,
                frame_h=frame_h,
                frame_w=frame_w,
                box_padding_ratio=self.config.box_padding_ratio,
            )
            (left_box, right_box), (v_left, v_right) = assign_eye_slots(eye_boxes, ref_cx)

            if left_box is not None:
                flat_crops[idx, 0] = self._crop_and_resize(flat_rgb[idx], left_box)
            if right_box is not None:
                flat_crops[idx, 1] = self._crop_and_resize(flat_rgb[idx], right_box)
            flat_mask[idx, 0] = v_left
            flat_mask[idx, 1] = v_right

        if self.norm_mean is not None and self.norm_std is not None:
            flat_crops = (flat_crops - self.norm_mean) / self.norm_std

        # Guarantee zero-fill on missing eye slots after normalization:
        # (B * T, 2) -> (B * T, 2, 1, 1, 1)
        mask_broadcast = flat_mask.reshape(b * t, 2, 1, 1, 1)
        flat_crops = flat_crops * mask_broadcast

        # (B * T, 2, 3, H_eye, W_eye) -> (B, T, 2, 3, H_eye, W_eye)
        eye_crops = flat_crops.reshape(b, t, 2, 3, h_eye, w_eye)
        # (B * T, 2) -> (B, T, 2)
        eye_mask = flat_mask.reshape(b, t, 2)
        return eye_crops, eye_mask
