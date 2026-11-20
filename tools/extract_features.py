"""Extract and cache per-video eye crops and masks using Stage 1 YOLOv8EyeCropper."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs.train import load_train_config
from configs.vision import DetectorConfig, load_vision_config
from data.dataset import VideoRecord, load_video_records
from models.detector import YOLOv8EyeCropper


def _read_image(path: str, to_rgb: bool) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if to_rgb else img_bgr


def extract_eye_crops_for_video(
    record: VideoRecord,
    cropper: YOLOv8EyeCropper,
    batch_size: int = 32,
    device: torch.device | None = None,
    io_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run YOLOv8EyeCropper over all ordered frames of one video and return uint8 crops & masks."""
    all_crops: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    paths = record.image_paths
    to_rgb = cropper.config.input_color_format.upper() == "RGB"

    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as pool:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            imgs = list(pool.map(lambda p: _read_image(p, to_rgb), batch_paths))

            # (1, T_chunk, H, W, 3) uint8
            batch_tensor = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0)
            if device is not None:
                batch_tensor = batch_tensor.to(device, non_blocking=True)

            # crops: (1, T_chunk, 2, 3, H_eye, W_eye) in [0, 1], mask: (1, T_chunk, 2)
            crops_f, mask_f = cropper(batch_tensor)
            crops_u8 = (
                (crops_f.squeeze(0).cpu() * 255.0)
                .round()
                .clamp(0.0, 255.0)
                .to(dtype=torch.uint8)
            )
            all_crops.append(crops_u8)
            all_masks.append(mask_f.squeeze(0).cpu().to(dtype=torch.float32))

    # (N, 2, 3, H_eye, W_eye) uint8 and (N, 2) float32
    video_crops = torch.cat(all_crops, dim=0)
    video_mask = torch.cat(all_masks, dim=0)
    return video_crops, video_mask


def cache_eye_crops_for_videos(
    records: Sequence[VideoRecord],
    cache_dir: str | Path,
    detector_config: DetectorConfig,
    batch_size: int = 32,
    overwrite: bool = False,
    detector: object | None = None,
    device: torch.device | None = None,
) -> int:
    """Extract and save per-video `{video_id}.pt` eye-crop caches, returning newly cached count."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        rec
        for rec in records
        if overwrite or not (out_dir / f"{rec.video_id}.pt").exists()
    ]
    if not missing:
        return 0

    # Cache unnormalized uint8 [0, 255] crops so Dataset can normalize and save 50% disk/IO
    raw_det_cfg = replace(detector_config, normalize_mean=None, normalize_std=None)
    cropper = YOLOv8EyeCropper(config=raw_det_cfg, detector=detector)
    if device is not None:
        cropper = cropper.to(device)
    cropper.eval()

    written = 0
    for rec in tqdm(missing, desc="Caching eye crops per video", unit="video"):
        video_crops, video_mask = extract_eye_crops_for_video(
            record=rec,
            cropper=cropper,
            batch_size=batch_size,
            device=device,
        )
        target_path = out_dir / f"{rec.video_id}.pt"
        tmp_path = out_dir / f".{rec.video_id}.pt.tmp"
        torch.save(
            {
                "video_id": rec.video_id,
                "label": rec.label,
                "eye_crops": video_crops,
                "eye_mask": video_mask,
            },
            tmp_path,
        )
        tmp_path.replace(target_path)
        written += 1

    return written


def main() -> None:
    """CLI entrypoint to extract and cache per-video eye crops from frames_manifest.csv."""
    parser = argparse.ArgumentParser(
        description="Extract and cache per-video YOLOv8 eye crops for training/evaluation."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    vision_cfg = load_vision_config(train_cfg.vision_config)
    records = load_video_records(train_cfg.labels_csv, train_cfg.frames_manifest_csv)
    if args.split != "all":
        records = [r for r in records if r.split == args.split]

    dev = (
        torch.device(train_cfg.device)
        if train_cfg.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    written = cache_eye_crops_for_videos(
        records=records,
        cache_dir=train_cfg.cache_dir,
        detector_config=vision_cfg.detector,
        batch_size=train_cfg.extract_batch_size,
        overwrite=args.overwrite,
        device=dev,
    )
    print(f"Cached eye crops for {written} videos into {train_cfg.cache_dir}.")


if __name__ == "__main__":
    main()
