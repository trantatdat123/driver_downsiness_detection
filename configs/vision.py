"""Configuration dataclasses and YAML loader for Stage 1 detector & CNN backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    """Hyperparameters for YOLOv8EyeCropper."""

    weights_path: str = "models/yolo_best.pt"
    imgsz: int = 320
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    eye_class_id: int = 0
    face_class_id: int = 1
    eye_crop_size: tuple[int, int] = (64, 64)
    box_padding_ratio: float = 0.0
    input_color_format: str = "RGB"
    normalize_mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] | None = (0.229, 0.224, 0.225)
    device: str | None = None

    def __post_init__(self) -> None:
        fmt = self.input_color_format.upper()
        if fmt not in {"RGB", "BGR"}:
            raise ValueError(f"Unsupported input_color_format: {self.input_color_format}")
        if len(self.eye_crop_size) != 2 or any(s <= 0 for s in self.eye_crop_size):
            raise ValueError(f"eye_crop_size must be positive (H, W), got {self.eye_crop_size}")
        if (self.normalize_mean is None) != (self.normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must both be set or both be None")
        if self.normalize_std is not None and any(s <= 0 for s in self.normalize_std):
            raise ValueError(f"normalize_std values must be > 0, got {self.normalize_std}")


SUPPORTED_BACKBONE_ARCHS = (
    "resnet18",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "custom_resnet",
)


@dataclass(frozen=True)
class BackboneConfig:
    """Hyperparameters for Stage 2 SharedEyeEncoder CNN backbone."""

    arch: str = "resnet18"
    pretrained: bool = True
    freeze_early_layers: bool = True
    in_channels: int = 3
    embedding_dim: int = 64
    base_channels: int = 16
    layers: tuple[int, int, int, int] = (2, 2, 2, 1)
    small_input_stem: bool = True
    dropout: float = 0.35
    zero_init_residual: bool = True
    norm_type: str = "group"
    num_groups: int = 8

    def __post_init__(self) -> None:
        if self.arch.lower() not in SUPPORTED_BACKBONE_ARCHS:
            raise ValueError(
                f"Unsupported backbone arch '{self.arch}'. Choose from {SUPPORTED_BACKBONE_ARCHS}."
            )
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {self.in_channels}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {self.base_channels}")
        if len(self.layers) != 4 or any(n <= 0 for n in self.layers):
            raise ValueError(f"layers must be 4 positive ints, got {self.layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.norm_type not in {"batch", "group"}:
            raise ValueError(f"Unsupported norm_type '{self.norm_type}'. Choose 'batch' or 'group'.")
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {self.num_groups}")


@dataclass(frozen=True)
class VisionConfig:
    """Combined configuration for eye detection/cropping and CNN encoding."""

    detector: DetectorConfig
    backbone: BackboneConfig


def _parse_tuple_3_float(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values}")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_vision_config(config_path: str | Path) -> VisionConfig:
    """Load DetectorConfig and BackboneConfig from a YAML file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    det_raw = raw.get("detector", {})
    bb_raw = raw.get("backbone", {})

    eye_crop_raw = det_raw.get("eye_crop_size", [64, 64])
    detector = DetectorConfig(
        weights_path=str(det_raw.get("weights_path", "models/yolo_best.pt")),
        imgsz=int(det_raw.get("imgsz", 320)),
        conf_thresh=float(det_raw.get("conf_thresh", 0.25)),
        iou_thresh=float(det_raw.get("iou_thresh", 0.45)),
        eye_class_id=int(det_raw.get("eye_class_id", 0)),
        face_class_id=int(det_raw.get("face_class_id", 1)),
        eye_crop_size=(int(eye_crop_raw[0]), int(eye_crop_raw[1])),
        box_padding_ratio=float(det_raw.get("box_padding_ratio", 0.0)),
        input_color_format=str(det_raw.get("input_color_format", "RGB")),
        normalize_mean=_parse_tuple_3_float(det_raw.get("normalize_mean", [0.485, 0.456, 0.406])),
        normalize_std=_parse_tuple_3_float(det_raw.get("normalize_std", [0.229, 0.224, 0.225])),
        device=det_raw.get("device", None),
    )

    layers_raw = bb_raw.get("layers", [2, 2, 2, 1])
    backbone = BackboneConfig(
        arch=str(bb_raw.get("arch", "resnet18")).lower(),
        pretrained=bool(bb_raw.get("pretrained", True)),
        freeze_early_layers=bool(bb_raw.get("freeze_early_layers", True)),
        in_channels=int(bb_raw.get("in_channels", 3)),
        embedding_dim=int(bb_raw.get("embedding_dim", 64)),
        base_channels=int(bb_raw.get("base_channels", 16)),
        layers=(int(layers_raw[0]), int(layers_raw[1]), int(layers_raw[2]), int(layers_raw[3])),
        small_input_stem=bool(bb_raw.get("small_input_stem", True)),
        dropout=float(bb_raw.get("dropout", 0.35)),
        zero_init_residual=bool(bb_raw.get("zero_init_residual", True)),
        norm_type=str(bb_raw.get("norm_type", "group")).lower(),
        num_groups=int(bb_raw.get("num_groups", 8)),
    )
    return VisionConfig(detector=detector, backbone=backbone)
