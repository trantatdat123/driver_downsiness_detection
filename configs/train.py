"""Configuration dataclass and YAML loader for Stage 2+3 video-sequence training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and paths for training on per-video ordered image sequences."""

    # Data & split
    labels_csv: str = "data/labels.csv"
    frames_manifest_csv: str = "data/frames_manifest.csv"
    cache_dir: str = "data/cache/eye_crops"
    val_ratio: float = 0.2
    window_size: int = 30
    temporal_stride: int = 3
    window_step_train: int = 25
    window_step_eval: int = 15

    # Model config paths
    vision_config: str = "configs/vision.yaml"
    temporal_config: str = "configs/temporal_gru.yaml"

    # Optimization & runtime
    seed: int = 42
    input_mode: str = "crop_cache"
    auto_build_cache: bool = True
    extract_batch_size: int = 32
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 0
    backbone_lr: float = 3.0e-4
    temporal_lr: float = 3.0e-4
    weight_decay: float = 2.0e-2
    grad_clip_norm: float = 1.0
    freeze_backbone: bool = False
    use_pos_weight: bool = True
    augment: bool = True
    label_smoothing: float = 0.05
    warmup_epochs: int = 3
    early_stopping_patience: int = 8
    preload_cache: bool = True
    gpu_augment: bool = True
    channels_last: bool = True
    use_compile: bool = False
    prefetch_factor: int = 2
    amp: bool = True
    eval_threshold: float = 0.5
    selection_metric: str = "auroc"
    checkpoint_dir: str = "checkpoints"
    device: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.window_size <= 0 or self.temporal_stride <= 0:
            raise ValueError("window_size and temporal_stride must be > 0")
        if self.window_step_train <= 0 or self.window_step_eval <= 0:
            raise ValueError("window_step values must be > 0")
        if self.input_mode not in {"crop_cache", "raw"}:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        if self.selection_metric not in {"f1", "recall", "auroc", "accuracy"}:
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be > 0")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}"
            )
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0, got {self.prefetch_factor}")


def load_train_config(config_path: str | Path) -> TrainConfig:
    """Load TrainConfig from a YAML file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    data_sec = raw.get("data", {})
    cfg_sec = raw.get("configs", {})
    trn_sec = raw.get("train", {})

    return TrainConfig(
        labels_csv=str(data_sec.get("labels_csv", "data/labels.csv")),
        frames_manifest_csv=str(
            data_sec.get("frames_manifest_csv", "data/frames_manifest.csv")
        ),
        cache_dir=str(data_sec.get("cache_dir", "data/cache/eye_crops")),
        val_ratio=float(data_sec.get("val_ratio", 0.2)),
        window_size=int(data_sec.get("window_size", 30)),
        temporal_stride=int(data_sec.get("temporal_stride", 3)),
        window_step_train=int(data_sec.get("window_step_train", 25)),
        window_step_eval=int(data_sec.get("window_step_eval", 15)),
        vision_config=str(cfg_sec.get("vision_config", "configs/vision.yaml")),
        temporal_config=str(cfg_sec.get("temporal_config", "configs/temporal_gru.yaml")),
        seed=int(trn_sec.get("seed", 42)),
        input_mode=str(trn_sec.get("input_mode", "crop_cache")),
        auto_build_cache=bool(trn_sec.get("auto_build_cache", True)),
        extract_batch_size=int(trn_sec.get("extract_batch_size", 32)),
        epochs=int(trn_sec.get("epochs", 30)),
        batch_size=int(trn_sec.get("batch_size", 16)),
        num_workers=int(trn_sec.get("num_workers", 0)),
        backbone_lr=float(trn_sec.get("backbone_lr", 3.0e-4)),
        temporal_lr=float(trn_sec.get("temporal_lr", 3.0e-4)),
        weight_decay=float(trn_sec.get("weight_decay", 2.0e-2)),
        grad_clip_norm=float(trn_sec.get("grad_clip_norm", 1.0)),
        freeze_backbone=bool(trn_sec.get("freeze_backbone", False)),
        use_pos_weight=bool(trn_sec.get("use_pos_weight", True)),
        augment=bool(trn_sec.get("augment", True)),
        label_smoothing=float(trn_sec.get("label_smoothing", 0.05)),
        warmup_epochs=int(trn_sec.get("warmup_epochs", 3)),
        early_stopping_patience=int(trn_sec.get("early_stopping_patience", 8)),
        preload_cache=bool(trn_sec.get("preload_cache", True)),
        gpu_augment=bool(trn_sec.get("gpu_augment", True)),
        channels_last=bool(trn_sec.get("channels_last", True)),
        use_compile=bool(trn_sec.get("use_compile", False)),
        prefetch_factor=int(trn_sec.get("prefetch_factor", 2)),
        amp=bool(trn_sec.get("amp", True)),
        eval_threshold=float(trn_sec.get("eval_threshold", 0.5)),
        selection_metric=str(trn_sec.get("selection_metric", "auroc")),
        checkpoint_dir=str(trn_sec.get("checkpoint_dir", "checkpoints")),
        device=trn_sec.get("device", None),
    )
