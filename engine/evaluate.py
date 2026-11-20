"""Evaluate a trained Stage 2+3 checkpoint on subject-independent val or test splits."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.temporal import TemporalConfig
from configs.train import TrainConfig
from configs.vision import BackboneConfig, DetectorConfig, VisionConfig
from data.dataset import BatchEyePreprocessor, load_video_records, split_train_val_videos
from engine.metrics import aggregate_video_predictions, seed_everything
from engine.train_temporal import (
    build_window_loader,
    evaluate_epoch,
    resolve_device,
)
from models.backbone import SharedEyeEncoder
from models.detector import YOLOv8EyeCropper
from models.temporal import TemporalHead
from tools.extract_features import cache_eye_crops_for_videos


def _filter_dataclass_kwargs(cls: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _reconstruct_configs(
    payload: dict[str, Any],
) -> tuple[TrainConfig, VisionConfig, TemporalConfig]:
    t_dict = dict(payload["train_config"])
    v_dict = dict(payload["vision_config"])
    det_raw = dict(v_dict["detector"])
    bb_raw = dict(v_dict["backbone"])
    tmp_raw = dict(payload["temporal_config"])

    det_raw["eye_crop_size"] = tuple(det_raw["eye_crop_size"])
    if det_raw.get("normalize_mean") is not None:
        det_raw["normalize_mean"] = tuple(det_raw["normalize_mean"])
    if det_raw.get("normalize_std") is not None:
        det_raw["normalize_std"] = tuple(det_raw["normalize_std"])
    bb_raw["layers"] = tuple(bb_raw["layers"])
    # Old checkpoints saved before pretrained/norm_type existed used custom_resnet + BatchNorm2d
    bb_raw.setdefault("arch", "custom_resnet")
    bb_raw.setdefault("pretrained", False)
    bb_raw.setdefault("freeze_early_layers", False)
    bb_raw.setdefault("norm_type", "batch")

    train_cfg = TrainConfig(**_filter_dataclass_kwargs(TrainConfig, t_dict))
    vision_cfg = VisionConfig(
        detector=DetectorConfig(**_filter_dataclass_kwargs(DetectorConfig, det_raw)),
        backbone=BackboneConfig(**_filter_dataclass_kwargs(BackboneConfig, bb_raw)),
    )
    temporal_cfg = TemporalConfig(**_filter_dataclass_kwargs(TemporalConfig, tmp_raw))
    return train_cfg, vision_cfg, temporal_cfg


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    threshold: float | None = None,
    input_mode: str | None = None,
    device_str: str | None = None,
    save_csv: str | Path | None = None,
    detector: Any | None = None,
) -> dict[str, Any]:
    """Load checkpoint and evaluate on subject-independent `val` or `test` split."""
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg, vision_cfg, temporal_cfg = _reconstruct_configs(payload)

    overrides: dict[str, Any] = {}
    if input_mode is not None:
        overrides["input_mode"] = input_mode
    if device_str is not None:
        overrides["device"] = device_str
    if overrides:
        train_cfg = replace(train_cfg, **overrides)

    seed_everything(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    eval_thresh = threshold if threshold is not None else train_cfg.eval_threshold

    records = load_video_records(train_cfg.labels_csv, train_cfg.frames_manifest_csv)
    _, val_videos, test_videos = split_train_val_videos(
        records, val_ratio=train_cfg.val_ratio, seed=train_cfg.seed
    )
    target_videos = val_videos if split == "val" else test_videos

    if train_cfg.input_mode == "crop_cache" and train_cfg.auto_build_cache:
        cache_eye_crops_for_videos(
            records=target_videos,
            cache_dir=train_cfg.cache_dir,
            detector_config=vision_cfg.detector,
            batch_size=train_cfg.extract_batch_size,
            overwrite=False,
            detector=detector,
            device=device,
        )

    use_gpu_preproc = bool(train_cfg.input_mode == "crop_cache" and train_cfg.gpu_augment)
    preprocessor: BatchEyePreprocessor | None = None
    if use_gpu_preproc:
        preprocessor = BatchEyePreprocessor(
            normalize_mean=vision_cfg.detector.normalize_mean,
            normalize_std=vision_cfg.detector.normalize_std,
            augment=False,
        ).to(device)

    loader, _ = build_window_loader(
        target_videos,
        train_cfg,
        vision_cfg,
        is_train=False,
        return_uint8=use_gpu_preproc,
    )

    cropper: YOLOv8EyeCropper | None = None
    if train_cfg.input_mode == "raw":
        cropper = YOLOv8EyeCropper(config=vision_cfg.detector, detector=detector).to(device)
        cropper.eval()

    # Avoid downloading ImageNet weights during eval since state_dict is loaded immediately
    eval_backbone_cfg = replace(vision_cfg.backbone, pretrained=False)
    encoder = SharedEyeEncoder(eval_backbone_cfg).to(device)
    temporal_head = TemporalHead(temporal_cfg).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"])
    temporal_head.load_state_dict(payload["temporal_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    eval_loss, win_metrics, vid_metrics, (all_vids, all_labels, all_probs) = evaluate_epoch(
        loader=loader,
        input_mode=train_cfg.input_mode,
        cropper=cropper,
        encoder=encoder,
        temporal_head=temporal_head,
        criterion=criterion,
        device=device,
        threshold=eval_thresh,
        use_amp=bool(train_cfg.amp and device.type == "cuda"),
        preprocessor=preprocessor,
    )

    if save_csv is not None:
        ordered_vids, vid_labels, vid_probs = aggregate_video_predictions(
            all_vids, all_labels, all_probs
        )
        out_csv = Path(save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["video_id", "label", "mean_prob", "pred"]
            )
            writer.writeheader()
            for vid, lbl, prob in zip(ordered_vids, vid_labels, vid_probs):
                writer.writerow(
                    {
                        "video_id": vid,
                        "label": int(lbl),
                        "mean_prob": round(float(prob), 6),
                        "pred": int(prob >= eval_thresh),
                    }
                )

    print(
        f"[Eval split={split} | backbone={vision_cfg.backbone.arch} | "
        f"arch={temporal_cfg.arch} | videos={len(target_videos)}] loss={eval_loss:.4f}"
    )
    print(
        "  Window-level -> "
        f"Acc={win_metrics['accuracy']:.4f} | P={win_metrics['precision']:.4f} | "
        f"R={win_metrics['recall']:.4f} | F1={win_metrics['f1']:.4f} | "
        f"AUROC={win_metrics['auroc']:.4f}"
    )
    print(
        "  Video-level  -> "
        f"Acc={vid_metrics['accuracy']:.4f} | P={vid_metrics['precision']:.4f} | "
        f"R={vid_metrics['recall']:.4f} | F1={vid_metrics['f1']:.4f} | "
        f"AUROC={vid_metrics['auroc']:.4f}"
    )

    return {
        "split": split,
        "backbone": vision_cfg.backbone.arch,
        "arch": temporal_cfg.arch,
        "loss": eval_loss,
        "window_metrics": win_metrics,
        "video_metrics": vid_metrics,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on val or test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--input-mode", type=str, default=None, choices=["crop_cache", "raw"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        threshold=args.threshold,
        input_mode=args.input_mode,
        device_str=args.device,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
