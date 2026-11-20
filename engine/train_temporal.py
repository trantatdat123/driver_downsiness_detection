"""Train Stage 2 (SharedEyeEncoder) + Stage 3 (TemporalHead) per video image sequence."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.temporal import TemporalConfig, load_temporal_config
from configs.train import TrainConfig, load_train_config
from configs.vision import VisionConfig, load_vision_config
from data.dataset import (
    BatchEyePreprocessor,
    VideoEyeCropWindowDataset,
    VideoRawWindowDataset,
    VideoRecord,
    build_video_windows,
    load_video_records,
    preload_eye_crop_cache,
    split_train_val_videos,
)
from engine.metrics import (
    aggregate_video_predictions,
    compute_binary_metrics,
    seed_everything,
)
from models.backbone import SharedEyeEncoder
from models.detector import YOLOv8EyeCropper
from models.temporal import TemporalHead
from tools.extract_features import cache_eye_crops_for_videos


def resolve_device(device_str: str | None) -> torch.device:
    """Select CUDA when available unless an explicit device string is provided."""
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_cuda_performance(device: torch.device) -> None:
    """Enable cuDNN autotuner and TF32 matmul/conv paths on CUDA GPUs."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _unwrap_state_dict(module: nn.Module) -> dict[str, Any]:
    """Return clean state_dict even if module was wrapped by torch.compile."""
    orig = getattr(module, "_orig_mod", module)
    return orig.state_dict()


def build_window_loader(
    records: list[VideoRecord],
    train_cfg: TrainConfig,
    vision_cfg: VisionConfig,
    is_train: bool,
    preloaded_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    return_uint8: bool = False,
) -> tuple[DataLoader[Any], list[int]]:
    """Build DataLoader over sliding windows sampled inside each video_id."""
    step = train_cfg.window_step_train if is_train else train_cfg.window_step_eval
    windows = build_video_windows(
        records=records,
        window_size=train_cfg.window_size,
        temporal_stride=train_cfg.temporal_stride,
        window_step=step,
    )
    labels = [w.label for w in windows]

    if train_cfg.input_mode == "crop_cache":
        use_aug = bool(is_train and train_cfg.augment)
        jitter = (train_cfg.window_step_train // 2) if use_aug else 0
        dataset: Any = VideoEyeCropWindowDataset(
            windows=windows,
            cache_dir=train_cfg.cache_dir,
            normalize_mean=vision_cfg.detector.normalize_mean,
            normalize_std=vision_cfg.detector.normalize_std,
            augment=use_aug,
            temporal_jitter=jitter,
            return_uint8=return_uint8,
            preloaded_cache=preloaded_cache,
        )
    else:
        dataset = VideoRawWindowDataset(
            records=records,
            windows=windows,
            color_format=vision_cfg.detector.input_color_format,
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": train_cfg.batch_size,
        "shuffle": is_train,
        "num_workers": train_cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if train_cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = train_cfg.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, labels


def _forward_batch(
    batch: tuple[Any, ...],
    input_mode: str,
    cropper: YOLOv8EyeCropper | None,
    encoder: nn.Module,
    temporal_head: nn.Module,
    device: torch.device,
    freeze_backbone: bool,
    preprocessor: BatchEyePreprocessor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if input_mode == "crop_cache":
        eye_crops, eye_mask, labels, video_ids = batch
        eye_crops = eye_crops.to(device, non_blocking=True)
        eye_mask = eye_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if preprocessor is not None:
            eye_crops, eye_mask = preprocessor(eye_crops, eye_mask)
    else:
        if cropper is None:
            raise RuntimeError("YOLOv8EyeCropper is required when input_mode='raw'")
        frames, labels, video_ids = batch
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        eye_crops, eye_mask = cropper(frames)

    if freeze_backbone:
        with torch.no_grad():
            frame_feats = encoder(eye_crops, eye_mask)
    else:
        frame_feats = encoder(eye_crops, eye_mask)

    # (B, W, 2*D + 2) -> (B, W, 1)
    logits = temporal_head(frame_feats)
    # Many-to-One loss uses the causal prediction at the final window step: (B, W, 1) -> (B,)
    last_logits = logits[:, -1, 0]
    return last_logits, labels, list(video_ids)


def train_one_epoch(
    loader: DataLoader[Any],
    input_mode: str,
    cropper: YOLOv8EyeCropper | None,
    encoder: nn.Module,
    temporal_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
    freeze_backbone: bool,
    label_smoothing: float = 0.0,
    scaler: torch.amp.GradScaler | None = None,
    preprocessor: BatchEyePreprocessor | None = None,
) -> float:
    """Run one training epoch over per-video sliding windows."""
    if freeze_backbone:
        encoder.eval()
    else:
        encoder.train()
    temporal_head.train()
    if preprocessor is not None:
        preprocessor.train()

    total_loss = 0.0
    total_items = 0
    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            last_logits, labels, _ = _forward_batch(
                batch=batch,
                input_mode=input_mode,
                cropper=cropper,
                encoder=encoder,
                temporal_head=temporal_head,
                device=device,
                freeze_backbone=freeze_backbone,
                preprocessor=preprocessor,
            )
            if label_smoothing > 0.0:
                targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                targets = labels
            loss = criterion(last_logits, targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0.0:
                params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()

        bsz = int(labels.shape[0])
        total_loss += float(loss.item()) * bsz
        total_items += bsz

    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate_epoch(
    loader: DataLoader[Any],
    input_mode: str,
    cropper: YOLOv8EyeCropper | None,
    encoder: nn.Module,
    temporal_head: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
    preprocessor: BatchEyePreprocessor | None = None,
) -> tuple[
    float,
    dict[str, float],
    dict[str, float],
    tuple[list[str], list[int], list[float]],
]:
    """Evaluate on val/test windows and return loss, window metrics, video metrics, and raw preds."""
    encoder.eval()
    temporal_head.eval()
    if preprocessor is not None:
        preprocessor.eval()

    total_loss = 0.0
    total_items = 0
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_vids: list[str] = []
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch in tqdm(loader, desc="Eval", leave=False):
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            last_logits, labels, video_ids = _forward_batch(
                batch=batch,
                input_mode=input_mode,
                cropper=cropper,
                encoder=encoder,
                temporal_head=temporal_head,
                device=device,
                freeze_backbone=True,
                preprocessor=preprocessor,
            )
            loss = criterion(last_logits.float(), labels.float())
            probs = torch.sigmoid(last_logits.float())

        bsz = int(labels.shape[0])
        total_loss += float(loss.item()) * bsz
        total_items += bsz
        all_labels.extend(int(x) for x in labels.detach().cpu().tolist())
        all_probs.extend(float(x) for x in probs.detach().cpu().tolist())
        all_vids.extend(video_ids)

    win_metrics = compute_binary_metrics(all_labels, all_probs, threshold=threshold)
    _, vid_labels, vid_probs = aggregate_video_predictions(all_vids, all_labels, all_probs)
    vid_metrics = compute_binary_metrics(vid_labels, vid_probs, threshold=threshold)
    return total_loss / max(1, total_items), win_metrics, vid_metrics, (all_vids, all_labels, all_probs)


def run_training(
    train_cfg: TrainConfig,
    vision_cfg: VisionConfig,
    temporal_cfg: TemporalConfig,
    detector: Any | None = None,
    preloaded_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> Path:
    """Execute full subject-independent training and return best checkpoint path."""
    expected_dim = 2 * vision_cfg.backbone.embedding_dim + 2
    if temporal_cfg.input_dim != expected_dim:
        raise ValueError(
            f"TemporalConfig.input_dim ({temporal_cfg.input_dim}) must match "
            f"2 * embedding_dim + 2 ({expected_dim})"
        )

    seed_everything(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    _configure_cuda_performance(device)

    records = load_video_records(train_cfg.labels_csv, train_cfg.frames_manifest_csv)
    train_videos, val_videos, test_videos = split_train_val_videos(
        records, val_ratio=train_cfg.val_ratio, seed=train_cfg.seed
    )
    print(
        f"Split summary | Train: {len({r.subject_id for r in train_videos})} subj / "
        f"{len(train_videos)} videos | Val: {len({r.subject_id for r in val_videos})} subj / "
        f"{len(val_videos)} videos | Test (held-out): "
        f"{len({r.subject_id for r in test_videos})} subj / {len(test_videos)} videos"
    )

    if train_cfg.input_mode == "crop_cache" and train_cfg.auto_build_cache:
        cache_eye_crops_for_videos(
            records=train_videos + val_videos,
            cache_dir=train_cfg.cache_dir,
            detector_config=vision_cfg.detector,
            batch_size=train_cfg.extract_batch_size,
            overwrite=False,
            detector=detector,
            device=device,
        )

    active_cache = preloaded_cache
    if train_cfg.input_mode == "crop_cache" and train_cfg.preload_cache and active_cache is None:
        t_load = time.perf_counter()
        active_cache = preload_eye_crop_cache(
            video_ids=[r.video_id for r in train_videos + val_videos],
            cache_dir=train_cfg.cache_dir,
        )
        print(
            f"Preloaded {len(active_cache)} cached videos into RAM in "
            f"{time.perf_counter() - t_load:.2f}s"
        )

    use_gpu_preproc = bool(train_cfg.input_mode == "crop_cache" and train_cfg.gpu_augment)
    preprocessor: BatchEyePreprocessor | None = None
    if use_gpu_preproc:
        preprocessor = BatchEyePreprocessor(
            normalize_mean=vision_cfg.detector.normalize_mean,
            normalize_std=vision_cfg.detector.normalize_std,
            augment=train_cfg.augment,
        ).to(device)

    train_loader, train_labels = build_window_loader(
        train_videos,
        train_cfg,
        vision_cfg,
        is_train=True,
        preloaded_cache=active_cache,
        return_uint8=use_gpu_preproc,
    )
    val_loader, val_labels = build_window_loader(
        val_videos,
        train_cfg,
        vision_cfg,
        is_train=False,
        preloaded_cache=active_cache,
        return_uint8=use_gpu_preproc,
    )
    print(
        f"Windows | Train: {len(train_labels)} (pos={sum(train_labels)}) | "
        f"Val: {len(val_labels)} (pos={sum(val_labels)}) | Arch: {temporal_cfg.arch}"
    )

    cropper: YOLOv8EyeCropper | None = None
    if train_cfg.input_mode == "raw":
        cropper = YOLOv8EyeCropper(config=vision_cfg.detector, detector=detector).to(device)
        cropper.eval()

    encoder: nn.Module = SharedEyeEncoder(vision_cfg.backbone).to(device)
    if train_cfg.channels_last and device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)
    temporal_head: nn.Module = TemporalHead(temporal_cfg).to(device)

    enc_total = sum(p.numel() for p in encoder.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    tmp_params = sum(p.numel() for p in temporal_head.parameters() if p.requires_grad)

    compiled = False
    if train_cfg.use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        if sys.platform == "win32":
            print("Warning: Skipping torch.compile on Windows; running eager CUDA.")
        else:
            try:
                encoder = torch.compile(encoder)
                temporal_head = torch.compile(temporal_head)
                compiled = True
            except Exception as exc:
                print(f"Warning: torch.compile failed ({exc}); falling back to eager mode.")

    bb_cfg = vision_cfg.backbone
    print(
        f"Params | Backbone ({bb_cfg.arch}, pretrain={bb_cfg.pretrained}, "
        f"freeze_early={bb_cfg.freeze_early_layers}): {enc_params:,}/{enc_total:,} trainable | "
        f"Temporal ({temporal_cfg.arch}): {tmp_params:,} | "
        f"Augment: {train_cfg.augment} (gpu={use_gpu_preproc}) | "
        f"LabelSmooth: {train_cfg.label_smoothing} | Compile: {compiled}"
    )

    if train_cfg.freeze_backbone:
        for param in encoder.parameters():
            param.requires_grad = False
        param_groups = [
            {"params": [p for p in temporal_head.parameters() if p.requires_grad], "lr": train_cfg.temporal_lr}
        ]
    else:
        enc_trainable = [p for p in encoder.parameters() if p.requires_grad]
        tmp_trainable = [p for p in temporal_head.parameters() if p.requires_grad]
        param_groups = []
        if enc_trainable:
            param_groups.append({"params": enc_trainable, "lr": train_cfg.backbone_lr})
        param_groups.append({"params": tmp_trainable, "lr": train_cfg.temporal_lr})

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=train_cfg.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.weight_decay)

    warmup_ep = min(train_cfg.warmup_epochs, max(0, train_cfg.epochs - 1))
    if warmup_ep > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, train_cfg.epochs - warmup_ep)
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_ep]
            )
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, train_cfg.epochs)
        )

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight_tensor: torch.Tensor | None = None
    if train_cfg.use_pos_weight and n_pos > 0 and n_neg > 0:
        pos_weight_tensor = torch.tensor([float(n_neg / n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    eval_criterion = nn.BCEWithLogitsLoss()

    use_amp = bool(train_cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{vision_cfg.backbone.arch.lower()}_{temporal_cfg.arch.lower()}"
    best_path = ckpt_dir / f"{run_tag}_best.pt"
    last_path = ckpt_dir / f"{run_tag}_last.pt"
    best_score = -1.0
    epochs_no_improve = 0
    last_payload: dict[str, Any] | None = None

    for epoch in range(1, train_cfg.epochs + 1):
        t_ep = time.perf_counter()
        train_loss = train_one_epoch(
            loader=train_loader,
            input_mode=train_cfg.input_mode,
            cropper=cropper,
            encoder=encoder,
            temporal_head=temporal_head,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=train_cfg.grad_clip_norm,
            freeze_backbone=train_cfg.freeze_backbone,
            label_smoothing=train_cfg.label_smoothing,
            scaler=scaler,
            preprocessor=preprocessor,
        )
        val_loss, val_win, val_vid, _ = evaluate_epoch(
            loader=val_loader,
            input_mode=train_cfg.input_mode,
            cropper=cropper,
            encoder=encoder,
            temporal_head=temporal_head,
            criterion=eval_criterion,
            device=device,
            threshold=train_cfg.eval_threshold,
            use_amp=use_amp,
            preprocessor=preprocessor,
        )
        scheduler.step()
        ep_sec = time.perf_counter() - t_ep
        win_per_s = (len(train_labels) + len(val_labels)) / max(1e-6, ep_sec)

        score = val_win[train_cfg.selection_metric]
        print(
            f"[Epoch {epoch:02d}/{train_cfg.epochs:02d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"win_F1={val_win['f1']:.4f} win_R={val_win['recall']:.4f} "
            f"win_AUC={val_win['auroc']:.4f} | "
            f"vid_F1={val_vid['f1']:.4f} vid_AUC={val_vid['auroc']:.4f} | "
            f"time={ep_sec:.1f}s ({win_per_s:.0f} win/s)"
        )

        last_payload = {
            "epoch": epoch,
            "backbone": vision_cfg.backbone.arch,
            "arch": temporal_cfg.arch,
            "encoder_state_dict": _unwrap_state_dict(encoder),
            "temporal_state_dict": _unwrap_state_dict(temporal_head),
            "vision_config": asdict(vision_cfg),
            "temporal_config": asdict(temporal_cfg),
            "train_config": asdict(train_cfg),
            "val_window_metrics": val_win,
            "val_video_metrics": val_vid,
        }
        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            torch.save(last_payload, best_path)
        else:
            epochs_no_improve += 1
            if (
                train_cfg.early_stopping_patience > 0
                and epochs_no_improve >= train_cfg.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch:02d} "
                    f"(no improvement in {train_cfg.selection_metric} for "
                    f"{train_cfg.early_stopping_patience} epochs)."
                )
                break

    if last_payload is not None:
        torch.save(last_payload, last_path)

    print(
        f"Training complete. Best checkpoint ({train_cfg.selection_metric}={best_score:.4f}): {best_path}"
    )
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2+3 drowsiness detector per ordered video image sequence."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--temporal-config", type=Path, default=None)
    parser.add_argument(
        "--backbone-arch",
        type=str,
        default=None,
        choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_large", "custom_resnet"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-mode", type=str, default=None, choices=["crop_cache", "raw"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f) or {}

    if "temporal" in raw_yaml and "train" not in raw_yaml:
        # Allow `python engine/train_temporal.py --config configs/temporal_gru.yaml`
        train_cfg = load_train_config(Path("configs/train.yaml"))
        temporal_cfg = load_temporal_config(args.config)
    else:
        train_cfg = load_train_config(args.config)
        t_path = (
            args.temporal_config
            if args.temporal_config is not None
            else Path(train_cfg.temporal_config)
        )
        temporal_cfg = load_temporal_config(t_path)

    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.input_mode is not None:
        overrides["input_mode"] = args.input_mode
    if args.freeze_backbone:
        overrides["freeze_backbone"] = True
    if args.device is not None:
        overrides["device"] = args.device
    if overrides:
        train_cfg = replace(train_cfg, **overrides)

    vision_cfg = load_vision_config(train_cfg.vision_config)
    if args.backbone_arch is not None:
        vision_cfg = replace(
            vision_cfg, backbone=replace(vision_cfg.backbone, arch=args.backbone_arch)
        )
    run_training(train_cfg=train_cfg, vision_cfg=vision_cfg, temporal_cfg=temporal_cfg)


if __name__ == "__main__":
    main()
