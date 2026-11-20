"""Dataset, subject-independent train/val/test split, and per-video sliding windows."""

from __future__ import annotations

import csv
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VideoRecord:
    """Ordered image sequence belonging to a single video_id."""

    video_id: str
    subject_id: str
    label: int
    split: str
    image_paths: tuple[str, ...]
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class WindowSpec:
    """Sliding window index specification within a single video_id."""

    video_id: str
    label: int
    positions: tuple[int, ...]


def load_video_records(
    labels_csv: str | Path,
    frames_manifest_csv: str | Path,
) -> list[VideoRecord]:
    """Load per-video ordered image sequences from labels.csv and frames_manifest.csv."""
    labels_path = Path(labels_csv)
    manifest_path = Path(frames_manifest_csv)
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frames manifest CSV: {manifest_path}")

    video_meta: dict[str, tuple[int, str]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vid = str(row["video_id"])
            video_meta[vid] = (int(row["label"]), str(row["split"]).strip().lower())

    grouped_frames: dict[str, list[tuple[int, int, str]]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vid = str(row["video_id"])
            frame_idx = int(row["frame_idx"])
            label = int(row["label"])
            image_path = str(row["image_path"])
            grouped_frames.setdefault(vid, []).append((frame_idx, label, image_path))

    records: list[VideoRecord] = []
    for vid in sorted(video_meta):
        expected_label, split = video_meta[vid]
        frames = grouped_frames.get(vid, [])
        if not frames:
            raise ValueError(f"Video '{vid}' in labels.csv has no rows in frames_manifest.csv")
        frames.sort(key=lambda item: item[0])

        for _, frame_label, _ in frames:
            if frame_label != expected_label:
                raise ValueError(
                    f"Label mismatch for video '{vid}': {frame_label} vs {expected_label}"
                )

        records.append(
            VideoRecord(
                video_id=vid,
                subject_id=vid.lower(),
                label=expected_label,
                split=split,
                image_paths=tuple(item[2] for item in frames),
                frame_indices=tuple(item[0] for item in frames),
            )
        )
    return records


def split_train_val_videos(
    records: Sequence[VideoRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
    """Split train records into subject-independent train/val sets, keeping test untouched."""
    train_pool = [r for r in records if r.split == "train"]
    test_videos = [r for r in records if r.split == "test"]

    by_subject: dict[str, list[VideoRecord]] = {}
    for rec in train_pool:
        by_subject.setdefault(rec.subject_id, []).append(rec)

    # Stratify subjects by their label composition (e.g., both classes vs drowsy-only)
    strata: dict[tuple[int, ...], list[str]] = {}
    for subj_id in sorted(by_subject):
        profile = tuple(sorted({r.label for r in by_subject[subj_id]}))
        strata.setdefault(profile, []).append(subj_id)

    rng = random.Random(seed)
    val_subjects: set[str] = set()
    for profile in sorted(strata):
        subjs = list(strata[profile])
        rng.shuffle(subjs)
        if len(subjs) <= 1:
            continue
        n_val = max(1, int(round(len(subjs) * val_ratio)))
        n_val = min(n_val, len(subjs) - 1)
        val_subjects.update(subjs[:n_val])

    train_subjects = set(by_subject) - val_subjects
    test_subjects = {r.subject_id for r in test_videos}

    if not train_subjects.isdisjoint(val_subjects):
        raise AssertionError("Subject leakage detected between train and val splits")
    if not (train_subjects | val_subjects).isdisjoint(test_subjects):
        raise AssertionError("Subject leakage detected with test split")

    train_videos = [r for r in train_pool if r.subject_id in train_subjects]
    val_videos = [r for r in train_pool if r.subject_id in val_subjects]
    return train_videos, val_videos, test_videos


def build_video_windows(
    records: Sequence[VideoRecord],
    window_size: int,
    temporal_stride: int,
    window_step: int,
) -> list[WindowSpec]:
    """Create sliding windows strictly within each video_id (never crossing videos)."""
    if window_size <= 0 or temporal_stride <= 0 or window_step <= 0:
        raise ValueError("window_size, temporal_stride, and window_step must be > 0")

    span = (window_size - 1) * temporal_stride + 1
    windows: list[WindowSpec] = []

    for rec in records:
        n = len(rec.image_paths)
        if n == 0:
            continue

        if n < span:
            end = n - 1
            positions = tuple(
                max(0, end - (window_size - 1 - k) * temporal_stride)
                for k in range(window_size)
            )
            windows.append(
                WindowSpec(video_id=rec.video_id, label=rec.label, positions=positions)
            )
            continue

        end_indices = list(range(span - 1, n, window_step))
        if end_indices[-1] != n - 1:
            end_indices.append(n - 1)

        for end in end_indices:
            positions = tuple(
                end - (window_size - 1 - k) * temporal_stride
                for k in range(window_size)
            )
            windows.append(
                WindowSpec(video_id=rec.video_id, label=rec.label, positions=positions)
            )

    return windows


class WindowEyeAugmenter:
    """Temporally consistent data augmentation for paired eye-crop windows.

    Applies identical photometric and geometric perturbations across all `W` frames
    in a window so the CNN cannot memorize static lighting, skin tone, or crop framing
    while preserving temporal eyelid dynamics and left/right spatial slot semantics.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        grayscale_prob: float = 0.15,
        affine_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        eye_dropout_prob: float = 0.1,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.affine_prob = affine_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.eye_dropout_prob = eye_dropout_prob

    def __call__(
        self, crops: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment `(W, 2, 3, H, W_eye)` float32 crops in `[0, 1]` and `(W, 2)` mask."""
        # 1. Window-consistent horizontal flip + left/right slot swap
        if random.random() < self.hflip_prob:
            # Flip width dim and swap slot 0 <-> slot 1: (W, 2, 3, H, W_eye)
            crops = torch.flip(crops, dims=[-1])[:, [1, 0], :, :, :]
            # Swap visibility mask slots: (W, 2) -> (W, 2)
            mask = mask[:, [1, 0]]

        # 2. Window-consistent photometric jitter (brightness, contrast, saturation)
        if random.random() < self.color_jitter_prob:
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.7, 1.3)

            crops = crops * brightness
            # Spatial mean per frame/eye/channel: (W, 2, 3, 1, 1)
            mean_val = crops.mean(dim=(-2, -1), keepdim=True)
            crops = (crops - mean_val) * contrast + mean_val

            # Luminance: (W, 2, 1, H, W_eye)
            gray = (
                0.299 * crops[:, :, 0:1, :, :]
                + 0.587 * crops[:, :, 1:2, :, :]
                + 0.114 * crops[:, :, 2:3, :, :]
            )
            crops = crops * saturation + gray * (1.0 - saturation)

        # 3. Random grayscale to suppress skin-tone / color shortcuts
        if random.random() < self.grayscale_prob:
            # (W, 2, 1, H, W_eye) -> (W, 2, 3, H, W_eye)
            gray = (
                0.299 * crops[:, :, 0:1, :, :]
                + 0.587 * crops[:, :, 1:2, :, :]
                + 0.114 * crops[:, :, 2:3, :, :]
            )
            crops = gray.expand_as(crops).clone()

        # 4. Small window-consistent affine perturbation (scale, rotation, translation)
        if random.random() < self.affine_prob:
            w_len, num_eyes, c, h, w_eye = crops.shape
            angle_rad = math.radians(random.uniform(-8.0, 8.0))
            scale = random.uniform(0.92, 1.08)
            tx = random.uniform(-0.08, 0.08)
            ty = random.uniform(-0.08, 0.08)

            cos_a = math.cos(angle_rad) / scale
            sin_a = math.sin(angle_rad) / scale
            # (1, 2, 3) -> (W * 2, 2, 3)
            theta = torch.tensor(
                [[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]],
                dtype=crops.dtype,
            ).expand(w_len * num_eyes, 2, 3)

            # (W, 2, 3, H, W_eye) -> (W * 2, 3, H, W_eye)
            flat = crops.reshape(w_len * num_eyes, c, h, w_eye)
            grid = F.affine_grid(theta, size=flat.size(), align_corners=False)
            warped = F.grid_sample(
                flat, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            # (W * 2, 3, H, W_eye) -> (W, 2, 3, H, W_eye)
            crops = warped.reshape(w_len, num_eyes, c, h, w_eye)

        # 5. Mild Gaussian sensor noise
        if random.random() < self.noise_prob:
            crops = crops + torch.randn_like(crops) * self.noise_std

        crops = crops.clamp(0.0, 1.0)

        # 6. Random single-eye occlusion dropout to regularize and train missing-eye robustness
        if random.random() < self.eye_dropout_prob:
            drop_slot = random.randint(0, 1)
            mask = mask.clone()
            mask[:, drop_slot] = 0.0

        return crops, mask


def preload_eye_crop_cache(
    video_ids: Sequence[str],
    cache_dir: str | Path,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Preload per-video uint8 eye-crop and float32 mask tensors into RAM once."""
    root = Path(cache_dir)
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for vid in sorted(set(video_ids)):
        cache_path = root / f"{vid}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing eye-crop cache for video '{vid}': {cache_path}. "
                "Run tools/extract_features.py first or enable auto_build_cache."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        cache[vid] = (payload["eye_crops"], payload["eye_mask"])
    return cache


class BatchEyePreprocessor(nn.Module):
    """GPU-vectorized uint8->float32 conversion, window-consistent augmentation, and normalization.

    Input:
        crops: (B, W, 2, 3, H, W_eye) uint8 [0, 255] or float32 [0, 1]
        mask:  (B, W, 2) float32 in {0.0, 1.0}
    Output:
        crops: (B, W, 2, 3, H, W_eye) normalized float32 with missing eyes zeroed
        mask:  (B, W, 2) float32 in {0.0, 1.0}
    """

    def __init__(
        self,
        normalize_mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        augment: bool = True,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        grayscale_prob: float = 0.15,
        affine_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        eye_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.affine_prob = affine_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.eye_dropout_prob = eye_dropout_prob

        if normalize_mean is not None and normalize_std is not None:
            # (3,) -> (1, 1, 1, 3, 1, 1) for (B, W, 2, 3, H, W_eye)
            mean_t = torch.tensor(normalize_mean, dtype=torch.float32).view(1, 1, 1, 3, 1, 1)
            std_t = torch.tensor(normalize_std, dtype=torch.float32).view(1, 1, 1, 3, 1, 1)
            self.register_buffer("norm_mean", mean_t)
            self.register_buffer("norm_std", std_t)
        else:
            self.norm_mean: torch.Tensor | None = None
            self.norm_std: torch.Tensor | None = None

    @torch.no_grad()
    def forward(
        self, crops: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert uint8 crops, optionally augment per-window on GPU, normalize, and zero-mask."""
        b, w_len, num_eyes, c, h, w_eye = crops.shape
        device = crops.device

        if crops.dtype == torch.uint8:
            # (B, W, 2, 3, H, W_eye) uint8 -> float32 [0, 1]
            crops = crops.to(dtype=torch.float32).mul_(1.0 / 255.0)
        else:
            crops = crops.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)

        if self.training and self.augment:
            # 1. Window-consistent random horizontal flip + L/R slot swap (only selected samples)
            if self.hflip_prob > 0.0:
                flip_idx = (torch.rand(b, device=device) < self.hflip_prob).nonzero(as_tuple=True)[0]
                if flip_idx.numel() > 0:
                    crops[flip_idx] = torch.flip(crops[flip_idx], dims=[-1])[:, :, [1, 0], :, :, :]
                    mask[flip_idx] = mask[flip_idx][:, :, [1, 0]]

            # 2. Window-consistent color/lighting jitter (only selected samples)
            if self.color_jitter_prob > 0.0:
                col_idx = (torch.rand(b, device=device) < self.color_jitter_prob).nonzero(as_tuple=True)[0]
                n_col = int(col_idx.numel())
                if n_col > 0:
                    sub = crops[col_idx]
                    brightness = torch.empty((n_col, 1, 1, 1, 1, 1), device=device).uniform_(0.7, 1.3)
                    contrast = torch.empty((n_col, 1, 1, 1, 1, 1), device=device).uniform_(0.7, 1.3)
                    saturation = torch.empty((n_col, 1, 1, 1, 1, 1), device=device).uniform_(0.7, 1.3)

                    sub = sub * brightness
                    mean_val = sub.mean(dim=(-2, -1), keepdim=True)
                    sub = (sub - mean_val) * contrast + mean_val
                    gray = (
                        0.299 * sub[:, :, :, 0:1, :, :]
                        + 0.587 * sub[:, :, :, 1:2, :, :]
                        + 0.114 * sub[:, :, :, 2:3, :, :]
                    )
                    crops[col_idx] = sub * saturation + gray * (1.0 - saturation)

            # 3. Window-consistent random grayscale (only selected samples)
            if self.grayscale_prob > 0.0:
                gray_idx = (torch.rand(b, device=device) < self.grayscale_prob).nonzero(as_tuple=True)[0]
                if gray_idx.numel() > 0:
                    sub = crops[gray_idx]
                    gray = (
                        0.299 * sub[:, :, :, 0:1, :, :]
                        + 0.587 * sub[:, :, :, 1:2, :, :]
                        + 0.114 * sub[:, :, :, 2:3, :, :]
                    )
                    crops[gray_idx] = gray.expand_as(sub)

            # 4. Window-consistent small spatial affine perturbation (only selected samples)
            if self.affine_prob > 0.0:
                aff_idx = (torch.rand(b, device=device) < self.affine_prob).nonzero(as_tuple=True)[0]
                n_aff = int(aff_idx.numel())
                if n_aff > 0:
                    angle = torch.empty(n_aff, device=device).uniform_(
                        -math.radians(8.0), math.radians(8.0)
                    )
                    scale = torch.empty(n_aff, device=device).uniform_(0.92, 1.08)
                    tx = torch.empty(n_aff, device=device).uniform_(-0.08, 0.08)
                    ty = torch.empty(n_aff, device=device).uniform_(-0.08, 0.08)

                    cos_a = torch.cos(angle) / scale
                    sin_a = torch.sin(angle) / scale
                    # (N_aff, 2, 3)
                    theta = torch.zeros((n_aff, 2, 3), dtype=crops.dtype, device=device)
                    theta[:, 0, 0] = cos_a
                    theta[:, 0, 1] = -sin_a
                    theta[:, 0, 2] = tx
                    theta[:, 1, 0] = sin_a
                    theta[:, 1, 1] = cos_a
                    theta[:, 1, 2] = ty

                    # (N_aff, 2, 3) -> (N_aff * W * 2, 2, 3)
                    theta_exp = (
                        theta.view(n_aff, 1, 1, 2, 3)
                        .expand(n_aff, w_len, num_eyes, 2, 3)
                        .reshape(n_aff * w_len * num_eyes, 2, 3)
                    )
                    # (N_aff, W, 2, 3, H, W_eye) -> (N_aff * W * 2, 3, H, W_eye)
                    flat = crops[aff_idx].reshape(n_aff * w_len * num_eyes, c, h, w_eye)
                    grid = F.affine_grid(theta_exp, size=flat.size(), align_corners=False)
                    warped = F.grid_sample(
                        flat, grid, mode="bilinear", padding_mode="border", align_corners=False
                    )
                    # (N_aff * W * 2, 3, H, W_eye) -> (N_aff, W, 2, 3, H, W_eye)
                    crops[aff_idx] = warped.reshape(n_aff, w_len, num_eyes, c, h, w_eye)

            # 5. Mild Gaussian sensor noise (only selected samples)
            if self.noise_prob > 0.0:
                noise_idx = (torch.rand(b, device=device) < self.noise_prob).nonzero(as_tuple=True)[0]
                if noise_idx.numel() > 0:
                    sub = crops[noise_idx]
                    crops[noise_idx] = sub + torch.randn_like(sub) * self.noise_std

            crops.clamp_(0.0, 1.0)

            # 6. Random single-eye occlusion dropout
            if self.eye_dropout_prob > 0.0:
                drop_idx = (torch.rand(b, device=device) < self.eye_dropout_prob).nonzero(as_tuple=True)[0]
                if drop_idx.numel() > 0:
                    drop_slots = torch.randint(0, 2, (drop_idx.numel(),), device=device)
                    mask[drop_idx, :, drop_slots] = 0.0

        if self.norm_mean is not None and self.norm_std is not None:
            crops = (crops - self.norm_mean) / self.norm_std

        # Guarantee zero-fill on missing eye slots: (B, W, 2) -> (B, W, 2, 1, 1, 1)
        crops = crops * mask.reshape(b, w_len, num_eyes, 1, 1, 1)
        return crops, mask


class VideoEyeCropWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Load sliding windows from per-video cached eye-crop `.pt` files."""

    def __init__(
        self,
        windows: Sequence[WindowSpec],
        cache_dir: str | Path,
        normalize_mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        max_cached_videos: int = 64,
        augment: bool = False,
        temporal_jitter: int = 0,
        return_uint8: bool = False,
        preloaded_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.windows = list(windows)
        self.cache_dir = Path(cache_dir)
        self.max_cached_videos = max(1, int(max_cached_videos))
        self.return_uint8 = bool(return_uint8)
        self.augmenter = WindowEyeAugmenter() if (augment and not self.return_uint8) else None
        self.temporal_jitter = max(0, int(temporal_jitter))
        self._preloaded_cache = preloaded_cache
        self._mem_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

        if normalize_mean is not None and normalize_std is not None:
            # (3,) -> (1, 1, 3, 1, 1) for (W, 2, 3, H_eye, W_eye)
            self._mean: torch.Tensor | None = torch.tensor(
                normalize_mean, dtype=torch.float32
            ).view(1, 1, 3, 1, 1)
            self._std: torch.Tensor | None = torch.tensor(
                normalize_std, dtype=torch.float32
            ).view(1, 1, 3, 1, 1)
        else:
            self._mean = None
            self._std = None

    def _load_video_cache(self, video_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._preloaded_cache is not None and video_id in self._preloaded_cache:
            return self._preloaded_cache[video_id]

        if video_id in self._mem_cache:
            self._mem_cache.move_to_end(video_id)
            return self._mem_cache[video_id]

        cache_path = self.cache_dir / f"{video_id}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing eye-crop cache for video '{video_id}': {cache_path}. "
                "Run tools/extract_features.py first or enable auto_build_cache."
            )

        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        entry = (payload["eye_crops"], payload["eye_mask"])
        self._mem_cache[video_id] = entry
        if len(self._mem_cache) > self.max_cached_videos:
            self._mem_cache.popitem(last=False)
        return entry

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        spec = self.windows[idx]
        video_crops, video_mask = self._load_video_cache(spec.video_id)
        num_frames = int(video_crops.shape[0])
        pos_idx = torch.tensor(spec.positions, dtype=torch.long)

        if self.temporal_jitter > 0 and num_frames > 1:
            shift = random.randint(-self.temporal_jitter, self.temporal_jitter)
            pos_idx = (pos_idx + shift).clamp(0, num_frames - 1)

        # (W, 2)
        mask = video_mask[pos_idx].to(dtype=torch.float32)
        label = torch.tensor(float(spec.label), dtype=torch.float32)

        if self.return_uint8:
            # Keep compact uint8 (W, 2, 3, H_eye, W_eye) for fast 4x smaller PCIe transfer to GPU
            crops_u8 = video_crops[pos_idx]
            return crops_u8, mask, label, spec.video_id

        # (W, 2, 3, H_eye, W_eye) uint8 -> float32 [0, 1]
        crops = video_crops[pos_idx].to(dtype=torch.float32) / 255.0

        if self.augmenter is not None:
            crops, mask = self.augmenter(crops, mask)

        if self._mean is not None and self._std is not None:
            crops = (crops - self._mean) / self._std

        # Guarantee zero-fill on missing eye slots: (W, 2) -> (W, 2, 1, 1, 1)
        crops = crops * mask.reshape(mask.shape[0], 2, 1, 1, 1)
        return crops, mask, label, spec.video_id


class VideoRawWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load sliding windows directly from ordered raw image files per video_id."""

    def __init__(
        self,
        records: Sequence[VideoRecord],
        windows: Sequence[WindowSpec],
        color_format: str = "RGB",
    ) -> None:
        self.records_by_id = {r.video_id: r for r in records}
        self.windows = list(windows)
        self.color_format = color_format.upper()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        spec = self.windows[idx]
        rec = self.records_by_id[spec.video_id]
        frames_np: list[np.ndarray] = []

        for pos in spec.positions:
            img_path = rec.image_paths[pos]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Unable to read image: {img_path}")
            if self.color_format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_np.append(img)

        # (W, H, W_img, 3) uint8
        stacked = np.stack(frames_np, axis=0)
        frames_tensor = torch.from_numpy(stacked)
        label = torch.tensor(float(spec.label), dtype=torch.float32)
        return frames_tensor, label, spec.video_id
