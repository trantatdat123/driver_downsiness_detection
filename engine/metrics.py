"""Reproducibility and binary classification metrics for window/video evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall (Sensitivity), F1-Score, and AUROC."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        }

    pred = (yp >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (yt == 1)))
    fp = int(np.sum((pred == 1) & (yt == 0)))
    fn = int(np.sum((pred == 0) & (yt == 1)))
    tn = int(np.sum((pred == 0) & (yt == 0)))

    accuracy = float((tp + tn) / max(1, yt.size))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    if len(np.unique(yt)) >= 2:
        auroc = float(roc_auc_score(yt, yp))
    else:
        auroc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


def aggregate_video_predictions(
    video_ids: Sequence[str],
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean-aggregate window probabilities into per-video predictions."""
    probs_by_vid: dict[str, list[float]] = defaultdict(list)
    label_by_vid: dict[str, int] = {}

    for vid, yt, yp in zip(video_ids, y_true, y_prob):
        probs_by_vid[vid].append(float(yp))
        label_by_vid[vid] = int(yt)

    ordered_vids = sorted(probs_by_vid)
    vid_labels = np.array([label_by_vid[v] for v in ordered_vids], dtype=np.int64)
    vid_probs = np.array(
        [float(np.mean(probs_by_vid[v])) for v in ordered_vids], dtype=np.float64
    )
    return ordered_vids, vid_labels, vid_probs
