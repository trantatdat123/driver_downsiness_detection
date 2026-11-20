"""Scan class image folders, group frames by leading video letters, and write CSV labels."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration loaded from preprocess.yaml."""

    raw_data_dir: Path
    class_folders: dict[str, int]
    image_extensions: tuple[str, ...]
    filename_pattern: str
    output_labels_csv: Path
    output_frames_manifest_csv: Path


def load_preprocess_config(config_path: Path) -> PreprocessConfig:
    """Load preprocessing configuration from a YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return PreprocessConfig(
        raw_data_dir=Path(raw["raw_data_dir"]),
        class_folders={str(k): int(v) for k, v in raw["class_folders"].items()},
        image_extensions=tuple(ext.lower() for ext in raw["image_extensions"]),
        filename_pattern=str(raw["filename_pattern"]),
        output_labels_csv=Path(raw["output_labels_csv"]),
        output_frames_manifest_csv=Path(raw["output_frames_manifest_csv"]),
    )


def _resolve_path(path: Path, root: Path) -> Path:
    """Resolve relative paths against root."""
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_video_and_frame_manifests(
    config: PreprocessConfig,
    base_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group images by leading letters, sort numerically, and export video/frame CSVs."""
    root = base_dir or Path.cwd()
    raw_dir = _resolve_path(config.raw_data_dir, root)
    pattern = re.compile(config.filename_pattern)
    valid_exts = set(config.image_extensions)

    grouped: dict[str, list[tuple[tuple[int, ...], str, int, str, str]]] = {}
    labels_by_vid: dict[str, int] = {}

    for class_name, label in config.class_folders.items():
        folder = raw_dir / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for entry in folder.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in valid_exts:
                continue

            match = pattern.match(entry.stem)
            if match is None:
                raise ValueError(f"Invalid filename '{entry.name}' in '{folder}'.")

            video_id = match.group("video_id")
            if video_id in labels_by_vid and labels_by_vid[video_id] != label:
                raise ValueError(f"Video '{video_id}' appears in multiple class folders.")
            labels_by_vid[video_id] = label

            nums = tuple(int(x) for x in re.findall(r"\d+", match.groupdict().get("rest", "")))
            try:
                rel_path = entry.relative_to(root).as_posix()
            except ValueError:
                rel_path = entry.as_posix()

            grouped.setdefault(video_id, []).append((nums, entry.stem, label, class_name, rel_path))

    video_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for video_id in sorted(grouped):
        items = sorted(grouped[video_id], key=lambda x: (x[0], x[1]))
        label, class_name = items[0][2], items[0][3]
        video_rows.append(
            {
                "video_id": video_id,
                "label": label,
                "class_name": class_name,
                "num_frames": len(items),
            }
        )
        for frame_idx, (_, _, _, _, image_path) in enumerate(items, start=1):
            frame_rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "label": label,
                    "image_path": image_path,
                }
            )

    _write_csv(
        _resolve_path(config.output_labels_csv, root),
        ["video_id", "label", "class_name", "num_frames"],
        video_rows,
    )
    _write_csv(
        _resolve_path(config.output_frames_manifest_csv, root),
        ["video_id", "frame_idx", "label", "image_path"],
        frame_rows,
    )
    return video_rows, frame_rows


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build video label CSV from image folders.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    videos, frames = build_video_and_frame_manifests(cfg)
    print(f"Processed {len(videos)} videos ({len(frames)} frames).")


if __name__ == "__main__":
    main()
