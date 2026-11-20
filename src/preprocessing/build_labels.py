"""Scan class image folders, group frames by leading video letters, and write CSV labels."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RAW_DIR_ENV_VAR: str = "DDD_RAW_DIR"  #?: biến môi trường ghi đè raw_data_dir (vd: trỏ sang Google Drive / ổ ngoài)

FrameItem = tuple[tuple[int, ...], str, int, str, str]  #?: (số thứ tự natural sort, stem, label, class_name, image_path)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration loaded from preprocess.yaml."""

    raw_data_dir: Path
    class_folders: dict[str, int]
    image_extensions: tuple[str, ...]
    filename_pattern: str
    output_labels_csv: Path
    output_frames_manifest_csv: Path
    kaggle_dataset: str | None = None


#@ Đọc configs/preprocess.yaml → PreprocessConfig; raw_data_dir bị ghi đè nếu có biến môi trường DDD_RAW_DIR
def load_preprocess_config(config_path: Path) -> PreprocessConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    raw_dir: str = os.environ.get(RAW_DIR_ENV_VAR) or str(raw["raw_data_dir"])
    return PreprocessConfig(
        raw_data_dir=Path(raw_dir),
        class_folders={str(k): int(v) for k, v in raw["class_folders"].items()},
        image_extensions=tuple(ext.lower() for ext in raw["image_extensions"]),
        filename_pattern=str(raw["filename_pattern"]),
        output_labels_csv=Path(raw["output_labels_csv"]),
        output_frames_manifest_csv=Path(raw["output_frames_manifest_csv"]),
        kaggle_dataset=raw.get("kaggle_dataset"),
    )


#@ Resolve đường dẫn tương đối theo root repo, giữ nguyên nếu đã tuyệt đối
def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


#@ Ghi rows ra CSV, tự tạo thư mục cha nếu chưa có
def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


#@ Đọc cột split (video_id → train/test) từ labels.csv cũ nếu có, để chạy lại không làm mất kết quả chia tập
def _load_existing_splits(labels_csv: Path) -> dict[str, str]:
    if not labels_csv.exists():
        return {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader: csv.DictReader = csv.DictReader(f)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            return {}
        return {row["video_id"]: row["split"] for row in reader if row.get("split")}


#@ Gom ảnh theo nhóm chữ cái đầu (video_id), sort số học tự nhiên, xuất labels.csv + frames_manifest.csv
#@ image_path được ghi tương đối theo root repo (vd: data/raw/Drowsy/A0001.png) để chạy được trên mọi máy
#@ Cột split của labels.csv cũ (ADR-004) được giữ lại; video mới chưa có split sẽ để trống
def build_video_and_frame_manifests(
    config: PreprocessConfig,
    base_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root: Path = (base_dir or Path.cwd()).resolve()
    raw_dir: Path = resolve_path(config.raw_data_dir, root).resolve()
    pattern: re.Pattern[str] = re.compile(config.filename_pattern)
    valid_exts: set[str] = set(config.image_extensions)

    grouped: dict[str, list[FrameItem]] = {}
    labels_by_vid: dict[str, int] = {}

    for class_name, label in config.class_folders.items():
        folder: Path = raw_dir / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}. Chạy `python src/download_data.py` trước.")

        for entry in folder.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in valid_exts:
                continue

            match: re.Match[str] | None = pattern.match(entry.stem)
            if match is None:
                raise ValueError(f"Invalid filename '{entry.name}' in '{folder}'.")

            video_id: str = match.group("video_id")
            if video_id in labels_by_vid and labels_by_vid[video_id] != label:
                raise ValueError(f"Video '{video_id}' appears in multiple class folders.")
            labels_by_vid[video_id] = label

            nums: tuple[int, ...] = tuple(int(x) for x in re.findall(r"\d+", match.groupdict().get("rest", "")))
            rel_path: str
            try:
                rel_path = entry.relative_to(root).as_posix()
            except ValueError:
                rel_path = entry.as_posix()  #! raw_dir nằm ngoài repo (DDD_RAW_DIR) → manifest chứa đường dẫn tuyệt đối, không portable

            grouped.setdefault(video_id, []).append((nums, entry.stem, label, class_name, rel_path))

    labels_csv: Path = resolve_path(config.output_labels_csv, root)
    splits: dict[str, str] = _load_existing_splits(labels_csv)

    video_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for video_id in sorted(grouped):
        items: list[FrameItem] = sorted(grouped[video_id], key=lambda x: (x[0], x[1]))
        vid_label: int = items[0][2]
        vid_class: str = items[0][3]
        video_rows.append(
            {
                "video_id": video_id,
                "label": vid_label,
                "class_name": vid_class,
                "num_frames": len(items),
                "split": splits.get(video_id, ""),
            }
        )
        for frame_idx, (_, _, _, _, image_path) in enumerate(items, start=1):
            frame_rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "label": vid_label,
                    "image_path": image_path,
                }
            )

    _write_csv(labels_csv, ["video_id", "label", "class_name", "num_frames", "split"], video_rows)
    _write_csv(
        resolve_path(config.output_frames_manifest_csv, root),
        ["video_id", "frame_idx", "label", "image_path"],
        frame_rows,
    )
    return video_rows, frame_rows


#@ CLI entrypoint: python src/preprocessing/build_labels.py --config configs/preprocess.yaml
def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Build video label CSV from image folders.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    args: argparse.Namespace = parser.parse_args()

    cfg: PreprocessConfig = load_preprocess_config(args.config)
    videos: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    videos, frames = build_video_and_frame_manifests(cfg)
    print(f"Processed {len(videos)} videos ({len(frames)} frames).")


if __name__ == "__main__":
    main()
