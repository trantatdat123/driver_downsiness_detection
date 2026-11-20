"""Test đường dẫn data portable: manifest tương đối, giữ cột split, tìm đúng thư mục dataset tải về."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from download_data import find_dataset_root, is_raw_ready
from preprocessing.build_labels import (
    RAW_DIR_ENV_VAR,
    PreprocessConfig,
    build_video_and_frame_manifests,
    load_preprocess_config,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


#@ Tạo repo giả với data/raw/{Drowsy,Non Drowsy} chứa vài ảnh rỗng
def _make_fake_repo(tmp_path: Path) -> PreprocessConfig:
    files: dict[str, list[str]] = {"Drowsy": ["A0002.png", "A0010.png", "A0001.png"], "Non Drowsy": ["a0001.png"]}
    for cls, names in files.items():
        folder: Path = tmp_path / "data" / "raw" / cls
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).touch()
    return PreprocessConfig(
        raw_data_dir=Path("data/raw"),
        class_folders={"Non Drowsy": 0, "Drowsy": 1},
        image_extensions=(".png",),
        filename_pattern=r"^(?P<video_id>[A-Za-z]+)(?P<rest>.*)$",
        output_labels_csv=Path("data/processed/labels.csv"),
        output_frames_manifest_csv=Path("data/processed/frames_manifest.csv"),
    )


#@ image_path trong manifest phải tương đối theo root và sort số học tự nhiên
def test_manifest_paths_are_relative_and_sorted(tmp_path: Path) -> None:
    cfg: PreprocessConfig = _make_fake_repo(tmp_path)
    frames: list[dict]
    _, frames = build_video_and_frame_manifests(cfg, base_dir=tmp_path)
    paths_a: list[str] = [r["image_path"] for r in frames if r["video_id"] == "A"]
    assert paths_a == ["data/raw/Drowsy/A0001.png", "data/raw/Drowsy/A0002.png", "data/raw/Drowsy/A0010.png"]
    assert all(not Path(r["image_path"]).is_absolute() for r in frames)


#@ Chạy lại build_labels không được xóa cột split đã chia theo subject (ADR-004)
def test_rebuild_preserves_existing_split(tmp_path: Path) -> None:
    cfg: PreprocessConfig = _make_fake_repo(tmp_path)
    labels_csv: Path = tmp_path / cfg.output_labels_csv
    labels_csv.parent.mkdir(parents=True)
    labels_csv.write_text("video_id,label,class_name,num_frames,split\nA,1,Drowsy,3,test\na,0,Non Drowsy,1,test\n")
    videos: list[dict]
    videos, _ = build_video_and_frame_manifests(cfg, base_dir=tmp_path)
    assert {v["video_id"]: v["split"] for v in videos} == {"A": "test", "a": "test"}
    with labels_csv.open() as f:
        assert next(csv.reader(f))[-1] == "split"


#@ Biến môi trường DDD_RAW_DIR ghi đè raw_data_dir trong YAML
def test_env_var_overrides_raw_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RAW_DIR_ENV_VAR, "/mnt/drive/ddd")
    cfg: PreprocessConfig = load_preprocess_config(REPO_ROOT / "configs" / "preprocess.yaml")
    assert cfg.raw_data_dir == Path("/mnt/drive/ddd")


#@ find_dataset_root phải tìm được thư mục lồng nhau kiểu ".../Driver Drowsiness Dataset (DDD)/"
def test_find_dataset_root_nested(tmp_path: Path) -> None:
    nested: Path = tmp_path / "versions" / "1" / "Driver Drowsiness Dataset (DDD)"
    for cls in ("Drowsy", "Non Drowsy"):
        (nested / cls).mkdir(parents=True)
        (nested / cls / "x.png").touch()
    assert find_dataset_root(tmp_path, ["Drowsy", "Non Drowsy"]) == nested
    assert is_raw_ready(nested, ["Drowsy", "Non Drowsy"])
    assert not is_raw_ready(tmp_path, ["Drowsy", "Non Drowsy"])


#@ Manifest đã commit không còn đường dẫn tuyệt đối kiểu G:/ của máy cũ
def test_committed_manifest_is_portable() -> None:
    manifest: Path = REPO_ROOT / "data" / "processed" / "frames_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        paths: list[str] = [row["image_path"] for row in csv.DictReader(f)]
    assert len(paths) == 41793
    assert all(p.startswith("data/raw/") for p in paths)
