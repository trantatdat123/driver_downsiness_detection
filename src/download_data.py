"""Tải Driver Drowsiness Dataset (DDD) từ Kaggle về data/raw/{Drowsy,Non Drowsy}.

Cách dùng (chạy từ root repo):
    python src/download_data.py                  # tải nếu data/raw chưa đủ
    python src/download_data.py --force          # xóa data/raw cũ và tải lại
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import kagglehub

from preprocessing.build_labels import PreprocessConfig, load_preprocess_config, resolve_path

ROOT: Path = Path(__file__).resolve().parents[1]  #?: src/download_data.py → root repo
DEFAULT_CONFIG: Path = ROOT / "configs" / "preprocess.yaml"


#@ Tìm thư mục chứa đủ tất cả class folder (vd: ".../Driver Drowsiness Dataset (DDD)") bên trong bản tải về
def find_dataset_root(download_dir: Path, class_names: list[str]) -> Path:
    candidates: list[Path] = [download_dir, *(p for p in download_dir.rglob("*") if p.is_dir())]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in class_names):
            return candidate
    raise FileNotFoundError(f"Không tìm thấy các thư mục {class_names} trong {download_dir}")


#@ Kiểm tra data/raw đã có đủ class folder và mỗi folder có ít nhất 1 file chưa
def is_raw_ready(raw_dir: Path, class_names: list[str]) -> bool:
    return all((raw_dir / name).is_dir() and any((raw_dir / name).iterdir()) for name in class_names)


#@ Tải dataset qua kagglehub thẳng vào thư mục tạm cạnh raw_dir, rồi move các class folder sang raw_dir
#@ Dùng output_dir + move (không qua cache ~/.cache/kagglehub) để không tốn gấp đôi dung lượng đĩa
def download_to_raw(config: PreprocessConfig, raw_dir: Path, force: bool = False) -> None:
    class_names: list[str] = list(config.class_folders)
    if config.kaggle_dataset is None:
        raise ValueError("Thiếu `kaggle_dataset` trong configs/preprocess.yaml")

    if is_raw_ready(raw_dir, class_names) and not force:
        print(f"[skip] {raw_dir} đã có đủ {class_names}. Dùng --force để tải lại.")
        return

    staging: Path = raw_dir.parent / "_kaggle_download"  #?: nằm trong data/ nên cùng ổ đĩa → move là rename, không copy
    if staging.exists():
        shutil.rmtree(staging)  #? dọn bản tải dở của lần chạy trước bị ngắt
    kagglehub.dataset_download(config.kaggle_dataset, force_download=True, output_dir=str(staging))
    src_root: Path = find_dataset_root(staging, class_names)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in class_names:
        dst: Path = raw_dir / name
        if dst.exists():
            shutil.rmtree(dst)  #! xóa folder cũ để không lẫn ảnh của lần tải trước
        shutil.move(str(src_root / name), str(dst))
        print(f"  {name}: {sum(1 for _ in dst.iterdir())} files → {dst}")
    shutil.rmtree(staging)


#@ Đối chiếu frames_manifest.csv với ảnh đã tải: in số ảnh bị thiếu (0 = khớp hoàn toàn)
def verify_manifest(config: PreprocessConfig) -> int:
    manifest: Path = resolve_path(config.output_frames_manifest_csv, ROOT)
    if not manifest.exists():
        print(f"[warn] Chưa có {manifest}, bỏ qua bước đối chiếu.")
        return 0
    with manifest.open("r", encoding="utf-8", newline="") as f:
        paths: list[Path] = [resolve_path(Path(row["image_path"]), ROOT) for row in csv.DictReader(f)]
    missing: list[Path] = [p for p in paths if not p.exists()]
    print(f"Manifest: {len(paths) - len(missing)}/{len(paths)} ảnh tồn tại, thiếu {len(missing)}.")
    for p in missing[:5]:
        print("  missing:", p)
    return len(missing)


#@ CLI: tải DDD về raw_data_dir trong config rồi đối chiếu với frames_manifest.csv
def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Download DDD dataset into data/raw.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Tải lại và ghi đè data/raw")
    args: argparse.Namespace = parser.parse_args()

    cfg: PreprocessConfig = load_preprocess_config(args.config)
    raw_dir: Path = resolve_path(cfg.raw_data_dir, ROOT)
    download_to_raw(cfg, raw_dir, force=args.force)
    verify_manifest(cfg)


if __name__ == "__main__":
    main()
