# Driver Drowsiness Detection

Phát hiện buồn ngủ khi lái xe bằng YOLOv8 + CNN + Temporal Head. Kiến trúc chi tiết xem [ARCHITECTURE.md](ARCHITECTURE.md), trạng thái dự án xem [MEMORY.md](MEMORY.md).

---

## 1. Cài đặt môi trường (lần đầu clone repo)

### Bước 1: Tạo và kích hoạt virtual environment

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell / CMD)
```

**Giải thích:**
- Dùng **Python 3.12**. Bản 3.13/3.14 chưa được mọi thư viện ML hỗ trợ đầy đủ.
- `.venv/` đã nằm trong `.gitignore` nên không bị commit lên git.
- Sau khi kích hoạt, đầu dòng terminal sẽ hiện `(.venv)`. Kiểm tra bằng `which python` (Windows: `where python`), kết quả phải trỏ vào `.venv/`.

### Bước 2: Cài thư viện theo phần cứng của máy

| Máy | Lệnh |
|---|---|
| Mac (Apple Silicon, dùng GPU qua MPS) / máy không có GPU NVIDIA | `pip install -r requirements/cpu.txt` |
| Windows / Linux có GPU NVIDIA | `pip install -r requirements/cuda.txt` |

**Giải thích:**
- Cả 2 file đều include `requirements/base.txt` (các thư viện chung), chỉ khác nhau ở bản **torch**.
- `cuda.txt` cài `torch==2.15.0+cu132`, cần driver NVIDIA hỗ trợ CUDA ≥ 13.2. Chạy `nvidia-smi` xem dòng `CUDA Version`. Nếu driver cũ hơn thì cập nhật driver. Nếu driver mới hơn và muốn dùng bản `cu134`, sửa mọi chỗ `cu132` trong `cuda.txt` (index URL, torch, torchvision) thành `cu134`.

### Bước 3: Kiểm tra cài đặt

```bash
python -c "import torch, ultralytics, cv2; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| MPS', torch.backends.mps.is_available())"
```

**Giải thích:** máy NVIDIA phải ra `CUDA True`, máy Mac phải ra `MPS True`. Nếu báo `ModuleNotFoundError` thì có thể bạn chưa kích hoạt `.venv` (xem Bước 1).

### Bước 4: Tải dữ liệu

```bash
python src/download_data.py
```

**Giải thích:**
- Script tải [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) từ Kaggle (~3 GB zip, ~2.7 GB sau giải nén) về `data/raw/{Drowsy,Non Drowsy}/`.
- Script tải zip vào `data/_kaggle_download/`, giải nén rồi chuyển ảnh sang `data/raw/` và xoá thư mục tạm, nên cần trống khoảng **6 GB** trong lúc tải.
- Cuối cùng script đối chiếu với `data/processed/frames_manifest.csv`. Kết quả đúng là `41793/41793 ảnh tồn tại, thiếu 0`.
- Nếu đã có data thì script bỏ qua. Dùng `--force` để tải lại.
- Nếu muốn để data ở chỗ khác (ổ ngoài, Google Drive…), đặt biến môi trường `DDD_RAW_DIR=/đường/dẫn/tới/folder` thay vì sửa code. Lưu ý là khi đó cần chạy lại `build_labels.py`, và manifest sẽ chứa đường dẫn tuyệt đối.

### Bước 5: Chạy test

```bash
pytest tests/ -v
```

---

## 2. Cài thêm thư viện & freeze `requirements`

### Cấu trúc file

```text
requirements/
├── base.txt   # mọi thư viện chung, đã ghim version (sinh bằng pip freeze, KHÔNG có torch)
├── cpu.txt    # -r base.txt + torch/torchvision bản thường (Mac MPS / CPU)
└── cuda.txt   # -r base.txt + torch/torchvision bản CUDA (index download.pytorch.org)
```

Torch được tách riêng vì mỗi máy cần một bản build khác nhau (`2.15.0` trên Mac, `2.15.0+cu132` trên máy NVIDIA). Nếu freeze cả torch vào một file, máy khác sẽ cài nhầm bản CPU hoặc không cài được.

### Khi cài thêm một thư viện mới

```bash
pip install <tên-package>
```

```bash
pip freeze | grep -viE "^(torch|torchvision|triton)==|^nvidia-" > requirements/base.txt
```

Windows PowerShell (không có `grep`):

```powershell
pip freeze | Select-String -NotMatch -Pattern '^(torch|torchvision|triton)==|^nvidia-' | ForEach-Object { $_.Line } | Set-Content requirements/base.txt
```

**Giải thích:**
- Lệnh `pip freeze` liệt kê mọi package đang cài trong `.venv` kèm version chính xác.
- `grep -viE ...` bỏ đi torch, torchvision, triton và các gói `nvidia-*`. Đây là các gói phụ thuộc phần cứng, đã được quản lý trong `cpu.txt` và `cuda.txt`.
- Nhớ **kích hoạt `.venv` trước khi freeze**. Nếu không, bạn sẽ freeze nhầm package của Python hệ thống.
- Commit `requirements/base.txt` cùng với code dùng thư viện mới.

### Khi nâng version torch

Sửa tay version trong **cả** `cpu.txt` **và** `cuda.txt` (torch và torchvision phải đi theo cặp tương thích). Trước khi sửa, kiểm tra bản CUDA mới có trên [download.pytorch.org/whl/torch](https://download.pytorch.org/whl/torch/).

### Khi kéo code mới về (đồng bộ với người khác)

```bash
git pull
```

```bash
pip install -r requirements/cpu.txt      # hoặc requirements/cuda.txt
```

**Giải thích:** chạy lại lệnh cài mỗi khi `requirements/` có thay đổi. Pip sẽ chỉ cài thêm hoặc nâng cấp những gì khác đi, các package đã đúng version thì bỏ qua.

---

## 3. Cấu trúc dữ liệu

```text
data/
├── raw/                    # ảnh gốc (gitignored), tải bằng src/download_data.py
│   ├── Drowsy/             # 22,348 ảnh
│   └── Non Drowsy/         # 19,445 ảnh
└── processed/
    ├── labels.csv          # video_id, label, class_name, num_frames, split (train/test theo người lái)
    └── frames_manifest.csv # video_id, frame_idx, label, image_path (tương đối theo root repo)
```

Sinh lại CSV từ ảnh (giữ nguyên cột `split` đã chia):

```bash
python src/preprocessing/build_labels.py --config configs/preprocess.yaml
```
