# Bộ Nhớ Dự Án (`MEMORY.md`)

## 1. Các Quyết Định Kiến Trúc Đã Chốt (ADRs)
* **ADR-001 (Stage 1 YOLOv8 & Visibility Mask):** Fine-tune YOLOv8n/s với 1 class `eye` (kèm tùy chọn `face` làm mốc `face_cx`) trên cả mắt mở và mắt nhắm. Bit `[v_L, v_R]` là visibility mask (`0.0`/`1.0`) sinh từ hậu xử lý (không phải nhánh phân loại nhắm/mở của YOLO), dùng để nhân triệt tiêu embedding rỗng sau CNN và ghép vào `f_t` kích thước `2D+2`.
* **ADR-002 (Nhãn mức Video & Chiến lược Training Cách 1):** Dữ liệu ảnh được trích xuất từ video với ký hiệu chữ cái bắt đầu làm `video_id` (hai thư mục `non_drowsy` và `drowsy` dùng các chữ cái khác nhau) và gán 1 nhãn nhị phân cho cả video (`0`: `non_drowsy`, `1`: `drowsy`). `TemporalHead` giữ đầu ra `(B, T, 1)` đúng chuẩn kiến trúc; khi train dùng **Many-to-One** trên cửa sổ trượt dài (`logits[:, -1, :]` với `y_clip`) kết hợp `temporal_stride` để phủ `3–6 giây` lịch sử.
* **ADR-003 (Preprocessing & Label Generator):** `preprocessing/build_labels.py` đọc `configs/preprocess.yaml`, quét `data/raw/{non_drowsy,drowsy}`, gom frame theo nhóm chữ cái bắt đầu `^(?P<video_id>[A-Za-z]+)(?P<rest>.*)$`, chặn trùng `video_id` giữa 2 thư mục, sort số học tự nhiên theo các số phía sau (`1..N`) và xuất `data/labels.csv` (`video_id,label,class_name,num_frames`) + `data/frames_manifest.csv` (`video_id,frame_idx,label,image_path`). Có notebook `preprocessing/inspect_and_run.ipynb` (4 cell, không GUI) để chạy và kiểm tra trực quan.

## 2. Nhật Ký Thực Nghiệm
| Ngày | Thử nghiệm | Backbone / Temporal | Split | Frame/Clip Metrics | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| — | Chưa chạy thực nghiệm huấn luyện | — | — | — | Mới khởi tạo cấu trúc `preprocessing/` |
| 2026-11-16 | Rút gọn `preprocessing/` & unit test | — | — | `pytest tests/ -v`: 4 passed | Chốt gom video theo chữ cái đầu, kiểm tra notebook `NB_OK: 4` |

## 3. Danh Sách Việc Cần Làm (TODO)
- [x] Khởi tạo bộ tài liệu `AGENTS.md`, `RULES.md`, `ARCHITECTURE.md`, `MEMORY.md`
- [x] Tạo cấu trúc `preprocessing/`, `configs/preprocess.yaml`, `data/raw/{drowsy,non_drowsy}`, `preprocessing/inspect_and_run.ipynb`, sinh `data/labels.csv` + `data/frames_manifest.csv` (đã bỏ `tests/test_preprocessing_labels.py` theo yêu cầu)
- [ ] Nạp dữ liệu ảnh thực tế vào `data/raw/` và chạy `python -m preprocessing.build_labels --config configs/preprocess.yaml`
- [ ] Xây dựng `data/splitter.py` chia tập độc lập theo `video_id`/`subject_id` (tách riêng người lái/video giữa Train/Val/Test)
- [ ] Xây dựng `models/detector.py` (`YOLOv8EyeCropper`), `models/backbone.py` (`SharedEyeEncoder`), `models/temporal.py` (`CausalGRU` / `CausalTransformer`)
- [ ] Viết test causality (`tests/test_causality.py`) và test mất dấu mắt 0/1/2 (`tests/test_detector_fallback.py`)
