# Hướng Dẫn Vận Hành Dành Cho AI Coding Agents (`AGENTS.md`)

Tài liệu này quy định **quy trình hành động, phạm vi trách nhiệm và các yêu cầu bắt buộc** đối với mọi AI coding agent khi làm việc trên repository này. Chi tiết kiến trúc xem tại `ARCHITECTURE.md`, luật viết code xem tại `RULES.md`, và trạng thái dự án xem tại `MEMORY.md`.

---

## 1. Quy Trình Hành Động Bắt Buộc (Agent Workflow)

Trước khi viết hoặc chỉnh sửa bất kỳ dòng code nào, agent **BẮT BUỘC** thực hiện tuần tự 5 bước sau:

1. **Đọc Ngữ Cảnh & Trạng Thái:**
   * Đọc `MEMORY.md` để nắm các quyết định kiến trúc đã chốt (ADRs), các lỗi biên (edge cases) đã biết và danh sách việc cần làm hiện tại.
   * Đọc `RULES.md` và `ARCHITECTURE.md` để đảm bảo tuân thủ chuẩn tensor shape và tính nhân quả (causality).
2. **Kiểm Tra Cấu Hình & Cấu Trúc Thư Mục:**
   * Kiểm tra file cấu hình trong `configs/` trước khi thêm tham số mới. Không tự ý tạo file rời rạc ngoài cấu trúc chuẩn (`configs/`, `data/`, `models/`, `engine/`, `inference/`, `tools/`, `tests/`).
3. **Lập Kế Hoạch & Xác Nhận Thay Đổi Lớn:**
   * Với các tác vụ thay đổi cấu trúc model, định dạng dữ liệu cache hoặc pipeline huấn luyện, hãy trình bày ngắn gọn kế hoạch thay đổi (file bị ảnh hưởng, shape đầu vào/đầu ra) trước khi thực thi.
4. **Thực Thi Code & Kiểm Thử Tự Động:**
   * Viết code kèm unit test tương ứng trong `tests/`.
   * Chạy kiểm thử (`pytest`) để xác minh:
     * Không rò rỉ dữ liệu tương lai (future-frame leakage) trong mô hình chuỗi thời gian.
     * Pipeline không crash khi YOLOv8 phát hiện `0`, `1` hoặc `2` mắt.
     * Xuất thử nghiệm sang ONNX thành công nếu có chỉnh sửa kiến trúc model.
5. **Cập Nhật Bộ Nhớ Dự Án (`MEMORY.md`):**
   * Sau khi hoàn thành một tính năng, sửa một lỗi phức tạp hoặc chạy xong một thực nghiệm, **BẮT BUỘC** cập nhật kết quả vào bảng thực nghiệm và đánh dấu hoàn thành trong mục TODO của `MEMORY.md`.

---

## 2. Yêu Cầu & Trách Nhiệm Của Agent Theo Từng Tác Vụ

### Khi xây dựng hoặc sửa đổi Model (`models/`)
* Luôn giữ tính module hóa: tách biệt `YOLOv8EyeCropper`, `SharedEyeEncoder` (CNN) và `TemporalHead` (`CausalGRU` / `CausalTransformer`).
* Mọi tensor biến đổi phức tạp phải có comment chú thích kích thước (shape) ngay bên cạnh.
* Đảm bảo đầu ra luôn là dự đoán theo từng khung hình `(B, T, 1)` phục vụ bài toán phân loại chuỗi many-to-many.

### Khi xử lý Dữ liệu & Huấn luyện (`data/`, `engine/`, `tools/`)
* Luôn kiểm tra và đảm bảo việc chia tập Train/Val/Test được thực hiện theo **`subject_id` (tách biệt theo người lái)**. Nếu phát hiện logic chia ngẫu nhiên theo frame/clip, agent phải dừng lại và sửa ngay.
* Ưu tiên hỗ trợ luồng **trích xuất và lưu cache đặc trưng** (`tools/extract_features.py`) trước khi huấn luyện end-to-end để tối ưu thời gian chạy GPU.

### Khi viết Code Suy Luận Thời Gian Thực (`inference/`)
* Tách biệt luồng đọc camera (I/O) và luồng chạy mô hình (Compute).
* Đo và in ra độ trễ trung bình mỗi frame (ms/frame) và tốc độ FPS khi kiểm thử pipeline suy luận.

---

## 3. Các Hành Động Bị Cấm Tuyệt Đối

* **KHÔNG** tự ý thay đổi framework lõi sang TensorFlow/Keras (chỉ dùng hệ sinh thái PyTorch + Ultralytics).
* **KHÔNG** dùng RNN hai chiều (`bidirectional=True`) hoặc Transformer không có causal mask cho bài toán dự đoán từng frame.
* **KHÔNG** xóa bỏ hoặc bỏ qua frame khi YOLOv8 không bắt được mắt (phải dùng zero-padding kèm binary mask `[v_L, v_R]`).
* **KHÔNG** hardcode đường dẫn dữ liệu, ngưỡng phát hiện hoặc siêu tham số (hyperparameters) trực tiếp trong code model.
* **KHÔNG** tự ý xóa lịch sử thực nghiệm trong `MEMORY.md`.

---

## 4. Bộ Lệnh Tiêu Chuẩn Agent Được Phép Dùng

```bash
# Chạy toàn bộ unit test (bắt buộc sau khi sửa models/ hoặc data/)
pytest tests/ -v

# Trích xuất và cache vector đặc trưng từ YOLOv8 + CNN
python tools/extract_features.py --config configs/extract.yaml

# Huấn luyện mô hình chuỗi thời gian (Pha 1)
python engine/train_temporal.py --config configs/temporal_gru.yaml

# Đánh giá trên tập test độc lập theo người lái
python engine/evaluate.py --checkpoint checkpoints/best.pt --split test

# Chạy suy luận thử nghiệm trên video hoặc webcam
python inference/stream_runner.py --source 0 --onnx models/exported/pipeline.onnx
```
