# Quy Chuẩn Lập Trình & Ràng Buộc Kỹ Thuật (`RULES.md`)

Tài liệu này định nghĩa các quy chuẩn kỹ thuật bắt buộc cho toàn bộ mã nguồn trong hệ thống phát hiện tài xế buồn ngủ (`YOLOv8 Crop mắt -> CNN -> RNN / Causal Transformer`).

---

## 1. Chuẩn Mực Mã Nguồn & Framework

* **Framework bắt buộc:** Sử dụng **PyTorch** (`torch`, `torchvision`, `timm`) cho toàn bộ mạng nơ-ron và `ultralytics` cho YOLOv8. Nghiêm cấm cài đặt hoặc import TensorFlow/Keras.
* **Định kiểu & Cấu hình:**
  * Sử dụng Python 3.10+ với đầy đủ type hints (`torch.Tensor`, `tuple[int, ...]`, `Optional[...]`).
  * Toàn bộ siêu tham số (kích thước ảnh cắt mắt, `hidden_size`, `window_size`, `learning_rate`, ngưỡng cảnh báo) phải được quản lý qua file YAML hoặc `@dataclass` trong `configs/`.
  * Luôn gọi hàm cố định seed (`seed_everything(seed)`) ở đầu mọi script huấn luyện và đánh giá.
* **Chú thích kích thước Tensor (Tensor Shape Comments):**
  * Mọi phép biến đổi tensor (`view`, `reshape`, `permute`, `cat`, `stack`) đều **PHẢI** có comment ghi rõ shape trước và sau khi biến đổi.
  * *Ví dụ đúng:*
    ```python
    # (B, T, 2, C, H, W) -> (B * T * 2, C, H, W)
    flat_crops = eye_crops.reshape(b * t * 2, c, h, w)
    ```

---

## 2. Quy Tắc Tính Nhân Quả Theo Thời Gian (Temporal Causality)

Hệ thống dự đoán trạng thái buồn ngủ **tại từng frame $t$** để khớp với luồng camera trực tiếp (online streaming). Do đó, mô hình tại bước $t$ **TUYỆT ĐỐI KHÔNG** được nhìn thấy thông tin từ các frame $t+1, t+2, \dots$:

* **Đối với nhánh RNN (`GRU` / `LSTM`):**
  * **CẤM** sử dụng `bidirectional=True`. Chỉ dùng RNN một chiều (`bidirectional=False`).
  * Hỗ trợ truyền và trả về `hidden_state` (`h_n`, `c_n`) để duy trì trạng thái liên tục giữa các chunk frame khi chạy real-time.
* **Đối với nhánh Transformer:**
  * **BẮT BUỘC** áp dụng mặt nạ nhân quả tam giác trên (`causal attention mask` với giá trị `-inf` ở các vị trí $j > i$, hoặc `is_causal=True` trong PyTorch SDPA/TransformerEncoder).
  * **BẮT BUỘC** có unit test kiểm tra gradient: thay đổi giá trị đầu vào tại frame $t+1$ phải cho ra đạo hàm bằng `0` đối với đầu ra tại frame $t$.

---

## 3. Xử Lý Phát Hiện Mắt Từ YOLOv8 & Mất Dấu (Missing Detections)

* **Không bao giờ giả định YOLOv8 luôn bắt đủ 2 mắt:** Trong thực tế (quay đầu, chói kính, thiếu sáng), YOLO có thể trả về `0`, `1` hoặc `2` bounding box.
* **Thứ tự không gian cố định:**
  * Slot `0` luôn là **mắt bên trái khung hình** (`image-left eye`), Slot `1` luôn là **mắt bên phải khung hình** (`image-right eye`), sắp xếp theo tọa độ tâm ngang $c_x$.
  * Khi chỉ phát hiện `1` mắt: so sánh $c_x$ với tâm khuôn mặt (hoặc tâm khung hình nếu không có box mặt) để gán đúng vào slot `0` hoặc `1`. Slot còn lại điền tensor `0.0`.
* **Cơ chế Masking:**
  * Luôn tạo vector cờ hợp lệ `eye_mask` có giá trị `{0.0, 1.0}` kích thước `(B, T, 2)` tương ứng `[v_L, v_R]`.
  * Sau khi qua CNN, nhân embedding với mask (`e_L = e_L * v_L`) để triệt tiêu nhiễu bias từ ảnh rỗng trước khi nối với vector mask: `f_t = [e_L || e_R || v_L || v_R]`.

---

## 4. Toàn Vẹn Dữ Liệu & Đánh Giá Mô Hình

* **Chia tập độc lập theo người lái (Subject-Independent Split):**
  * Tập Train, Validation và Test **BẮT BUỘC** phân tách theo `subject_id` (mỗi người chỉ xuất hiện ở duy nhất 1 tập).
  * Nghiêm cấm chia ngẫu nhiên theo frame hoặc theo đoạn video ngắn của cùng một người (gây rò rỉ dữ liệu — data leakage).
* **Chỉ số đánh giá ưu tiên:**
  * Báo cáo đầy đủ ở mức **Frame-level**: `Recall (Sensitivity)`, `Precision`, `F1-Score`, `AUROC`.
  * Báo cáo bổ sung ở mức **Event-level**: `False Alarm Rate (FAR / phút)` và độ trễ phát hiện buồn ngủ (Detection Latency tính bằng giây).
  * Ưu tiên tối đa hóa **Recall** (không bỏ sót cơn buồn ngủ) trong giới hạn FAR cho phép.

---

## 5. Tiêu Chuẩn Triển Khai Thời Gian Thực (Deployment)

* **Tách luồng I/O và Compute:** Luồng đọc webcam dùng hàng đợi ghi đè 1 phần tử (`maxsize=1`) để luôn lấy frame mới nhất, tránh dồn ứ độ trễ.
* **Bộ đệm trượt (Ring Buffer):** Sử dụng `collections.deque(maxlen=W)` để lưu cửa sổ trượt `W` frame gần nhất.
* **Tương thích ONNX:** Không viết logic điều khiển Python động phụ thuộc giá trị tensor bên trong hàm `forward()` của model để đảm bảo `torch.onnx.export` chạy mượt mà.
