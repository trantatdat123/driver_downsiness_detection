# Thiết Kế Kiến Trúc Hệ Thống — Phát Hiện Tài Xế Buồn Ngủ (`ARCHITECTURE.md`)

## 1. Tổng Quan Luồng Xử Lý End-to-End

Hệ thống phân tích luồng video trực diện tài xế qua 3 giai đoạn học sâu nhân quả (causal) kết hợp 1 bộ máy ra quyết định theo thời gian:

```mermaid
flowchart TD
    Input["Khung hình gốc (H x W x 3)"] --> Stage1

    Stage1["Giai đoạn 1: Phát hiện vùng mắt (YOLOv8)
    • Bắt bounding box mắt trái & mắt phải
    • Sắp xếp trái-phải theo tọa độ tâm x
    • Xử lý mất dấu (0/1/2 mắt) bằng mask"]

    Stage1 -->|"Ảnh mắt trái (64x64x3)"| Stage2
    Stage1 -->|"Ảnh mắt phải (64x64x3)"| Stage2

    Stage2["Giai đoạn 2: Trích xuất đặc trưng (CNN)
    • Backbone dùng chung trọng số (Shared)
    • Trích xuất vector e_L, e_R thuộc R^D
    • Triệt tiêu embedding nếu mắt bị mất"]

    Stage2 -->|"Nối vector: f_t = concat(e_L, e_R, v_L, v_R) thuộc R^(2D+2)"| Stage3

    Stage3["Giai đoạn 3: Mô hình chuỗi thời gian
    • Cửa sổ trượt (f_{t-W+1}, ..., f_t)
    • Unidirectional GRU/LSTM HOẶC Causal Transformer Encoder
    • Dự đoán logit buồn ngủ từng frame z_t"]

    Stage3 -->|"Sigmoid -> Xác suất p_t thuộc [0, 1]"| Stage4

    Stage4["Giai đoạn 4: Bộ máy quyết định cảnh báo
    • Làm mượt xác suất bằng EMA
    • Tính tỷ lệ nhắm mắt cửa sổ (PERCLOS)
    • Cơ chế trễ Hysteresis & Cooldown"]
```

---

## 2. Hợp Đồng Kích Thước Tensor (Tensor Shape Contracts)

Ký hiệu: $B$ = batch size, $T$ = độ dài chuỗi/cửa sổ thời gian (`window_size`), $H_{\text{eye}} = W_{\text{eye}} = 64$ (cấu hình qua YAML), $D$ = chiều embedding của mỗi mắt.

| Giai đoạn | Thành phần | Đầu vào | Đầu ra | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `YOLOv8EyeCropper` | Raw frames `(B, T, H, W, 3)` | `eye_crops`: `(B, T, 2, 3, H_eye, W_eye)`<br>`eye_mask`: `(B, T, 2)` | Slot `0` = image-left, Slot `1` = image-right; mắt mất điền `0.0` và `mask=0.0` |
| **Stage 2** | `SharedEyeEncoder` | `eye_crops` flatten -> `(B*T*2, 3, H_eye, W_eye)` | `frame_feats`: `(B, T, 2*D + 2)` | `e = e * mask[..., None]`; ghép `[e_L \|\| e_R \|\| v_L \|\| v_R]` |
| **Stage 3** | `TemporalHead` | `frame_feats`: `(B, T, 2*D + 2)` | `logits`: `(B, T, 1)` | Dự đoán many-to-many nhân quả; không trả về `sigmoid` trong `forward()` |
| **Stage 4** | Decision Engine | `p_t = sigmoid(z_t)` $\in [0, 1]$ | Trạng thái cảnh báo `{ALERT, DROWSY}` | EMA + PERCLOS + Hysteresis/Cooldown |

---

## 3. Chi Tiết Thiết Kế Từng Giai Đoạn

### 3.1 Giai đoạn 1 — `YOLOv8EyeCropper` (`models/`)
* Dùng Ultralytics YOLOv8n/s phát hiện bounding box mắt (và tùy chọn box khuôn mặt để làm mốc phân biệt trái/phải khi chỉ thấy 1 mắt).
* **Quy ước thứ tự không gian:**
  * Khi có `2` mắt: sắp xếp theo tọa độ tâm ngang $c_x$ tăng dần (`slot 0`: $c_x$ nhỏ hơn / bên trái khung hình; `slot 1`: $c_x$ lớn hơn / bên phải khung hình).
  * Khi có `1` mắt: so sánh $c_x$ với tâm khuôn mặt (hoặc $W/2$ nếu không có box mặt) để gán đúng slot `0` hoặc `1`; slot còn lại zero-padding.
  * Khi có `0` mắt: giữ nguyên frame (không drop), điền `0.0` cho cả 2 crop và đặt `eye_mask = [0.0, 0.0]`.

### 3.2 Giai đoạn 2 — `SharedEyeEncoder` (`models/`)
* Một backbone CNN nhẹ dùng chung trọng số (ví dụ `MobileNetV3-Small`, `ResNet18`, `EfficientNet-B0` qua `torchvision`/`timm`) xử lý đồng thời cả 2 mắt qua batch gộp `(B * T * 2, 3, H_eye, W_eye)`.
* Bắt buộc nhân embedding với `eye_mask` sau CNN (`e_L = e_L * v_L`, `e_R = e_R * v_R`) để triệt tiêu vector bias sinh ra từ ảnh `0.0` của mắt bị mất dấu, sau đó mới nối `[v_L, v_R]` thành `f_t` kích thước `2D + 2`.

### 3.3 Giai đoạn 3 — `TemporalHead` (`models/`)
Hỗ trợ 2 biến thể hoán đổi qua config nhưng cùng giao diện đầu vào `(B, T, 2*D + 2)` và đầu ra `(B, T, 1)`:
* **`CausalGRU` / `CausalLSTM`:** `bidirectional=False`, hỗ trợ nhận và trả về `hidden_state` (`h_n`, `c_n`) để suy luận liên tục theo từng frame trên luồng real-time và reset khi đổi video/tài xế.
* **`CausalTransformer`:** Thêm Positional Encoding theo bước thời gian, sử dụng `nn.TransformerEncoder` với mặt nạ nhân quả tam giác trên (`-inf` với $j > i$, hoặc `is_causal=True`) nhằm chặn tuyệt đối rò rỉ thông tin tương lai.

### 3.4 Giai đoạn 4 — Bộ Máy Quyết Định Cảnh Báo (`inference/`)
* **EMA Smoothing:** $\tilde{p}_t = \alpha \cdot p_t + (1 - \alpha) \cdot \tilde{p}_{t-1}$.
* **PERCLOS & Hysteresis:** Kết hợp tỷ lệ frame vượt ngưỡng trong cửa sổ trượt `W` và ngưỡng bật/tắt trễ (`thresh_on > thresh_off`) kèm thời gian `cooldown` để tránh cảnh báo nhấp nháy (flickering).

---

## 4. Cấu Trúc Thư Mục Chuẩn

```text
.
├── AGENTS.md                   # Quy trình vận hành cho AI coding agents
├── ARCHITECTURE.md             # Tài liệu thiết kế kiến trúc (file này)
├── RULES.md                    # Quy chuẩn lập trình & ràng buộc kỹ thuật
├── MEMORY.md                   # Trạng thái dự án, ADRs & nhật ký thực nghiệm
├── configs/                    # File cấu hình YAML (extract.yaml, temporal_gru.yaml, ...)
├── data/                       # Dataset, DataLoader, chia tập theo subject_id
├── models/                     # YOLOv8EyeCropper, SharedEyeEncoder, TemporalHead, pipeline ONNX
├── engine/                     # Huấn luyện (train_temporal.py), đánh giá (evaluate.py)
├── inference/                  # Suy luận thời gian thực (stream_runner.py, tách luồng I/O)
├── tools/                      # Script tiền xử lý & cache đặc trưng (extract_features.py)
└── tests/                      # Unit tests (causality, missing-eye 0/1/2, split leakage, ONNX)
```

---

## 5. Chiến Lược Huấn Luyện 2 Pha & Suy Luận Real-Time

1. **Pha 1 — Trích xuất & Cache Đặc Trưng (`tools/extract_features.py`):** Chạy YOLOv8 + CNN (hoặc lưu eye crops / feature vectors) một lần theo đúng thứ tự frame để tăng tốc độ thử nghiệm kiến trúc `TemporalHead` và tránh lặp lại chi phí I/O video.
2. **Pha 2 — Huấn luyện Chuỗi Thời Gian & Fine-tune (`engine/train_temporal.py`):** Huấn luyện `TemporalHead` (và fine-tune `SharedEyeEncoder` với learning rate nhỏ hơn nếu dùng raw/crop cache) trên các tập chia độc lập theo `subject_id`.
   * **Chế độ nhãn mức Video/Clip (Many-to-One trên cửa sổ trượt — Mặc định):** Khi dataset là ảnh cắt từ video và chỉ có 1 nhãn `y_clip` $\in \{0, 1\}$ cho cả đoạn video, `Dataset` sắp xếp ảnh theo đúng `frame_index` trong từng `video_id` (không ghép chéo video) và lấy mẫu theo `temporal_stride` (ví dụ hạ từ 30 FPS xuống `5–10 FPS`) để cửa sổ `W` phủ đủ `3–6 giây` lịch sử. `TemporalHead` vẫn xuất `logits` `(B, W, 1)` đúng hợp đồng kiến trúc, còn vòng lặp train lấy bước cuối `logits[:, -1, :]` `(B, 1)` tính `BCEWithLogitsLoss` với `y_clip`.
3. **Suy Luận Thời Gian Thực (`inference/stream_runner.py`):**
   * Luồng Producer (Camera I/O) đẩy frame mới nhất vào hàng đợi `maxsize=1` (ghi đè frame cũ).
   * Luồng Consumer (Compute) chạy Stage 1–2 trên đúng `1` frame hiện tại, đẩy $f_t$ vào `collections.deque(maxlen=W)` (hoặc cập nhật `hidden_state` của GRU), chạy Stage 3 + Stage 4 (đọc `sigmoid(logits[:, -1, :])`), và log `ms/frame` cùng `FPS`.
