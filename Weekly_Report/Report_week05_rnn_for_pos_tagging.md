# Report_Week05 — RNN For Pos Tagging

## I. Giới thiệu và Phương pháp Triển khai

File notebook của lab này được đặt ở thư mục `/notebook/lab5_rnn_for_pos_tagging.ipynb`. Nhằm mục đích xây dựng một mô hình Mạng Nơ-ron Hồi quy Đơn giản (Simple Recurrent Neural Network - RNN) sử dụng PyTorch để thực hiện nhiệm vụ Gán nhãn Từ loại (UPOS Tagging) trên tập dữ liệu UD English-EWT.

### 1. Nêu rõ các bước triển khai chi tiết

### A. Tải và Tiền xử lý Dữ liệu

* **Hàm load_conllu:** Đọc và phân tích cú pháp file .conllu, trích xuất các cặp (word, upos_tag).

* **Xây dựng Từ điển:** Tạo word_to_ix và tag_to_ix. Đã thêm các token đặc biệt:

        - <PAD> (0): Dùng để đệm chuỗi và bị bỏ qua trong tính toán loss/accuracy.

        - <UNK> (1): Dùng cho các từ không có trong từ điển (OOV).

### B. Xây dựng Data Pipeline

    Lớp POSDataset: Chuyển đổi các từ và nhãn thành các tensor chỉ số (index tensors).

    Hàm collate_fn: Đây là bước quan trọng để xử lý độ dài câu khác nhau. Sử dụng torch.nn.utils.rnn.pad_sequence với batch_first=True và padding_value=PAD_IDX để đệm tất cả các mẫu trong lô về cùng độ dài.

### C. Kiến trúc Mô hình Simple RNN

Mô hình SimpleRNNForTokenClassification bao gồm ba thành phần chính:

1. **nn.Embedding:** Chuyển đổi chỉ số từ thành vector nhúng.

2. **nn.RNN:** Xử lý chuỗi vector nhúng để tạo ra các vector ẩn theo thứ tự thời gian.

3. **nn.Linear:** Ánh xạ vector ẩn sang không gian nhãn (Logits).

---

## II. Log Kết quả Huấn luyện

### 1. Cách chạy code và ghi log kết quả

Mô hình được huấn luyện trong 10 Epoch. Log kết quả hiệu suất sau mỗi Epoch được ghi lại như sau:
| Epoch | Train Loss (TB) | Train Accuracy (TB) | Dev Accuracy | Ghi chú |
| :---: | :-------------: | :-----------------: | :----------: | :--- |
| 01 | 1.119 | 0.659 | 0.742 | |
| 02 | 0.622 | 0.800 | 0.799 | |
| 03 | 0.466 | 0.851 | 0.820 | |
| 04 | 0.368 | 0.882 | 0.836 | |
| 05 | 0.300 | 0.904 | 0.846 | |
| 06 | 0.247 | 0.921 | 0.854 | |
| 07 | 0.206 | 0.935 | 0.860 | |
| 08 | 0.174 | 0.945 | 0.864 | |
| 09 | 0.147 | 0.953 | 0.865 | **Mô hình Tốt nhất** |
| 10 | 0.124 | 0.961 | 0.864 | |	

### 2. Giải thích các kết quả thu được

* **Học tập và Hội tụ:** Train Loss giảm đều từ 1.119 xuống 0.124 và Train Accuracy tăng mạnh lên 0.961, cho thấy mô hình học tốt trên dữ liệu huấn luyện.

* **Điểm Tối ưu và Quá khớp (Overfitting):** Độ chính xác trên tập Dev (Dev Acc) đạt đỉnh là 0.865 tại Epoch 9. Tại Epoch 10, Train Acc vẫn tăng nhưng Dev Acc lại giảm nhẹ (0.864), xác nhận hiện tượng quá khớp bắt đầu xảy ra. Do đó, mô hình tốt nhất được chọn là mô hình sau Epoch 9.

---

## III. KẾT QUẢ THỰC HIỆN

Mục	Giá trị
Độ chính xác trên tập dev:	0.865
Ví dụ dự đoán câu mới:	
– Câu: “I love NLP”	– Dự đoán: [('i', 'PRON'), ('love', 'VERB'), ('nlp', 'NOUN')]

---

## IV. Khó khăn gặp phải và cách giải quyết

| Khó khăn | Mô tả chi tiết | Cách giải quyết (Kỹ thuật) |
| :--- | :--- | :--- |
| **Lỗi Path (`FileNotFoundError`)** | Lỗi không tìm thấy file `.conllu` do đường dẫn tương đối không chính xác so với thư mục làm việc hiện tại. | Điều chỉnh đường dẫn tương đối bằng cách thêm `../` (quay lại cấp thư mục) hoặc sử dụng `pathlib` để xác định vị trí file chính xác. |
| **Độ dài Chuỗi Khác nhau** | Không thể xử lý các chuỗi có độ dài khác nhau trong cùng một batch. | Triển khai `collate_fn` tùy chỉnh sử dụng `torch.nn.utils.rnn.pad_sequence` để đệm chuỗi và nhãn bằng `PAD_IDX`. |
| **Tính Loss/Accuracy trên Padding** | Các token đệm làm sai lệch kết quả đánh giá. | Sử dụng `nn.CrossEntropyLoss(ignore_index=PAD_IDX)` cho hàm mất mát và viết hàm `calculate_accuracy` tùy chỉnh để bỏ qua các chỉ số `PAD_IDX` (0). |

---

## V. Tài liệu Tham khảo

1. **Tập dữ liệu:** Universal Dependencies (UD) English-EWT corpus (Nguồn dữ liệu chuẩn).
2. **Kiến trúc Mô hình:** Simple Recurrent Neural Network (RNN) – Kiến trúc cơ bản của học sâu cho dữ liệu tuần tự.
3. **Thư viện:** PyTorch Documentation (để triển khai các lớp nn.RNN, nn.Embedding, nn.CrossEntropyLoss và hàm pad_sequence).