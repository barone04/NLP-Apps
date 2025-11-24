# Report_Week05 — RNN For Name Entity Recognition (NER)


## 1. Các bước triển khai

Chương trình của bài lab này được đặt tại thư mục `lab5_rnn_for_ner.ipynb` .Hệ thống được xây dựng theo quy trình 5 bước chuẩn của bài toán Sequence Labeling sử dụng PyTorch:

1.  **Tải và Tiền xử lý dữ liệu:**

      * Sử dụng thư viện `datasets` để tải bộ dữ liệu **CoNLL-2003**.
      * Trích xuất câu (tokens) và nhãn (NER tags).
      * Xây dựng bộ từ điển (`Vocabulary`): Ánh xạ từ vựng sang index, bổ sung token `<UNK>` (cho từ lạ) và `<PAD>` (cho padding).
      * Xây dựng ánh xạ nhãn (`Tag Map`): Ánh xạ các nhãn `B-PER`, `I-ORG`, `O`... sang số nguyên.

2.  **Tạo Dataset và DataLoader:**

      * Xây dựng class `NERDataset` kế thừa từ `torch.utils.data.Dataset`.
      * Triển khai hàm `collate_fn` sử dụng `pad_sequence` để đồng bộ độ dài các câu trong cùng một batch. Các vị trí đệm của nhãn được gán giá trị `-1`.

3.  **Xây dựng Kiến trúc Mô hình:**

      * Mô hình `SimpleRNNForTokenClassification` gồm 3 lớp:
          * `nn.Embedding`: Chuyển đổi index từ vựng sang vector.
          * `nn.LSTM` (Long Short-Term Memory): Xử lý chuỗi tuần tự để nắm bắt ngữ cảnh.
          * `nn.Linear`: Lớp kết nối đầy đủ để phân loại nhãn cho từng token.

4.  **Huấn luyện (Training):**

      * Loss Function: `CrossEntropyLoss` với tham số `ignore_index=-1` (bỏ qua các vị trí padding khi tính lỗi).
      * Optimizer: `Adam`.
      * Quy trình: Forward pass → Tính Loss → Backward pass → Cập nhật trọng số. Đồng thời tính Accuracy cho cả tập Train và Validation.

5.  **Đánh giá và Dự đoán:**

      * Tính toán độ chính xác trên tập Validation (loại bỏ padding).
      * Thực hiện inference trên câu mới nhập vào.

## 2. Cách chạy code và ghi log kết quả

### Cách chạy code

Đảm bảo đã cài đặt các thư viện cần thiết (`torch`, `datasets`, `numpy`). Chạy file script chính bằng lệnh:

```bash
python main.py
```

### Log kết quả thực nghiệm

Dưới đây là log ghi lại quá trình huấn luyện qua 3 Epochs và kết quả chạy thử nghiệm:

```text
Epoch: 01 | Train Loss: 0.033 | Train Acc: 0.990 | Val Acc: 0.921
Epoch: 02 | Train Loss: 0.018 | Train Acc: 0.995 | Val Acc: 0.913
Epoch: 03 | Train Loss: 0.013 | Train Acc: 0.997 | Val Acc: 0.911

Sentence: VNU University is located in Hanoi
Token           Predicted Label
------------------------------
VNU             B-ORG
University      I-ORG
is              O
located         O
in              O
Hanoi           B-MISC
```

## 3. Giải thích các kết quả thu được

Dựa vào log kết quả trên, ta có các phân tích sau:

**Về chỉ số Huấn luyện (Training Metrics):**

  * **Train Acc rất cao (99.0% → 99.7%):** Mô hình có khả năng học và ghi nhớ dữ liệu huấn luyện rất tốt. Loss giảm sâu xuống 0.013 cho thấy mô hình gần như khớp hoàn toàn với tập train.

**Về chỉ số Kiểm thử (Validation Metrics):**

  * **Val Acc cao nhưng có xu hướng giảm nhẹ (92.1% → 91.1%):** Mặc dù độ chính xác trên tập validation đạt mức tốt (>91%), nhưng việc Train Acc tăng trong khi Val Acc giảm nhẹ là dấu hiệu sớm của hiện tượng **Overfitting** (Qúa khớp). Mô hình bắt đầu "học vẹt" các đặc điểm nhiễu của tập train thay vì học các quy luật tổng quát, dẫn đến khả năng suy luận trên dữ liệu mới kém đi một chút.
  * *Lưu ý:* Trong bài toán NER, nhãn 'O' (Outside) chiếm đa số. Việc độ chính xác cao (91%) đôi khi bị sai lệch do mô hình dự đoán đúng phần lớn nhãn 'O'.

**Về kết quả Dự đoán (Inference):**

  * Câu: *"VNU University is located in Hanoi"*
  * **"VNU University" (→ B-ORG, I-ORG):** Dự đoán chính xác. Mô hình nhận diện được đây là một Tổ chức (Organization).
  * **"Hanoi" (→ B-MISC):** Dự đoán chưa chính xác về loại nhãn. "Hanoi" là địa danh, lẽ ra phải là `B-LOC`. Tuy nhiên, mô hình lại gán nhãn `B-MISC` (Miscellaneous - Thực thể hỗn hợp). Điều này có thể do từ "Hanoi" ít xuất hiện trong tập train (CoNLL-2003 chủ yếu là dữ liệu tin tức phương Tây) hoặc do ngữ cảnh chưa đủ mạnh để phân biệt giữa LOC và MISC.

## 4. Các khó khăn gặp phải và cách giải quyết

1.  **Khó khăn:** *Xử lý độ dài câu không đồng nhất trong Batch.*

      * **Giải quyết:** Sử dụng hàm `pad_sequence` trong `collate_fn` để thêm padding vào cuối câu ngắn.

2.  **Khó khăn:** *Mô hình học sai do tính toán Loss cả vào phần Padding.*

      * **Giải quyết:** Thiết lập `ignore_index=-1` trong hàm `CrossEntropyLoss`. Điều này chỉ đạo PyTorch bỏ qua, không tính gradient tại các vị trí padding, giúp mô hình tập trung vào dữ liệu thực.

3.  **Khó khăn:** *Overfitting xuất hiện nhanh.*

      * **Giải quyết (Đề xuất):** Có thể thêm lớp `Dropout` vào sau tầng Embedding hoặc LSTM, hoặc sử dụng kỹ thuật `Early Stopping` (dừng huấn luyện khi Val Loss không giảm nữa) để ngăn chặn overfitting.

## 5. Nguồn tham khảo

Báo cáo sử dụng kiến thức và tài liệu từ các nguồn sau:

1.  **Hugging Face Datasets Documentation:** Hướng dẫn tải và xử lý bộ dữ liệu CoNLL-2003.
2.  **PyTorch Documentation:** Tài liệu kỹ thuật về `nn.LSTM` và `nn.CrossEntropyLoss`.
3.  **Jurafsky, D., & Martin, J. H. (2023).** *Speech and Language Processing*. Chương về Sequence Labeling.

## 6. Mô hình sử dụng

  * **Kiến trúc:** Mô hình được xây dựng "from scratch" (tự định nghĩa) gồm các lớp: Embedding → LSTM → Linear Layer.
  * **Dữ liệu huấn luyện:** Bộ dữ liệu CoNLL-2003.

-----

### Kết quả cuối cùng

• **Độ chính xác trên tập validation:** 91.1% (tại Epoch 3)

• **Ví dụ dự đoán câu mới:**
– **Câu:** “VNU University is located in Hanoi”
– **Dự đoán:**
| Token | Predicted Label |
| :--- | :--- |
| VNU | **B-ORG** |
| University | **I-ORG** |
| is | O |
| located | O |
| in | O |
| Hanoi | **B-MISC** |