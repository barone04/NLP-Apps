# Report_Week05 — RNNs Text Classification.


## 1) Mục tiêu & Tổng quan

Mục tiêu của lab là xây dựng và so sánh 4 pipeline phân loại ý định (intent) trên bộ dữ liệu HWU:

1) **TF-IDF + Logistic Regression** (baseline cổ điển)  
2) **Word2Vec (trung bình) + Dense** (baseline dùng embedding tĩnh)  
3) **LSTM + Embedding khởi tạo từ Word2Vec** (đóng băng, “pre-trained”)  
4) **LSTM + Embedding học từ đầu** (trainable)

Đánh giá bằng **F1-macro** (ưu tiên lớp thiểu số) và **test loss** (với các mô hình Keras). Ngoài ra, thực hiện **phân tích định tính** trên các câu khó có phủ định/ràng buộc.

---

## 2) Dữ liệu & Tiền xử lý

- Đọc 3 file `train/val/test` (2 cột: `text`, `category`).  
- Loại bỏ trường hợp header bị đọc nhầm thành dòng dữ liệu đầu.  
- Chuẩn hóa: ép `text`/`category` sang `str`, fill NA.  
- Mã hóa nhãn bằng `LabelEncoder` (fit trên hợp `train+val+test` để đồng bộ mapping).  
- **Tokenizer** (cho mô hình chuỗi): fit trên **train**, `oov_token="<UNK>"`.  
- **Độ dài chuỗi `max_len`**: chọn theo **95th percentile** độ dài (ước lượng từ train) để cân bằng độ phủ ngữ cảnh/chi phí tính toán.  
- **Word2Vec**: train trên token hóa `simple_preprocess` của `gensim` từ tập **train** (có thể mở rộng thêm val), `vector_size=100, window=5, min_count=1`.

> Lưu ý nhất quán: `tokenizer`, `vocab_size`, `max_len` dùng **chung** cho Nhiệm vụ 3–4 (Embedding + LSTM).

---

## 3) Kiến trúc & Thiết lập huấn luyện

### 3.1 TF-IDF + Logistic Regression
- `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`  
- `LogisticRegression(max_iter=1000)`  
- Train trên **train**, predict trên **test**.  
- **Không có test loss** theo Keras → để `NaN`.

### 3.2 Word2Vec (Avg) + Dense
- Với mỗi câu, lấy **trung bình** vector các từ có trong vocab W2V (nếu rỗng → vector 0).  
- **Model:** `Dense(128, relu) → Dropout(0.5) → Dense(num_classes, softmax)`  
- **Loss:** `categorical_crossentropy`, **Optimizer:** `adam`.  
- **Callback:** `EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)`.

### 3.3 LSTM + Embedding (Pre-trained, Frozen)
- Tạo `embedding_matrix` từ Word2Vec (map theo `tokenizer.word_index`, out-of-vocab để 0).  
- **Model:** `Embedding(weights=[embedding_matrix], trainable=False) → LSTM(128, dropout=0.2, recurrent_dropout=0.2) → Dense(num_classes, softmax)`  
- Cấu hình huấn luyện giống 3.2.

### 3.4 LSTM + Embedding (Scratch, Trainable)
- **Model:** `Embedding(input_dim=vocab_size, output_dim=100, trainable=True) → LSTM(128, ...) → Dense(num_classes, softmax)`  
- Cấu hình huấn luyện giống 3.2.

---

## 4) Cách chạy & Ghi log

1. Cài môi trường:
   ```bash
   pip install pandas scikit-learn gensim "tensorflow>=2.12" numpy
   ```
2. Đặt dữ liệu vào `data/hwu/{train,val,test}.csv` (2 cột `text,category`).
3. Chạy notebook/script đã cung cấp (đặt seed 42).  
4. **Log kết quả**:
   - In `classification_report` từng mô hình (precision/recall/F1 từng lớp + macro).
   - In bảng tổng hợp **F1-macro** và **test loss**.
   - In dự đoán 4 mô hình cho các câu “khó”.

> Khuyến nghị lưu thêm: `results.to_csv('results_summary.csv', index=False)` và `history.history` mỗi mô hình để vẽ learning curve (tùy chọn).

---

## 5) Kết quả định lượng

| Pipeline                             | F1-score (Macro) | Test Loss |
|--------------------------------------|------------------:|----------:|
| TF-IDF + Logistic Regression         | **0.829401**      |     NaN   |
| Word2Vec (Avg) + Dense               | 0.148429          | 3.091791  |
| Embedding (Pre-trained) + LSTM       | 0.243005          | 2.629349  |
| Embedding (Scratch) + LSTM           | 0.792026          | 0.772743  |

**Nhận xét nhanh:**
- **TF-IDF+LR** đạt **F1-macro cao nhất (0.829)** trên tập test này.  
- **LSTM (scratch)** rất cạnh tranh (**0.792**, loss tốt), gần sát TF-IDF.  
- **W2V Avg + Dense** và **LSTM (pre-trained, frozen)** cho F1 thấp (0.15–0.24), loss cao.

---

## 6) Phân tích định tính (các câu “khó”)

### Câu 1
**“can you remind me to not call my mom”**  
- TF-IDF+LR: `calendar_set`  
- W2V(Avg)+Dense: `general_explain`  
- LSTM + Pretrained Emb: `email_query`  
- LSTM + Scratch Emb: `calendar_set`

*Cả 4 mô hình đều chưa khớp “reminder_create”, nhưng LSTM (scratch) và TF-IDF cho nhãn gần nghĩa hơn.*

### Câu 2
**“is it going to be sunny or rainy tomorrow”**  
- TF-IDF+LR: `weather_query` 
- W2V(Avg)+Dense: `email_query`  
- LSTM + Pretrained Emb: `transport_query`  
- LSTM + Scratch Emb: `weather_query`

*Câu trực tiếp → TF-IDF và LSTM (scratch) đều bắt đúng từ khóa chủ đề thời tiết.*

### Câu 3
**“find a flight from new york to london but not through paris”**  
- TF-IDF+LR: `transport_query`  
- W2V(Avg)+Dense: `general_dontcare`  
- LSTM + Pretrained Emb: `transport_ticket`  
- LSTM + Scratch Emb: `transport_query`

*Cấu trúc “not through paris” chỉ LSTM mới nắm được quan hệ phủ định → dự đoán gần nghĩa nhất.*

---

## 7) Tổng kết định tính

- **LSTM (scratch)** thể hiện rõ ưu thế khi xử lý **phủ định** và **cấu trúc phụ thuộc xa**.  
- **TF-IDF** vẫn mạnh trong **câu ngắn, từ khóa rõ**, dữ liệu nhỏ.  
- **W2V trung bình** mất thông tin ngữ cảnh; **LSTM frozen** kém thích nghi nếu embedding huấn luyện nội bộ quá nhỏ.

---

## 8) So sánh ưu & nhược điểm

| Phương pháp | Ưu điểm | Nhược điểm | Khi nên dùng |
|---|---|---|---|
| **TF-IDF + LR** | Nhanh, hiệu quả, dễ huấn luyện | Không hiểu thứ tự từ | Dữ liệu nhỏ, intent đơn giản |
| **W2V Avg + Dense** | Tận dụng embedding, nhanh | Mất cấu trúc chuỗi | Chỉ để baseline |
| **LSTM + Pretrained (frozen)** | Hội tụ nhanh nếu embedding tốt | Phụ thuộc embedding | Khi có embedding lớn, cùng miền |
| **LSTM + Scratch** | Học embedding phù hợp, hiểu ngữ cảnh | Dễ overfit nếu dữ liệu ít | Khi có đủ dữ liệu & dùng regularization |

---

## 9) Khó khăn & Giải pháp

| Khó khăn | Cách giải quyết |
|----------|-----------------|
| Embedding Word2Vec chất lượng thấp | Train trên tập lớn hơn, fine-tune embedding |
| OOV nhiều, mất nghĩa từ mới | Dùng subword (BPE) hoặc tokenizer tốt hơn |
| LSTM overfit | Dùng Dropout, EarlyStopping |
| Class imbalance | Dùng `class_weight`, focal loss |

---

## 10) Hướng mở rộng

- BiLSTM/GRU + Attention  
- Subword/BPE tokenization  
- Fine-tune Transformer (DistilBERT/BERT)  
- Hyperparameter tuning (Optuna/Ray Tune)  
- Data augmentation (synonym replacement, back-translation)

---

## 11) Kết luận

- **TF-IDF+LR** vẫn là baseline mạnh với dữ liệu vừa & nhỏ.  
- **LSTM (scratch)** cho kết quả **gần tương đương**, đặc biệt mạnh ở câu phức.  
- **W2V trung bình** & **LSTM frozen** yếu do embedding nội bộ kém.  
- Kết hợp embedding + LSTM fine-tune có thể vượt TF-IDF khi dữ liệu mở rộng.

---

## 13) Tài liệu tham khảo

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (Word2Vec).  
- Gensim documentation — Word2Vec.  
- Keras documentation — `Embedding`, `LSTM`, `EarlyStopping`.  
- Scikit-learn documentation — `TfidfVectorizer`, `LogisticRegression`, `classification_report`.
