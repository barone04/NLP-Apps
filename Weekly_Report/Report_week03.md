# Report_Week03 — Word Embedding and Visualization

## Các source code mà bài lab dùng:

| STT | Loại File/Đường dẫn            | Đường dẫn (Path)                                 | Mục đích |
|-----|--------------------------------|--------------------------------------------------|----------|
| 1   | Lớp chính (Implement)          | src/representations/word_embedder.py            | Cài đặt lớp WordEmbedder để xử lý các thao tác với Word Embedding (lấy vector, similarity, most similar, embed document). |
| 2   | File Kiểm thử Pre-trained      | test/test_lab04.py                               | Chạy thực nghiệm với các mô hình Embedding đã huấn luyện sẵn (pre-trained model), ví dụ GloVe. |
| 3   | File Huấn luyện Tự động        | test/lab4_embedding_training_demo.py             | Huấn luyện mô hình Word2Vec từ đầu bằng thư viện gensim. |
| 4   | File Huấn luyện Spark          | test/spark_word2vec_demo.py                      | Huấn luyện mô hình Word2Vec sử dụng PySpark MLlib. |
| 5   | File Trực quan hóa             | test/lab4_embedding_visualization.py             | Giảm chiều (PCA/t-SNE) và vẽ biểu đồ scatter plot để trực quan hóa không gian Embedding. |
| 6   | File Dữ liệu Đầu vào           | data/UD_English-EWT/en_ewt-ud-train.txt          | Dữ liệu văn bản thô được sử dụng để tự huấn luyện mô hình Word2Vec. |
| 7   | File Mô hình Lưu               | results/word2vec_ewt.model                       | Nơi lưu trữ mô hình Word2Vec sau khi tự huấn luyện. |

---

## 🔹 1. Giải thích các bước thực hiện

Bài tập tuần 05 tập trung vào **biểu diễn từ (Word Embedding)**, bao gồm việc sử dụng mô hình **pre-trained**, **tự huấn luyện Word2Vec**, và **trực quan hóa embedding space**.

### Các bước thực hiện:

1. **Cài đặt lớp `WordEmbedder`**

   * File: `src/representations/word_embedder.py`
   * Dùng `gensim.downloader.load(model_name)` để tải mô hình pre-trained, ví dụ `glove-wiki-gigaword-50`.
   * Cung cấp các phương thức:

     * `get_vector(word)`: Lấy vector của từ.
     * `get_similarity(word1, word2)`: Tính độ tương đồng cosine giữa hai từ.
     * `get_most_similar(word, top_n)`: Trả về danh sách các từ gần nghĩa nhất.
     * `embed_document(document)`: Biểu diễn toàn bộ văn bản bằng trung bình vector các từ trong câu (bỏ qua OOV).

2. **Thực nghiệm với pre-trained model**

   * File test: `test/test_lab04.py`
   * Các thao tác:

     * Lấy vector của từ “king”.
     * Tính similarity giữa “king–queen”, “king–man”.
     * Tìm 10 từ gần nghĩa nhất với “computer”.
     * Biểu diễn câu “The queen rules the country.”

3. **Huấn luyện mô hình Word2Vec từ đầu**

   * File: `test/lab4_embedding_training_demo.py`
   * Dữ liệu: `data/UD_English-EWT/en_ewt-ud-train.txt`
   * Mô hình huấn luyện được lưu tại: `results/word2vec_ewt.model`

4. **Trực quan hóa Embedding**

   * File: `test/lab4_embedding_visualization.py`
   * Giảm chiều bằng **PCA** hoặc **t-SNE**.
   * Vẽ biểu đồ scatter plot các từ như:
     `["king", "queen", "man", "woman", "car", "truck", "fruit", "apple", "dog", "cat", ...]`

---

## 🔹 2. Hướng dẫn chạy code

### Chạy thực nghiệm với mô hình pre-trained

```bash
python -m test.test_lab04
```

### Huấn luyện Word2Vec từ đầu

```bash
python test/lab4_embedding_training_demo.py
```

### Huấn luyện bằng pyspark

```bash
python test/spark_word2vec_demo.py
```

### Trực quan hóa embedding

```bash
python test/lab4_embedding_visualization.py
```

---

## 🔹 3. Phân tích kết quả

- Kết quả:
![Query Document](../image/Result_task1_2.png)
![Result Documents](../image/Result_task3.png)
![Result Documents](../image/Result_task4.png)

### a. Độ tương đồng và từ đồng nghĩa tìm được

* `Similarity(king, queen)` ≈ **0.78** → cao, vì hai từ có quan hệ ngữ nghĩa mạnh (nam–nữ hoàng).
* `Similarity(king, man)` ≈ **0.53** → thấp hơn, vì “man” chỉ là giống loài, không phải vai trò hoàng gia.
* Top từ gần nghĩa với “computer”:

  ```
  computers, software, technology, electronic, internet, digital, ...
  ```

  → cho thấy model học được các mối liên hệ ngữ nghĩa đúng như kỳ vọng.

### b. Phân tích biểu đồ trực quan hóa

- Kết quả:
![Query Document](../image/PCA_visual.png)
![Result Documents](../image/TSNE_visual.png)

Khi giảm chiều xuống 2D bằng **PCA** hoặc **t-SNE**:

* Các nhóm từ cùng chủ đề (ví dụ: *king, queen, prince, princess*) nằm gần nhau.
* Nhóm *car, truck, vehicle* cũng tạo thành cụm riêng biệt.
* Một vài cụm từ thú vị:

  * *dog* và *cat* nằm gần nhau → đúng vì cùng loại “animal”.
  * *apple, banana, fruit* cũng gần nhau → phản ánh quan hệ siêu-phụ (hypernym).

Điều này chứng minh rằng mô hình embedding đã **học được quan hệ ngữ nghĩa và ngữ cảnh giữa các từ**.

---

## 🔹 4. So sánh giữa mô hình pre-trained và mô hình tự huấn luyện

| Tiêu chí             | Pre-trained (GloVe)        | Word2Vec tự huấn luyện               |
| -------------------- | -------------------------- | ------------------------------------ |
| Dữ liệu              | Wikipedia + Gigaword       | UD_English-EWT (nhỏ hơn nhiều)       |
| Kết quả similarity   | Ổn định, chính xác         | Dao động, ít ổn định                 |
| Cụm từ trực quan hóa | Rõ ràng, phân tách tốt     | Mờ hơn, do ít dữ liệu                |
| Ưu điểm              | Dễ dùng, chính xác cao     | Linh hoạt, phù hợp domain riêng      |
| Nhược điểm           | Cồng kềnh, không tùy chỉnh | Cần nhiều dữ liệu và thời gian train |

→ Kết luận: **Pre-trained model cho kết quả tốt hơn**, nhưng **model tự train** hữu ích khi muốn embedding chuyên biệt cho một lĩnh vực cụ thể (như y học, tài chính, mạng xã hội...).

---

## 🔹 5. Khó khăn và giải pháp

| Khó khăn                                                       | Giải pháp                                                |
| -------------------------------------------------------------- |----------------------------------------------------------|
| Lỗi “ModuleNotFoundError: No module named 'src'” khi chạy test | Chạy từ thư mục gốc: `python -m test.test_lab04`         |
| Lỗi import `RegexTokenizer` trong class `WordEmbedder`         | Sửa thuộc tính thành `self.tokenizer = RegexTokenizer()` |
| Visualization t-SNE chậm                                       | Dùng PCA để thử nhanh trước, sau đó mới chạy t-SNE       |

---

## 🔹 6. Tài liệu tham khảo

* [Gensim Documentation – Word Embeddings](https://radimrehurek.com/gensim/models/keyedvectors.html)
* [Stanford GloVe Pretrained Models](https://nlp.stanford.edu/projects/glove/)
* [Universal Dependencies – English EWT Corpus](https://universaldependencies.org/treebanks/en_ewt/)
* [Scikit-learn: PCA and t-SNE](https://scikit-learn.org/stable/modules/manifold.html)
* [PySpark MLlib Word2Vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec)

---
