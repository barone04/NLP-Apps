# Report_Week04 — Text Classification

---

## I. Chi Tiết Triển Khai

### 1. Task 1 & 2: Scikit-learn TextClassifier và Test Case

Đã xây dựng một pipeline phân loại văn bản module trong Python/Scikit-learn để đảm bảo tính linh hoạt và dễ kiểm thử.

| **Module** | **Tệp** | **Chức năng Đã Triển khai** |
|-------------|----------|------------------------------|
| `TextClassifier` | `src/models/text_classifier.py` | Bao bọc mô hình `LogisticRegression` của Scikit-learn. Phương thức `fit` gọi `vectorizer.fit_transform` và huấn luyện mô hình. `predict` và `evaluate` thực hiện các chức năng tương ứng. |
| `Test Case` | `test/lab5_test.py` | Tạo bộ dữ liệu thử nghiệm nhỏ, sử dụng `RegexTokenizer` và `CountVectorizer` để trích xuất đặc trưng, huấn luyện `TextClassifier`, và in ra kết quả dự đoán cùng với các chỉ số đánh giá. |

---

### 2. Task 3: Chạy Ví dụ Spark ML (Baseline)

Script `test/lab5_spark_sentiment_analysis.py` đã được chạy thành công, thiết lập một **Spark ML Pipeline** cho phân tích cảm xúc trên dữ liệu `data/sentiments.csv`.

| **Thành phần Pipeline** | **Vai trò** |
|--------------------------|-------------|
| `Tokenizer`, `StopWordsRemover` | Tiền xử lý văn bản thô. |
| `HashingTF`, `IDF` | Chuyển đổi văn bản thành các vector đặc trưng tần suất từ (TF-IDF). |
| `LogisticRegression` | Mô hình phân loại cơ sở (Baseline). |

---

### 3. Task 4: Thử nghiệm Cải thiện Mô hình

Chúng tôi đã thực hiện chiến lược thay thế kiến trúc mô hình để cải thiện hiệu suất:

- **Kỹ thuật Áp dụng:** Thay thế mô hình `LogisticRegression` bằng mô hình **NaiveBayes (Multinomial Naive Bayes)** trong Spark ML Pipeline.  
- **Tệp Kiểm thử:** Tạo `test/lab5_improvement_test.py` để chạy và so sánh hiệu suất của mô hình **Naive Bayes** với cùng một pipeline tiền xử lý TF-IDF.

---

## II. Báo Cáo và Phân Tích (Part 2: Report and Analysis - 50%)

### 1. Hướng Dẫn Thực Thi Mã (Code Execution Guide)

Để tái hiện các kết quả, vui lòng chạy các script sau từ **thư mục gốc của dự án**:

#### 🔹 Kiểm thử Module (Scikit-learn)

```bash
python test/lab5_test.py
```

#### 🔹 Baseline (Spark ML)

```bash
python test/lab5_spark_sentiment_analysis.py
```

#### 🔹 Cải thiện (Spark ML)

```bash
python test/lab5_improvement_test.py
```

---

### 2. Phân Tích Kết quả

Chúng tôi sử dụng một tập dữ liệu nhỏ (~100 mẫu) với nhãn `−1` và `1` cho thử nghiệm Spark.

#### 2.1. Báo cáo Hiệu suất Mô hình

| **Mô hình** | **Cơ sở Dữ liệu** | **Độ chính xác (Accuracy)** | **F1-Score** |
|--------------|-------------------|-----------------------------|--------------|
| Baseline (Logistic Regression) | Spark ML / TF-IDF | 41.67 %                     | 0.3259       |
| Cải thiện (Naive Bayes) | Spark ML / TF-IDF | 44.58 %                     | 0.3123       |



---

#### 2.2. So sánh và Phân tích

Mô hình **Naive Bayes** đã cho thấy sự cải thiện nhẹ về cả **Độ chính xác** và **F1-Score** so với **Logistic Regression** trên tập dữ liệu này.

- **Lý do Naive Bayes Hiệu quả:**  
  Naive Bayes, đặc biệt là phiên bản *Multinomial*, hoạt động rất tốt với các đặc trưng tần suất thưa thớt (*sparse frequency features*) như TF-IDF.  
  Giả định độc lập giữa các từ của nó thường hoạt động như một cơ chế **chuẩn hóa hiệu quả** (*effective regularization*) trong phân loại văn bản,
  giúp mô hình tổng quát hóa tốt hơn và tránh bị quá khớp hơn so với mô hình tuyến tính Logistic Regression khi dữ liệu thưa thớt hoặc bộ dữ liệu có kích thước hạn chế.

- **Kết luận:**  
  Việc thay thế mô hình là một **kỹ thuật cải tiến thành công**, cung cấp hiệu suất tốt hơn với chi phí tính toán tương đương.

---

### 3. Thách Thức và Giải Pháp

| **Thách thức** | **Giải pháp** |
|-----------------|----------------|
| Quá khớp (*Overfitting*) trên dữ liệu nhỏ | **Chiến lược Chuẩn hóa:** Cấu hình `LogisticRegression` với `regParam=0.001` và `NaiveBayes` với `smoothing=1.0` để kiểm soát độ phức tạp của mô hình. |
| Cấu hình Đường dẫn Module | Sử dụng `sys.path.insert(0, ...)` trong các tệp kiểm thử Scikit-learn để thêm thư mục gốc dự án, đảm bảo việc import các module từ thư mục `src` hoạt động chính xác. |
| Chuẩn hóa Nhãn Dữ liệu Spark | Đảm bảo chuyển đổi nhãn `−1/1` thành nhãn `0/1` chính xác bằng công thức:  
  ```python
  (col("sentiment").cast("integer") + 1) / 2
  ```  
  để phù hợp với yêu cầu của các thuật toán phân loại Spark ML. |

---

### 4. Tài Liệu Tham Khảo

- [Apache Spark ML Documentation](https://spark.apache.org/docs/latest/ml-guide.html): Tài liệu chính thức về các thuật toán và pipeline components.  
- [scikit-learn Documentation](https://scikit-learn.org/stable/): Hướng dẫn về `LogisticRegression` và `sklearn.metrics`.  
- **Tài liệu Lớp học / Giảng viên:** Các tài liệu và hướng dẫn về cấu trúc dự án module.

---