# Kho lưu trữ Môn học Xử lý Ngôn ngữ Tự nhiên & Ứng dụng (NLP & Apps)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Spark](https://img.shields.io/badge/Apache_Spark-3.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![License](https://img.shields.io/badge/License-MIT-green)

Chào mừng bạn đến với kho lưu trữ chính thức của môn học **Xử lý Ngôn ngữ Tự nhiên & Ứng dụng** (Natural Language Processing & Applications).

Dự án này là tập hợp toàn diện các tài liệu thực hành, mã nguồn triển khai mô hình, báo cáo tiến độ hàng tuần và các bộ dữ liệu được sử dụng trong suốt học kỳ. Nội dung bao trùm từ các kỹ thuật NLP cơ bản đến các mô hình Học sâu nâng cao (RNN, Transformer) và xử lý ngôn ngữ quy mô lớn với Apache Spark.

---

## Cấu trúc Dự án

Dưới đây là tổng quan về tổ chức thư mục để bạn dễ dàng điều hướng:

```text
NLP-Apps/
├── data/                  # Dữ liệu thô và đã xử lý (VD: UD_English, Sentiments)
├── notebook/              # Jupyter Notebooks cho các bài lab tương tác (PyTorch, Transformers)
├── spark_labs/            # Dự án Big Data NLP (Scala/SBT & Apache Spark)
├── src/                   # Mã nguồn Python chính (Core logic)
│   ├── core/              # Các class cơ bản và tiện ích chung
│   ├── models/            # Triển khai mô hình (Custom Architectures, Wrappers)
│   ├── preprocessing/     # Tokenization, Cleaning, Data preparation
│   └── representations/   # Biểu diễn văn bản (TF-IDF, Embeddings)
├── test/                  # Unit tests, demo scripts và file thực thi lab
├── Weekly_Report/         # Báo cáo tiến độ và nghiên cứu lý thuyết hàng tuần
├── requirements.txt       # Danh sách thư viện Python phụ thuộc
└── README.md              # Tài liệu dự án
````

-----

## Lộ trình & Danh sách Bài thực hành

Dự án này được thiết kế theo lộ trình từ cơ bản đến nâng cao. Dưới đây là danh sách các chủ đề chính (tương ứng với các thư mục trong `notebook/` và `test/`):

1.  **Cơ bản về NLP:** Tiền xử lý văn bản, Tokenization, Regex.
2.  **Biểu diễn văn bản:** Bag-of-Words, TF-IDF, Word Embeddings (Word2Vec, GloVe).
3.  **Mô hình phân loại:** Text Classification (Sentiment Analysis).
4.  **Gán nhãn chuỗi (Sequence Labeling):** POS Tagging & Named Entity Recognition (NER).
5.  **Học sâu (Deep Learning):** RNN, LSTM, GRU cho NLP.
6.  **Mô hình hiện đại:** Giới thiệu về Attention, Transformers và BERT.
7.  **Big Data NLP:** Xử lý dữ liệu văn bản lớn với Apache Spark (trong thư mục `spark_labs/`).

-----

## Bắt đầu 

Để chạy được mã nguồn trong repo này, hãy làm theo các bước sau.

### 1\. Yêu cầu tiên quyết

  * **Python:** Phiên bản 3.8 trở lên.
  * **Java (JDK 8 hoặc 11):** Bắt buộc để chạy các bài Lab về Spark.
  * **SBT (Scala Build Tool):** Để biên dịch project trong `spark_labs`.

### 2\. Cài đặt

**Bước 1: Clone kho lưu trữ**

```bash
git clone [https://github.com/barone04/NLP-Apps.git](https://github.com/barone04/NLP-Apps.git)
cd NLP-Apps
```

**Bước 2: Cài đặt thư viện Python**

```bash
pip install -r requirements.txt
```

**Bước 3: Tải các mô hình ngôn ngữ cần thiết**
Một số bài lab yêu cầu mô hình có sẵn của spaCy:

```bash
python -m spacy download en_core_web_sm
```

-----

## Hướng dẫn Chạy

### Chạy các bài Lab Python

Bạn có hai cách để chạy mã nguồn Python:

**Cách 1: Sử dụng Jupyter Notebook (Khuyên dùng cho học tập)**
Dùng để xem trực quan hóa dữ liệu và chạy từng dòng code.

```bash
jupyter notebook
# Sau đó mở các file .ipynb trong thư mục 'notebook/'
```

**Cách 2: Chạy Scripts kiểm thử/Demo**
Dùng để huấn luyện mô hình hoặc chạy các pipeline hoàn chỉnh.

```bash
# Ví dụ: Demo huấn luyện Word Embedding (Lab 4)
python test/lab4_embedding_training_demo.py

# Ví dụ: Chạy kiểm thử cho Lab 1
python test/lab1_test.py
```

### Chạy các bài Lab Big Data (Apache Spark)

Các bài lab này nằm trong thư mục `spark_labs/` và được cấu trúc như một dự án Scala chuẩn.

1.  Di chuyển vào thư mục:
    ```bash
    cd spark_labs
    ```
2.  Chạy ứng dụng với `sbt`:
    ```bash
    sbt run
    ```
    *Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để tải các thư viện Spark và Scala.*

-----

## Tài nguyên tham khảo 

Dưới đây là các tài liệu và khóa học bổ trợ cho nội dung trong repo này:

**Sách (Books):**

  * *Jurafsky & Martin* - Speech and Language Processing (3rd Ed.)
  * *Manning et al.* - Introduction to Information Retrieval
  * *Ian Goodfellow* - Deep Learning

**Khóa học (Courses):**

  * CS224n: Natural Language Processing with Deep Learning (Stanford)
  * Natural Language Processing Specialization (Coursera/DeepLearning.AI)

**Công cụ & Frameworks:**

  * [PyTorch](https://pytorch.org/) - Deep Learning Framework
  * [Hugging Face Transformers](https://huggingface.co/) - State-of-the-art NLP models
  * [Apache Spark](https://spark.apache.org/) - Unified Engine for large-scale data analytics

-----

## 👤 Tác giả

  * **GitHub:** [barone04](https://www.google.com/search?q=https://github.com/barone04)
  * **Môn học:** NLP & Applications
  * **Trường:** [Tên trường của bạn - Tùy chọn]

-----

## Đóng góp

Mọi đóng góp (Pull Requests) hoặc báo lỗi (Issues) đều được hoan nghênh. Xin vui lòng mở một issue để thảo luận về những thay đổi lớn trước khi thực hiện.

-----

*Kho lưu trữ này phục vụ mục đích giáo dục và nghiên cứu.*
