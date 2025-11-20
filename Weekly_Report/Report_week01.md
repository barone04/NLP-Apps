# Report_Week01 — TOKENIZATION VÀ COUNT VECTORIZATION.

**Học phần:** Xử lý ngôn ngữ tự nhiên (NLP) **Bài thực hành:** Lab 1 & Lab 2

## 1. Các bước triển khai

Dựa trên yêu cầu của Lab 1 và Lab 2, hệ thống đã được triển khai theo
cấu trúc mô-đun với các bước cụ thể như sau:

### Bước 1: Thiết lập cấu trúc và Interface (Giao diện)

Trước khi đi vào chi tiết thuật toán, tôi đã định nghĩa các lớp cơ sở
trừu tượng (abstract base classes) để đảm bảo tính nhất quán.

-   Tạo file `src/core/interfaces.py`.
-   Định nghĩa lớp `Tokenizer` với phương thức trừu tượng
    `tokenize(self, text: str) -> list[str]`.
-   Định nghĩa lớp `Vectorizer` với các phương thức trừu tượng: `fit`,
    `transform`, và `fit_transform`.

### Bước 2: Triển khai Tokenization (Lab 1)

Mục tiêu là chuyển đổi văn bản thô thành danh sách các tokens.

-   **Simple Tokenizer**
    -   Tạo file `src/preprocessing/simple_tokenizer.py`
    -   Thực hiện: lowercase, split theo khoảng trắng và tách dấu câu cơ
        bản.
-   **Regex Tokenizer**
    -   Tạo file `src/preprocessing/regex_tokenizer.py`
    -   Sử dụng regex `\w+|[^\w\s]` để tách từ mạnh mẽ hơn.

### Bước 3: Triển khai Count Vectorization (Lab 2)

Mục tiêu là biểu diễn văn bản dưới dạng vector Bag-of-Words.

-   Tạo file `src/representations/count_vectorizer.py`
-   `CountVectorizer` nhận đối tượng `Tokenizer`.
-   **fit**: xây vocabulary từ corpus.
-   **transform**: chuyển văn bản thành vector đếm.

## 2. Cách chạy code và log kết quả

### 2.1 Kiểm thử Lab 1

``` bash
python main.py
```

**Log mô phỏng:**

``` text
--- Testing Tokenizers ---
Input: "Hello, world! This is a test."

[SimpleTokenizer Output]:
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

[RegexTokenizer Output]:
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: "NLP is fascinating... isn't it?"
[SimpleTokenizer Output]:
['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']

[RegexTokenizer Output]:
['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
```

### 2.2 Kiểm thử Lab 2

Test file: `test/lab2_test.py`

**Corpus mẫu:**

``` python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
```

**Log mô phỏng:**

``` text
--- Testing CountVectorizer ---
Using Tokenizer: RegexTokenizer

1. Learned Vocabulary:
i, love, nlp, ., programming, is, a, subfield, of, ai

2. Document-Term Matrix:
Doc 1: [1,1,1,1,0,0,0,0,0,0]
Doc 2: [1,1,0,1,1,0,0,0,0,0]
Doc 3: [0,0,1,1,0,1,1,1,1,1]
```

## 3. Giải thích kết quả

-   SimpleTokenizer hoạt động tốt nhưng kém trong trường hợp phức tạp.
-   RegexTokenizer tách từ chi tiết hơn.
-   BOW tạo vector thưa, mất thứ tự từ.

## 4. Khó khăn và cách giải quyết

1.  **Tách dấu câu:** thêm khoảng trắng hoặc xử lý thủ công.
2.  **Thứ tự vocabulary thay đổi:** sort token.
3.  **Token OOV:** bỏ qua khi transform.

## 5. Nguồn tham khảo

- Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed. draft).

    Tham khảo Chương 2 (Regular Expressions, Text Normalization, Edit Distance) để hiểu sâu về quy trình chuẩn hóa văn bản và Tokenization.

    Tham khảo Chương 6 (Vector Semantics) để nắm bắt lý thuyết về biểu diễn văn bản dạng vector.

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

    Tham khảo Chương 6 (Scoring, term weighting and the vector space model) để hiểu rõ mô hình Bag-of-Words và cách xây dựng ma trận Document-Term.

- Scikit-learn Documentation. Feature extraction: Text features.

    Tham khảo cách thiết kế lớp CountVectorizer trong thư viện Scikit-learn để mô phỏng lại giao diện (interface) và logic fit/transform trong bài thực hành này.

    Link: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
