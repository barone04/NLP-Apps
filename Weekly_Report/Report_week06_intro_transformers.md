# Report_Week06 — Intro Transformers


## 1. Các bước triển khai và Ghi log kết quả

Để thực hiện bài lab này, môi trường Google Colab/Local đã được cài đặt
các thư viện cần thiết:

``` bash
!pip install transformers torch tf-keras
```

### Bài 1: Khôi phục Masked Token (Masked Language Modeling)

**Mục tiêu:** Sử dụng mô hình BERT (`bert-base-uncased`) để dự đoán từ
bị che `[MASK]` trong câu: *"Hanoi is the \[MASK\] of Vietnam."*

**Code triển khai:**

``` python
from transformers import pipeline
mask_filler = pipeline("fill-mask", model="bert-base-uncased")
input_sentence = "Hanoi is the [MASK] of Vietnam."
predictions = mask_filler(input_sentence, top_k=5)
# (Code in kết quả như bài mẫu)
```

**Log kết quả thực tế:**

> Câu gốc: Hanoi is the \[MASK\] of Vietnam.
>
> 1.  Từ dự đoán: '**capital**' \| Độ tin cậy: **0.9991** -\> Câu: hanoi
>     is the capital of vietnam.
> 2.  Từ dự đoán: 'center' \| Độ tin cậy: 0.0001
> 3.  Từ dự đoán: 'birthplace' \| Độ tin cậy: 0.0001
> 4.  Từ dự đoán: 'headquarters' \| Độ tin cậy: 0.0001
> 5.  Từ dự đoán: 'city' \| Độ tin cậy: 0.0001

### Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

**Mục tiêu:** Sử dụng mô hình GPT (`gpt2`) để sinh văn bản từ câu mồi
(prompt).

**Code triển khai:**

``` python
from transformers import pipeline, set_seed
generator = pipeline("text-generation", model="gpt2")
prompt = "The best thing about learning NLP is"
output = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
# (Code in kết quả như bài mẫu)
```

**Log kết quả thực tế:**

> Prompt: 'The best thing about learning NLP is' Văn bản sinh ra: "The
> best thing about learning NLP is **it means that you can pick whatever
> you want---whether it's writing in-person or taking a video. Once I
> got to try NLP, I found a lot of interesting content I wanted to learn
> by**"

### Bài 3: Tính toán Vector biểu diễn câu (Sentence Representation)

**Mục tiêu:** Tính vector nhúng (embedding) trung bình của câu bằng
PyTorch và model `bert-base-uncased`.

**Code triển khai:**

``` python
# (Code setup model, tokenizer và thực hiện Mean Pooling như bài mẫu)
# Input sentences được tokenize và đưa vào model
```

**Log kết quả thực tế:**

> Input IDs shape: **torch.Size(\[1, 7\])** Kích thước vector biểu diễn
> câu: **torch.Size(\[1, 768\])** 5 giá trị đầu tiên của vector:
> **tensor(\[-0.2424, -0.3832, -0.0138, -0.2991, -0.2145\])**

------------------------------------------------------------------------

## 2. Giải thích kết quả & Trả lời câu hỏi

### Phân tích Bài 1 (Masked LM)

**1. Mô hình đã dự đoán đúng từ "capital" không?**

-   **Có.** Dự đoán chính xác tuyệt đối.
-   **Giải thích:** Token "capital" có độ tin cậy (score) lên tới
    **0.9991** (tức 99.91%), vượt trội hoàn toàn so với các dự đoán khác
    (chỉ 0.0001). Điều này cho thấy mô hình `bert-base-uncased` đã học
    rất kỹ kiến thức địa lý rằng "Hanoi" là thủ đô của "Vietnam".

**2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ
này?**

-   Do cơ chế **Attention hai chiều (Bidirectional Context)**. Để điền
    đúng từ vào giữa câu, mô hình bắt buộc phải "nhìn thấy" đồng thời cả
    chủ ngữ phía trước ("Hanoi") và bổ ngữ phía sau ("Vietnam"). Kiến
    trúc Encoder cho phép mô hình tiếp cận toàn bộ câu cùng lúc để hiểu
    ngữ cảnh đầy đủ.

### Phân tích Bài 2 (Text Generation)

**1. Kết quả sinh ra có hợp lý không?**

-   **Về ngữ pháp:** Hoàn toàn hợp lý, câu văn trôi chảy, đúng cấu trúc
    tiếng Anh.
-   **Về ngữ nghĩa:** Nội dung hơi lan man ("writing in-person or taking
    a video" - viết trực tiếp hoặc quay video) và chưa thực sự tập trung
    sâu vào kỹ thuật NLP. Câu kết thúc bị cụt ("learn by") do giới hạn
    `max_length=50`.
-   **Đánh giá:** Đây là đặc trưng của dòng GPT-2 (mô hình cũ), nó giỏi
    kết nối từ ngữ trôi chảy nhưng khả năng duy trì logic dài hạn kém
    hơn các mô hình hiện đại như GPT-4.

**2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ
này?**

-   Do cơ chế **Causal Masking (Che giấu tương lai)**. Khi viết văn bản,
    chúng ta viết từ trái sang phải, từ sau nối tiếp từ trước. Decoder
    được thiết kế để chỉ nhìn thấy các từ đã sinh ra trong quá khứ,
    không nhìn thấy tương lai, phù hợp hoàn hảo cho tác vụ sinh từ kế
    tiếp (Next Token Prediction/Autoregressive).

### Phân tích Bài 3 (Sentence Representation)

**1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Tương ứng
tham số nào?**

-   Kích thước vector là **768** (theo log `torch.Size([1, 768])`).
-   Con số này tương ứng với tham số **`hidden_size`** của mô hình
    `bert-base`. Mỗi token đi qua BERT sẽ được mã hóa thành một vector
    768 chiều.

**2. Tại sao chúng ta cần sử dụng `attention_mask` khi thực hiện Mean
Pooling?**

-   Để đảm bảo tính toán trung bình chính xác. Trong lô dữ liệu (batch),
    các câu ngắn được thêm các token đệm (padding, giá trị 0) để bằng độ
    dài câu dài nhất. Nếu tính trung bình cộng cả các số 0 này, vector
    kết quả sẽ bị sai lệch giá trị thực. `attention_mask` giúp đánh dấu
    vị trí từ thật để chỉ tính toán trên đó.

------------------------------------------------------------------------

## 3. Khó khăn gặp phải và Cách giải quyết

Trong quá trình thực hiện, tôi đã gặp và xử lý các vấn đề sau:

1.  **Lỗi tương thích thư viện:**

    -   *Vấn đề:* Code báo lỗi `RuntimeError` liên quan đến `keras 3`
        khi import transformers.
    -   *Giải quyết:* Đã cài đặt gói tương thích ngược bằng lệnh
        `pip install tf-keras` và khởi động lại môi trường.

2.  **Sự khác biệt về Token đặc biệt:**

    -   *Vấn đề:* Ban đầu pipeline mặc định (thường là RoBERTa) báo lỗi
        thiếu token `<mask>`.
    -   *Giải quyết:* Đã chỉ định rõ `model="bert-base-uncased"` trong
        pipeline để sử dụng đúng token `[MASK]` theo yêu cầu bài tập.

3.  **Số lượng Token đầu vào (Bài 3):**

    -   *Vấn đề:* Log hiển thị `Input IDs shape: [1, 7]`.
    -   *Giải thích:* Câu đầu vào *"This is a sample sentence"* (không
        có dấu chấm) sẽ được tách thành: `[CLS]`, `this`, `is`, `a`,
        `sample`, `sentence`, `[SEP]`. Tổng cộng đúng 7 token. Điều này
        giúp tôi hiểu rõ hơn về cách Tokenizer xử lý dấu câu và các
        token đặc biệt.

------------------------------------------------------------------------

## 4. Nguồn tham khảo & Model

-   **Model sử dụng:**
    -   `bert-base-uncased`: Mô hình BERT cơ bản (chữ thường) của
        Google.
    -   `gpt2`: Mô hình GPT thế hệ 2 của OpenAI.
-   **Nguồn tham khảo:**
    -   Hugging Face Transformers Documentation:
        https://huggingface.co/docs/transformers
    -   Tài liệu bài giảng môn học NLP.