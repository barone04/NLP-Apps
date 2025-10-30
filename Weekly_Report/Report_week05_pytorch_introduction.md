# Báo Cáo Lab 5: Giới Thiệu PyTorch và Cơ Bản về Tensor & Autograd 📊

Báo cáo này trình bày kết quả thực thi và phân tích các bài tập cơ bản trong PyTorch, bao gồm thao tác Tensor, tự động tính đạo hàm (Autograd) và xây dựng kiến trúc mạng nơ-ron đơn giản (`torch.nn`).

## I. Giải Thích Các Bước Triển Khai (Implementation Steps)

Các bước triển khai được thực hiện thông qua file `lab5-pytorch-introduction.ipynb`, tập trung vào các khái niệm cốt lõi của PyTorch.

### 1. Phần 1: Thao tác cơ bản với Tensor

* **Task 1.1 (Tạo Tensor):** Tensor được tạo từ danh sách Python (`list`) và mảng NumPy (`numpy.array`) bằng `torch.tensor()` và `torch.from_numpy()`. Đồng thời, kiểm tra các thuộc tính cơ bản như `shape`, `dtype`, và `device`.
* **Task 1.2 (Phép toán):** Thực hiện các phép toán cơ bản như cộng phần tử (`+`), nhân vô hướng (`*`), và nhân ma trận (`@`) với toán tử chuyển vị (`.T`).
* **Task 1.3 (Indexing & Slicing):** Truy cập các phần tử, hàng, và cột của Tensor bằng cú pháp slicing tiêu chuẩn của Python (ví dụ: `x_data[0]`, `x_data[:, 1]`).
* **Task 1.4 (Thay đổi hình dạng):** Sử dụng `tensor.reshape(16, 1)` để thay đổi hình dạng của Tensor $4 \times 4$ thành $16 \times 1$.

### 2. Phần 2: Tự động tính Đạo hàm (Autograd)

* **Task 2.1 (Thực hành):** Khởi tạo Tensor `x` với `requires_grad=True` để PyTorch theo dõi các phép toán. Tính toán biến phụ thuộc $z = 3(x+2)^2$. Gọi `z.backward()` để tính đạo hàm $\frac{\partial z}{\partial x}$ và kiểm tra kết quả tại `x.grad`.

### 3. Phần 3: Xây dựng Mô hình với `torch.nn`

* **Task 3.1 (nn.Linear):** Khởi tạo và kiểm tra lớp tuyến tính (`nn.Linear`) để xác nhận sự thay đổi hình dạng đầu vào (input) $(3, 5)$ thành đầu ra (output) $(3, 2)$.
* **Task 3.2 (nn.Embedding):** Khởi tạo và kiểm tra lớp nhúng (`nn.Embedding`) để ánh xạ các chỉ số số nguyên (`input_indices`) thành các vector nhúng dày đặc (dense vectors).
* **Task 3.3 (nn.Module):** Xây dựng lớp **`MyFirstModel`** kế thừa từ `nn.Module`, kết hợp các lớp `nn.Embedding`, `nn.Linear`, và hàm kích hoạt `nn.ReLU()` để tạo ra một kiến trúc mạng nơ-ron cơ bản.

---

## II. Hướng Dẫn Thực Thi Mã (Code Execution Guide)

Các bước thực thi được thực hiện trong môi trường Jupyter/IPython.

1.  **Môi trường:** Đảm bảo thư viện **PyTorch** (`torch`), **NumPy** (`numpy`) đã được cài đặt.
2.  **Thực thi:** Mở file `lab5-pytorch-introduction.ipynb` trong Jupyter Notebook hoặc môi trường tương thích (như Google Colab/Kaggle).
3.  **Xem Kết quả:** Chạy từng ô code theo thứ tự. Kết quả (bao gồm hình dạng, kiểu dữ liệu, kết quả phép toán, đạo hàm và cấu trúc mô hình) sẽ được in ra ngay sau mỗi ô.

---

## III. Phân Tích Kết quả và Giải quyết Thách thức

### 1. Phân tích Task 2.1: Autograd

* **Biểu thức tính toán:** $z = 3(x+2)^2$
* **Giá trị tại $x=1$:** $y = 1+2 = 3$; $z = 3 \times 3^2 = 27$.
* **Tính đạo hàm:** $\frac{\partial z}{\partial x} = 6(x+2)$
* **Giá trị đạo hàm tại $x=1$:** $\frac{\partial z}{\partial x} = 6(1+2) = 18$.
* **Kết quả:** `x.grad` hiển thị giá trị **`tensor([18.])`**, xác nhận tính toán đạo hàm tự động của PyTorch là chính xác.

### 2. Trả lời Câu hỏi Task 2.1 (Backward() lần 2)

* **Vấn đề:** Khi gọi `z.backward()` lần nữa.
* **Kết quả:** Sẽ gây ra lỗi `RuntimeError: grad can be implicitly created only for scalar outputs` (Hoặc lỗi về việc biểu đồ đã bị giải phóng).
* **Phân tích:** PyTorch sử dụng **biểu đồ tính toán động** (dynamic computation graph). Sau khi tính toán đạo hàm ngược (`backward()`) lần đầu tiên, biểu đồ mặc định bị **xóa** để giải phóng bộ nhớ. Do đó, lần gọi thứ hai thất bại vì biểu đồ cần thiết không còn tồn tại.

### 3. Phân tích Cấu trúc Mô hình (Task 3.3)

| Lớp | Chức năng | Input Shape (Ví dụ) | Output Shape (Ví dụ) |
| :--- | :--- | :--- | :--- |
| **`nn.Embedding`** | Ánh xạ chỉ số thành vector 16 chiều. | $(1, 4)$ (Batch size, Sequence length) | $(1, 4, 16)$ |
| **`nn.Linear`** (linear) | Chuyển đổi $16 \to 8$ chiều. | $(1, 4, 16)$ | $(1, 4, 8)$ |
| **`nn.ReLU`** | Hàm kích hoạt phi tuyến. | $(1, 4, 8)$ | $(1, 4, 8)$ |
| **`nn.Linear`** (output) | Chuyển đổi $8 \to 2$ chiều (Output layer). | $(1, 4, 8)$ | $(1, 4, 2)$ |

* **Kết quả:** Mô hình đã được xây dựng thành công, xác nhận kích thước Tensor đầu ra $(1, 4, 2)$ là chính xác, nơi $1$ là batch size, $4$ là sequence length, và $2$ là output dimension.

---

## IV. Thách thức và Giải pháp

| Thách thức | Giải pháp |
| :--- | :--- |
| **Kiểu dữ liệu Tensor** | **Vấn đề:** Lớp `nn.Embedding` yêu cầu đầu vào là Tensor kiểu `LongTensor` (số nguyên) cho các chỉ số. | **Giải pháp:** Sử dụng `torch.LongTensor([1, 5, 0, 8])` để khởi tạo chính xác các chỉ số đầu vào. |
| **Nhân ma trận** | **Vấn đề:** Để nhân ma trận trong PyTorch, cần đảm bảo điều kiện kích thước phù hợp (`n x m` và `m x p`). | **Giải pháp:** Sử dụng toán tử `@` và `.T` (chuyển vị) để thực hiện `x_data @ x_data.T`, đảm bảo ma trận có kích thước nhân hợp lệ. |

---

## VI. Tài liệu Tham khảo

1.  **PyTorch Documentation:** Hướng dẫn chính thức về Tensor, Autograd và `torch.nn`.
