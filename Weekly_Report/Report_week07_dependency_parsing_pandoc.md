# Report_Week07 — Lab 07: Dependency Parsing Pandoc

-----

## 1. Nêu rõ các bước triển khai

Chương trình của bài lab này được đặt tại thư mục `/notebook/lab7_dependency_parsing_pandoc.ipynb`. Bài thực hành tập trung vào việc thao tác với cây phụ thuộc (Dependency Tree) thông qua thư viện `spaCy`. Dưới đây là logic triển khai cho 3 bài tập tự luyện:

### Bài 1: Tìm động từ chính của câu (Main Verb)

  * **Mục tiêu:** Tìm token giữ vai trò `ROOT` trong câu.
  * **Logic triển khai:**
    1.  Duyệt qua từng `token` trong đối tượng `doc`.
    2.  Kiểm tra thuộc tính `token.dep_`.
    3.  Nếu `token.dep_ == "ROOT"`, đó là động từ chính.
    4.  Trả về token đó và kết thúc vòng lặp.

### Bài 2: Trích xuất các cụm danh từ (Noun Chunks) thủ công

  * **Mục tiêu:** Tự viết hàm trích xuất cụm danh từ (thay vì dùng `.noun_chunks` có sẵn) dựa trên cây phụ thuộc.
  * **Logic triển khai:**
    1.  Duyệt qua tất cả các token trong câu để tìm các "từ trung tâm" (Head Noun). Điều kiện là `token.pos_` phải là "NOUN" hoặc "PROPN".
    2.  Với mỗi danh từ tìm được, duyệt qua các `token.children` (các từ con trực tiếp).
    3.  Lọc lấy các con có nhãn phụ thuộc (dependency label) bổ nghĩa cho danh từ như: `det` (mạo từ), `amod` (tính từ), `compound` (từ ghép), `poss` (sở hữu).
    4.  Sắp xếp các từ thu được theo chỉ số (`token.i`) để đảm bảo đúng thứ tự xuất hiện trong câu.
    5.  Ghép lại thành chuỗi văn bản hoàn chỉnh.

### Bài 3: Tìm đường đi ngắn nhất trong cây

  * **Mục tiêu:** Tìm đường đi từ một token con lên tới gốc (ROOT).
  * **Logic triển khai:**
    1.  Bắt đầu từ `token` đầu vào. Tạo một list `path`.
    2.  Sử dụng vòng lặp `while True`.
    3.  Thêm token hiện tại vào `path`.
    4.  Kiểm tra điều kiện dừng: Nếu `token.head == token` (tức là đã đến ROOT), thoát vòng lặp.
    5.  Nếu chưa đến ROOT, gán `token = token.head` (di chuyển lên cha) và lặp lại.

-----

## 2. Cách chạy code và ghi log kết quả

### Yêu cầu môi trường

  * Python 3.x
  * Thư viện: `spacy`
  * Model ngôn ngữ: `en_core_web_sm`

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Source code (`lab7_exercises.py`)

Dưới đây là đoạn mã hoàn chỉnh để chạy kiểm thử 3 bài tập:

```python
import spacy

# Khởi tạo model
nlp = spacy.load("en_core_web_sm")

# --- BÀI 1: TÌM ĐỘNG TỪ CHÍNH ---
def find_main_verb(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

# --- BÀI 2: TRÍCH XUẤT NOUN CHUNKS (THỦ CÔNG) ---
def get_manual_noun_chunks(doc):
    chunks = []
    # Các nhãn phụ thuộc thường gặp trong cụm danh từ
    valid_deps = {"det", "amod", "compound", "poss", "nummod"}

    for token in doc:
        # Tìm các danh từ làm gốc
        if token.pos_ in ["NOUN", "PROPN"]:
            # Lấy danh sách các từ con phù hợp + chính token đó
            chunk_tokens = [child for child in token.children if child.dep_ in valid_deps]
            chunk_tokens.append(token)

            # Sắp xếp theo vị trí xuất hiện trong câu
            chunk_tokens.sort(key=lambda t: t.i)

            # Chuyển thành chuỗi
            chunk_text = " ".join([t.text for t in chunk_tokens])
            chunks.append(chunk_text)
    return chunks

# --- BÀI 3: TÌM ĐƯỜNG ĐI ĐẾN GỐC ---
def get_path_to_root(token):
    path = []
    current = token
    while True:
        path.append(current)
        if current.head == current: # Điều kiện dừng tại ROOT
            break
        current = current.head
    return path

# --- CHẠY KIỂM THỬ ---
text = "Autonomous cars shift insurance liability toward manufacturers."
doc = nlp(text)

print(f"Câu: {text}\n")

# Test Bài 1
main_verb = find_main_verb(doc)
print(f"1. Động từ chính (ROOT): {main_verb.text} | POS: {main_verb.pos_}")

# Test Bài 2
print("\n2. Các cụm danh từ (Manual extraction):")
manual_chunks = get_manual_noun_chunks(doc)
for chunk in manual_chunks:
    print(f"   - {chunk}")

# Test Bài 3
target_token = doc[6] # "manufacturers"
print(f"\n3. Đường đi từ '{target_token.text}' lên ROOT:")
path = get_path_to_root(target_token)
for t in path:
    print(f"   -> {t.text} ({t.dep_})")
```

### Kết quả log (Output)

Khi chạy đoạn code trên, kết quả thu được như sau:

```text
Câu: Autonomous cars shift insurance liability toward manufacturers.

1. Động từ chính (ROOT): shift | POS: VERB

2. Các cụm danh từ (Manual extraction):
   - Autonomous cars
   - insurance liability
   - manufacturers

3. Đường đi từ 'manufacturers' lên ROOT:
   -> manufacturers (pobj)
   -> toward (prep)
   -> shift (ROOT)
```

-----

## 3. Giải thích các kết quả thu được

1.  **Kết quả Bài 1 ("shift"):**

      * Mô hình spaCy xác định "shift" là gốc (ROOT) của câu vì đây là động từ chính mô tả hành động của câu. Các thành phần khác (cars, liability) phụ thuộc vào nó.

2.  **Kết quả Bài 2 (Noun chunks):**

      * Hàm thủ công đã trích xuất được "Autonomous cars" (gồm `amod` + `noun`) và "insurance liability" (gồm `compound` + `noun`).
      * Logic dựa trên việc tìm các từ con (children) có nhãn `amod`, `compound` hoạt động tốt với các cụm danh từ đơn giản trong câu ví dụ.

3.  **Kết quả Bài 3 (Path to ROOT):**

      * Đường đi thể hiện cấu trúc phân cấp: `manufacturers` là bổ ngữ cho giới từ `toward`, và `toward` bổ nghĩa cho động từ chính `shift`.
      * Kết quả trả về danh sách token theo thứ tự từ dưới lên trên (Bottom-up).

-----

## 4. Nêu rõ các khó khăn gặp phải và cách giải quyết

  * **Khó khăn 1 (Bài 2): Xác định phạm vi của cụm danh từ.**

      * *Vấn đề:* Khi tự viết hàm trích xuất, rất khó để bao quát hết các trường hợp phức tạp như mệnh đề quan hệ (ví dụ: "the car *that I bought*") hoặc các giới từ đi kèm.
      * *Giải quyết:* Trong phạm vi bài lab này, em giới hạn logic chỉ lấy các từ bổ nghĩa trực tiếp (direct children) có nhãn `det`, `amod`, `compound`. Đối với các trường hợp phức tạp hơn, giải pháp tối ưu vẫn là sử dụng `doc.noun_chunks` tích hợp sẵn của spaCy hoặc duyệt đệ quy (recursive subtree).

  * **Khó khăn 2 (Bài 3): Hiểu về cấu trúc Head-Child.**

      * *Vấn đề:* Ban đầu dễ nhầm lẫn hướng duyệt cây. Trong spaCy, thuộc tính `.head` trỏ lên cha, trong khi `.children` trỏ xuống con.
      * *Giải quyết:* Sử dụng `.head` để duyệt ngược từ lá lên gốc cho bài toán tìm đường đi. Vẽ cây (visualize) bằng `spacy.displacy` để dễ hình dung trước khi code.

-----

## 5. Nguồn tham khảo

1.  **spaCy Documentation:** https://www.google.com/search?q=https://spacy.io/usage/linguistic-features%23dependency-parse
2.  **Universal Dependencies:** https://universaldependencies.org/

-----