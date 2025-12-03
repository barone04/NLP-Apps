# Report_Week07 — Lab X: TỔNG QUAN VỀ CÔNG NGHỆ TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)

**Môn học:** [Natural Language Processing and Applications]  
**Tuần:** 12 - Nội dung bổ sung: Kỹ năng tự nghiên cứu  
**Sinh viên thực hiện:** Phan Đinh Thái Bảo

---

## 1. Giới thiệu chung và Bối cảnh

Tổng hợp tiếng nói (Text-To-Speech - TTS) là bài toán chuyển đổi văn bản đầu vào thành tín hiệu âm thanh lời nói tương ứng. Trong bối cảnh trí tuệ nhân tạo (AI) phát triển bùng nổ, TTS không chỉ dừng lại ở việc đọc văn bản mà đã tiến tới khả năng sao chép giọng nói (Voice Cloning) và biểu đạt cảm xúc phức tạp.

Báo cáo này tập trung phân tích 3 cấp độ phát triển của TTS, các thách thức kỹ thuật, giải pháp pipeline tối ưu và khía cạnh đạo đức AI.

---

## 2. Các hướng tiếp cận và Mức độ phát triển (Levels)

### Level 1: Phương pháp dựa trên luật và Thống kê (Rule-based / Statistical Parametric)
Đây là thế hệ TTS đầu tiên, tiêu biểu là kỹ thuật **Concatenative Synthesis** (ghép nối từ vựng) và **HMM** (Hidden Markov Models).

* **Đặc điểm:** Hệ thống hoạt động dựa trên việc ghép nối các đoạn âm thanh nhỏ hoặc sinh tham số dựa trên luật thống kê.
* **Ưu điểm:** Tốc độ phản hồi cực nhanh, yêu cầu phần cứng thấp, hoạt động ổn định.
* **Nhược điểm:** Giọng nói thiếu tự nhiên, ngữ điệu "robot", khó tạo cảm xúc.
* **Tài liệu tham khảo:** *Zen, H., et al. (2009). "Statistical parametric speech synthesis".*

### Level 2: Deep Learning & Cá nhân hóa (Neural TTS with Fine-tuning)
Sự ra đời của Deep Learning đã chuyển dịch TTS sang hướng **Neural Rendering** (Tacotron, FastSpeech).

* **Cách triển khai (Pipeline):**
    1.  Huấn luyện một mô hình gốc (Base Model) trên tập dữ liệu lớn.
    2.  Người dùng ghi âm một lượng dữ liệu nhỏ (vài chục phút).
    3.  Thực hiện **Fine-tuning** (tinh chỉnh) trọng số để model học đặc trưng giọng người đó.
* **Ưu điểm:** Độ tự nhiên rất cao, tiệm cận giọng người thật. Mô hình sau khi fine-tune chạy tốn ít tài nguyên hơn so với các mô hình Zero-shot lớn.
* **Nhược điểm:** Yêu cầu người dùng phải bỏ công sức ghi âm và chờ thời gian huấn luyện.
* **Tài liệu tham khảo:** *Shen, J., et al. (2018). "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions".*

### Level 3: Few-shot / Zero-shot Learning (Large Audio Models)
Hướng tiếp cận hiện đại nhất (VALL-E, XTTS), coi bài toán TTS tương tự như bài toán mô hình ngôn ngữ lớn (LLM).

* **Cách triển khai:** Sử dụng cơ chế **In-context Learning**. Chỉ cần cung cấp mẫu âm thanh 3-5 giây (prompt), mô hình tự động "clone" giọng nói mà không cần cập nhật trọng số (Zero-shot).
* **Ưu điểm:** Trải nghiệm người dùng tốt nhất (chỉ cần vài giây ghi âm), khả năng sao chép giọng tức thì.
* **Nhược điểm:** Tốn tài nguyên tính toán lớn (Computationally expensive), độ trễ cao hơn, độ ổn định chưa tuyệt đối.
* **Tài liệu tham khảo:** *Wang, C., et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers".*

---

## 3. Tổng hợp Ưu/Nhược điểm và Use Cases

| Level | Công nghệ lõi | Ưu điểm | Nhược điểm | Trường hợp sử dụng |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | HMM / Ghép nối | Nhanh, nhẹ, rẻ | Thiếu tự nhiên, máy móc | Thiết bị nhúng, IoT, thông báo công cộng. |
| **Level 2** | Neural Net + Fine-tune | Tự nhiên, tối ưu tài nguyên inference | Cần dữ liệu training, thời gian chờ | Trợ lý ảo, Sách nói, Voice-over chuyên nghiệp. |
| **Level 3** | Audio LLM / Zero-shot | Tiện lợi, linh hoạt, clone nhanh | Nặng, tốn GPU, rủi ro deepfake | Game, Metaverse, Lồng tiếng phim tự động. |

---

## 4. Thách thức và Tối ưu hóa Pipeline

Để tối đa hóa ưu điểm và giảm thiểu nhược điểm, các nghiên cứu hiện đại tập trung vào:

### 4.1. Tốc độ và Tài nguyên
* **Vấn đề:** Các mô hình tự nhiên (Level 2, 3) thường nặng và chậm.
* **Giải pháp:** Chuyển từ mô hình *Autoregressive* (sinh tuần tự) sang *Non-autoregressive* (sinh song song) như **FastSpeech 2**. Sử dụng kỹ thuật **Knowledge Distillation** để nén mô hình lớn (Teacher) thành mô hình nhỏ (Student) để chạy trên điện thoại.

### 4.2. Tính đa ngôn ngữ (Multilingual)
* **Giải pháp:** Sử dụng biểu diễn âm vị quốc tế (**IPA**) làm trung gian. Điều này giúp mô hình học được đặc trưng phát âm chung, cho phép một giọng nói tiếng Việt có thể "nói" tiếng Anh trôi chảy (Cross-lingual) mà không cần dữ liệu huấn luyện song ngữ lớn.

### 4.3. Cảm xúc và Ngữ điệu
* **Giải pháp:** Tích hợp module **Prosody Control**. Trích xuất vector cảm xúc (Style Embeddings) từ giọng mẫu và điều khiển cao độ (pitch), năng lượng (energy) của giọng sinh ra.

---

## 5. Đạo đức nghiên cứu (AI Ethics)

Sự phát triển của Level 3 tạo ra nguy cơ **Deepfake** và giả mạo giọng nói. Cộng đồng nghiên cứu đề xuất các tiêu chuẩn:

1.  **Audio Watermarking:** Nhúng các tín hiệu ẩn (không nghe thấy bằng tai thường) vào đầu ra của AI. Các hệ thống phát hiện (Detector) có thể nhận diện watermark để phân biệt giọng thật/giả.
2.  **Quản lý truy cập:** Các mô hình Zero-shot mạnh mẽ thường được cung cấp dưới dạng API có kiểm soát thay vì công khai mã nguồn (Open source) hoàn toàn để tránh lạm dụng.

---

## 6. Kết luận

Nghiên cứu TTS đang ở giai đoạn chuyển giao thú vị. Trong khi **Level 1** vẫn giữ vai trò nền tảng cho các thiết bị cấu hình thấp, **Level 2** đang là tiêu chuẩn vàng cho các ứng dụng thương mại cần chất lượng cao. **Level 3** mở ra tương lai của sự sáng tạo không giới hạn nhưng đi kèm trách nhiệm về quản lý đạo đức và tối ưu hóa chi phí vận hành.

---

## 7. Tài liệu tham khảo

1.  Zen, H., Tokuda, K., & Black, A. W. (2009). Statistical parametric speech synthesis. *Speech Communication*.
2.  Ren, Y., et al. (2019). FastSpeech: Fast, Robust and Controllable Text to Speech. *NeurIPS*.
3.  Wang, C., et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E). *arXiv*.
4.  Tan, X., et al. (2021). A Survey on Neural Speech Synthesis. *arXiv*.