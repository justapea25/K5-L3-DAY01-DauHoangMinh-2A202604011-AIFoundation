# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604011
- **Họ và tên:** Dau Hoang Minh
- **SEED** (in ở ô đầu notebook): 6281

## 1. Bằng chứng

| Tệp                                                                       | Có? |
| -------------------------------------------------------------------------- | ---- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☑   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước        | ☑   |

Bộ kiểm tra tự động: 15/15 đạt. Đủ 10 dự đoán, không thiếu dự đoán nào.

## 2. Kết quả chính

| Cách                                      | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| ------------------------------------------ | ------: | --------: | ---------: | ---------: |
| 1. BoW + LogReg                            |   0.914 |     0.919 |      0.050 |          0 |
| 2. BoW + MLP                               |   0.956 |     0.974 |      0.093 |          0 |
| 3a. Encoder (MiniLM) + kNN                 |   0.879 |     0.974 |    565.659 |          0 |
| 3b. Encoder (MiniLM) + LogReg              |   0.943 |     0.947 |      9.035 |          0 |
| 3c. Encoder (MiniLM) + MLP head            |   0.952 |     0.923 |      8.262 |          0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`)    |  0.974* |     0.974 |   1899.229 |     0.0075 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) |  0.974* |     0.974 |   1805.414 |     0.0750 |
| 4c. LLM nhỏ few-shot                      |  0.974* |     0.974 |   1423.679 |     0.0220 |

\* Hàng LLM chỉ đo trên 40 tin của tập SUB, nên cần so với các cách khác bằng cột F1 SUB 40. Với 40 tin, chênh lệch F1 dưới khoảng 0.05 được coi là hòa. Chi phí LLM dựa trên ước tính token (số từ / 0.75).

- **Đạt p95 < 50 ms:** BoW + LogReg, BoW + MLP, 3b và 3c.
- **Đường Pareto:** BoW + LogReg và BoW + MLP.

## 3. Dự đoán và thực tế

**`hard_eval` — độ chính xác của LLM trên tập khó**

- **Đã đoán:** LLM zero-shot 80–90%, LLM phòng thủ 85–95%.
- **Thực tế:** cả hai đạt 100% trên 50 case gốc.
- **Vì sao lệch:** đánh giá thấp khả năng đọc hiểu ý định của LLM, và các kiểu bypass trong bộ này khá đơn giản với model. Kết quả này không có nghĩa LLM miễn nhiễm: khi thêm 6 case tự viết, 4a và 4d đều sai 3/56 case.

**`hard_eval` — nhóm khó nhất với BoW**

- **Đã đoán:** né từ khóa, vì token không khớp vocab.
- **Thực tế:** spam giọng tự nhiên khó nhất (LogReg 10%, MLP 0%). Né từ khóa đạt LogReg 17%, MLP 0%.
- **Vì sao lệch:** tin giọng tự nhiên không có từ khóa spam quen thuộc nên BoW không có tín hiệu để bắt, không chỉ các tin bị biến đổi ký tự mới làm khó nó.

**`ml_lr` — đường loss với `lr=50`**

- **Đã đoán:** loss giảm vọt rồi dao động mạnh ở mức cao.
- **Thực tế:** loss bắt đầu khoảng 2.5, giảm mạnh rồi giảm mượt về 0.004, không dao động.
- **Vì sao lệch:** bước đầu lớn làm `w` tăng mạnh. `w` lớn làm sigmoid bão hòa, phần lớn tin được đoán đúng với p gần 0 hoặc 1, nên error gần 0.

## 4. Phân tích 50 case khó

Số dưới đây tính trên 50 case gốc.

**BoW + MLP tụt nhiều nhất:** F1 từ 0.974 xuống 0.324 (giảm 0.650). Tiếp theo là BoW + LogReg, từ 0.919 xuống 0.333 (giảm 0.586). Nhóm Embedding giảm khoảng 0.2–0.25, còn LLM không giảm (0.974 lên 1.000).

Điểm yếu nằm ở spam giọng tự nhiên, thao túng và né từ khóa:

| Nhóm                   | BoW + LogReg | BoW + MLP |
| ----------------------- | -----------: | --------: |
| Spam giọng tự nhiên  |          10% |        0% |
| Thao túng / lừa đảo |          12% |       25% |
| Né từ khóa           |          17% |        0% |

Hai case BoW + MLP đoán sai:

- **Case #29 (né từ khóa):** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m`. Chữ bị thay bằng số nên các token như `w0n`, `cl4im` không khớp với từ trong vocab. Model mất các từ vốn đẩy về spam.
- **Case #16 (spam giọng tự nhiên):** `Long time no see! I've started a little side business with skincare, you'd love it, can I send you the catalogue link?`. Tin viết như bạn cũ và không có từ khóa spam. BoW chỉ đếm từ rời rạc, mất thứ tự và ngữ cảnh, nên không nhận ra ý định bán hàng.

**Nhận xét chung:** test thường có cùng phân phối với train và có spam mẫu gần giống nhau ở cả hai tập, nên model có thể đạt điểm cao chỉ nhờ nhớ từ khóa.

**Khi thêm 6 case tự viết (56 case):**

| Model                        |    F1 | Accuracy |
| ---------------------------- | ----: | -------: |
| BoW + LogReg                 | 0.300 |    50.0% |
| BoW + MLP                    | 0.326 |    48.2% |
| 3a. Emb + kNN                | 0.758 |    71.4% |
| 3b. Emb + LogReg             | 0.679 |    67.9% |
| 3c. Emb + MLP head           | 0.643 |    64.3% |
| 4a. LLM zero-shot            | 0.955 |    94.6% |
| 4d. LLM + prompt phòng thủ | 0.954 |    94.6% |

## 5. Thử thách cá nhân

**Đề #6:** sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết hàm parse JSON, chạy lại trên 10 tin SUB, so token và chi phí với bản 1 từ, rồi trả lời có đáng không.

**Cách làm:** sửa prompt để yêu cầu chỉ trả JSON có hai trường `label` và `reason`. Viết `parse_classification_response`, trả nhãn `unknown` khi JSON lỗi hoặc nhãn không hợp lệ. Gọi `gpt-6-luna` với `max_tokens=150` trên `texts_te[SUB[:10]]`.

**Kết quả:**

- Độ chính xác:
  - Cả 10 tin đều có JSON đọc được: 5 spam, 5 ham, không có `unknown`.
  - Đúng 10/10 tin, 0 unknown nhưng test data vẫn còn quá ít.
  - `reason` khá hợp lý và có xét ngữ cảnh, không chỉ dựa vào từ khóa. Tin 0 được nhận ra là lời xin lỗi cá nhân, tin 8 là hình thức dụ người nhận gửi SMS để nhận tiền thưởng.
- Độ trễ trung bình là 1.419 s/tin, so với khoảng 1.20 s/tin của bản 1 từ (4a, đo trên 40 tin).
- So token và chi phí:
  - Bản JSON có token ra ước tính là 21, so với 1.3 của bản 1.4 từ. Chi phí dự kiến cũng sẽ tăng hơn, khoảng gấp đôi so với bản có 1 từ.
- Chi phí và độ trể có tăng một chút nhưng mà đáng để thêm vào vì giúp người kiểm duyệt và phát triển hiểu đúng cách model suy luận để rà soát kiểm tra khiếu nại.

## 6. Phương án cho sản phẩm thật

**Chọn:** 3b (Encoder + LogReg) làm mô hình chính, thêm router sang LLM cho các tin lưỡng lự.

**Số liệu:**

- **Độ trễ:** 3b có p95 9.0 ms, đạt yêu cầu. kNN (566 ms) và LLM (1424–1899 ms) không đạt.
- **Chất lượng:** trên test thường, BoW + MLP (F1 0.956), 3b (0.943) và 3c (0.952) coi như hòa. Trên tập khó (56 case), BoW + MLP tụt còn 0.326, còn 3b đạt 0.679.
- **Chi phí nếu chỉ dùng LLM** (300 triệu tin/tháng, số xấp xỉ): 4a khoảng $2,250, 4c khoảng $6,600, 4b khoảng $22,500.
- **Rào cản chính của LLM là độ trễ:** chạy song song khoảng 139 lời gọi (116 × 1.2 s) giải quyết được thông lượng, nhưng p95 mỗi tin vẫn khoảng 1.4–1.9 s, gấp khoảng 28–38 lần SLA.

**Thiết kế router:** chuẩn hóa đầu vào → 3b phân loại → tin có p trong khoảng [0.3, 0.7] chuyển sang LLM dùng prompt phòng thủ → nếu LLM timeout thì lấy nhãn của 3b. Ví dụ chuyển 4% số tin thì sẽ giảm thiểu được chi phí đi , so với $2,250 nếu dùng LLM cho mọi tin.

**Đánh đổi chấp nhận:**

- 3b chậm hơn BoW + MLP (9.0 ms so với 0.09 ms).
- F1 tập khó của 3b vẫn thấp (0.679), nên cần router hỗ trợ.
- Thêm chi phí API và độ phức tạp vận hành cho nhánh LLM.

## 7. AI log và điều đã học



| Phần | Bạn hỏi AI gì (prompt)                                                                                                      | AI trả lời/viết gì                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Bạn đã sửa / kiểm chứng gì                                                                                                                               |
| ----- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4     | "Cách chạy đo cho model nhận diện spam với model GPT6-luna, cần đánh giá kêt quả trên những phương diện nào" | **Cách chạy đo**Dùng đúng tập test: chạy trên tập test mà model chưa thấy và không lấy ví dụ few-shot từ đó. Nếu cần tiết kiệm, lấy một tập con cố định để các lần so sánh dùng cùng dữ liệu.Cố định cấu hình: dùng temperature=0, max_tokens nhỏ nhưng đủ cho nhãn, và với GPT-6 thì đặt reasoning_effort="none". Ghi lại phiên bản prompt và model.Gọi từng tin: với mỗi tin, tạo prompt, gọi API (có retry khi gặp lỗi tạm thời), rồi parse câu trả lời thành spam, ham hoặc không rõ.Ghi log từng lần gọi: lưu dự đoán, câu trả lời gốc, độ trễ và số token.So sánh với các mốc: chạy thêm zero-shot và few-shot, model lớn và model nhỏ, cùng một baseline rẻ như BoW + LogReg.Ghi dự đoán trước khi chạy: cách này giúp bạn không giải thích theo kết quả sau khi đã thấy số.**Các phương diện cần đánh giá**Chất lượng: Precision, recall, F1, confusion matrix — Model chặn nhầm và bỏ sót bao nhiêu tin?Độ trễ: p50, p95 — Có đạt yêu cầu SLA không?Thông lượng: Số tin/giây — Có theo kịp lưu lượng thật không, kể cả rate limit?Chi phí: USD/1.000 tin, chi phí/tháng — Khi chạy ở quy mô thật thì hết bao nhiêu tiền?Độ tin cậy của đầu ra: Tỉ lệ n_invalid, từ chối trả lời, lỗi API — Có bao nhiêu câu trả lời không dùng được?Độ bền: F1 theo từng nhóm case khó — Model có chịu được né từ khóa, lừa đảo giọng tự nhiên và prompt injection không?Độ ổn định: Mức thay đổi giữa các lần chạy — Chạy lại có ra cùng kết quả không? | Mình có chạy và kiểm tra kêt quả thực tế, có bị chặn khi max_token không, kết quả thực tế so với phần mdoel embedding và BoW trước ra sao |

Qua bài này t hiểu được cách so sánh và đánh giá các phương pháp cho một model đơn giản nhận diện spam, mỗi phương pháp sẽ có những ưu nhược điểm khác nhau, cần phải so sánh và phân tích các sự tradeoff để đưa ra phương án cho thực tế
