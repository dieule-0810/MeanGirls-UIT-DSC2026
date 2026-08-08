# Hành vi thực tế của `scoring.py` (BTC)

> Kết quả dò bằng cách chạy trực tiếp hàm `eval_retrieval` của BTC với các input biên.
> Đọc file này trước khi sửa bất cứ dòng nào trong `src/evaluate.py` hoặc `src/make_submission.py`.

Truth dùng để dò: `{"q1": ["100"], "q2": ["200","201"]}`

| Trường hợp nộp | Recall | Precision | Ghi chú |
|---|---|---|---|
| Đúng hoàn toàn | 1.000 | 1.000 | |
| Nộp đủ 5 doc/câu | 1.000 | 0.300 | Recall không đổi, Precision sụt mạnh |
| 6 doc ở q1 | 0.500 | 0.500 | q1 bị gán 0 cả hai, q2 vẫn tính bình thường |
| `answer: []` ở q1 | 0.500 | 0.500 | Giống hệt trường hợp vi phạm |
| **`["100","100","100"]`** | 1.000 | **0.667** | 🔴 Trùng lặp KHÔNG được khử |
| **`[100]` (int)** | **0.000** | **0.000** | 🔴 Không lỗi, không cảnh báo |
| Thiếu 1 câu hỏi | — | — | 💥 `Exception: Samples in predict not match with reference` |
| Đủ số lượng nhưng sai qid | — | — | 💥 `TypeError: object of type 'NoneType' has no len()` |
| Thiếu key `answer` | — | — | 💥 `KeyError: 'answer'` |

---

## Ba kết luận vận hành

### 1. `doc_id` kiểu int → 0 điểm toàn bài, im lặng

Đây là chế độ hỏng nguy hiểm nhất vì **không có tín hiệu nào cả**. Leaderboard hiện 0.0 và team sẽ đi
tìm bug trong mô hình suốt nhiều ngày. `context_*.json` để `id` là int, nhãn để str → khoảng cách này
có thật và rất dễ vấp.

→ `make_submission.py` assert `isinstance(d, str)` cho từng phần tử.

### 2. Trùng lặp tính vào cả giới hạn 5 lẫn mẫu số Precision

Mẫu số Precision là `len(y_pred[k])` — độ dài **list**, không phải set. Điều kiện `<= 5` cũng dùng độ dài list.
Nên `["A","A","A","A","A","A"]` vừa bị coi là vi phạm 6 doc, vừa chỉ đóng góp 1 doc thật.

Trùng lặp rất dễ phát sinh khi gộp kết quả nhiều retriever (RRF, ensemble) — đúng thứ team sẽ làm từ Tuần 2.

→ `make_submission.py` dedupe **giữ nguyên thứ tự** rồi mới cắt còn 5.

### 3. Sai cấu trúc → CRASH, không phải 0 điểm

Ba dạng lỗi cấu trúc đều làm container chấm điểm văng exception. Trên CodaLab điều này hiện ra là
**submission Failed**, và nhiều khả năng vẫn **tiêu một lượt nộp**.

Ở private test chỉ có **3 lượt/ngày** → một submission hỏng vì thiếu 1 câu hỏi là mất 1/3 ngân sách ngày hôm đó.

→ `make_submission.py` bắt buộc nhận `expected_qids` và assert khớp **chính xác** tập khoá, không chỉ số lượng.

---

## Hai điều mã chấm KHÔNG làm

**Không quan tâm thứ tự.** Toàn bộ dùng phép giao tập hợp. Không có MRR, không có NDCG, không có
điểm theo vị trí. Doc đúng nằm ở vị trí 1 hay 5 trong list đều như nhau.

> Hệ quả: đừng tốn công tối ưu thứ tự *bên trong* top-5. Toàn bộ giá trị nằm ở việc **doc đúng có lọt vào
> tập 5 hay không**, và ở **kích thước tập trả về**. (Dòng `return {'mrr': 0.0, ...}` bị comment trong
> `scoring.py` cho thấy phiên bản trước từng dùng MRR — đã bỏ.)

**Không kiểm tra doc_id có tồn tại trong corpus.** Nộp `"999999999"` không gây lỗi, chỉ đơn giản không khớp.
Nên một bug sinh ra id rác sẽ âm thầm làm tụt điểm chứ không báo gì.

→ `make_submission.py` cảnh báo (không chặn) nếu doc_id không có trong `corpus_clean.jsonl`.

---

## Về `metadata.yaml`

```yaml
command: python3 scoring.py
```

Xác nhận BTC chạy đúng `scoring.py` này trong container tại `/app`. Tên file trong bài nộp được đọc từ
`metadata['files']['input']` phía reference (team không thấy được) — theo tài liệu Task là `submission.json`.

→ Giữ đúng `submission.json`, đặt ở **gốc** file zip, không nằm trong thư mục con.
