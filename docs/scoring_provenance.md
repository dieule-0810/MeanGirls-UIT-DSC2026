# Nguồn gốc mã chấm của BTC

> Bắt buộc theo `tests/test_scoring.py::test_provenance_recorded`.
> Nếu `vendor/btc_scoring/scoring.py` đổi mà file này không đổi theo → CI đỏ.

| Trường | Giá trị |
|---|---|
| File | `vendor/btc_scoring/scoring.py` |
| sha256 | `84d7e4459469b9d7d183d577b30c9668a184a7d4a7382a5fd8a8ae7379d7083a` |
| Nguồn | `Scoring-Program-Task-LegalIR.zip` — bundle chính thức của BTC |
| Khớp bundle | ✅ hash file trong `vendor/` **trùng** hash file trong bundle |
| `metadata.yaml` | `command: python3 scoring.py` — xác nhận đúng file này được chạy |
| Timestamp trong zip | `scoring.py` 2026-08-03 · `metadata.yaml` 2025-08-19 (template cũ) |
| Số phiên bản tồn tại | **Một.** BTC không phát hành bản thứ hai. |
| Người dò | P4, 13/08/2026 |

## Về câu "BTC thêm ràng buộc vào mã nguồn"

Q&A của BTC có câu: *"BTC quyết định thêm ràng buộc vào trong mã nguồn của chương
trình đánh giá"*. Câu này **không** có nghĩa là tồn tại hai phiên bản mà các đội
lần lượt nhận được. Bundle phát hành cho thí sinh đã có sẵn luật `<= 5` ngay từ
đầu; các đội chưa bao giờ cầm bản nào khác.

Vì vậy **không cần truy ngày tải**. Câu hỏi đáng hỏi không phải "P1 tải lúc nào"
mà là "file trong repo có đúng là file BTC chạy không" — và hash trả lời trực
tiếp câu đó, mạnh hơn mọi bằng chứng về thời điểm.

`test_provenance_recorded` vì thế kiểm **hash khớp**, không kiểm ngày tháng.

---

## Kết quả dò — chạy trực tiếp `eval_retrieval`

Fixture tại `tests/fixtures/`. Tái lập: `pytest tests/test_scoring.py -v -s`.

### Một câu hỏi, gold = `{"q1": ["100"]}`

| Nộp gì | Recall | Precision | Ghi chú |
|---|---:|---:|---|
| `["100"]` | 1.0000 | 1.0000 | |
| `["100","2","3","4","5"]` | 1.0000 | 0.2000 | 1/5 |
| `["100","100","100"]` | 1.0000 | 0.3333 | 1/3 — trùng lặp không khử |
| `[100]` (int) | 0.0000 | 0.0000 | 🔴 im lặng, không lỗi |
| `[]` | 0.0000 | 0.0000 | không crash |
| 6 doc | 0.0000 | 0.0000 | chỉ câu này bị 0 |
| 5 doc / 5 gold | 1.0000 | 1.0000 | **5 hợp lệ**, luật là `> 5` |

### Precision theo số doc nộp (gold luôn nằm trong tập)

| k | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Recall | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Precision | 1.000 | 0.500 | 0.333 | 0.250 | **0.200** |

Recall bất biến theo k → **nộp thừa không được thưởng**. Đây là nền tảng của bộ
calibration Tuần 5: mọi doc thứ 2–5 chỉ có thể làm giảm Precision, trừ khi nó
thật sự là gold.

### Fixture gốc của P1, gold = `{"q1": ["100"], "q2": ["200","201"]}`

| Nộp gì | Recall | Precision |
|---|---:|---:|
| 5 doc mỗi câu, trúng hết | 1.0000 | 0.3000 |
| q1 = `["100","100","100"]`, q2 đúng | 1.0000 | 0.6667 |
| q1 đúng, q2 nộp 6 doc | 0.5000 | 0.5000 |

**Hai con số 0.300 và 0.667 trong `plan.md` mục 0.4 là ĐÚNG** — trung bình trên
hai câu có số gold khác nhau: `mean(1/5, 2/5) = 0.300` và `mean(1/3, 2/2) = 0.667`.
Bảng 0.4 chỉ thiếu chú thích rằng fixture có 2 câu hỏi. Không có lỗi số học.

> ⚠️ Vì 0.300 là hiện vật của fixture 2 câu, **không được dùng nó làm Precision
> kỳ vọng của tập test**. Trên phân bố gold thật (`eda_notes.md` mục 4:
> Σ|gold| = 7.637 trên 7.000 câu), trần Precision khi luôn nộp 5 doc là
> `1,0910 / 5 = 0,2182`, và thực tế còn thấp hơn vì Recall < 1.

### Lỗi cấu trúc — CRASH, không phải 0 điểm

| Nộp gì | Kết quả |
|---|---|
| Thiếu 1 câu hỏi | in `Samples in predict not match with reference` → `Exception` |
| Đủ số lượng nhưng sai qid | `TypeError: object of type 'NoneType' has no len()` |
| Thừa qid nhưng đủ đếm | `TypeError` (như trên) |
| Thiếu key `answer` | `KeyError: 'answer'` |

Traceback luôn chỉ vào **dòng 31** (recall) kể cả khi nguyên nhân là thừa qid,
vì recall duyệt `ids_truth` trước và chạm `y_pred.get(k) → None` sớm hơn.
Đừng đi tìm ở dòng precision.

---

## Bốn kết luận đọc thẳng từ mã nguồn

### 1. Vi phạm >5 doc → zero theo TỪNG CÂU

Ba nguồn độc lập đồng thuận: (a) `scoring.py` dòng 31–32 dùng list comprehension
`for k in ids_truth`, không có nhánh nào cho toàn bài; (b) chạy thật ra
`0.5000/0.5000`; (c) Data Overview mục 5: *"gán Recall và Precision bằng 0 cho
câu hỏi đó… trung bình trên tất cả các câu hỏi, bao gồm cả câu vi phạm"*.

Câu *"kết quả của submission sẽ là 0"* trong Q&A là cách nói tắt, không phải đặc tả.

**Đường "cả submission = 0" vẫn tồn tại**, nhưng ở nhánh exception (thiếu/sai/thừa
qid, thiếu key `answer`) → container chấm chết → **Failed**, tệ hơn 0 điểm. Đây
mới là chế độ hỏng cần canh.

→ `make_submission.py` vẫn fail-loud khi thấy câu >5 doc. Không phải vì hình phạt
nặng — dưới per-question chỉ mất 1/1000 điểm — mà vì **>5 doc là dấu hiệu tầng
fusion hỏng**, và cắt âm thầm sẽ giấu bug đó vĩnh viễn. Chốt chặn này giữ kể cả
nếu BTC bỏ luật >5.

### 2. Kiểm tra qid chỉ so SỐ LƯỢNG, không so tập khoá

```python
if len(ids_preds) != len(ids_truth):
```

Nộp đủ 1.000 câu nhưng một qid lệch → lọt qua kiểm tra này rồi chết ở `len(None)`.
→ `make_submission.py` phải assert **bằng nhau về tập hợp**, không phải về `len`.
Lỗi này chỉ nổ ở private test nếu tập qid khác public.

### 3. `set(y_true[k]) & set(y_pred[k])` — so sánh theo KIỂU

`100 != "100"`, và **cả hai chiều**: gold int / pred str cũng 0 điểm. Corpus để
`id` là int, nhãn để str → khoảng cách này có thật.
→ Ép `str()` **tại điểm đọc corpus**, không phải ở bước cuối.

### 4. Mẫu số Precision là `len(y_pred[k])` — độ dài LIST

Trùng lặp vừa chiếm chỗ trong hạn mức 5, vừa phình mẫu số. RRF/ensemble sinh
trùng lặp tự nhiên từ Tuần 2.
→ `dedupe_keep_order()` **rồi mới** cắt còn 5. Làm ngược lại sẽ để lọt < 5 doc thật.

## Hai điều mã chấm KHÔNG làm

**Không dùng thứ tự.** Toàn bộ là phép giao tập hợp. Dòng `return {'mrr': 0.0, ...}`
bị comment cho thấy phiên bản trước từng dùng MRR — đã bỏ.
→ Đừng tối ưu thứ tự *bên trong* top-5. Giá trị nằm ở việc gold có lọt vào tập 5
hay không, và ở **kích thước** tập trả về.

**Không kiểm tra doc_id có tồn tại trong corpus.** Nộp `"999999999"` không gây lỗi,
chỉ đơn giản không khớp. Bug sinh id rác sẽ âm thầm làm tụt điểm.
→ `make_submission.py` cảnh báo (không chặn) nếu doc_id không có trong `corpus_clean.jsonl`.

## Về `main()`

```python
input_data = read_json(os.path.join(prediction_dir, metadata['files']['input']))
```

Tên file bài nộp đọc từ `metadata.json` phía reference (đội thi không thấy được).
Theo tài liệu Task là `submission.json`. Sai tên → `FileNotFoundError` → Failed.
→ Giữ đúng `submission.json`, đặt ở **gốc** zip, không nằm trong thư mục con.
