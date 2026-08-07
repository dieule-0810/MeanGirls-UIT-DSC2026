# INTERFACES.md — Hợp đồng dữ liệu giữa 4 thành viên

> **Chốt ngày 07/08/2026. Đây là phần KHÓA của repo.**
> Cây thư mục có thể đổi bất cứ lúc nào (`git mv` là xong). Những định nghĩa dưới đây thì không —
> đổi một dòng ở đây là làm hỏng code của 3 người còn lại. Muốn đổi: nêu ở standup, cả team đồng ý, sửa file này TRƯỚC khi sửa code.

---

## 0. Quy ước tối thượng: `doc_id` LUÔN LÀ `str`

Mã chấm của BTC dùng `set(y_true[k]) & set(y_pred[k])`. Nhãn trong `train.json` là **string** (`"177504"`),
nhưng trường `id` trong `context_*.json` là **int** (`177504`).

Nếu nộp `[177504]` thay vì `["177504"]`:

```
{'precision': 0.0, 'recall': 0.0}     ← KHÔNG có lỗi, KHÔNG có cảnh báo
```

Đã kiểm chứng bằng cách chạy trực tiếp `scoring.py` của BTC. Đây là cách mất trắng một lượt nộp mà không ai
biết tại sao. Vì vậy:

- **Ép `str(...)` ngay tại điểm đọc** `context_*.json`, không ép ở cuối pipeline.
- Mọi hàm nhận/trả `doc_id` đều dùng `str`.
- `make_submission.py` có assert chặn kiểu, fail loud.

`qid` (khoá câu hỏi) cũng luôn là `str` — nó vốn là khoá JSON nên đã là str, chỉ cần đừng cast sang int ở giữa đường.

---

## 1. `corpus_clean.jsonl` — P2 sản xuất

Một dòng = một văn bản. **8.532 dòng, không hơn không kém.**

```jsonc
{
  "doc_id": "740",          // str, BẮT BUỘC. Lấy từ trường `id`, ép str.
  "name": "Quyet-dinh-...", // str | null. context_69.json THIẾU trường này → dùng .get()
  "link": "https://...",    // str | null
  "text": "BỘ Y TẾ ..."     // str, đã normalize, đã bỏ boilerplate
}
```

Sinh bởi: `python -m src.data.parse_corpus`

**Kiểm tra bắt buộc sau khi chạy**: script tự in ra số dòng và checksum SHA-256. Ghi checksum vào README —
đây là thứ để BTC (và chính team, 6 tuần sau) xác minh bước tiền xử lý không đổi.

---

## 2. `chunks.jsonl` — P2 sản xuất

Một dòng = một chunk.

```jsonc
{
  "chunk_id": "740::0003",  // str = f"{doc_id}::{position:04d}". Duy nhất toàn corpus.
  "doc_id": "740",          // str, trỏ ngược về corpus_clean
  "position": 3,            // int, thứ tự trong văn bản, bắt đầu từ 0
  "text": "Điều 3. ..."     // str
}
```

Sinh bởi: `python -m src.data.chunker`

> `chunk_id` tách bằng `::` chứ không phải `_` vì `doc_id` là số và ta cần tách ngược được chắc chắn.

---

## 3. Retriever — P3 sở hữu

Mọi retriever (BM25, dense, hybrid) đều phải khớp giao diện này. P4 viết reranker dựa trên nó,
P1 viết eval dựa trên nó — không ai được biết retriever bên trong là gì.

```python
class BaseRetriever:
    def index(self, chunks: list[dict]) -> None:
        """chunks: đọc từ chunks.jsonl. Gọi một lần."""

    def search(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        """
        Trả về, CHO MỖI query, danh sách (doc_id, score) đã:
          - gộp từ chunk lên document (chiến lược gộp là chuyện nội bộ của retriever)
          - loại trùng doc_id
          - sắp xếp score giảm dần
          - cắt còn top_k
        doc_id là str. len(result) == len(queries).
        """
```

**Gộp chunk → doc là trách nhiệm của retriever, không phải của người gọi.** Lý do: mỗi retriever có thang điểm
khác nhau (BM25 vs cosine), cách gộp tối ưu cũng khác. Ép ra ngoài sẽ rò rỉ chi tiết nội bộ.

---

## 4. Predictions — định dạng nội bộ toàn pipeline

```python
Predictions = dict[str, list[str]]   # {qid: [doc_id, ...]}
```

Phẳng, không lồng. `{"answer": ...}` **chỉ xuất hiện ở bước cuối** trong `make_submission.py`.
Đừng để cấu trúc của BTC lan vào trong code.

---

## 5. Evaluation — P1 sở hữu

```python
from src.evaluate import eval_official

eval_official(preds: Predictions, truth: dict[str, list[str]]) -> dict
# → {'recall': float, 'precision': float}
```

Đây là **bản sao chính xác** logic `scoring.py` của BTC, giữ nguyên mọi hành vi biên (kể cả những chỗ trông
như bug). Không ai được "sửa cho hợp lý". Nếu thấy chỗ nào lạ → đọc `docs/scoring_behaviour.md`.

---

## 6. Config

Một file YAML cho mỗi thí nghiệm, đặt trong `configs/`. **Không hằng số hard-code trong code.**
Mọi script nhận `--config configs/xxx.yaml`.

```yaml
exp_id: v0.1_bm25          # trùng tên file, ghi vào experiments.csv
seed: 42
paths:
  corpus_raw: data/selected-contexts
  corpus_clean: data/corpus_clean.jsonl
  chunks: data/chunks.jsonl
```

---

## 7. Tên file trung gian — cố định, không ai được đổi

| File | Ai sinh | Ai dùng |
|---|---|---|
| `data/corpus_clean.jsonl` | P2 | P3, P4 |
| `data/chunks.jsonl` | P2 | P3, P4 |
| `data/holdout.json` | P2 | tất cả |
| `data/train_split.json` | P2 | P3, P4 |
| `data/embeddings.npy` | P3 | P3 |
| `outputs/<exp_id>/predictions.json` | P3/P4 | P1 |
| `outputs/<exp_id>/submission.zip` | P1 | CodaLab |

---

## 8. Ai chạm vào file nào

Tránh conflict khi 4 người làm song song:

| Thư mục | Chủ sở hữu | Người khác |
|---|---|---|
| `src/data/` | P2 | PR + review |
| `src/retrieval/` | P3 | PR + review |
| `src/rerank/` | P4 | PR + review |
| `src/evaluate.py`, `src/make_submission.py`, `src/verify_env.py` | **P1, khoá** | không sửa trực tiếp, báo P1 |
| `configs/` | ai tạo thí nghiệm thì tạo file mới | không sửa file của người khác |
| `INTERFACES.md` | cả team đồng thuận | — |
