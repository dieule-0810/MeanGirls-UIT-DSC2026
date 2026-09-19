# AGENTS.md

Hướng dẫn cho Codex khi làm việc trong repo này.

## 1. Repo này là gì

Bài dự thi **DSC2026 — Task 1: Legal Information Retrieval** (truy hồi văn bản pháp luật tiếng Việt).
Input: 1 câu hỏi. Output: **tối đa 5 `doc_id`** từ corpus 8.532 văn bản. Metric chính **Recall**,
tie-break **Precision** (`src/evaluate.py`).

Team 4 người, vai trò = **quyền sở hữu file**, xem `plan.md` mục 4:

| Vai | Trách nhiệm | Thư mục sở hữu |
|---|---|---|
| P1 | Infra, eval, submission, release/Docker, nộp bài | `src/evaluate.py`, `src/make_submission.py`, `src/verify_env.py` (**KHOÁ**) |
| P2 | Parser, normalize, chunker, holdout, hard negative | `src/data/` |
| P3 | First-stage retrieval — KPI **Recall@50** | `src/retrieval/` |
| P4 | Rerank, calibration số lượng doc, error analysis | `src/rerank/` |

Ngôn ngữ tài liệu và comment trong repo: **tiếng Việt**. Giữ nguyên quy ước đó khi thêm code.

## 2. Ba tài liệu phải đọc trước khi sửa code

1. `INTERFACES.md` — hợp đồng dữ liệu giữa 4 người. Đây là phần **khoá** của repo; sửa một dòng ở đó
   là làm hỏng code của 3 người còn lại. Đổi thì phải sửa `INTERFACES.md` **trước**, có đồng thuận team.
2. `docs/scoring_behaviour.md` — hành vi thật của mã chấm BTC, đo bằng cách chạy `vendor/btc_scoring/scoring.py`.
3. `plan.md` — kế hoạch 6 tuần, ma trận thực nghiệm, ràng buộc bài báo khoa học (mục 0.6).

## 3. Bốn bất biến không được vi phạm

Tất cả đều đã được kiểm chứng trên mã chấm thật (`tests/test_scoring.py`, fixtures ở `tests/fixtures/`):

1. **`doc_id` và `qid` LUÔN là `str`.** `context_*.json` để `id` là int, nhãn là str; mã chấm dùng
   `set(a) & set(b)` nên `100 != "100"` → **0 điểm im lặng, không lỗi, không cảnh báo**. Ép `str()`
   **tại điểm đọc**, không ép ở cuối pipeline.
2. **Trùng lặp không được khử bởi mã chấm.** Mẫu số Precision là `len(list)`, không phải `len(set)`;
   giới hạn 5 cũng đếm theo list. Dedupe **giữ thứ tự** rồi mới cắt 5.
3. **> 5 doc, list rỗng → câu đó 0 cả Recall lẫn Precision** (không huỷ cả submission — xem `docs/over_limit_verdict.txt`).
4. **Thiếu qid / thừa qid / thiếu key `answer` → mã chấm CRASH** → CodaLab báo *Failed* và có thể vẫn tiêu
   một lượt nộp (private test chỉ 3 lượt/ngày).

Hệ quả cho việc tối ưu: mã chấm **không dùng thứ tự** (không MRR/NDCG). Đừng tối ưu thứ tự bên trong
top-5 — giá trị nằm ở "doc đúng có lọt vào tập hay không" và ở "kích thước tập trả về".

## 4. Lệnh hay dùng

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.11.x, verify_env fail nếu khác
pip install -r requirements.txt                        # lõi: numpy, scipy, PyYAML
pip install -r requirements-p3.txt                     # tuỳ chọn: pyvi / underthesea cho tokenizer P3

python scripts/fetch_data.py --check-only    # dữ liệu BTC đã đủ chưa
python scripts/fetch_data.py --from ~/Downloads   # đã tải tay từ Drive → chỉ cần chỉ chỗ
python -m src.data.parse_corpus              # → data/corpus_clean.jsonl  (8532 dòng)
python -m src.data.chunker                   # → data/chunks.jsonl        (525023 chunk)
python -m src.data.split_data                # → holdout 1000 / train_split 5689 / error_pool 300

python -m src.verify_env            # phải in "✅ Môi trường OK"
python scripts/smoke_test.py        # verify_env + toàn bộ pytest — CHẠY TRƯỚC MỌI THỨ
python -m pytest tests/ -q          # test riêng
python scripts/run_v0.1.py          # pipeline end-to-end → outputs/v0.1_bm25/submission.zip

# P3 — first-stage retrieval
python -m src.retrieval.bm25 --config configs/v0.1_bm25.yaml \
    --questions data/holdout.json --out outputs/tmp/preds.json
python scripts/bench_retrieval.py --config configs/v0.2_bm25_tokenizer.yaml   # lưới tokenizer × pooling
python scripts/bench_retrieval.py --demo                                      # chạy trên corpus giả, không cần data/
```

## 5. Trạng thái hiện tại (cập nhật khi đổi)

- Có đủ: `src/data/` (P2), `evaluate.py` / `make_submission.py` / `verify_env.py` (P1),
  khung `BaseRetriever` + BM25 + tokenizer tiếng Việt (P3), EDA 12 mục, test mã chấm.
- `data/` **không bao giờ commit** (dữ liệu BTC, `.gitignore` đã chặn `data/`, `*.json`, `*.jsonl`, checkpoint).
  Dữ liệu nằm trên Drive team → `scripts/fetch_data.py` (xem `configs/data_sources.yaml`).

### Ba chỗ đang gãy / lệch, cần chốt ở standup (đừng vá lặng lẽ rồi quên)

1. **Tên trường lệch INTERFACES.md.** `src/data/` ghi `id`/`passage`, hợp đồng ghi `doc_id`/`text`;
   `chunks.jsonl` không có `position` và `chunk_id` không zero-pad (`740::3` thay vì `740::0003`).
   `src/common/io.py` đang dịch tạm để tầng retrieval chạy được. Nhưng `src/evaluate.py:179` và
   `src/make_submission.py:173` (file KHOÁ của P1) đọc thẳng `["doc_id"]` → **`KeyError` nếu truyền `--corpus`**.
   Hoặc P2 đổi tên trường, hoặc sửa `INTERFACES.md` trước rồi sửa cả ba nơi — phải chọn một.
2. **`scripts/run_v0.1.py` không chạy được nữa**: nó gọi `python -m src.data.parse_corpus --config …`
   nhưng module của P2 không nhận `--config`, và gọi `src.data.split_holdout` trong khi file thật tên
   `split_data.py`. Chạy từng bước bằng tay cho tới khi P1/P2 sửa.
3. **Thiếu `src/data/__init__.py`** — hiện vẫn import được nhờ namespace package, nhưng đó là may mắn
   chứ không phải thiết kế.

## 6. Quy ước code

- **Không hard-code hằng số.** Mỗi thí nghiệm = 1 file YAML trong `configs/`, script nhận `--config`.
- Mọi retriever khớp `BaseRetriever` (`INTERFACES.md` mục 3): `index(chunks)` và
  `search(queries, top_k) -> list[list[tuple[str, float]]]`, đã gộp chunk→doc, đã dedupe, đã sort giảm dần,
  đã cắt `top_k`. **Gộp chunk→doc là việc nội bộ của retriever**, không đẩy ra ngoài.
- Định dạng dự đoán nội bộ toàn pipeline: `Predictions = dict[str, list[str]]` (phẳng). Cấu trúc
  `{"answer": [...]}` của BTC **chỉ xuất hiện** trong `make_submission.py`.
- Tên file trung gian cố định (`INTERFACES.md` mục 7) — không tự đổi.
- Fail loud: thà dừng với thông báo rõ còn hơn nộp một file sai im lặng.

## 7. Git

`git_rules.md` là bản đầy đủ. Tóm tắt:

- Nhánh `<role>/<mo-ta-ngan>`, ví dụ `p3/feature-tokenizer-vi`. **Không push thẳng `main`**, luôn PR + 1 review.
  PR đụng file KHOÁ của P1 thì bắt buộc P1 review.
- Conventional Commits: `<type>(<scope>): <subject>` — type `feat|fix|exp|docs|style|refactor|test|chore`,
  scope gợi ý `data|retrieval|rerank|eval|submission|docker|docs`. Subject viết tiếng Việt.
- Mỗi thí nghiệm ghi **một dòng** vào `experiments.csv` (13 cột). Ba cột cuối bắt buộc điền, không được
  ghi chung chung kiểu "cải thiện recall": `nhom_so_sanh` (`co_dien|dl_from_scratch|llm_leverage_embed|
  llm_leverage_rerank|chien_luoc_du_lieu`), `gia_thuyet_lien_quan`, `diem_yeu_khac_phuc_tu_exp_truoc`.
  File này hay conflict → `git pull --rebase` trước khi thêm, không xoá dòng người khác.

## 8. Ràng buộc từ BTC cần nhớ khi đề xuất giải pháp

- **Cấm gọi API mô hình/dịch vụ bên thứ ba bên trong pipeline dự thi.** Trọng số phải tự tải, tự chạy.
- Ngân sách **< 4B tham số cho toàn pipeline**, tính cả lớp embedding. LoRA/quantization không hợp lệ hoá
  model > 4B. Model phải được BTC duyệt trước (danh sách + revision hash: `docs/param_audit.md`).
- Không augmentation, không dữ liệu ngoài.
- Pin **revision hash** của repo HuggingFace trong config, không chỉ tên repo.
- Kết quả phải tái lập: mọi submission thật sinh ra từ một Git tag.
