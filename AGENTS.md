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
pip install -r requirements-dense.txt                  # torch, transformers — cần cho v0.8 (dense)
pip install -r requirements-p3.txt                     # tuỳ chọn: pyvi / underthesea cho lưới tokenizer

python scripts/smoke_test.py        # verify_env + toàn bộ pytest (170 test) — CHẠY TRƯỚC MỌI THỨ
python -m pytest tests/ -q          # test riêng

# Dữ liệu (P2) — docs/exclusion_decisions.json do eda.py sinh, bị .gitignore chặn
python scripts/eda.py --out outputs/eda/eda_notes.md          # → docs/exclusion_decisions.json
python -m src.data.parse_corpus                               # → data/corpus_clean.jsonl (8.507 dòng)
python -m src.data.chunker --strategy strict --out data/chunks.jsonl   # 432.142 chunk — kho của v0.8
python -m src.data.chunker                                    # strategy loose: 524.422 chunk — kho v0.1–v0.6
python -m src.data.split_data                                 # → holdout 1000 / dev 1000 / error_pool 300 / train_split 4689

# Pipeline chính thức v0.8 (BM25 + dense, RRF mức doc, calibrate)
python scripts/run_e2e.py --questions data/private-official.json          # một lệnh, từ dữ liệu thô
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml --questions data/dev.json --eval
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_demo.yaml --demo --eval   # không cần data/, torch

# Chọn siêu tham số — chọn trên train_split, báo cáo MỘT lần trên dev
python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --limit-fit 0
python scripts/fit_calibration.py --ranking <train_split ranking_full.json> --questions data/train_split.json \
    --verify-ranking <dev ranking_full.json> --verify-questions data/dev.json --max-recall-drop 0.003

# Embedding (GPU, một lần)
python -u scripts/encode_corpus.py --config configs/v0.4_dense.yaml --resume       # hoặc --shard K/N trên Kaggle

# P3 — lưới tokenizer × pooling
python scripts/bench_retrieval.py --demo
python scripts/bench_retrieval.py --config configs/v0.2_bm25_tokenizer.yaml --questions data/dev.json

# Tái lập bài nộp
python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml            # kiểm hash + cấu trúc
python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml --commands # in chuỗi lệnh đã chạy
```

## 5. Trạng thái hiện tại (cập nhật 09/10/2026)

- **Pipeline chính thức: v0.8** (`configs/v0.8_hybrid_rrf.yaml`) — BM25 `syllable_bigram` + dense
  `AITeamVN/Vietnamese_Embedding_v2`, RRF mức doc (k=20, w 0,4/0,6), bộ quyết định số lượng doc
  (θ=0,502793). dev: R@5 0,9384 · BTC recall 0,9354 / precision 0,2451. Chưa đo holdout, chưa
  có kê khai/tag cho lượt nộp private v0.8.
- **`data/chunks.jsonl` hiện là kho `strict`** (432.142 chunk). Số v0.1–v0.6 đo trên kho `loose`
  (524.422). Embedding đã encode trên kho `strict`; dùng kho khác thì `DenseRetriever` dừng.
- **Config kế thừa bằng `extends`** (`src/common/config.py`). Mọi script đọc config qua
  `load_config()` — KHÔNG `yaml.safe_load()` thẳng (sẽ thấy khoá `extends` mà không thấy phần kế
  thừa). `configs/base.yaml` giữ `paths` + `top_k` chung; `v0.3_bm25_best.yaml` là khối BM25 chuẩn,
  `v0.4_dense.yaml` là khối dense chuẩn, nguồn của hybrid kế thừa hai khối đó bằng
  `extends: file.yaml#retrieval.bm25`.
- **Hợp nhất nhiều nguồn là `src/retrieval/hybrid.py`, một `type: hybrid` trong YAML — KHÔNG phải
  một tầng của runner** (INTERFACES §3). `scripts/tune_rrf.py` chốt `w`/`rrf_k` trên `train_split`
  rồi đo **một lần** trên `dev`, và in **độ lạc quan** của việc quét trên chính tập đo.
- Nguồn kNN câu hỏi (v0.6, private LB 0,8818) vẫn là script rời `scripts/p5_knn_fuse.py`, CHƯA
  được đưa vào `hybrid.py` — chưa đo v0.8 + kNN.
- Mỗi bài nộp thật cần một bản kê khai `docs/releases/*.yaml`; kiểm bằng `scripts/verify_release.py`,
  lượt nộp mới thì thêm khối `runs:` rồi chạy `--record`.
- `data/` **không bao giờ commit** (dữ liệu BTC). Cài đặt dữ liệu: README mục 4.

### Chỗ còn treo (đừng vá lặng lẽ rồi quên)

1. Lượt nộp private v0.8 (`outputs/private_hybrid_doc_calibrated/`) chưa có kê khai, chưa có tag,
   chưa ghi điểm LB vào `experiments.csv`. Repo hiện **không có git tag nào** (kể cả tag mà kê khai
   v0.6 nhắc tới).
2. Điểm BM25 lệch ~1e-4 giữa hai tiến trình vì thứ tự từ vựng phụ thuộc `PYTHONHASHSEED`
   (thứ hạng không đổi). Cần khớp từng byte thì đặt `PYTHONHASHSEED=0`.
3. Ma trận bài báo còn trống ô `dl_from_scratch` (docs/pipeline_e2e_plan.md vòng 4).

## 6. Quy ước code

- **Không hard-code hằng số.** Mỗi thí nghiệm = 1 file YAML trong `configs/`, chỉ ghi phần KHÁC
  config nó kế thừa; script nhận `--config` và đọc bằng `src.common.config.load_config`.
- Mọi retriever khớp `BaseRetriever` (`INTERFACES.md` mục 3): chỉ viết `index(chunks)` và
  `_score_chunks(queries, n)`; khung lo gộp chunk→doc, dedupe, sort, cắt `top_k`, `check_contract()`.
  Tham số gom trong dataclass (`PoolingConfig`, `BM25Config`, `EncoderConfig`, `FusionConfig`);
  dựng từ YAML qua `Retriever.from_spec(dict)` / `build_retriever(spec)`, không gọi constructor
  với hàng chục kwargs.
- Định dạng dự đoán nội bộ toàn pipeline: `Predictions = dict[str, list[str]]` (phẳng). Cấu trúc
  `{"answer": [...]}` của BTC **chỉ xuất hiện** trong `make_submission.py`.
- Helper dùng chung: `src.common.runinfo` (`git_commit`, `guard_holdout`, `append_experiment_rows`),
  `src.common.io` (`load_chunks`, `load_questions`, `load_labelled`, `load_corpus_ids`),
  `src.common.demo` (corpus giả cho `--demo`). Đừng chép lại trong script mới.
- Docstring theo **Google style** (dòng tóm tắt ngay sau `"""`, rồi `Args:` / `Returns:` /
  `Raises:` khi tham số không hiển nhiên), nội dung tiếng Việt.
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
