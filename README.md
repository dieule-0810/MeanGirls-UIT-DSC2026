# DSC2026 — Task 1: Legal Information Retrieval

> ⚠️ **README này là deliverable chính gửi BTC.** Nó được *kiểm thử*, không phải được *viết*.
> Quy tắc: người dry-run chỉ được làm đúng những gì ghi ở đây. Vướng chỗ nào thì ghi vào mục 10
> — mỗi câu phải hỏi là một lỗ hổng đã tìm ra. Máy sạch để dry-run: notebook Kaggle/Colab mới
> (x86_64, không state cũ).

Đầu vào: một câu hỏi pháp luật tiếng Việt. Đầu ra: **tối đa 5 `doc_id`** trong kho 8.532 văn bản
(8.507 sau khi loại văn bản rỗng/trùng). Chấm bằng Recall (chính) và Precision (phá hoà) theo đúng
`scoring.py` của BTC — hành vi biên của mã chấm: `docs/scoring_behaviour.md`.

---

## 1. Tóm tắt phương pháp

**Bản hiện tại: v0.8** — `configs/v0.8_hybrid_rrf.yaml`

```
câu hỏi ─┬─► BM25 (syllable_bigram, gộp mean_top2) ──► top-200 doc ─┐
         │                                                            ├─► RRF mức doc (k=20, w 0,4/0,6)
         └─► Dense Vietnamese_Embedding_v2 (cosine, mean_top2) ──► top-200 doc ─┘          │
                                                                                            ▼
                                       bộ quyết định số lượng doc (khe hở điểm, θ=0,502793) ──► 1..5 doc_id
```

1. **Kho chunk**: chunk theo tiêu đề `Điều N.`, Điều dài thì cắt cửa sổ trượt 256 từ / gối 64
   bên trong Điều; văn bản không có Điều thì cắt cửa sổ trượt toàn văn bản → 432.142 chunk.
2. **BM25** trên chunk, tokenizer âm tiết + bigram âm tiết (không cần thư viện tách từ), gộp
   chunk→doc bằng trung bình 2 chunk tốt nhất.
3. **Dense** zero-shot `AITeamVN/Vietnamese_Embedding_v2` (pin revision), cosine trên chunk,
   cùng cách gộp. Embedding kho encode một lần (`scripts/encode_corpus.py`).
4. **Hợp nhất RRF** trên thứ hạng doc của hai nguồn. `w` và `k` chọn trên `train_split`, báo
   cáo một lần trên `dev` (`scripts/tune_rrf.py`).
5. **Bộ quyết định số lượng doc**: cắt ở khe hở điểm đã chuẩn hoá đầu tiên vượt θ; θ fit trên
   `train_split` với ngân sách giảm recall ≤ 0,003 (`scripts/fit_calibration.py`).

Không fine-tune, không reranker, không dữ liệu ngoài, không augmentation, không API bên thứ ba.

| Release | Phương pháp | dev R@5 | dev BTC R / P | holdout R@5 | LB R / P | Kho chunk |
|---|---|---|---|---|---|---|
| v0.1 | BM25 regex, gộp max | 0,7863 | 0,7863 / 0,1642 | — | — | 524.422 |
| v0.3 | BM25 syllable_bigram, mean_top2 | 0,8555 | 0,8555 / 0,1792 | 0,8449 | public 0,8566 / — | 524.422 |
| v0.4 | + bge-reranker-v2-m3, RRF | 0,8733 | — | 0,8559 | public 0,8566 / 0,1838 | 524.422 |
| v0.6 | + kNN câu hỏi train, RRF 3 nguồn | 0,9028 | — | 0,8855 | **private 0,8818 / 0,1884** | 524.422 |
| **v0.8** | BM25 + dense, RRF + calibrate | **0,9384** | **0,9354 / 0,2451** | _(chưa đo)_ | _(chưa ghi)_ | 432.142 |

Nguồn số: `experiments.csv`, `outputs/v0.8_hybrid_rrf/tune_rrf.md`,
`outputs/v0.8_hybrid_doc/calibration.json`. Kê khai bài nộp: `docs/releases/`.

**Ngân sách tham số**: 567.754.752 (chỉ model dense; BM25 và RRF không có tham số học được) —
14,2% trần 4 tỷ của BTC. Chi tiết: `docs/model_card.md`, `docs/param_audit.md`.

---

## 2. Yêu cầu hệ thống

| Hạng mục | Giá trị |
|---|---|
| OS | Ubuntu 22.04 / macOS 14+ (x86_64 hoặc ARM64) |
| Python | **3.11.x** — `verify_env.py` dừng nếu khác |
| GPU | Chỉ cần cho bước encode kho chunk (`encode`). Truy vấn chạy được trên CPU / Apple MPS |
| RAM | ≥ 16 GB (ma trận BM25 ~1,1 GB + embedding fp32 ~1,65 GB) |
| Đĩa trống | ≥ 10 GB (embedding 1,77 GB, cache token, trọng số model 2,3 GB) |

**Thời gian chạy** (MacBook M4, đo 09/10/2026 trừ dòng có ghi chú):

| Bước | Thời gian |
|---|---|
| Chunking 8.507 văn bản (`--strategy strict`) | 12 giây |
| Encode 432.142 chunk | _ước tính_ ~2–2,5 giờ trên T4 (docs/runbook_e2e.md) — làm một lần |
| `run_pipeline.py` trên 1.000 câu dev (index BM25 từ cache token + nạp embedding + truy vấn + calibrate) | 1 phút 30 giây |
| Lần đầu tách từ BM25 (chưa có cache) | _chưa đo lại_ — vài phút |

---

## 3. Cài đặt

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt                           # numpy, scipy, PyYAML
pip install -r requirements-dense.txt                     # torch, transformers — cần cho v0.8
python -m src.verify_env                                  # phải in "✅ Môi trường OK"
```

`requirements-p3.txt` (pyvi, underthesea) **không** cần cho đường nộp bài — chỉ dùng cho lưới
benchmark tokenizer.

---

## 4. Chuẩn bị dữ liệu

Giải nén dữ liệu BTC vào `data/`:

```
data/
├── selected-contexts/        ← giải nén selected-contexts.zip (8.532 file context_*.json)
├── train.json
├── public-official.json
└── private-official.json
```

Dữ liệu **không** được commit (`.gitignore` chặn `data/`) — đây là dữ liệu của BTC.

---

## 5. Trọng số

Không có bước tải riêng: `transformers` tự tải `AITeamVN/Vietnamese_Embedding_v2` ở **revision
`18b44161e041bf1d3a333ab5144b5b7b93f914d2`** (pin trong `configs/v0.4_dense.yaml`) lần đầu cần
đến. Pin revision vì tác giả có thể cập nhật repo HuggingFace làm kết quả lệch mà không ai biết.

---

## 6. Smoke test — **chạy cái này TRƯỚC**

```bash
pip install pytest
python scripts/smoke_test.py          # verify_env + 170 test, ~5 giây, không cần data/
```

Phải in `✅ PASS`. Test gồm việc kiểm `src/evaluate.py` khớp `scoring.py` của BTC trên các ca biên
(`tests/test_scoring.py`), và mọi config trong `configs/` đọc được. Fail thì **dừng lại**.

---

## 7. Chạy từ dữ liệu thô đến `submission.zip`

```bash
python scripts/run_e2e.py --questions data/private-official.json
```

Một lệnh, chạy tuần tự `verify_env → eda → parse → chunk → split → encode → predict`, dừng ngay
ở bước đầu tiên lỗi. Bước `eda` tự bỏ qua nếu đã có `docs/exclusion_decisions.json`; bước `encode`
tự bỏ qua nếu đã có `data/embeddings.npy`. Ra: `outputs/v0.8_hybrid_rrf/submission.zip`.

Chạy từng bước bằng tay (cùng lệnh mà `run_e2e.py` gọi, xem `--dry-run`):

```bash
python scripts/eda.py --out outputs/eda/eda_notes.md          # sinh docs/exclusion_decisions.json
python -m src.data.parse_corpus                               # → data/corpus_clean.jsonl
python -m src.data.chunker --strategy strict --out data/chunks.jsonl
python -m src.data.split_data                                 # → holdout/dev/error_pool/train_split
python -u scripts/encode_corpus.py --config configs/v0.4_dense.yaml --resume   # GPU
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \
    --questions data/private-official.json --submission
```

Đo trên dev thay vì nộp: `--questions data/dev.json --eval`. Encode trên nhiều tài khoản Kaggle
song song: `docs/kaggle_encode.ipynb` + `--shard K/N`, ghép bằng `scripts/merge_embeddings.py`.

---

## 8. Con số kỳ vọng (để đối chiếu ngay)

Lệch ở bước nào thì lỗi nằm ở bước đó — đừng chạy tiếp.

| Kiểm tra | Giá trị kỳ vọng |
|---|---|
| Số dòng `data/corpus_clean.jsonl` | `8507` |
| SHA-256 `data/corpus_clean.jsonl` | `365306b58c0c14e87adce81a2b383a2245002faacb41b95928ddff4db0d73945` |
| Số dòng `data/chunks.jsonl` (`--strategy strict`) | `432142` |
| SHA-256 `data/chunks.jsonl` (`--strategy strict`) | `086868bd7c5a6f86d3fce57914c6af8b001ab52df12a7cdf650b0d8463a47334` |
| Vân tay kho chunk trong `data/embeddings.meta.json` | `9efb4ecf6b194a66d73ed24e` |
| holdout / dev / error_pool / train_split | `1000 / 1000 / 300 / 4689` |
| dev, `run_pipeline.py --eval` | `Recall@5 0.9384 · Recall@50 0.9857 · BTC recall 0.9354 precision 0.2451` |

Kho chunk mặc định của v0.1–v0.6 (`python -m src.data.chunker`, chiến lược `loose`): 524.422
chunk, SHA-256 `032a66ae23b1b64be38f54536bdc365067f364e79bfc4587e0038a48cfaf985e`.

---

## 9. Tái lập việc chọn siêu tham số (không bắt buộc để verify kết quả)

Không có bước huấn luyện trọng số. Hai thứ được CHỌN từ dữ liệu, đều chọn trên `train_split` và
báo cáo trên `dev`:

```bash
# (w, rrf_k, fuse_level) của RRF
python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --limit-fit 0

# θ của bộ quyết định số lượng doc: cần ranking của cả tập fit lẫn tập báo cáo
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \
    --questions data/train_split.json --out-dir outputs/v0.8_hybrid_doc/train_split
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml --questions data/dev.json
python scripts/fit_calibration.py \
    --ranking outputs/v0.8_hybrid_doc/train_split/ranking_full.json --questions data/train_split.json \
    --verify-ranking outputs/v0.8_hybrid_rrf/ranking_full.json --verify-questions data/dev.json \
    --max-recall-drop 0.003 --out outputs/v0.8_hybrid_doc/calibration.json   # → θ = 0.502793
```

`ranking_full.json` được ghi TRƯỚC bước cắt số lượng doc, nên dùng được để fit θ dù config đang
bật `pipeline.calibrate`.

---

## 10. Troubleshooting

> Chỉ điền bằng **lỗi thật gặp khi chạy**, không phải lỗi tưởng tượng.

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `verify_env` báo sai Python | Máy dùng 3.10/3.12 | Tạo venv bằng đúng `python3.11` |
| `ls data/selected-contexts \| wc -l` ≠ 8532 | Giải nén tạo thêm một cấp thư mục | `mv data/selected-contexts/*/*.json data/selected-contexts/` |
| `Không tìm thấy docs/exclusion_decisions.json` | File do `scripts/eda.py` sinh, bị `.gitignore` chặn | Chạy `python scripts/eda.py --out outputs/eda/eda_notes.md` |
| `embeddings.npy encode từ MỘT BỘ CHUNK KHÁC` | `chunks.jsonl` dựng bằng chiến lược khác lần encode | Dựng lại bằng `--strategy strict`, hoặc encode lại |
| Điểm BM25 lệch ở chữ số thập phân thứ 4 giữa hai lần chạy, thứ hạng không đổi | Thứ tự từ vựng phụ thuộc `PYTHONHASHSEED` ⇒ thứ tự cộng float32 đổi | Cần khớp từng byte thì đặt `PYTHONHASHSEED=0` |
| Recall ≈ 0.000 nhưng không có lỗi | 🔴 `doc_id` bị ép thành int ở đâu đó | Xem `docs/scoring_behaviour.md`; `make_submission.py` lẽ ra đã chặn |

---

## 11. Cấu trúc repo

```
configs/          base.yaml + mỗi thí nghiệm 1 YAML (kế thừa bằng `extends`, src/common/config.py)
src/
  common/         config (extends), io (file trung gian), runinfo (commit, chặn holdout), demo
  data/           parse_corpus, chunker (loose | strict), chunker_dieu, split_data   [P2]
  retrieval/      base, bm25, dense, hybrid, tokenizers, enrich                       [P3]
  rerank/         cross_encoder, calibrate                                            [P4]
  evaluate.py     bản sao chính xác scoring.py của BTC                           [P1, KHOÁ]
  make_submission.py  chốt chặn cuối trước CodaLab                               [P1, KHOÁ]
  verify_env.py                                                                  [P1, KHOÁ]
scripts/          run_e2e, run_pipeline, encode_corpus, tune_rrf, fit_calibration, ...
                  p4_*/p5_*: chuỗi thí nghiệm của P4 (đã sinh các số v0.2–v0.6)
tests/            mã chấm BTC, retriever, hybrid, calibrate, config, chunker
docker/           Dockerfile — đặc tả môi trường
docs/             scoring_behaviour, reproduce, model_card, data_statement, releases/, ...
INTERFACES.md     hợp đồng dữ liệu giữa 4 người — phần KHOÁ của repo
experiments.csv   mọi thí nghiệm, kèm commit
```

## Giấy phép

MIT — xem `LICENSE`. Repo để **private** trong thời gian thi (Điều 11 cấm trao đổi mã nguồn giữa các đội);
chuyển public trong vòng 30 ngày sau chung kết (13/11/2026) theo Điều 10.
