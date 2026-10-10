# reproduce.md — tái lập kết quả

> Cập nhật 09/10/2026 cho pipeline v0.8. Nguyên tắc: **mọi lệnh ở đây là lệnh ĐÃ CHẠY THẬT**;
> chỗ nào chưa xác nhận thì ghi rõ. README mục *Sử dụng* là bản rút gọn của tài liệu này.

---

## 0. Môi trường

```
Python 3.11 (src/verify_env.py chặn cứng)
pip install -r requirements.txt          # numpy · scipy · torch 2.14.0 · transformers 5.17.0 · pyvi · underthesea
python scripts/smoke_test.py             # kỳ vọng: 170 passed
```

pyvi/underthesea chỉ cần cho lưới benchmark tokenizer, không cần cho đường nộp bài
(`syllable_bigram` không dùng thư viện ngoài).

---

## 1. Dữ liệu

Đặt dữ liệu BTC vào `data/`: `selected-contexts/`, `train.json`, `public-official.json`,
`private-official.json` (không file nào được commit).

```bash
python scripts/eda.py --out outputs/eda/eda_notes.md     # sinh docs/exclusion_decisions.json
python -m src.data.parse_corpus                          # → data/corpus_clean.jsonl
python -m src.data.chunker --strategy strict --out data/chunks.jsonl
python -m src.data.split_data                            # → holdout/dev/error_pool/train_split + train_{1000,2500,4689}
```

`docs/exclusion_decisions.json` (25 văn bản loại, 11 câu vùng chết, 4 cụm trùng) bị `.gitignore`
chặn — phải sinh bằng `eda.py` trước `parse_corpus` và `split_data`. Báo cáo EDA ghi vào
`outputs/eda/eda_notes.md`.

**Kỳ vọng** (đã kiểm 09/10/2026):

| File | Dòng | SHA-256 |
|---|---:|---|
| `corpus_clean.jsonl` | 8.507 | `365306b58c0c14e87adce81a2b383a2245002faacb41b95928ddff4db0d73945` |
| `chunks.jsonl` — `--strategy strict` (v0.8) | 432.142 | `086868bd7c5a6f86d3fce57914c6af8b001ab52df12a7cdf650b0d8463a47334` |
| `chunks.jsonl` — `--strategy loose` (v0.1–v0.6) | 524.422 | `032a66ae23b1b64be38f54536bdc365067f364e79bfc4587e0038a48cfaf985e` |

Bảy file chia tập tái tạo khớp từng byte với bản đang dùng (kiểm 09/10/2026).

### ⚠️ Hai kho chunk, một tên file

`data/chunks.jsonl` là tên đã khoá (INTERFACES.md §7), nhưng nội dung đổi một lần:

* **v0.1–v0.6** đo trên kho `loose` (`python -m src.data.chunker`, 524.422 chunk) — nhận cả trích
  dẫn chéo "theo Điều 5 Luật ..." làm ranh giới.
* **v0.8** đo trên kho `strict` (`python -m src.data.chunker_dieu` hoặc `--strategy strict`,
  432.142 chunk) — chỉ nhận tiêu đề `Điều N. `. Embedding dense đã encode trên kho này; dùng kho
  khác thì `DenseRetriever` dừng với lỗi "vân tay không khớp".

Hai kho cho BM25 gần như như nhau (dev R@5 0,8555 vs 0,8547; holdout 0,8449 vs 0,8475 — không có
ý nghĩa thống kê). Muốn tái lập một con số, dựng đúng kho mà dòng `experiments.csv` đó đã dùng.

⚠️ SHA-256 kho `loose` ở trên **khác** giá trị ghi trong `docs/releases/v0.6_private.yaml`
(`34b426dd…`, 576.070.530 byte): kho đó sinh từ một phiên bản chunker trước refactor
`c59e98d`. Số dòng vẫn là 524.422; nội dung từng chunk đã được xác nhận trùng ngày 17/09.
Chunker hiện tại (trước và sau refactor 09/10) cho cùng `032a66ae…`.

---

## 2. Embedding (GPU, một lần)

```bash
python -u scripts/encode_corpus.py --config configs/v0.4_dense.yaml --dry-run    # ước tính
python -u scripts/encode_corpus.py --config configs/v0.4_dense.yaml --resume
```

Hoặc chia 8 mảnh trên 4 tài khoản Kaggle bằng `docs/kaggle_encode.ipynb` (mỗi tài khoản: import
notebook, thêm dataset chứa `data/chunks.jsonl`, secret `RCLONE_CONF_B64`, GPU T4 ×2 + Internet ON,
sửa `SHARDS_THIS_ACCOUNT` rồi *Save & Run All*; chạy lại thì notebook bỏ qua mảnh đã xong), rồi
`python scripts/merge_embeddings.py`. Kỳ vọng
`data/embeddings.meta.json`: `n_chunks 432142`, `chunk_fingerprint 9efb4ecf6b194a66d73ed24e`,
`revision 18b44161e041bf1d3a333ab5144b5b7b93f914d2`.

---

## 3. Đường nộp bài v0.8

```bash
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml --questions data/dev.json --eval
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \
    --questions data/private-official.json --submission
```

**Kỳ vọng trên dev** (đã chạy 09/10/2026, khớp từng byte `ranking_full.json` và
`predictions.json` giữa code trước và sau refactor):

```
Recall@5 0.9384 · Recall@20 0.9786 · Recall@50 0.9857
Chấm như BTC: recall=0.9354 precision=0.2451
```

BM25 một mình (`configs/v0.3_bm25_best.yaml`) trên cùng kho: `Recall@5 0.8547 · Recall@50 0.9708`.

### 3.1 Chọn siêu tham số (đã chạy, kết quả nằm trong config)

```bash
python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --limit-fit 0
#   → fuse_level=doc · rrf_k=20 · w_bm25=0,4 · dev R@5 0,9384 · độ lạc quan +0,0050

python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \
    --questions data/train_split.json --out-dir outputs/v0.8_hybrid_doc/train_split
python scripts/fit_calibration.py \
    --ranking outputs/v0.8_hybrid_doc/train_split/ranking_full.json --questions data/train_split.json \
    --verify-ranking outputs/v0.8_hybrid_rrf/ranking_full.json --verify-questions data/dev.json \
    --max-recall-drop 0.003 --out outputs/v0.8_hybrid_doc/calibration.json
#   → θ = 0.502793 (đã chạy lại 09/10/2026, khớp từng khoá với calibration.json cũ)
```

### 3.2 Tính tất định

Điểm BM25 có thể lệch ở chữ số thập phân thứ 4 giữa hai tiến trình (thứ tự từ vựng dựng từ
`set` chuỗi, phụ thuộc `PYTHONHASHSEED`, kéo theo thứ tự cộng float32). Thứ hạng và dự đoán
không đổi. Cần khớp từng byte `ranking_full.json` thì đặt `PYTHONHASHSEED=0`.

---

## 4. Tái lập các bản cũ

Các bản v0.4–v0.6 đi qua chuỗi script `p4_*`/`p5_*`, KHÔNG qua `run_pipeline.py`, và dùng kho
chunk `loose`. Chuỗi lệnh đầy đủ nằm trong kê khai, in bằng:

```bash
python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml --commands
```

Tóm tắt (chi tiết: `RUN_PRIVATE_KNN.md`):

| Bản | Lệnh chính | Số kỳ vọng |
|---|---|---|
| v0.3 | `p4_build_ranking --config configs/v0.3_bm25_best.yaml --top-k 50` | dev R@5 0,8555 |
| v0.4 | `p4_rerank --model bge-m3 --rerank-top 20 --chunks 1 --prepend-name` → `p4_fuse --w 0.6 --rrf-k 60` | dev 0,8733 · holdout 0,8559 |
| v0.6 | `p5_knn_fuse --memory data/train.json` (wb .6 · wr .4 · wk .1) | holdout 0,8855 · private LB 0,8818 |

`p4_fuse.py` BẮT BUỘC có `--w` trên tập không nhãn — thiếu nó script quét `w` trên chính tập thi.
`p4_rerank --model` nhận alias trong `src/rerank/cross_encoder.MODELS` (nơi pin revision), không
nhận repo id.

Các kiểm tra từng treo ở bản trước của tài liệu này đã có kết quả trong `experiments.csv`:
`candidate_chunks` 2000 vs null (`v0.5_candnull_*`, không khác biệt có ý nghĩa), tách
`fold_tone` khỏi khoản +0,0439 (`v0.4_syl_notonefold`: +0,0429 tokenizer, +0,0010 fold_tone),
`max_length` 1024 cho reranker (`v0.5_maxlen1024_devsub300`).

---

## 5. Ghi chú về `outputs/`

`outputs/*` bị `.gitignore` chặn; mọi file ranking chỉ tồn tại trên ổ đĩa từng người. Thứ duy
nhất trong repo ràng một bài nộp vào đầu vào sinh ra nó là kê khai `docs/releases/*.yaml`
(kiểm bằng `scripts/verify_release.py`). Lượt nộp v0.8 (`outputs/private_hybrid_doc_calibrated/`)
**chưa có kê khai** và chưa có tag.

`outputs/v0.2_bm25_tok/` là chạy demo (corpus giả 429 chunk) — không dùng cho kết luận nào.

---

## 6. Sự cố đã gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `verify_env` báo sai Python | Máy dùng 3.10/3.12 | Tạo venv bằng đúng `python3.11` |
| `ls data/selected-contexts \| wc -l` ≠ 8532 | Giải nén tạo thêm một cấp thư mục | `mv data/selected-contexts/*/*.json data/selected-contexts/` |
| `Không tìm thấy docs/exclusion_decisions.json` | File do `scripts/eda.py` sinh, bị `.gitignore` chặn | `python scripts/eda.py --out outputs/eda/eda_notes.md` |
| `embeddings.npy encode từ MỘT BỘ CHUNK KHÁC` | `chunks.jsonl` dựng bằng chiến lược khác lần encode | Dựng lại bằng `--strategy strict`, hoặc encode lại |
| Recall ≈ 0 nhưng không có lỗi | `doc_id` bị ép thành int ở đâu đó | Xem `docs/scoring_behaviour.md`; `make_submission.py` lẽ ra đã chặn |
