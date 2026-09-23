# Chạy private với kNN câu hỏi (v0.6) — 23/09

Chép `scripts/p5_knn_fuse.py` vào repo. Không cần cài thêm gì, không cần GPU cho bước kNN (~1 phút).

## Bước 0 — dựng hai file đầu vào (nếu chưa có từ lượt nộp trước)

```bash
py -m scripts.p4_build_ranking --config configs/v0.3_bm25_best.yaml \
    --questions data/private-official.json --out outputs/v0.6_private/bm25_top50.json --top-k 50          # ~7 phút CPU
py -m scripts.p4_rerank --model bge-m3 --ranking outputs/v0.6_private/bm25_top50.json \
    --questions data/private-official.json --rerank-top 20 --chunks 1 --prepend-name --batch-size 8 \
    --out outputs/v0.6_private/bge_top20.json              # ~75 phút GPU, crash cuối là bình thường
```

## Lượt A — BM25 + bge + kNN (holdout 0,8559 → 0,8917)

```bash
py -m scripts.p5_knn_fuse --questions data/private-official.json --memory data/train.json \
    --bm25 outputs/v0.6_private/bm25_top50.json --rerank outputs/v0.6_private/bge_top20.json \
    --out outputs/v0.6_private/knn_rrf.json
py -m scripts.p4_to_preds --ranking outputs/v0.6_private/knn_rrf.json --questions data/private-official.json --out outputs/v0.6_private/preds_A.json
py -m src.make_submission --preds outputs/v0.6_private/preds_A.json --questions data/private-official.json \
    --corpus data/corpus_clean.jsonl --out outputs/v0.6_private/A/submission.zip
py -m scripts.p4_check_submission --zip outputs/v0.6_private/A/submission.zip --questions data/private-official.json --corpus data/corpus_clean.jsonl
```

## Lượt B — BM25 + kNN, không GPU (holdout 0,8449 → 0,8762)

Giống lượt A, bỏ `--rerank`, đổi tên file ra thành `knn_bm25.json` / `preds_B.json` / `B/submission.zip`.
Nếu bge chưa chạy xong thì **nộp B trước** để có mốc.
Q=data/private-official.json
py -m scripts.p4_fuse --bm25 outputs/v0.6_private/bm25_top50.json \
    --rerank outputs/v0.6_private/bge_top20.json --questions data/private-official.json \
    --top-k 20 --w 0.6 --rrf-k 60 --out outputs/v0.6_private/rrf_w06.json
py -m scripts.p4_to_preds --ranking outputs/v0.6_private/rrf_w06.json --questions data/private-official.json --out outputs/v0.6_private/preds_C.json
py -m src.make_submission --preds outputs/v0.6_private/preds_C.json --questions data/private-official.json \
    --corpus data/corpus_clean.jsonl --out outputs/v0.6_private/C/submission.zip
py -m scripts.p4_check_submission --zip outputs/v0.6_private/C/submission.zip --questions data/private-official.json --corpus data/corpus_clean.jsonl

## Kiểm nhanh trước khi nộp

- Log phải in `bộ nhớ kNN: 7000 câu` và **không** in dòng ⚠️ về qid trùng bộ nhớ (private không nằm trong train).
- `make_submission` in `Phân bố: {5: N}`.

## Tái lập số đo (cho bài báo / BTC)

```bash
py -m scripts.p5_knn_fuse --questions data/dev.json --memory data/train_split.json \
    --bm25 outputs/v0.3_bm25_best/ranking_full.json --rerank outputs/p4_rerank/dev_bge_top20_syl.json --out /tmp/d.json      # 0,9028
py -m scripts.p5_knn_fuse --questions data/holdout.json --memory data/train_split.json \
    --bm25 outputs/p4_bm25_ranking/holdout_bm25_top50.json --rerank outputs/p4_rerank/holdout_bge_top20_syl.json --out /tmp/h.json  # 0,8855
```
