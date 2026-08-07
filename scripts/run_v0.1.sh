#!/usr/bin/env bash
# v0.1 — chạy end-to-end bằng MỘT lệnh: bash scripts/run_v0.1.sh
# Điều kiện nghiệm thu release (INTERFACES / kế hoạch Track B).
set -euo pipefail

CONFIG="${1:-configs/v0.1_bm25.yaml}"
OUT_DIR="outputs/v0.1_bm25"
mkdir -p "$OUT_DIR"

echo "══ 0/5  Kiểm tra môi trường ══"
python -m src.verify_env

echo -e "\n══ 1/5  Parse corpus ══"
python -m src.data.parse_corpus --config "$CONFIG"

echo -e "\n══ 2/5  Chunking ══"
python -m src.data.chunker --config "$CONFIG"

echo -e "\n══ 3/5  Tách held-out ══"
python -m src.data.split_holdout --config "$CONFIG"

echo -e "\n══ 4/5  BM25 trên held-out (đo chất lượng) ══"
python -m src.retrieval.bm25 --config "$CONFIG" \
    --questions data/holdout.json --out "$OUT_DIR/preds_holdout.json"
python -m src.evaluate --preds "$OUT_DIR/preds_holdout.json" \
    --truth data/holdout.json --corpus data/corpus_clean.jsonl

echo -e "\n══ 5/5  BM25 trên public test → submission.zip ══"
python -m src.retrieval.bm25 --config "$CONFIG" \
    --questions data/public-official.json --out "$OUT_DIR/preds_public.json"
python -m src.make_submission --preds "$OUT_DIR/preds_public.json" \
    --questions data/public-official.json --corpus data/corpus_clean.jsonl \
    --out "$OUT_DIR/submission.zip"

echo -e "\n✅ Xong. Nộp: $OUT_DIR/submission.zip"
echo "   Ghi kết quả vào experiments.csv kèm commit SHA: git rev-parse --short HEAD"
