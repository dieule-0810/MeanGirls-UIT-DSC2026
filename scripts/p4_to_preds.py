#!/usr/bin/env python3
"""Ranking [[doc_id, score, ...], ...] → preds {qid: [doc_id, ...]} cho src.make_submission.

Mắt xích còn thiếu giữa `p4_fuse` / `p4_build_ranking` và `src/make_submission.py`:
ranking là list-of-list, `--preds` cần list phẳng. Chuyển tay bằng one-liner trong
terminal là chỗ dễ nhất để một lần nộp bài chết vì lý do ngớ ngẩn, nên nó ở đây,
có kiểm tra, và chạy được lại.

    python -m scripts.p4_to_preds \
        --ranking outputs/v0.4_submit/public_rrf_w06.json \
        --questions data/public-official.json \
        --out outputs/v0.4_submit/public_preds.json

Rồi:

    python -m src.make_submission \
        --preds outputs/v0.4_submit/public_preds.json \
        --questions data/public-official.json \
        --corpus data/corpus_clean.jsonl \
        --out outputs/v0.4_submit/submission.zip

KHÔNG cắt còn 5 ở đây. `make_submission` khử trùng lặp RỒI mới cắt — mẫu số của
Precision là `len(list)` sau khi cắt, nên thứ tự hai thao tác đó đổi kết quả.
Cắt sớm ở đây là giành việc của nó và làm hỏng đúng cái nó canh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=10,
                    help="số doc lấy từ ranking trước khi giao cho make_submission "
                         "khử trùng lặp rồi cắt 5. Để dư một ít cho an toàn.")
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    rank = json.load(open(a.ranking, encoding="utf-8"))
    qids = [str(x) for x in q]

    missing = [x for x in qids if x not in rank]
    if missing:
        raise SystemExit(
            f"❌ {len(missing)} qid không có trong ranking (vd {missing[:3]}). "
            f"make_submission sẽ báo THIẾU và dừng — nhưng biết ngay ở đây thì rõ "
            f"nguyên nhân hơn: ranking dựng trên tập câu hỏi khác.")

    preds, n_short = {}, 0
    for qid in qids:
        rows = rank[qid][:a.depth]
        docs = [str(r[0]) for r in rows]
        if len({*docs}) < 5:
            n_short += 1
        preds[qid] = docs

    if n_short:
        print(f"⚠️  {n_short} câu có ít hơn 5 doc phân biệt trong top-{a.depth}. "
              f"Recall bị chặn trên cho các câu đó — kiểm xem ranking có bị cụt không "
              f"(đầu ra RRF chỉ sâu bằng --top-k lúc hợp nhất).")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    print(f"✅ {a.out}  ·  {len(preds)} câu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
