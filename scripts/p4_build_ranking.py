#!/usr/bin/env python3
"""P4: sinh ranking (top-K doc + điểm + chunk đại diện) từ BM25 của P3.

Vì sao cần file này: `search()` trả về cấp document và vứt mất chunk nào đã
thắng — mà reranker phải biết chấm chunk nào. Đây chính là lỗ hổng
INTERFACES §3b. File này KHÔNG sửa src/retrieval/ (P3 sở hữu), chỉ dùng lại
API công khai `candidates()` + `_doc_ids_arr`, rồi tự gộp chunk→doc theo max
và GIỮ chunk_id thắng cuộc.

Đầu ra: {qid: [[doc_id, score, chunk_id], ...]} — chunk_id là chunk có điểm
cao nhất của doc đó, dùng làm baseline cho P4-4. Lưu ý K-CHUNKPICK: đây chỉ
là baseline; reranker nên chấm NHIỀU chunk rồi max-pool, xem P4_TASKS §P4-4.

Chạy:
    python -m scripts.p4_build_ranking --config configs/v0.1_bm25.yaml \
        --questions data/error_pool.json \
        --out outputs/p4_bm25_ranking/error_pool_top50.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.retrieval.bm25 import retriever_from_config

REPO = Path(__file__).resolve().parents[1]


def load_chunks(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def build(retriever, qids: list[str], texts: list[str], top_k: int) -> dict:
    """Gộp chunk→doc bằng max, giữ chunk_id của chunk thắng.

    `candidates()` trả (idx, scores) đã sắp giảm dần theo điểm, nên chunk đầu
    tiên gặp của mỗi doc CHÍNH LÀ chunk max — không cần argmax lại.
    """
    doc_arr = retriever._doc_ids_arr
    assert doc_arr is not None, "Gọi .index() trước"
    n_cand = max(retriever.candidate_chunks, top_k)

    out: dict[str, list] = {}
    batch = 64
    for s in range(0, len(qids), batch):
        chunk_of_q = retriever.candidates(texts[s : s + batch], n_cand)
        for j, (idx, sc) in enumerate(chunk_of_q):
            qid = qids[s + j]
            if idx.size == 0:
                out[qid] = []
                continue
            seen: dict[str, tuple[float, str]] = {}
            for i, score in zip(idx, sc):
                d = str(doc_arr[int(i)])
                if d not in seen:  # lần đầu gặp = điểm cao nhất của doc này
                    seen[d] = (float(score), retriever.chunk_ids[int(i)])
            ranked = sorted(seen.items(), key=lambda kv: (-kv[1][0], kv[0]))
            out[qid] = [[d, sc_, cid] for d, (sc_, cid) in ranked[:top_k]]
        print(f"  {min(s + batch, len(qids))}/{len(qids)}", flush=True)
    return out


def recall_at(rank: dict, questions: dict, k: int) -> float | None:
    """Recall@k nếu tập câu hỏi có nhãn; None nếu là tập test."""
    vals = []
    for qid, item in questions.items():
        gold = {str(a) for a in (item.get("answer") or [])}
        if not gold:
            return None
        got = {d for d, _, _ in rank.get(str(qid), [])[:k]}
        vals.append(len(gold & got) / len(gold))
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.1_bm25.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--limit-chunks", type=int, default=None,
                    help="chỉ để smoke test, KHÔNG dùng khi chạy thật")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    chunks_path = REPO / cfg["paths"]["chunks"]

    t0 = time.time()
    chunks = load_chunks(chunks_path, a.limit_chunks)
    print(f"Đọc {len(chunks):,} chunk trong {time.time()-t0:.1f}s")

    r = retriever_from_config(cfg)
    t0 = time.time()
    r.index(chunks)
    print(f"Index xong trong {time.time()-t0:.1f}s — {r.n_chunks:,} chunk / {r.n_docs:,} doc")

    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    qids = [str(q) for q in questions]
    texts = [questions[q]["question"] for q in questions]

    t0 = time.time()
    rank = build(r, qids, texts, a.top_k)
    dt = time.time() - t0
    print(f"Truy vấn {len(qids)} câu trong {dt:.1f}s ({dt/len(qids)*1000:.0f} ms/câu)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rank, ensure_ascii=False), encoding="utf-8")
    print(f"✅ {out}")

    for k in (5, 20, a.top_k):
        v = recall_at(rank, questions, k)
        if v is not None:
            print(f"   Recall@{k:<3} = {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
