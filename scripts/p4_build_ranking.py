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


def build(retriever, qids: list[str], texts: list[str], top_k: int,
          pool: str | None = None, chunks_per_doc: int = 1) -> dict:
    """Gộp chunk→doc, giữ (các) chunk_id đại diện của mỗi doc.

    HAI THỨ TÁCH BẠCH, đừng lẫn:
      • THỨ HẠNG doc  ← do chiến lược gộp quyết định (max/sum/mean_topN/logsumexp).
        Uỷ quyền cho `pool_candidates()` của P3 — INTERFACES §3 nói rõ gộp chunk→doc
        là trách nhiệm của retriever. P4 chỉ ĐO và CHỌN, không hiện thực lại.
      • chunk_id đại diện ← LUÔN xếp theo điểm BM25 của chunk, bất kể pool nào.
        Đây là mỏ neo cho reranker: nó cần đoạn văn bản cụ thể để chấm. Giữ cố định
        để đổi `pool` không kéo theo đổi luôn đoạn đem đi rerank — nếu không thì khi
        kết quả đổi ta không biết do cách gộp hay do đoạn văn bản khác.

    Định dạng ra (tương thích ngược):
        chunks_per_doc == 1 → [doc_id, score, chunk_id]              (3 phần tử)
        chunks_per_doc  > 1 → [doc_id, score, chunk_id, [cid, ...]]  (4 phần tử)
    Phần tử thứ 3 LUÔN là chunk tốt nhất, kể cả khi có phần tử thứ 4. Nhờ vậy mọi
    file cũ và mọi đoạn code đọc 3 phần tử vẫn chạy nguyên.

    Vì sao cần phần tử thứ 4: biến thể `n-chunk` của P4-3 chấm nhiều đoạn mỗi văn bản
    rồi max-pool điểm reranker. Cần thiết vì 17/300 câu error_pool có ≥2 chunk cùng
    văn bản HOÀ ĐIỂM TUYỆT ĐỐI — BM25 tuyên bố nó không phân biệt được, nên để BM25
    chọn một đoạn duy nhất đem đi rerank là để đồng xu chọn đầu vào cho reranker.

    Phá hoà: điểm cao nhất trước, hoà thì chunk_id nhỏ nhất (= đoạn sớm hơn trong văn
    bản). Bắt buộc tường minh — `candidates()` dùng argpartition khi candidate_chunks
    hữu hạn, mà argpartition XÁO TRỘN thứ tự giữa các phần tử bằng nhau.
    """
    doc_arr = retriever._doc_ids_arr
    assert doc_arr is not None, "Gọi .index() trước"
    assert chunks_per_doc >= 1
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

            # (-điểm, chunk_id) → sort tăng dần = điểm giảm dần, hoà thì id nhỏ trước
            per_doc: dict[str, list[tuple[float, str]]] = {}
            for i, score in zip(idx, sc):
                d = str(doc_arr[int(i)])
                per_doc.setdefault(d, []).append(
                    (-float(score), retriever.chunk_ids[int(i)])
                )

            # thứ hạng doc: uỷ quyền cho P3
            ranked = retriever.pool_candidates(idx, sc, top_k, pool=pool)

            rows = []
            for d, score in ranked:
                cids = [cid for _, cid in sorted(per_doc[d])[:chunks_per_doc]]
                rows.append(
                    [d, float(score), cids[0]]
                    if chunks_per_doc == 1
                    else [d, float(score), cids[0], cids]
                )
            out[qid] = rows
        print(f"  {min(s + batch, len(qids))}/{len(qids)}", flush=True)
    return out


def recall_at(rank: dict, questions: dict, k: int) -> float | None:
    """Recall@k nếu tập câu hỏi có nhãn; None nếu là tập test."""
    vals = []
    for qid, item in questions.items():
        gold = {str(a) for a in (item.get("answer") or [])}
        if not gold:
            return None
        # row là [doc, score, chunk_id] hoặc [doc, score, chunk_id, [cid...]]
        # tuỳ --chunks-per-doc, nên chỉ lấy phần tử đầu.
        got = {row[0] for row in rank.get(str(qid), [])[:k]}
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
    ap.add_argument("--chunks-per-doc", type=int, default=1,
                    help="lưu top-M chunk mỗi doc cho biến thể n-chunk của reranker. "
                         "M=1 giữ nguyên định dạng 3 phần tử; M>1 thêm phần tử thứ 4 "
                         "là danh sách chunk_id xếp theo điểm BM25 giảm dần.")
    ap.add_argument("--pool", default=None,
                    help="max | sum | mean_topN (vd mean_top3) | logsumexp. "
                         "Bỏ trống = lấy từ config. Chỉ đổi THỨ HẠNG doc; "
                         "chunk_id đại diện luôn là chunk BM25 cao nhất.")
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
    print(f"Gộp chunk→doc bằng: {a.pool or r.pool}")
    rank = build(r, qids, texts, a.top_k, pool=a.pool,
                 chunks_per_doc=a.chunks_per_doc)
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
