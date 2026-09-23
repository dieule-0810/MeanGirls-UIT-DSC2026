#!/usr/bin/env python3
"""Hợp nhất 3 nguồn: BM25 (chunk→doc) + reranker bge-m3 + kNN câu-hỏi-giống-câu-hỏi trên train.

GIẢ THUYẾT (H_kNN). ~70% văn bản vàng của dev/holdout đã từng là văn bản vàng của một câu
train (dev 0,707 · holdout 0,697). Các câu hỏi về cùng một văn bản dùng lại thuật ngữ của
nhau nhiều hơn là dùng lại thuật ngữ của văn bản. BM25 trên CÂU HỎI TRAIN (4.689–7.000 câu,
rất nhỏ) bù đúng điểm yếu đã đo ở stratified_baseline: văn bản vàng xuất hiện nhiều trong
train (luật dài, nhiều chunk) lại có Recall BM25 thấp nhất (freq>=11: 0,63–0,70).

    rank_knn: doc được xếp theo  Σ_{j ∈ top-N câu train giống nhất}  s_j / s_max · [doc ∈ gold_j]
    score(d) = wb/(k+rank_bm25) + wr/(k+rank_rerank) + wk/(k+rank_knn)     (RRF có trọng số)

Không có tham số học, không dữ liệu ngoài, không augmentation — chỉ dùng nhãn train BTC cấp
như một bộ nhớ láng giềng gần nhất (kNN classifier cổ điển).

SỐ ĐO (siêu tham số chọn trên dev, holdout chạm 1 lần):
    dev     n=1000 mem=train_split : RRF hiện tại 0,8733 → 0,9028
    holdout n=1000 mem=train_split : 0,8559 → 0,8855   (thắng 46 · thua 16)
    holdout n=1000 mem=split+dev   : 0,8559 → 0,8902   (bộ nhớ lớn hơn → lời hơn)
    holdout, KHÔNG reranker (wr=0, wk=0,07): BM25 0,8449 → 0,8707 / 0,8732

Private: dùng --memory data/train.json (đủ 7.000 câu).

    python -m scripts.p5_knn_fuse --questions data/private-official.json \
        --bm25 outputs/v0.6_private/bm25_top50.json \
        --rerank outputs/v0.6_private/bge_top20.json \
        --memory data/train.json --out outputs/v0.6_private/knn_fused.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from src.retrieval.tokenizers import Tokenizer

TOK = Tokenizer(name="syllable_bigram", fold_tone=True)


class QBM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avg = sum(self.dl) / max(1, len(self.dl))
        n = len(docs)
        df = collections.Counter(t for d in self.tf for t in d)
        self.idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}
        self.inv: dict[str, list] = collections.defaultdict(list)
        for i, d in enumerate(self.tf):
            for t, c in d.items():
                self.inv[t].append((i, c))

    def score(self, q: list[str]) -> dict[int, float]:
        sc: dict[int, float] = collections.defaultdict(float)
        for t in set(q):
            w = self.idf.get(t)
            if w is None:
                continue
            for i, c in self.inv[t]:
                sc[i] += w * c * (self.k1 + 1) / (
                    c + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return sc


def knn_doc_scores(bm: QBM25, mem_ids, mem, question, n_neighbors=50, exclude=None):
    sc = bm.score(TOK(question))
    top = sorted(sc.items(), key=lambda x: -x[1])[:n_neighbors + 1]
    out: dict[str, float] = collections.defaultdict(float)
    top = [(i, s) for i, s in top if mem_ids[i] != exclude][:n_neighbors]
    if not top:
        return out
    mx = top[0][1]
    for i, s in top:
        for d in mem[mem_ids[i]]["answer"]:
            out[str(d)] += s / mx
    return out


def fuse_one(bm25_rows, rr_rows, knn, wb, wr, wk, k, bm25_depth, knn_depth):
    big = 10 ** 6
    rb = {str(r[0]): i + 1 for i, r in enumerate(bm25_rows[:bm25_depth])}
    rr = {str(r[0]): i + 1 for i, r in enumerate(rr_rows[:20])} if rr_rows else {}
    ks = sorted(knn.items(), key=lambda x: -x[1])[:knn_depth]
    rk = {d: i + 1 for i, (d, _) in enumerate(ks)}
    docs = set(rb) | set(rr) | set(rk)
    sc = {d: wb / (k + rb.get(d, big)) + wr / (k + rr.get(d, big)) + wk / (k + rk.get(d, big))
          for d in docs}
    order = sorted(docs, key=lambda d: (-sc[d], rb.get(d, big)))
    return [[d, sc[d], ""] for d in order]


def recall_at(rank, q, kk=5):
    t = 0.0
    for qid, v in q.items():
        g = {str(x) for x in v["answer"]}
        t += len(g & {r[0] for r in rank[qid][:kk]}) / len(g)
    return t / len(q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--bm25", required=True, help="ranking BM25 top-50 (p4_build_ranking)")
    ap.add_argument("--rerank", default=None, help="ranking bge top-20 (p4_rerank); bỏ trống = không reranker")
    ap.add_argument("--memory", default="data/train.json", help="câu train có nhãn làm bộ nhớ kNN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wb", type=float, default=0.6)
    ap.add_argument("--wr", type=float, default=None, help="mặc định 0,4 nếu có --rerank, 0 nếu không")
    ap.add_argument("--wk", type=float, default=None, help="mặc định 0,10 nếu có --rerank, 0,07 nếu không")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--neighbors", type=int, default=50)
    ap.add_argument("--knn-depth", type=int, default=10)
    ap.add_argument("--bm25-depth", type=int, default=50)
    a = ap.parse_args()

    wr = a.wr if a.wr is not None else (0.4 if a.rerank else 0.0)
    wk = a.wk if a.wk is not None else (0.10 if a.rerank else 0.07)

    q = json.load(open(a.questions, encoding="utf-8"))
    bm25 = json.load(open(a.bm25, encoding="utf-8"))
    rr = json.load(open(a.rerank, encoding="utf-8")) if a.rerank else {}
    mem = json.load(open(a.memory, encoding="utf-8"))
    qids = [str(x) for x in q]

    miss = [x for x in qids if x not in bm25 or (a.rerank and x not in rr)]
    if miss:
        raise SystemExit(f"❌ {len(miss)} qid thiếu trong bm25/rerank (vd {miss[:3]})")
    overlap = set(qids) & set(mem)
    if overlap:
        print(f"⚠️  {len(overlap)} qid của tập hỏi nằm trong bộ nhớ → sẽ LOẠI chính nó khỏi láng giềng "
              f"(tránh rò nhãn khi đo trên dev/holdout).")

    mem_ids = list(mem)
    bm = QBM25([TOK(mem[i]["question"]) for i in mem_ids])
    print(f"bộ nhớ kNN: {len(mem_ids)} câu · wb={a.wb} wr={wr} wk={wk} "
          f"knn_depth={a.knn_depth} neighbors={a.neighbors} rrf_k={a.rrf_k}")

    out = {}
    for qid in qids:
        knn = knn_doc_scores(bm, mem_ids, mem, q[qid]["question"], a.neighbors, exclude=qid)
        out[qid] = fuse_one(bm25[qid], rr.get(qid), knn, a.wb, wr, wk,
                            a.rrf_k, a.bm25_depth, a.knn_depth)

    has_gold = all(q[x].get("answer") for x in qids)
    if has_gold:
        base = {x: [[str(r[0])] for r in bm25[x]] for x in qids}
        print(f"BM25 R@5 = {recall_at(base, q):.4f}  →  hợp nhất+kNN R@5 = {recall_at(out, q):.4f}")
    else:
        print("Tập không nhãn → không in Recall.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"✅ {a.out}  ·  {len(out)} câu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
