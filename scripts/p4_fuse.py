"""Hợp nhất thứ hạng BM25 và reranker (RRF) — chạy trên CPU, không cần GPU.

VÌ SAO KHÔNG THAY THẾ MÀ HỢP NHẤT. Đo được trên dev_sub300, bge-m3, top-20:
    R@1  +0,0178   R@3  +0,0089   R@5  −0,0150   R@10  −0,0133
Reranker giỏi ở đỉnh, phá ở khúc giữa: nó chọn quán quân tốt hơn BM25 nhưng dìm quá
sâu những văn bản nó không thích, đẩy gold từ hạng 4-5 xuống 12-18. Phép thử chẩn đoán
cho thấy nó phân biệt gold vs một văn bản lấy bừa ở mức 95% — nhưng bài toán thật là
phân biệt gold với BỐN văn bản khó nhất mà BM25 cũng xếp cao. Giỏi theo CẶP không kéo
theo giỏi theo DANH SÁCH.

Metric cuộc thi là Recall@5. Reranker cải thiện R@1 mà hại R@5 là đúng loại sai cho
bài này. Hợp nhất giữ bề rộng của BM25 và lấy phần đỉnh của reranker.

DÙNG RRF CHỨ KHÔNG CỘNG ĐIỂM vì hai thang điểm không so được: BM25 là điểm dương không
chặn trên, reranker là logit có dấu. Tệ hơn, file rerank có điểm TRỘN — phần đầu là
điểm reranker, phần đuôi giữ nguyên điểm BM25. RRF chỉ dùng THỨ HẠNG nên miễn nhiễm.

    score(d) = w/(k + rank_bm25(d)) + (1-w)/(k + rank_rerank(d))

w=1 là BM25 thuần, w=0 là reranker thuần. Quét w để tìm điểm cân bằng.

    python -m scripts.p4_fuse --bm25 outputs/p4_bm25_ranking/devsub_pool.json \\
        --rerank outputs/p4_rerank/devsub_bge_top20_noname.json \\
        --questions data/dev_sub300.json --top-k 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ranks_of(rank: dict, qid: str, limit: int) -> dict[str, int]:
    return {row[0]: i + 1 for i, row in enumerate(rank[qid][:limit])}


def fuse(bm25: dict, rr: dict, qids: list[str], w: float, k: int,
         limit: int) -> dict[str, list]:
    out = {}
    for qid in qids:
        rb = ranks_of(bm25, qid, limit)
        rr_ = ranks_of(rr, qid, limit)
        docs = set(rb) | set(rr_)
        big = limit + 1000  # doc vắng mặt ở một bên: coi như xếp rất sau, không loại hẳn
        scored = [
            (d, w / (k + rb.get(d, big)) + (1 - w) / (k + rr_.get(d, big)))
            for d in docs
        ]
        # hoà điểm → ưu tiên thứ hạng BM25, để w=1 tái lập BM25 chính xác
        scored.sort(key=lambda t: (-t[1], rb.get(t[0], big)))
        out[qid] = [[d, s, ""] for d, s in scored]
    return out


def recall_at(rank: dict, questions: dict, k: int) -> float:
    tot = 0.0
    for qid, v in questions.items():
        gold = {str(x) for x in v["answer"]}
        got = {r[0] for r in rank.get(str(qid), [])[:k]}
        tot += len(gold & got) / len(gold)
    return tot / len(questions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--rerank", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--top-k", type=int, default=20,
                    help="phải KHỚP --rerank-top lúc chạy rerank; ngoài phạm vi đó "
                         "hai thứ hạng giống hệt nhau nên hợp nhất vô nghĩa")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out", default=None, help="ghi thứ hạng của w tốt nhất")
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    bm25 = json.load(open(a.bm25, encoding="utf-8"))
    rr = json.load(open(a.rerank, encoding="utf-8"))
    qids = [str(x) for x in q]

    base5 = recall_at(bm25, q, 5)
    print(f"BM25 R@5 = {base5:.4f}  ·  trần R@{a.top_k} = "
          f"{recall_at(bm25, q, a.top_k):.4f}  ·  rrf_k = {a.rrf_k}\n")
    print(f"{'w':>5} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}   {'Δ R@5':>8}")

    best_w, best_r5, best_rank = None, -1.0, None
    for i in range(11):
        w = i / 10
        f = fuse(bm25, rr, qids, w, a.rrf_k, a.top_k)
        r5 = recall_at(f, q, 5)
        mark = ""
        if r5 > best_r5:
            best_w, best_r5, best_rank = w, r5, f
            mark = ""
        print(f"{w:>5.1f} {recall_at(f,q,1):>8.4f} {recall_at(f,q,3):>8.4f} "
              f"{r5:>8.4f} {recall_at(f,q,10):>8.4f}   {r5-base5:>+8.4f}{mark}")

    print(f"\nw tốt nhất = {best_w:.1f}  →  R@5 = {best_r5:.4f}  "
          f"({best_r5-base5:+.4f} so với BM25)")
    if best_r5 <= base5 + 1e-9:
        print("  ⚠️ Hợp nhất KHÔNG vượt được BM25 thuần ở K=5. Reranker không đóng góp")
        print("     gì cho metric của cuộc thi, dù nó có thể cải thiện R@1.")
        print("     Đừng đọc w tốt nhất như một lựa chọn — mọi w đều hoà hoặc thua.")
    elif best_w in (0.0, 1.0):
        print("  → Một đầu mút thắng: hợp nhất không mang lại gì, dùng thẳng bên thắng.")
    else:
        print("  → Hợp nhất VƯỢT cả hai đầu mút. Nhưng w chọn trên chính tập này nên")
        print("    con số LẠC QUAN. Cố định w rồi đo lại trên dev đầy đủ, và kiểm bằng")
        print("    scripts/p4_paired_test.py trước khi tin.")

    if a.out and best_rank is not None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(best_rank, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\n✅ {a.out} (w={best_w:.1f})")


if __name__ == "__main__":
    main()
