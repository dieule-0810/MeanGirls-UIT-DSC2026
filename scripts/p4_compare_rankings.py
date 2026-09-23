#!/usr/bin/env python3
"""So hai bảng xếp hạng theo TỪNG CÂU, không so bằng Recall.

Vì sao cần: hai hệ thống có thể cho Recall giống hệt mà vẫn trả về tập văn bản khác
nhau — nhóm đã gặp đúng chuyện đó trên leaderboard (730/1000 câu đổi tập doc, Recall
không đổi). Recall là một con số tổng hợp; nó che giấu việc bên trong đã đổi những gì.

Dùng để kiểm TƯƠNG ĐƯƠNG (hai cấu hình đáng lẽ phải cho kết quả y hệt):

    python -m scripts.p4_compare_rankings \\
        --a outputs/v0.5_cand/errpool_cand2000.json \\
        --b outputs/v0.5_cand/errpool_candnull.json \\
        --questions data/error_pool.json

Tương đương thật = 100% câu trùng khít ở MỌI độ sâu. Chỉ cần một câu lệch ở top-50
là giả định tương đương đã hỏng, dù Recall có giống nhau đến mấy.
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="bảng xếp hạng A (thường là bản hiện dùng)")
    ap.add_argument("--b", required=True, help="bảng xếp hạng B (bản đối chứng)")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    A = json.load(open(a.a, encoding="utf-8"))
    B = json.load(open(a.b, encoding="utf-8"))
    qids = [str(x) for x in q]

    miss = [x for x in qids if x not in A or x not in B]
    if miss:
        raise SystemExit(f"❌ {len(miss)} qid thiếu ở một trong hai file (vd {miss[:3]})")

    has_gold = all((q[x].get("answer") or None) is not None for x in q)
    depths = [1, 3, 5, 10, 20, 50]
    same_order = {d: 0 for d in depths}
    same_set = {d: 0 for d in depths}
    first_diff = []

    for qid in qids:
        da = [str(r[0]) for r in A[qid]]
        db = [str(r[0]) for r in B[qid]]
        for d in depths:
            if da[:d] == db[:d]:
                same_order[d] += 1
            if set(da[:d]) == set(db[:d]):
                same_set[d] += 1
        if da[:50] != db[:50] and len(first_diff) < 5:
            k = next((i for i in range(min(len(da), len(db))) if da[i] != db[i]), 0)
            first_diff.append((qid, k + 1, da[k] if k < len(da) else "-",
                               db[k] if k < len(db) else "-"))

    n = len(qids)
    print(f"So {n} câu  ·  {a.label_a} vs {a.label_b}\n")
    print(f"{'độ sâu':>7} {'trùng thứ tự':>14} {'trùng tập':>12}")
    for d in depths:
        print(f"{d:>7} {same_order[d]:>8}/{n}  {same_set[d]:>7}/{n}")

    if has_gold:
        def rec(R, k):
            t = 0.0
            for qid in qids:
                g = {str(x) for x in q[qid]["answer"]}
                t += len(g & {str(r[0]) for r in R[qid][:k]}) / len(g)
            return t / n
        print(f"\nRecall@5: {a.label_a} {rec(A,5):.4f}  ·  {a.label_b} {rec(B,5):.4f}")

    if same_order[50] == n:
        print(f"\n✅ TƯƠNG ĐƯƠNG: {n}/{n} câu trùng khít tới top-50.")
        return 0

    print(f"\n❌ KHÔNG tương đương: {n - same_order[50]}/{n} câu lệch trong top-50.")
    print("   Vài câu lệch sớm nhất (qid, hạng đầu tiên lệch, doc A, doc B):")
    for qid, k, x, y in first_diff:
        print(f"     {qid}  hạng {k}:  {x}  vs  {y}")
    print("\n   Recall giống nhau KHÔNG cứu được kết luận này — hai bên đang trả về")
    print("   tập văn bản khác nhau, nên giả định tương đương đã hỏng.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
