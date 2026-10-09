#!/usr/bin/env python3
"""
So hai lần chạy trên cùng tập câu hỏi. CHỦ SỞ HỮU: P3.

Câu hỏi thật sự cần trả lời sau khi có dense KHÔNG phải "dense có hơn BM25 không" — mà là
"dense có tìm được thứ BM25 BỎ SÓT không". Hai hệ ngang điểm nhưng sai ở những câu khác nhau
thì hợp nhất có lời; hai hệ ngang điểm và sai ở cùng những câu thì hợp nhất vô nghĩa.

Cột quan trọng nhất là **HỢP (union)**: trần trên của mọi cách hợp nhất. Union@50 mà không cao
hơn max(A, B) bao nhiêu thì đừng mất công viết hybrid.

    python scripts/compare_runs.py \\
        --a outputs/v0.3_bm25_best/ranking_full.json --name-a bm25 \\
        --b outputs/v0.4_dense/ranking_full.json     --name-b dense \\
        --questions data/dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_rank(p: Path) -> dict[str, list[str]]:
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(q): [str(r[0]) if isinstance(r, (list, tuple)) else str(r) for r in v] for q, v in raw.items()}


def recall(pred: dict[str, list[str]], gold: dict[str, list[str]], k: int) -> float:
    tot = 0.0
    for q, g in gold.items():
        if g:
            tot += len(set(pred.get(q, [])[:k]) & set(g)) / len(g)
    return tot / max(len(gold), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--ks", default="1,5,20,50")
    args = ap.parse_args()

    A, B = load_rank(REPO / args.a), load_rank(REPO / args.b)
    qs = json.loads((REPO / args.questions).read_text(encoding="utf-8"))
    gold = {str(q): [str(x) for x in v["answer"]] for q, v in qs.items() if str(q) in A and str(q) in B}
    ks = [int(x) for x in args.ks.split(",")]
    if not gold:
        raise SystemExit("❌ Hai file không có qid chung — chúng chạy trên hai tập khác nhau.")

    # Hợp: xen kẽ A,B rồi khử trùng. Không phải cách hợp nhất tốt nhất, nhưng là TRẦN TRÊN
    # rẻ nhất: nếu ngay cả trần cũng không hơn, mọi RRF tinh vi đều vô ích.
    U: dict[str, list[str]] = {}
    for q in gold:
        out, seen = [], set()
        for pair in zip(A[q], B[q]):
            for d in pair:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
        U[q] = out

    print(f"n = {len(gold)} câu · {args.questions}\n")
    print(f"{'':<14}" + "".join(f"{'R@'+str(k):>10}" for k in ks))
    for name, p in ((args.name_a, A), (args.name_b, B), ("HỢP (trần)", U)):
        print(f"{name:<14}" + "".join(f"{recall(p, gold, k):>10.4f}" for k in ks))

    for k in ks:
        ga = {q for q in gold if set(A[q][:k]) & set(gold[q])}
        gb = {q for q in gold if set(B[q][:k]) & set(gold[q])}
        only_a, only_b, both = len(ga - gb), len(gb - ga), len(ga & gb)
        head = max(recall(A, gold, k), recall(B, gold, k))
        gain = recall(U, gold, k) - head
        print(
            f"\n@{k}: cả hai đúng {both} · chỉ {args.name_a} {only_a} · chỉ {args.name_b} {only_b} · "
            f"cả hai sai {len(gold) - both - only_a - only_b}"
        )
        print(f"    dư địa hợp nhất @{k}: {gain:+.4f} so với hệ tốt hơn")
        if only_a + only_b == 0:
            print("    ⇒ hai hệ sai ở ĐÚNG cùng những câu: hợp nhất không thể giúp gì ở độ sâu này.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
