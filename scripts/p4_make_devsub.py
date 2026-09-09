"""Cắt tập con dev để sàng lọc reranker — PHÂN TẦNG, không ngẫu nhiên thuần.

Vì sao không lấy ngẫu nhiên 300 câu: tầng `freq>=11` chỉ chiếm 17,1% dev, nên mẫu
ngẫu nhiên 300 để lại ~51 câu ở đúng cái tầng ta quan tâm nhất (dư địa rerank +0,2982).
n=53 đã một lần lừa cả nhóm: từ error_pool ta suýt kết luận "BM25 luôn tìm thấy luật
khung" dựa trên R@50=0,9811 ở tầng n=53, rồi trên n=171 con số đó lật thành thấp nhất.
Lấy mẫu phân tầng giữ nguyên tỉ lệ bốn tầng, nên bảng phân tầng trên tập con vẫn đọc
được cùng cách với bảng trên dev đầy đủ.

⚠️ Đây là tập SÀNG LỌC, không phải tập báo cáo. Mọi con số cuối cùng phải đo lại trên
dev đầy đủ n=1000. Sai số ở tầng nhỏ nhất của tập con vào khoảng ±7%.

    python -m scripts.p4_make_devsub --n 300
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def tier(f: int) -> str:
    if f == 0:
        return "freq=0"
    if f <= 2:
        return "freq=1-2"
    if f <= 10:
        return "freq=3-10"
    return "freq>=11"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/dev.json")
    ap.add_argument("--train-split", default="data/train_split.json")
    ap.add_argument("--out", default="data/dev_sub300.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dev = json.load(open(a.dev, encoding="utf-8"))
    tr = json.load(open(a.train_split, encoding="utf-8"))
    freq = Counter(str(x) for v in tr.values() for x in v["answer"])

    # agg=min, khớp p4_stratified_recall.py — đổi là hai bảng hết so được với nhau
    buckets: dict[str, list[str]] = defaultdict(list)
    for qid, v in dev.items():
        t = tier(min(freq.get(str(x), 0) for x in v["answer"]))
        buckets[t].append(str(qid))

    rng = random.Random(a.seed)
    names = ["freq=0", "freq=1-2", "freq=3-10", "freq>=11"]
    picked: list[str] = []
    for t in names:
        ids = sorted(buckets[t])          # sorted trước khi shuffle = tái lập được
        rng.shuffle(ids)
        k = round(a.n * len(buckets[t]) / len(dev))
        picked += ids[:k]

    # làm tròn có thể lệch 1-2 câu; bù/bớt ở tầng lớn nhất để tổng đúng bằng n
    big = max(names, key=lambda t: len(buckets[t]))
    while len(picked) != a.n:
        pool = [i for i in sorted(buckets[big]) if i not in set(picked)]
        if len(picked) < a.n and pool:
            picked.append(pool[0])
        elif len(picked) > a.n:
            picked = [i for i in picked if i != [p for p in picked if p in buckets[big]][-1]]
        else:
            break

    sub = {q: dev[q] for q in picked}
    Path(a.out).write_text(
        json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"✅ {a.out}  —  {len(sub)} câu")
    print(f"{'tầng':<10} {'dev':>6} {'':>7}  {'tập con':>8} {'':>7}")
    for t in names:
        got = sum(
            1
            for q in sub
            if tier(min(freq.get(str(x), 0) for x in sub[q]["answer"])) == t
        )
        print(
            f"{t:<10} {len(buckets[t]):>6} {len(buckets[t])/len(dev):>6.1%}  "
            f"{got:>8} {got/len(sub):>7.1%}"
        )
    gold = sum(len(v["answer"]) for v in sub.values())
    print(f"\nΣ|gold| = {gold}  ·  mean = {gold/len(sub):.4f}  ·  "
          f"trần P@5 = {gold/len(sub)/5:.4f}")


if __name__ == "__main__":
    main()
