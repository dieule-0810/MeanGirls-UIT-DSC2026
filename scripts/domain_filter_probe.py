#!/usr/bin/env python3
"""
Lọc theo LĨNH VỰC trước/sau truy hồi — đo trần trên trước khi cắm. CHỦ SỞ HỮU: P3.

Ý tưởng: link của mọi văn bản chứa sẵn nhãn lĩnh vực
(`thuvienphapluat.vn/van-ban/<lĩnh-vực>/…`) — 100% văn bản có, 63 lĩnh vực, lớn nhất chỉ 22%.
Đây là metadata MIỄN PHÍ: 0 tham số, 0 model, không phải đăng ký, không đụng luật augmentation.

Vì sao đáng thử ĐÚNG LÚC NÀY: khoảng cách giữa R@50 (0,9700) và R@5 (0,8555) là lỗi XẾP HẠNG,
và nhóm lỗi thống trị là `R-ENTITY` — khớp đúng khung điều luật nhưng sai lĩnh vực. Lọc theo lĩnh
vực tấn công thẳng cơ chế đó: nó không tìm thêm, nó bỏ bớt thứ gây nhiễu ở đỉnh.

Ba chế độ, xếp theo mức rủi ro tăng dần:
  oracle  — lọc theo lĩnh vực CỦA GOLD. Không dùng được thật, chỉ để biết TRẦN TRÊN.
  hard    — đoán lĩnh vực bằng bỏ phiếu trong top-K rồi loại hết doc khác lĩnh vực. Đoán sai là
            mất câu đó VĨNH VIỄN — recall không lấy lại được.
  soft    — cộng thưởng điểm cho doc cùng lĩnh vực đoán được rồi xếp lại. Đoán sai chỉ tụt hạng,
            không loại hẳn. Đây là biến thể an toàn với Recall, metric chính của giải.

    python scripts/domain_filter_probe.py --ranking outputs/v0.3_bm25_best/ranking_full.json \\
        --questions data/dev.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.io import read_jsonl  # noqa: E402

LINK_RE = re.compile(r"thuvienphapluat\.vn/([a-z\-]+)/([a-z\-]+)/")


def domain_map(corpus: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in read_jsonl(corpus):
        did = str(d.get("doc_id", d.get("id")))
        m = LINK_RE.search(d.get("link") or "")
        out[did] = f"{m.group(1)}/{m.group(2)}" if m else "?"
    return out


def recall(preds: dict[str, list[str]], gold: dict[str, list[str]], k: int) -> float:
    tot = 0.0
    for q, g in gold.items():
        if not g:
            continue
        tot += len(set(preds.get(q, [])[:k]) & set(g)) / len(g)
    return tot / max(len(gold), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--vote-k", type=int, default=10, help="độ sâu bỏ phiếu đoán lĩnh vực")
    ap.add_argument("--alpha", type=float, default=0.15, help="mức thưởng của chế độ soft")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    dom = domain_map(REPO / args.corpus)
    rank = json.loads((REPO / args.ranking).read_text(encoding="utf-8"))
    qs = json.loads((REPO / args.questions).read_text(encoding="utf-8"))
    gold = {str(q): [str(a) for a in v["answer"]] for q, v in qs.items() if str(q) in rank}
    k = args.top_k

    rows = {q: [(str(r[0]), float(r[1])) for r in rank[q]] for q in gold}
    base = {q: [d for d, _ in v] for q, v in rows.items()}

    # ── oracle: lọc theo lĩnh vực của gold ───────────────────────────────────
    orc = {
        q: [d for d, _ in v if dom.get(d) in {dom.get(g) for g in gold[q]}] for q, v in rows.items()
    }

    # ── đoán lĩnh vực: bỏ phiếu có trọng số điểm trong top-K ─────────────────
    pred: dict[str, str] = {}
    for q, v in rows.items():
        vote: dict[str, float] = defaultdict(float)
        for d, s in v[: args.vote_k]:
            vote[dom.get(d, "?")] += s
        pred[q] = max(vote, key=vote.get) if vote else "?"

    hard = {q: [d for d, _ in v if dom.get(d) == pred[q]] for q, v in rows.items()}
    soft = {
        q: [d for d, _ in sorted(
            ((d, s * (1 + args.alpha) if dom.get(d) == pred[q] else s) for d, s in v),
            key=lambda t: -t[1],
        )]
        for q, v in rows.items()
    }

    n = len(gold)
    dom_ok = sum(1 for q in gold if pred[q] in {dom.get(g) for g in gold[q]})
    hit5 = {q for q in gold if set(base[q][:k]) & set(gold[q])}
    broken = sum(1 for q in hit5 if not set(hard[q][:k]) & set(gold[q]))

    print(f"n = {n} câu · độ sâu ranking = {len(next(iter(rows.values())))} · bỏ phiếu top-{args.vote_k}\n")
    print(f"{'chế độ':<22}{'R@1':>8}{'R@3':>8}{f'R@{k}':>8}{'R@20':>8}")
    for name, p in (("gốc", base), ("soft boost", soft), ("hard filter", hard), ("ORACLE lĩnh vực", orc)):
        print(f"{name:<22}" + "".join(f"{recall(p, gold, kk):>8.4f}" for kk in (1, 3, k, 20)))

    print(
        f"\nĐoán lĩnh vực đúng      : {dom_ok}/{n} ({dom_ok/n:.1%})\n"
        f"Câu đang đúng @{k} bị hard filter làm HỎNG: {broken} ({broken/max(len(hit5),1):.1%} số câu đang đúng)\n"
        f"Trần trên của lọc lĩnh vực (oracle) : {recall(orc, gold, k):.4f} "
        f"(+{recall(orc, gold, k) - recall(base, gold, k):+.4f} so với gốc)"
    )
    print(
        "\nĐọc kết quả: ORACLE là mức chỉ đạt được nếu đoán lĩnh vực KHÔNG BAO GIỜ sai — không phải "
        "con số dùng được. Khoảng cách giữa `soft` và `gốc` mới là phần thật sự lấy được hôm nay; "
        "`hard` chỉ đáng dùng nếu tỉ lệ làm hỏng ≈ 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
