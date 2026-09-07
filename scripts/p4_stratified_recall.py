#!/usr/bin/env python3
"""P4-3: Recall phân tầng theo tần suất gold trong train_split.

VÌ SAO: holdout/dev tách từ train, nên gold của chúng THEO CẤU TẠO nằm trong
3.105 doc đã từng là gold, trong khi 63,6% corpus chưa từng là gold. Public/
private test không có ràng buộc đó. Mọi thành phần học từ train_split sẽ có
Recall trên holdout cao hơn Recall thật — holdout mù với đúng cái nó cần đo.

CÁCH ĐỌC: tầng freq=0 là đại diện gần nhất cho phần corpus mô hình chưa thấy.
Nếu lợi thế của một mô hình biến mất ở tầng đó, lợi thế ấy phần lớn là GHI NHỚ
chứ không phải khái quát hoá. Đây là ô thí nghiệm trả lời trực tiếp câu hỏi
của BTC về deep learning thuần vs tận dụng LLM.

🔴 PHẢI chạy trên BM25 TRƯỚC mọi fine-tune. BM25 không học gì từ train_split
nên bốn tầng phải xấp xỉ bằng nhau. Nếu KHÔNG bằng nhau, đó tự nó là phát
hiện: freq đang đo độ khó nội tại của câu hỏi, và mọi so sánh sau phải trừ đi
phần chênh này. Không có đường cơ sở này thì kết luận sau không đứng vững.

Chạy:
    python -m scripts.p4_stratified_recall \
        --ranking outputs/p4_bm25_ranking/error_pool_top50.json \
        --questions data/error_pool.json --label "BM25 v0.1"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = REPO / "data" / "train_split.json"

# CHỐT — đừng đổi giữa chừng, đổi là mọi bảng cũ hết so được.
# Với câu nhiều gold dùng min: bảo thủ nhất, câu là "khó" nếu CÓ BẤT KỲ gold nào
# chưa từng thấy. Dùng max sẽ giấu mất đúng những câu ta cần soi.
AGG = min
TIERS = [(0, 0, "freq=0"), (1, 2, "freq=1-2"), (3, 10, "freq=3-10"), (11, 10**9, "freq>=11")]


def tier_of(f: int) -> str:
    for lo, hi, name in TIERS:
        if lo <= f <= hi:
            return name
    raise ValueError(f)


def gold_freq() -> Counter:
    """Số lần mỗi doc là gold trong train_split (KHÔNG tính holdout/dev/error_pool)."""
    data = json.loads(TRAIN_SPLIT.read_text(encoding="utf-8"))
    return Counter(str(a) for v in data.values() for a in (v.get("answer") or []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True, help="đầu ra của p4_build_ranking")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--label", default="", help="tên hệ thống, để in vào bảng")
    ap.add_argument("--ks", default="5,20,50")
    a = ap.parse_args()

    freq = gold_freq()
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    rank = json.loads(Path(a.ranking).read_text(encoding="utf-8"))
    ks = [int(x) for x in a.ks.split(",")]

    # Gán tầng một lần, dùng lại cho mọi K → tầng không đổi theo K.
    tier_of_q: dict[str, str] = {}
    for qid, item in questions.items():
        gold = [str(x) for x in (item.get("answer") or [])]
        if not gold:
            raise SystemExit(f"qid {qid} không có nhãn — tập test không dùng được ở đây.")
        tier_of_q[str(qid)] = tier_of(AGG(freq.get(g, 0) for g in gold))

    counts = Counter(tier_of_q.values())
    names = [n for _, _, n in TIERS]
    n_all = len(tier_of_q)

    print(f"\n{a.label or a.ranking}  ·  {n_all} câu  ·  agg={AGG.__name__}")
    print(f"{'K':>4} | " + " | ".join(f"{n:>12}" for n in names) + " |    toàn cục")
    print("-" * (7 + 15 * len(names) + 14))

    rows = {}
    for k in ks:
        acc = defaultdict(list)
        overall = []
        for qid, item in questions.items():
            qid = str(qid)
            gold = {str(x) for x in item["answer"]}
            got = {d for d, _, _ in rank.get(qid, [])[:k]}
            r = len(gold & got) / len(gold)
            acc[tier_of_q[qid]].append(r)
            overall.append(r)
        rows[k] = {t: sum(v) / len(v) for t, v in acc.items()}
        cells = " | ".join(
            f"{rows[k].get(n, float('nan')):>12.4f}" if n in rows[k] else f"{'—':>12}"
            for n in names
        )
        print(f"{k:>4} | {cells} |  {sum(overall)/len(overall):>10.4f}")

    print(f"{'n':>4} | " + " | ".join(f"{counts.get(n, 0):>12}" for n in names) + f" |  {n_all:>10}")

    # Chênh lệch tầng cao nhất ↔ thấp nhất, ở MỌI K.
    #
    # 🔴 Bản trước chỉ in spread ở K lớn nhất, và in kết luận "bốn tầng xấp xỉ bằng
    # nhau" dựa trên mỗi con số đó. Trên dev n=1000 điều đó gây hiểu sai nghiêm trọng:
    #     K=5 → 0,2166      K=20 → 0,0578      K=50 → 0,0368
    # Ở độ sâu ta THỰC SỰ nộp bài (K=5) bốn tầng chênh 21,7 điểm, còn ở K=50 thì gần
    # như bằng nhau. Chỉ nhìn K=50 sẽ kết luận "BM25 không thiên lệch theo tầng" —
    # ngược hẳn sự thật. Spread phải đọc theo từng K, và spread ở K=5 mới là con số
    # dùng làm đường cơ sở để trừ khi so với mô hình fine-tuned.
    print()
    spreads = {}
    for k in ks:
        have = [n for n in names if n in rows[k]]
        if len(have) < 2:
            continue
        spreads[k] = max(rows[k][n] for n in have) - min(rows[k][n] for n in have)
        flag = "⚠️ " if spreads[k] > 0.05 else "✓ "
        print(f"  {flag}spread giữa 4 tầng tại K={k:<3} = {spreads[k]:.4f}")

    if len(spreads) >= 2:
        k_lo, k_hi = min(spreads), max(spreads)
        if spreads[k_lo] > 0.05 and spreads[k_hi] <= 0.05:
            print(
                f"\n  ĐỌC: spread co lại từ {spreads[k_lo]:.4f} (K={k_lo}) xuống "
                f"{spreads[k_hi]:.4f} (K={k_hi}).\n"
                "  Ở độ sâu lớn bốn tầng bao phủ như nhau ⇒ chênh lệch KHÔNG nằm ở khâu\n"
                "  tìm kiếm mà nằm trọn ở khâu XẾP HẠNG. Đây là dư địa của reranker, và\n"
                f"  spread tại K={k_lo} ({spreads[k_lo]:.4f}) là đường cơ sở phải TRỪ ĐI\n"
                "  trước khi quy bất kỳ chênh lệch nào của mô hình fine-tuned cho ghi nhớ."
            )
        elif spreads[k_hi] > 0.05:
            print(
                f"\n  ⚠️ spread vẫn > 5% ở K={k_hi}. Bốn tầng KHÔNG bao phủ như nhau ⇒\n"
                "  freq đang đo cả độ khó nội tại lẫn hiệu ứng ghi nhớ. Không tách được\n"
                "  hai thứ đó bằng thí nghiệm này — ghi nhận là hạn chế, đừng kết luận."
            )
        else:
            print(
                "\n  ✓ Bốn tầng xấp xỉ bằng nhau ở MỌI K — đúng kỳ vọng với hệ thống không\n"
                "  học. Mọi chênh lệch của mô hình fine-tuned sau này đọc được là ghi nhớ."
            )

    print("\nMọi % ở đây là % TRONG TẦNG. Khi viết bài báo phải quy đổi lại theo")
    print("tỉ lệ thật của bốn tầng (error_taxonomy.md — Cảnh báo về lấy mẫu).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
