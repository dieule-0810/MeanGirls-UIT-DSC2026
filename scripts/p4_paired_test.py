"""So sánh hai bảng xếp hạng trên CÙNG một tập câu hỏi — bằng kiểm định CẶP.

Vì sao không dùng sai số biên (SE ≈ 1,3% ở n=1000): hai hệ thống chạy trên đúng
cùng 1000 câu hỏi, nên phần lớn phương sai là chung và triệt tiêu. Sai số của HIỆU
chỉ phụ thuộc vào các câu mà hai bên BẤT ĐỒNG. Bỏ qua điều này sẽ kết luận "chênh
2,4% nằm trong sai số 1,3%±, chưa đủ bằng chứng" — sai, vì 1,3% là sai số của từng
số riêng lẻ, không phải của hiệu.

Hai kiểm định, đọc cả hai:
  • bootstrap cặp  → khoảng tin cậy của hiệu Recall trung bình. Trả lời "chênh bao nhiêu".
  • McNemar        → chỉ nhìn các câu bất đồng. Trả lời "chênh có thật không".

Dùng lại được cho P4-8 (calibration) khi so "luôn nộp 5" với "nộp theo dự đoán".

    python -m scripts.p4_paired_test \
        --a outputs/p4_bm25_ranking/dev_top50_max.json \
        --b outputs/p4_bm25_ranking/dev_top50_mean_top3.json \
        --questions data/dev.json --k 5 --label-a max --label-b mean_top3
"""
from __future__ import annotations

import argparse
import json
import math
import random


def recalls(ranking: dict, questions: dict, k: int) -> dict[str, float]:
    out = {}
    for qid, v in questions.items():
        gold = {str(a) for a in v["answer"]}
        got = {d for d, _, _ in ranking[str(qid)][:k]}
        out[str(qid)] = len(gold & got) / len(gold)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="ranking A (đường cơ sở)")
    ap.add_argument("--b", required=True, help="ranking B (bản mới)")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    ra = recalls(json.load(open(a.a, encoding="utf-8")), q, a.k)
    rb = recalls(json.load(open(a.b, encoding="utf-8")), q, a.k)
    qids = sorted(ra)
    assert set(ra) == set(rb), "Hai ranking không cùng tập câu hỏi"

    diffs = [rb[i] - ra[i] for i in qids]
    n = len(diffs)
    mean_a = sum(ra.values()) / n
    mean_b = sum(rb.values()) / n
    obs = mean_b - mean_a

    print(f"Recall@{a.k} trên {n} câu  ·  {a.questions}")
    print(f"  {a.label_a:<12} = {mean_a:.4f}")
    print(f"  {a.label_b:<12} = {mean_b:.4f}")
    print(f"  hiệu         = {obs:+.4f}")

    # --- bootstrap cặp: lấy lại mẫu CÂU HỎI, giữ nguyên cặp (A,B) của mỗi câu
    rng = random.Random(a.seed)
    boots = []
    for _ in range(a.boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * a.boot)]
    hi = boots[int(0.975 * a.boot)]
    print(f"\n  bootstrap cặp ({a.boot} lần), KTC 95% của hiệu: [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print(f"  → {a.label_b} TỐT HƠN, khoảng tin cậy không chứa 0.")
    elif hi < 0:
        print(f"  → {a.label_b} TỆ HƠN, khoảng tin cậy không chứa 0.")
    else:
        print("  → Khoảng tin cậy CHỨA 0: chưa đủ bằng chứng để nói hai bên khác nhau.")
        print("    Chọn theo tiêu chí khác (tốc độ, tính đơn giản), đừng chọn theo con số này.")

    # --- McNemar: chỉ đếm câu bất đồng
    b_win = sum(1 for d in diffs if d > 0)
    a_win = sum(1 for d in diffs if d < 0)
    tie = n - b_win - a_win
    print(f"\n  McNemar: {a.label_b} thắng {b_win} câu · {a.label_a} thắng {a_win} câu · hoà {tie}")
    disc = a_win + b_win
    if disc == 0:
        print("  → Hai ranking cho kết quả y hệt trên mọi câu.")
        return
    # xấp xỉ chuẩn có hiệu chỉnh liên tục
    chi2 = (abs(b_win - a_win) - 1) ** 2 / disc
    p = math.erfc(math.sqrt(chi2 / 2))
    print(f"  → chi2 = {chi2:.3f}, p ≈ {p:.4g}", end="  ")
    print("(có ý nghĩa ở mức 0,05)" if p < 0.05 else "(CHƯA có ý nghĩa ở mức 0,05)")
    if disc < 25:
        print("    ⚠️ dưới 25 câu bất đồng — xấp xỉ chuẩn kém tin cậy, đọc bootstrap thay thế.")


if __name__ == "__main__":
    main()
