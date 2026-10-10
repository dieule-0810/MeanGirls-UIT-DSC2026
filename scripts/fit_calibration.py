#!/usr/bin/env python3
"""Tìm ngưỡng θ cho bộ quyết định số lượng doc (`src/rerank/calibrate.py`).

Bài toán CÓ RÀNG BUỘC, không phải "tối ưu một con số":

    max precision   sao cho   recall ≥ recall("luôn nộp max_count") − --max-recall-drop

Ngân sách recall khai TRƯỚC khi nhìn kết quả. Luật cắt với ngưỡng HẰNG đơn điệu theo θ (tăng θ
⇒ mọi câu trả nhiều doc hơn ⇒ recall chỉ tăng, precision chỉ giảm), nên θ* = θ NHỎ NHẤT còn thoả
ràng buộc — một phép quét một chiều, không có cực trị địa phương.

Fit trên `train_split`, báo cáo trên `dev`: `--verify-ranking` chạy bước báo cáo NGAY trong một
lệnh, với θ đã chốt ở tập fit và KHÔNG fit lại.

Typical usage example:

    python scripts/fit_calibration.py \\
        --ranking outputs/v0.8_hybrid_doc/train_split/ranking_full.json --questions data/train_split.json \\
        --verify-ranking outputs/v0.8_hybrid_doc/ranking_full.json --verify-questions data/dev.json \\
        --max-recall-drop 0.003 --out outputs/v0.8_hybrid_doc/calibration.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import repo_path  # noqa: E402
from src.common.runinfo import git_commit, guard_holdout  # noqa: E402
from src.evaluate import eval_official  # noqa: E402
from src.rerank.calibrate import (  # noqa: E402
    HARD_LIMIT,
    count_from_margins,
    count_histogram,
    margins,
    normalize_params,
)


@dataclass(frozen=True)
class CutRule:
    """Tham số luật cắt dùng chung cho fit và báo cáo."""

    max_count: int = HARD_LIMIT
    min_count: int = 1
    scale_rank: int = 10


@dataclass
class CalibSet:
    """Ranking + nhãn trên CÙNG tập qid, kèm khe hở đã tính sẵn (không phụ thuộc θ)."""

    name: str
    qids: list[str]
    rows: list[list]
    truth: dict[str, list[str]]
    margs: list[list[float]]

    def counts_at(self, theta: float, rule: CutRule) -> list[int]:
        """Áp luật cắt cho cả tập ở một θ — gọi thẳng `count_from_margins`, không chép lại luật."""
        thresholds = normalize_params(theta, rule.max_count, rule.min_count)
        return [
            min(count_from_margins(m, thresholds, rule.max_count, rule.min_count), max(1, len(row)))
            for m, row in zip(self.margs, self.rows)
        ]

    def base_counts(self, rule: CutRule) -> list[int]:
        """Đường cơ sở: luôn nộp `max_count` (hoặc ít hơn nếu ranking ngắn)."""
        return [min(rule.max_count, max(1, len(r))) for r in self.rows]

    def score(self, counts: list[int]) -> tuple[dict, dict[str, list[str]]]:
        """Chấm như BTC với số doc cho trước."""
        preds = {q: [str(d[0]) for d in row[:n]] for q, row, n in zip(self.qids, self.rows, counts)}
        return eval_official(preds, self.truth), preds


def load_calib_set(ranking_path: str, questions_path: str, rule: CutRule) -> CalibSet:
    """Nạp ranking + nhãn; lệch một qid là mã chấm BTC crash nên chặn ngay ở đây.

    Raises:
        SystemExit: qid thiếu trong ranking, hoặc file câu hỏi không có nhãn.
    """
    ranking = json.loads(repo_path(ranking_path).read_text(encoding="utf-8"))
    questions = json.loads(repo_path(questions_path).read_text(encoding="utf-8"))
    qids = [str(q) for q in questions]
    missing = [q for q in qids if q not in ranking]
    if missing:
        raise SystemExit(
            f"❌ {len(missing)} qid có trong {questions_path} nhưng không có trong {ranking_path} "
            f"(vd {missing[:3]}). Hai file phải sinh ra từ cùng một lần chạy."
        )
    truth = {}
    for q in qids:
        v = questions[q]
        if not isinstance(v, dict) or "answer" not in v:
            raise SystemExit(f"❌ qid {q} thiếu khoá 'answer' — file này không có nhãn, không fit được.")
        truth[q] = [str(x) for x in v["answer"]]
    rows = [ranking[q] for q in qids]
    margs = [margins([float(d[1]) for d in row], rule.max_count, rule.scale_rank) for row in rows]
    return CalibSet(questions_path, qids, rows, truth, margs)


def per_query_recall(preds: dict[str, list[str]], truth: dict[str, list[str]]) -> dict[str, float]:
    """Recall từng câu, chỉ để bootstrap cặp; trung bình được đối chiếu với `eval_official`."""
    return {
        q: len(set(gold) & set(preds[q])) / len(gold) if 0 < len(preds[q]) <= HARD_LIMIT else 0.0
        for q, gold in truth.items()
    }


def bootstrap_drop(base_r: dict[str, float], cal_r: dict[str, float], n_boot: int, seed: int) -> dict:
    """KTC95 của HIỆU recall theo cặp (cùng câu, cùng ranking, chỉ khác số doc trả về).

    Args:
        base_r: Recall từng câu của đường cơ sở.
        cal_r: Recall từng câu sau hiệu chỉnh.
        n_boot: Số lần lấy mẫu lại.
        seed: Hạt giống.

    Returns:
        `{"delta_recall", "ci95", "n_worse", "n_better"}`.
    """
    diffs = [cal_r[q] - base_r[q] for q in sorted(base_r)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "delta_recall": sum(diffs) / n,
        "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]],
        "n_worse": sum(1 for d in diffs if d < 0),
        "n_better": sum(1 for d in diffs if d > 0),
    }


def theta_grid(margs: list[list[float]], n_points: int) -> list[float]:
    """Lưới θ = phân vị của các khe hở quan sát được, cộng hai đầu mút.

    Raises:
        SystemExit: Không có khe hở nào (ranking chỉ 1 doc/câu).
    """
    observed = sorted({m for row in margs for m in row})
    if not observed:
        raise SystemExit("❌ Không có m_k nào — ranking chỉ có 1 doc/câu? Không có gì để hiệu chỉnh.")
    step = max(1, len(observed) // n_points)
    return sorted({observed[i] for i in range(0, len(observed), step)} | {0.0, observed[-1] + 1e-9})


@dataclass
class FitResult:
    """θ* và mọi thứ cần để báo cáo trên tập fit."""

    theta: float
    score: dict
    counts: list[int]
    preds: dict[str, list[str]]
    curve: list[dict]


def fit_theta(cset: CalibSet, rule: CutRule, floor: float, n_points: int) -> FitResult:
    """θ nhỏ nhất giữ recall ≥ `floor` (= precision cao nhất trong ngân sách).

    Raises:
        SystemExit: Không θ nào thoả ràng buộc.
    """
    curve, chosen = [], None
    for theta in theta_grid(cset.margs, n_points):
        counts = cset.counts_at(theta, rule)
        s, preds = cset.score(counts)
        curve.append({"theta": theta, "recall": s["recall"], "precision": s["precision"],
                      "mean_docs": statistics.mean(counts)})
        if chosen is None and s["recall"] >= floor:
            chosen = FitResult(theta, s, counts, preds, curve)
    if chosen is None:
        raise SystemExit(
            f"❌ Không θ nào giữ được recall >= {floor:.4f}. Nới --max-recall-drop, hoặc chấp nhận rằng "
            f"tầng này không có lợi cho ranking hiện tại — đó cũng là một kết quả, hãy ghi lại."
        )
    chosen.curve = curve
    return chosen


def report(tag: str, s: dict, counts: list[int], base: dict) -> None:
    """In một dòng recall/precision/số doc so với đường cơ sở."""
    spread = " ".join(f"{n}:{c}" for n, c in count_histogram(counts).items())
    print(
        f"  {tag:<12} recall={s['recall']:.4f} (Δ{s['recall'] - base['recall']:+.4f})  "
        f"precision={s['precision']:.4f} (×{s['precision'] / base['precision']:.2f})  "
        f"doc/câu={statistics.mean(counts):.2f}  [{spread}]"
    )


def _print_boot(b: dict) -> None:
    """In kết quả bootstrap cặp."""
    print(f"  bootstrap cặp ΔRecall = {b['delta_recall']:+.4f} KTC95 [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] · "
          f"{b['n_worse']} câu xấu đi, {b['n_better']} câu tốt lên")


def _print_curve_near(fit: FitResult) -> None:
    """In vài điểm đường cong quanh θ* để kiểm tính đơn điệu bằng mắt."""
    print("lân cận đường cong — θ tăng ⇒ recall tăng, precision giảm (đơn điệu, kiểm được bằng mắt):")
    i_star = next(i for i, c in enumerate(fit.curve) if c["theta"] == fit.theta)
    for c in fit.curve[max(0, i_star - 2): i_star + 3]:
        mark = " ←" if c["theta"] == fit.theta else ""
        print(f"    θ={c['theta']:.5f}  recall={c['recall']:.4f}  precision={c['precision']:.4f}  "
              f"doc/câu={c['mean_docs']:.2f}{mark}")


def verify_on(cset: CalibSet, theta: float, rule: CutRule, args: argparse.Namespace) -> dict:
    """Báo cáo trên tập khác với θ GIỮ NGUYÊN — không fit lại."""
    base_counts = cset.base_counts(rule)
    base, base_preds = cset.score(base_counts)
    counts = cset.counts_at(theta, rule)
    s, preds = cset.score(counts)
    print(f"\nTẬP BÁO CÁO — {cset.name} ({len(cset.qids)} câu), θ lấy từ tập fit, KHÔNG fit lại:")
    report("cơ sở", base, base_counts, base)
    report("hiệu chỉnh", s, counts, base)
    boot = bootstrap_drop(per_query_recall(base_preds, cset.truth), per_query_recall(preds, cset.truth), args.boot, args.seed)
    _print_boot(boot)
    if boot["delta_recall"] < -args.max_recall_drop:
        print("  ⚠️  Ở tập báo cáo recall tụt QUÁ ngân sách ⇒ θ không khái quát hoá. "
              "Đừng nộp bản này; hạ --max-recall-drop rồi fit lại.")
    return {"ranking": args.verify_ranking, "questions": args.verify_questions, "n": len(cset.qids),
            "base": base, "calibrated": s, "hist": count_histogram(counts), "bootstrap": boot}


def print_yaml_block(theta: float, rule: CutRule, fit_name: str) -> None:
    """In khối YAML để dán vào config (không hằng số trong code)."""
    print("\nKhối YAML để dán vào config:\n")
    print(f"pipeline:\n  calibrate:\n    thresholds: {theta:.6f}   # vectơ hằng, fit trên {Path(fit_name).name}")
    print(f"    max_count: {rule.max_count}\n    min_count: {rule.min_count}\n    scale_rank: {rule.scale_rank}")


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", required=True, help="ranking_full.json của TẬP FIT (train_split)")
    ap.add_argument("--questions", required=True, help="file câu hỏi CÓ NHÃN của tập fit")
    ap.add_argument("--verify-ranking", default=None, help="ranking của tập BÁO CÁO (dev) — không fit lại")
    ap.add_argument("--verify-questions", default=None)
    ap.add_argument("--max-recall-drop", type=float, default=0.003,
                    help="ngân sách recall, khai TRƯỚC khi nhìn kết quả (plan.md mục 3: ~0,3%%)")
    ap.add_argument("--max-count", type=int, default=HARD_LIMIT)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--scale-rank", type=int, default=10)
    ap.add_argument("--grid", type=int, default=400, help="số điểm quét θ (phân vị của m_k quan sát được)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="ghi tham số + báo cáo ra JSON")
    return ap.parse_args()


def main() -> int:
    """Điểm vào CLI: fit θ trên tập fit, báo cáo trên tập khác."""
    args = parse_args()
    guard_holdout(args.questions, False, purpose="fit")
    if args.verify_ranking and not args.verify_questions:
        raise SystemExit("❌ --verify-ranking cần --verify-questions đi kèm.")
    rule = CutRule(args.max_count, args.min_count, args.scale_rank)
    cset = load_calib_set(args.ranking, args.questions, rule)
    print(f"tập fit : {args.questions} · {len(cset.qids)} câu · ranking {args.ranking}")

    base_counts = cset.base_counts(rule)
    base, base_preds = cset.score(base_counts)
    base_r = per_query_recall(base_preds, cset.truth)
    if abs(sum(base_r.values()) / len(base_r) - base["recall"]) > 1e-9:
        raise AssertionError("recall từng câu không khớp eval_official — dừng, xem INTERFACES.md mục 5.")
    floor = base["recall"] - args.max_recall_drop
    print(f"\nđường cơ sở (luôn {rule.max_count} doc): recall={base['recall']:.4f} precision={base['precision']:.4f}")
    print(f"ngân sách recall: {args.max_recall_drop:.4f} ⇒ sàn recall = {floor:.4f}\n")

    fit = fit_theta(cset, rule, floor, args.grid)
    print(f"θ* = {fit.theta:.6f}  (θ nhỏ nhất còn thoả ràng buộc ⇒ precision cao nhất trong ngân sách)\n")
    _print_curve_near(fit)
    print(f"\nTẬP FIT ({len(cset.qids)} câu):")
    report("cơ sở", base, base_counts, base)
    report("hiệu chỉnh", fit.score, fit.counts, base)
    boot_fit = bootstrap_drop(base_r, per_query_recall(fit.preds, cset.truth), args.boot, args.seed)
    _print_boot(boot_fit)

    out = {
        "ngay": date.today().isoformat(), "commit": git_commit(),
        "fit": {"ranking": args.ranking, "questions": args.questions, "n": len(cset.qids)},
        "params": {"thresholds": fit.theta, "max_count": rule.max_count,
                   "min_count": rule.min_count, "scale_rank": rule.scale_rank},
        "max_recall_drop": args.max_recall_drop,
        "tap_fit": {"base": base, "calibrated": fit.score, "hist": count_histogram(fit.counts), "bootstrap": boot_fit},
        "curve": fit.curve,
    }
    if args.verify_ranking:
        vset = load_calib_set(args.verify_ranking, args.verify_questions, rule)
        out["tap_bao_cao"] = verify_on(vset, fit.theta, rule, args)
    print_yaml_block(fit.theta, rule, args.questions)
    if args.out:
        p = repo_path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
