#!/usr/bin/env python3
"""
Tìm ngưỡng cho bộ quyết định số lượng doc (`src/rerank/calibrate.py`). CHỦ SỞ HỮU: P4.

Bài toán CÓ RÀNG BUỘC, không phải "tối ưu một con số":

    max precision   sao cho   recall >= recall("luôn nộp max_count") - --max-recall-drop

Recall là metric chính, precision chỉ phá hoà ⇒ ngân sách recall là thứ người chạy khai
TRƯỚC, không phải thứ đọc ra sau khi đã nhìn kết quả.

VÌ SAO QUÉT MỘT CHIỀU LÀ ĐỦ. Luật cắt với ngưỡng HẰNG đơn điệu theo θ: tăng θ ⇒ mọi câu
trả về nhiều doc hơn ⇒ tập dự đoán mới là tập CHA của tập cũ ⇒ recall chỉ tăng, precision
chỉ giảm. Nên tồn tại đúng một biên θ*, và θ* = θ NHỎ NHẤT còn thoả ràng buộc recall.
Không có cực trị địa phương để mắc kẹt, không cần khởi tạo lại nhiều lần.

QUY TẮC PHƯƠNG PHÁP (docs/runbook_e2e.md): fit trên `train_split`, báo cáo trên `dev`,
`holdout` chạm đúng một lần. `--verify-ranking` chạy bước báo cáo NGAY TRONG một lệnh,
với θ đã chốt ở tập fit và KHÔNG fit lại — để không ai vô tình chọn tham số trên tập đo.

    python scripts/fit_calibration.py \
        --ranking outputs/v0.7_calib/train_split/ranking_full.json \
        --questions data/train_split.json \
        --verify-ranking outputs/v0.3_bm25_best/ranking_full.json \
        --verify-questions data/dev.json \
        --max-recall-drop 0.003 --out outputs/v0.7_calib/calibration.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluate import eval_official  # noqa: E402
from src.rerank.calibrate import (  # noqa: E402
    HARD_LIMIT,
    count_from_margins,
    count_histogram,
    margins,
    normalize_params,
)


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def load_pair(ranking_path: Path, questions_path: Path) -> tuple[list[str], list[list], dict]:
    """Ranking + nhãn trên CÙNG tập qid. Lệch một qid là mã chấm BTC crash ⇒ chặn ngay ở đây."""
    ranking = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
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
    return qids, [ranking[q] for q in qids], truth


def per_query_recall(preds: dict[str, list[str]], truth: dict[str, list[str]]) -> dict[str, float]:
    """
    Recall từng câu — chỉ để bootstrap cặp. Trung bình của nó được ĐỐI CHIẾU với
    `eval_official`; lệch là dừng, vì nghĩa là ta vừa tự diễn giải lại mã chấm, đúng
    việc INTERFACES.md mục 5 cấm.
    """
    out = {}
    for q, gold in truth.items():
        p = preds[q]
        out[q] = len(set(gold) & set(p)) / len(gold) if 0 < len(p) <= HARD_LIMIT else 0.0
    return out


def score_counts(qids, rows, truth, counts) -> tuple[dict, dict[str, list[str]]]:
    preds = {q: [str(d[0]) for d in row[:n]] for q, row, n in zip(qids, rows, counts)}
    return eval_official(preds, truth), preds


def counts_at(margs, rows, theta: float, max_count: int, min_count: int) -> list[int]:
    """
    Áp luật cắt cho cả tập ở một θ. Gọi thẳng `calibrate.count_from_margins` chứ KHÔNG
    chép lại luật: fit một luật rồi chạy một luật khác là cách hỏng không ai phát hiện ra.
    `margins` tính một lần ở ngoài vì nó không phụ thuộc θ.
    """
    thresholds = normalize_params(theta, max_count, min_count)
    return [
        min(count_from_margins(m, thresholds, max_count, min_count), max(1, len(row)))
        for m, row in zip(margs, rows)
    ]


def bootstrap_drop(base_r: dict[str, float], cal_r: dict[str, float], n_boot: int, seed: int) -> dict:
    """
    KTC95 của HIỆU recall theo cặp (cùng câu hỏi, cùng ranking, chỉ khác số doc trả về).
    Cùng quy ước với `scripts/p4_paired_test.py`: phần lớn phương sai là chung và triệt
    tiêu, sai số của HIỆU chỉ đến từ những câu hai bên bất đồng.
    """
    qids = sorted(base_r)
    diffs = [cal_r[q] - base_r[q] for q in qids]
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "delta_recall": obs,
        "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]],
        "n_worse": sum(1 for d in diffs if d < 0),
        "n_better": sum(1 for d in diffs if d > 0),
    }


def report(tag: str, s: dict, counts: list[int], base: dict) -> None:
    spread = " ".join(f"{n}:{c}" for n, c in count_histogram(counts).items())
    print(
        f"  {tag:<12} recall={s['recall']:.4f} (Δ{s['recall'] - base['recall']:+.4f})  "
        f"precision={s['precision']:.4f} (×{s['precision'] / base['precision']:.2f})  "
        f"doc/câu={statistics.mean(counts):.2f}  [{spread}]"
    )


def main() -> int:
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
    a = ap.parse_args()

    if "holdout" in Path(a.questions).name:
        raise SystemExit("❌ Không fit trên holdout. Đó là tập ĐO LẦN CUỐI (docs/runbook_e2e.md).")

    qids, rows, truth = load_pair(_abs(a.ranking), _abs(a.questions))
    margs = [margins([float(d[1]) for d in row], a.max_count, a.scale_rank) for row in rows]
    print(f"tập fit : {a.questions} · {len(qids)} câu · ranking {a.ranking}")

    # ── đường cơ sở: luôn nộp max_count ──────────────────────────────────────
    base_counts = [min(a.max_count, max(1, len(r))) for r in rows]
    base, base_preds = score_counts(qids, rows, truth, base_counts)
    base_r = per_query_recall(base_preds, truth)
    if abs(sum(base_r.values()) / len(base_r) - base["recall"]) > 1e-9:
        raise AssertionError("recall từng câu không khớp eval_official — dừng, xem INTERFACES.md mục 5.")
    print(f"\nđường cơ sở (luôn {a.max_count} doc): "
          f"recall={base['recall']:.4f} precision={base['precision']:.4f}")
    print(f"ngân sách recall: {a.max_recall_drop:.4f} ⇒ sàn recall = {base['recall'] - a.max_recall_drop:.4f}\n")

    # ── quét θ ───────────────────────────────────────────────────────────────
    observed = sorted({m for row in margs for m in row})
    if not observed:
        raise SystemExit("❌ Không có m_k nào — ranking chỉ có 1 doc/câu? Không có gì để hiệu chỉnh.")
    step = max(1, len(observed) // a.grid)
    grid = sorted({observed[i] for i in range(0, len(observed), step)} | {0.0, observed[-1] + 1e-9})

    floor = base["recall"] - a.max_recall_drop
    curve, chosen = [], None
    for theta in grid:
        counts = counts_at(margs, rows, theta, a.max_count, a.min_count)
        s, preds = score_counts(qids, rows, truth, counts)
        curve.append({"theta": theta, "recall": s["recall"], "precision": s["precision"],
                      "mean_docs": statistics.mean(counts)})
        if chosen is None and s["recall"] >= floor:
            chosen = (theta, s, counts, preds)   # θ nhỏ nhất thoả ràng buộc = precision cao nhất

    if chosen is None:
        raise SystemExit(
            f"❌ Không θ nào giữ được recall >= {floor:.4f}. Nới --max-recall-drop, hoặc chấp nhận "
            f"rằng tầng này không có lợi cho ranking hiện tại — đó cũng là một kết quả, hãy ghi lại."
        )
    theta, s_fit, counts_fit, preds_fit = chosen
    print(f"θ* = {theta:.6f}  (θ nhỏ nhất còn thoả ràng buộc ⇒ precision cao nhất trong ngân sách)\n")
    print("lân cận đường cong — θ tăng ⇒ recall tăng, precision giảm (đơn điệu, kiểm được bằng mắt):")
    i_star = next(i for i, c in enumerate(curve) if c["theta"] == theta)
    for c in curve[max(0, i_star - 2): i_star + 3]:
        mark = " ←" if c["theta"] == theta else ""
        print(f"    θ={c['theta']:.5f}  recall={c['recall']:.4f}  precision={c['precision']:.4f}  "
              f"doc/câu={c['mean_docs']:.2f}{mark}")

    print(f"\nTẬP FIT ({len(qids)} câu):")
    report("cơ sở", base, base_counts, base)
    report("hiệu chỉnh", s_fit, counts_fit, base)
    boot_fit = bootstrap_drop(base_r, per_query_recall(preds_fit, truth), a.boot, a.seed)
    print(f"  bootstrap cặp ΔRecall = {boot_fit['delta_recall']:+.4f} "
          f"KTC95 [{boot_fit['ci95'][0]:+.4f}, {boot_fit['ci95'][1]:+.4f}] · "
          f"{boot_fit['n_worse']} câu xấu đi, {boot_fit['n_better']} câu tốt lên")

    out = {
        "ngay": date.today().isoformat(),
        "commit": git_commit(),
        "fit": {"ranking": a.ranking, "questions": a.questions, "n": len(qids)},
        "params": {"thresholds": theta, "max_count": a.max_count,
                   "min_count": a.min_count, "scale_rank": a.scale_rank},
        "max_recall_drop": a.max_recall_drop,
        "tap_fit": {"base": base, "calibrated": s_fit,
                    "hist": count_histogram(counts_fit), "bootstrap": boot_fit},
        "curve": curve,
    }

    # ── báo cáo trên tập khác, θ GIỮ NGUYÊN ──────────────────────────────────
    if a.verify_ranking:
        if not a.verify_questions:
            raise SystemExit("❌ --verify-ranking cần --verify-questions đi kèm.")
        vq, vrows, vtruth = load_pair(_abs(a.verify_ranking), _abs(a.verify_questions))
        vmargs = [margins([float(d[1]) for d in row], a.max_count, a.scale_rank) for row in vrows]
        vbase_counts = [min(a.max_count, max(1, len(r))) for r in vrows]
        vbase, vbase_preds = score_counts(vq, vrows, vtruth, vbase_counts)
        vcounts = counts_at(vmargs, vrows, theta, a.max_count, a.min_count)
        vs, vpreds = score_counts(vq, vrows, vtruth, vcounts)
        print(f"\nTẬP BÁO CÁO — {a.verify_questions} ({len(vq)} câu), θ lấy từ tập fit, KHÔNG fit lại:")
        report("cơ sở", vbase, vbase_counts, vbase)
        report("hiệu chỉnh", vs, vcounts, vbase)
        boot_v = bootstrap_drop(per_query_recall(vbase_preds, vtruth),
                                per_query_recall(vpreds, vtruth), a.boot, a.seed)
        print(f"  bootstrap cặp ΔRecall = {boot_v['delta_recall']:+.4f} "
              f"KTC95 [{boot_v['ci95'][0]:+.4f}, {boot_v['ci95'][1]:+.4f}] · "
              f"{boot_v['n_worse']} câu xấu đi, {boot_v['n_better']} câu tốt lên")
        if boot_v["delta_recall"] < -a.max_recall_drop:
            print("  ⚠️  Ở tập báo cáo recall tụt QUÁ ngân sách ⇒ θ không khái quát hoá. "
                  "Đừng nộp bản này; hạ --max-recall-drop rồi fit lại.")
        out["tap_bao_cao"] = {"ranking": a.verify_ranking, "questions": a.verify_questions,
                              "n": len(vq), "base": vbase, "calibrated": vs,
                              "hist": count_histogram(vcounts), "bootstrap": boot_v}

    print("\nKhối YAML để dán vào config (INTERFACES.md mục 6 — không hằng số trong code):\n")
    print("pipeline:")
    print("  calibrate:")
    print(f"    thresholds: {theta:.6f}   # vectơ hằng, fit trên {Path(a.questions).name}")
    print(f"    max_count: {a.max_count}")
    print(f"    min_count: {a.min_count}")
    print(f"    scale_rank: {a.scale_rank}")

    if a.out:
        p = _abs(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
