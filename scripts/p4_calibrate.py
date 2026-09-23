"""Bộ quyết định SỐ LƯỢNG doc trả về — hiệu chỉnh ngưỡng margin trên tập dev.

Vì sao cần: Recall là metric chính, nhưng 92,1% câu chỉ có 1 đáp án đúng nên trần
Precision khi luôn nộp 5 doc chỉ là 0,2164. Khi nhiều đội hoà Recall — chuyện rất dễ
xảy ra vì Recall trên 1.000 câu có độ hạt thô — Precision quyết thứ hạng. Nộp ít doc
hơn cho những câu mà hệ thống TỰ TIN sẽ nâng Precision gần như miễn phí.

Tín hiệu tự tin: KHOẢNG CÁCH ĐIỂM giữa các doc liền kề, không phải điểm tuyệt đối.
Điểm tuyệt đối cao chỉ nói "câu này nhiều từ khớp", còn khoảng cách nói "ứng viên số 1
bỏ xa số 2" — đúng thứ ta cần.

Bốn cách đo margin, vì điểm BM25 luôn dương (~70) còn logit reranker có thể ÂM:
  spread  (mặc định)  (s_i − s_{i+1}) / (s_1 − s_N)   vô hướng-độc-lập, chạy được với
                                                       điểm âm; cùng một t có nghĩa
                                                       trên cả BM25 lẫn reranker
  rel                 (s_i − s_{i+1}) / s_1            chỉ dùng khi điểm luôn dương
  abs                 s_i − s_{i+1}                    t phụ thuộc thang điểm
  softmax             p_i − p_{i+1} sau softmax(τ)     nhạy với τ, để đối chứng

Hai quy tắc:
  top1     trả 1 doc nếu margin₁ > t, ngược lại trả max_docs. Đơn giản, dễ giải thích.
  cascade  trả n doc với n nhỏ nhất sao cho margin_n > t. Tổng quát hoá của top1,
           cho phép trả 2–3 doc ở câu có hai ứng viên ngang nhau rồi mới tụt dốc.

CẢNH BÁO PHƯƠNG PHÁP LUẬN: hiệu chỉnh t trên dev, xác nhận ĐÚNG MỘT LẦN trên holdout.
Quét t trên holdout rồi chọn t tốt nhất = tự bịt mắt mình. Script sẽ cảnh báo nếu
--questions trỏ vào holdout.

Mẫu số Precision là len(list) chứ KHÔNG phải len(set) — trùng lặp vừa tính vào giới hạn
5 doc vừa làm hỏng Precision, nên script khử trùng trước khi tính (plan.md, Bẫy 2).

    python -m scripts.p4_calibrate \
        --ranking outputs/p4_bm25_ranking/dev_top50_mean_top2.json \
        --questions data/dev.json --budget 0.003

    # chốt được t rồi thì xuất dự đoán để so bằng kiểm định cặp
    python -m scripts.p4_calibrate --ranking ... --questions ... \
        --pick-t 0.20 --dump-pred outputs/p4_calib/dev_pred_t020.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EPS = 1e-9


def load_ranking(path: str) -> dict[str, list[tuple[str, float]]]:
    """Đọc mọi định dạng ranking của team: phần tử là [doc_id, score, ...phần thừa]."""
    raw = json.load(open(path, encoding="utf-8"))
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, items in raw.items():
        seen: set[str] = set()
        keep: list[tuple[str, float]] = []
        for it in items:
            doc, score = str(it[0]), float(it[1])
            if doc in seen:      # khử trùng NGAY, giữ lần xuất hiện đầu (điểm cao nhất)
                continue
            seen.add(doc)
            keep.append((doc, score))
        out[str(qid)] = keep
    return out


def margins(scores: list[float], mode: str, tau: float) -> list[float]:
    """margins[i] = độ hụt từ doc thứ i xuống doc thứ i+1 (i tính từ 0)."""
    n = len(scores)
    if n < 2:
        return [float("inf")]
    if mode == "abs":
        denom = 1.0
    elif mode == "rel":
        denom = max(scores[0], EPS)
    elif mode == "spread":
        denom = max(scores[0] - scores[-1], EPS)
    elif mode == "softmax":
        m = max(scores)
        ex = [math.exp((s - m) / tau) for s in scores]
        z = sum(ex)
        p = [e / z for e in ex]
        return [p[i] - p[i + 1] for i in range(n - 1)] + [float("inf")]
    else:
        raise ValueError(f"margin mode lạ: {mode}")
    return [(scores[i] - scores[i + 1]) / denom for i in range(n - 1)] + [float("inf")]


def predict(
    ranking: dict, t: float, rule: str, mode: str, tau: float, max_docs: int
) -> dict[str, list[str]]:
    pred = {}
    for qid, items in ranking.items():
        docs = [d for d, _ in items[:max_docs]]
        scores = [s for _, s in items[:max_docs]]
        if not docs:
            pred[qid] = []
            continue
        mg = margins(scores, mode, tau)
        if rule == "top1":
            pred[qid] = docs[:1] if mg[0] > t else docs[:max_docs]
        elif rule == "cascade":
            n = max_docs
            for i, g in enumerate(mg[:max_docs]):
                if g > t:
                    n = i + 1
                    break
            pred[qid] = docs[:n]
        else:
            raise ValueError(f"rule lạ: {rule}")
    return pred


def evaluate(pred: dict, questions: dict, max_docs: int) -> tuple[float, float, float]:
    """Trả (Recall, Precision, số doc trung bình). Áp đúng luật BTC: >5 doc ⇒ câu đó 0 điểm."""
    R = P = ndocs = 0.0
    n = len(questions)
    for qid, v in questions.items():
        gold = {str(a) for a in v["answer"]}
        p = pred.get(str(qid), [])
        ndocs += len(p)
        if len(p) > 5:           # ràng buộc cứng của BTC, không phải cảnh báo
            continue
        if not p:
            continue
        hit = len(gold & set(p))
        R += hit / len(gold)
        P += hit / len(p)
    return R / n, P / n, ndocs / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True, help="file ranking [[doc_id, score, ...], ...]")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--rule", default="cascade", choices=["top1", "cascade"])
    ap.add_argument("--margin", default="spread", choices=["spread", "rel", "abs", "softmax"])
    ap.add_argument("--tau", type=float, default=1.0, help="chỉ dùng cho margin=softmax")
    ap.add_argument("--max-docs", type=int, default=5)
    ap.add_argument("--grid", default=None, help="danh sách t phẩy, mặc định tự sinh")
    ap.add_argument(
        "--budget",
        type=float,
        default=0.003,
        help="mức Recall được phép hy sinh (0.003 = 0,3%% theo plan.md muc 3)",
    )
    ap.add_argument("--pick-t", type=float, default=None, help="bỏ qua quét, dùng thẳng t này")
    ap.add_argument("--dump-pred", default=None, help="xuất dự đoán tại t đã chọn")
    ap.add_argument(
        "--score-from",
        default=None,
        help="THỨ TỰ lấy từ --ranking, nhưng ĐIỂM lấy từ file này. Dùng khi hệ thống "
        "xếp hạng tốt nhất (vd RRF) có điểm quá thô để đo độ tự tin, trong khi một "
        "model khác (vd reranker) cho điểm tách tốt hơn. Doc không có trong file "
        "điểm sẽ nhận -inf, tức bị coi là kém tự tin nhất.",
    )
    a = ap.parse_args()

    if "holdout" in a.questions:
        print("⚠️  ĐANG QUÉT TRÊN HOLDOUT. Hiệu chỉnh phải làm trên dev; holdout chỉ")
        print("    được xác nhận đúng một lần với t đã chốt (--pick-t). Xem §1.1.\n")

    q = json.load(open(a.questions, encoding="utf-8"))
    ranking = load_ranking(a.ranking)

    # Chỉ giữ câu được đánh giá — file ranking có thể rộng hơn tập câu hỏi.
    ranking = {str(k): v for k, v in ranking.items() if str(k) in {str(x) for x in q}}

    if a.score_from:
        src = load_ranking(a.score_from)
        n_miss = n_used = 0
        for qid, items in ranking.items():
            table = dict(src.get(qid, []))
            new = []
            for i, (doc, _) in enumerate(items):
                hit = doc in table
                if i < a.max_docs:                 # chỉ đếm phần THẬT SỰ ảnh hưởng margin
                    n_used += 1
                    n_miss += 0 if hit else 1
                new.append((doc, table[doc] if hit else float("-inf")))
            ranking[qid] = new
        pct = 100 * n_miss / max(n_used, 1)
        print(f"Điểm lấy từ: {a.score_from}")
        print(f"  Thiếu điểm trong top-{a.max_docs}: {n_miss}/{n_used} ({pct:.1f}%)")
        if pct > 5:
            print("⚠️  Doc thiếu điểm nhận −inf, làm hỏng mẫu số của margin=spread.")
            print("    Rerank phải phủ ÍT NHẤT top-{} của file thứ tự.".format(a.max_docs))

    missing = [k for k in q if str(k) not in ranking]
    if missing:
        raise SystemExit(f"Ranking thiếu {len(missing)} câu, vd {missing[:3]}")

    base_pred = {k: [d for d, _ in ranking[str(k)][:a.max_docs]] for k in q}
    R0, P0, N0 = evaluate(base_pred, q, a.max_docs)
    print(f"Ranking : {a.ranking}")
    print(f"Tập đo  : {a.questions}  (n={len(q)})")
    print(f"Quy tắc : {a.rule} · margin={a.margin}" + (f" τ={a.tau}" if a.margin == "softmax" else ""))
    print(f"\nĐường cơ sở (luôn nộp {a.max_docs}): Recall={R0:.4f}  Precision={P0:.4f}\n")

    if a.pick_t is not None:
        grid = [a.pick_t]
    elif a.grid:
        grid = [float(x) for x in a.grid.split(",")]
    elif a.margin in ("spread", "softmax"):
        # Với spread, các khoảng cách cộng lại bằng 1 nên t có nghĩa chạy tới ~0,95.
        # Lưới cũ dừng ở 0,50 làm hụt hẳn vùng tối ưu của điểm reranker (quanh 0,75).
        grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65,
                0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
    else:
        grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print(f"{'t':>6} {'Recall':>8} {'Precision':>10} {'ΔRecall':>9} {'ΔP %':>7} {'doc/câu':>8}")
    print("─" * 54)
    rows = []
    for t in grid:
        pred = predict(ranking, t, a.rule, a.margin, a.tau, a.max_docs)
        R, P, N = evaluate(pred, q, a.max_docs)
        rows.append((t, R, P, N))
        print(f"{t:>6.2f} {R:>8.4f} {P:>10.4f} {R-R0:>+9.4f} {100*(P/P0-1):>+6.1f}% {N:>8.2f}")

    ok = [r for r in rows if r[1] >= R0 - a.budget]
    print()
    if ok:
        best = max(ok, key=lambda r: r[2])
        print(f"→ Trong ngân sách ΔRecall ≥ −{a.budget:.4f}, t tốt nhất = {best[0]:.2f}")
        print(f"  Recall {R0:.4f} → {best[1]:.4f}   Precision {P0:.4f} → {best[2]:.4f} "
              f"({100*(best[2]/P0-1):+.1f}%)")
        print(f"  Xác nhận bằng kiểm định cặp trước khi tin — xem scripts/p4_paired_test.py")
    else:
        print(f"→ Không t nào giữ được Recall trong ngân sách {a.budget:.4f}.")
        print("  Nghĩa là tín hiệu margin của nguồn điểm này quá yếu để cắt bớt doc.")

    if a.dump_pred:
        t = a.pick_t if a.pick_t is not None else (best[0] if ok else grid[-1])
        pred = predict(ranking, t, a.rule, a.margin, a.tau, a.max_docs)
        out = Path(a.dump_pred)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({k: {"answer": v} for k, v in pred.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        over = sum(1 for v in pred.values() if len(v) > 5)
        print(f"\n✅ {out}  (t={t:.2f}, {over} câu vượt 5 doc)")

        # p4_paired_test đọc định dạng [[doc, score, chunk]], không đọc định dạng nộp bài.
        # Xuất thêm bản song song để kiểm định cặp chạy được ngay, khỏi viết script rời.
        sib = out.with_suffix(".ranking.json")
        sib.write_text(
            json.dumps(
                {k: [[d, 0.0, ""] for d in v] for k, v in pred.items()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"✅ {sib}  (để chạy scripts/p4_paired_test.py)")


if __name__ == "__main__":
    main()
