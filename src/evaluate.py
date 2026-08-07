"""
Bản sao chính xác logic chấm điểm của BTC (scoring.py).

⚠️ CHỦ SỞ HỮU: P1. Không sửa trực tiếp — báo P1.
⚠️ File này CỐ TÌNH giữ nguyên mọi hành vi biên của BTC, kể cả những chỗ trông như bug.
   Xem docs/scoring_behaviour.md trước khi thắc mắc.

Dùng:
    from src.evaluate import eval_official
    eval_official({"q1": ["100"]}, {"q1": ["100"]})
    # → {'recall': 1.0, 'precision': 1.0}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_ANSWERS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Lõi: khớp từng dòng với scoring.py của BTC
# ─────────────────────────────────────────────────────────────────────────────
def eval_official(
    preds: dict[str, list[str]],
    truth: dict[str, list[str]],
) -> dict[str, float]:
    """
    preds : {qid: [doc_id, ...]}   — định dạng NỘI BỘ (phẳng), không phải {"answer": ...}
    truth : {qid: [doc_id, ...]}

    Tái hiện chính xác:
      recall    = mean_k  |truth[k] ∩ preds[k]| / |truth[k]|   nếu 0 < len(preds[k]) <= 5, ngược lại 0
      precision = mean_k  |truth[k] ∩ preds[k]| / len(preds[k]) nếu 0 < len(preds[k]) <= 5, ngược lại 0

    Lưu ý các hành vi được giữ nguyên có chủ đích:
      - mẫu số precision là len(LIST), không phải len(set) → trùng lặp bị phạt
      - phép giao dùng set() → so sánh theo KIỂU, int 100 != str "100"
      - thiếu/thừa qid → raise, giống BTC (ở đó là crash container chấm điểm)
    """
    if len(preds) != len(truth):
        raise ValueError(
            f"Samples in predict not match with reference: "
            f"preds={len(preds)} vs truth={len(truth)}. "
            f"BTC sẽ raise Exception → submission FAILED, mất một lượt nộp."
        )

    missing = set(truth) - set(preds)
    extra = set(preds) - set(truth)
    if missing or extra:
        raise ValueError(
            f"Tập qid không khớp. Thiếu {len(missing)} (vd {list(missing)[:3]}), "
            f"thừa {len(extra)} (vd {list(extra)[:3]}). "
            f"BTC sẽ raise TypeError → submission FAILED."
        )

    recalls, precisions = [], []
    for k in truth:
        p = preds[k]
        n_pred = len(p)  # ← LIST, không phải set. Có chủ đích.
        if 0 < n_pred <= MAX_ANSWERS:
            hit = len(set(truth[k]) & set(p))
            recalls.append(hit / len(truth[k]))
            precisions.append(hit / n_pred)
        else:
            recalls.append(0.0)
            precisions.append(0.0)

    n = len(recalls)
    return {"recall": sum(recalls) / n, "precision": sum(precisions) / n}


# ─────────────────────────────────────────────────────────────────────────────
# Chẩn đoán: những gì BTC KHÔNG nói cho bạn biết
# ─────────────────────────────────────────────────────────────────────────────
def diagnose(
    preds: dict[str, list[str]],
    truth: dict[str, list[str]],
    corpus_ids: set[str] | None = None,
) -> dict:
    """
    Chạy CÙNG với eval_official mỗi lần đánh giá.
    Bắt các chế độ hỏng im lặng mà leaderboard chỉ hiện ra bằng một con số 0.
    """
    issues: list[str] = []

    n_int = sum(1 for v in preds.values() for d in v if not isinstance(d, str))
    if n_int:
        issues.append(
            f"🔴 {n_int} doc_id KHÔNG phải str → BTC cho 0 điểm im lặng. Ép str() ngay."
        )

    n_dup = sum(len(v) - len(set(v)) for v in preds.values())
    if n_dup:
        issues.append(
            f"🔴 {n_dup} doc_id trùng lặp → tính vào giới hạn 5 và làm tụt Precision."
        )

    n_over = sum(1 for v in preds.values() if len(v) > MAX_ANSWERS)
    if n_over:
        issues.append(f"🔴 {n_over} câu có > {MAX_ANSWERS} doc → mỗi câu bị 0 cả hai chỉ số.")

    n_empty = sum(1 for v in preds.values() if len(v) == 0)
    if n_empty:
        issues.append(f"🟡 {n_empty} câu trả về rỗng → Recall và Precision đều 0 cho các câu đó.")

    if corpus_ids is not None:
        unknown = {d for v in preds.values() for d in v if str(d) not in corpus_ids}
        if unknown:
            issues.append(
                f"🟡 {len(unknown)} doc_id không có trong corpus (vd {list(unknown)[:3]}) "
                f"→ không gây lỗi nhưng chắc chắn sai."
            )

    sizes = [len(v) for v in preds.values()]
    return {
        "issues": issues,
        "n_questions": len(preds),
        "avg_answers_per_q": sum(sizes) / len(sizes) if sizes else 0.0,
        "size_distribution": {i: sizes.count(i) for i in sorted(set(sizes))},
    }


def recall_at_k(
    ranked: dict[str, list[str]],
    truth: dict[str, list[str]],
    k: int,
) -> float:
    """
    Recall@k KHÔNG giới hạn 5 — dùng để P3 đo tầng 1 (KPI Recall@50/@100).
    Đây là TRẦN CỨNG của toàn hệ thống: doc không lọt top-k thì reranker không cứu được.
    """
    vals = [
        len(set(truth[q]) & set(ranked.get(q, [])[:k])) / len(truth[q])
        for q in truth
    ]
    return sum(vals) / len(vals)


def load_truth(path: str | Path) -> dict[str, list[str]]:
    """Đọc train.json ({qid: {question, answer}}) hoặc dạng phẳng ({qid: [ids]})."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for qid, v in raw.items():
        ans = v["answer"] if isinstance(v, dict) else v
        if ans is None:
            raise ValueError(f"qid {qid} có answer=null — đây là file test, không phải ground truth.")
        out[str(qid)] = [str(a) for a in ans]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Chấm điểm theo đúng công thức BTC.")
    ap.add_argument("--preds", required=True, help="{qid: [doc_id]} hoặc {qid: {answer: [...]}}")
    ap.add_argument("--truth", required=True, help="train.json hoặc holdout.json")
    ap.add_argument("--corpus", default=None, help="corpus_clean.jsonl để kiểm doc_id lạ")
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    preds = {
        str(k): [str(d) for d in (v["answer"] if isinstance(v, dict) else v)]
        for k, v in raw.items()
    }
    truth = load_truth(args.truth)

    corpus_ids = None
    if args.corpus:
        corpus_ids = {
            json.loads(line)["doc_id"]
            for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    # Chẩn đoán chạy trên dữ liệu THÔ (trước khi ép str) để bắt được lỗi kiểu
    raw_preds = {str(k): (v["answer"] if isinstance(v, dict) else v) for k, v in raw.items()}
    diag = diagnose(raw_preds, truth, corpus_ids)

    scores = eval_official(preds, truth)
    print(f"Recall    : {scores['recall']:.4f}   ← metric chính")
    print(f"Precision : {scores['precision']:.4f}   ← tie-break")
    print(f"\nSố câu hỏi: {diag['n_questions']}, trung bình {diag['avg_answers_per_q']:.2f} doc/câu")
    print(f"Phân bố kích thước: {diag['size_distribution']}")
    if diag["issues"]:
        print("\nCẢNH BÁO:")
        for it in diag["issues"]:
            print(f"  {it}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
