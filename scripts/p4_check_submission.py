#!/usr/bin/env python3
"""Kiểm submission.zip bằng CHÍNH mã chấm của BTC, trước khi nộp.

Hai chế độ:

  1. Cấu trúc (mọi tập).  Mở zip, kiểm đúng một file `submission.json`, đúng tập qid,
     không câu nào quá 5 doc, không doc_id lạ.

  2. Chấm điểm (tập có nhãn).  Gọi `vendor/btc_scoring/scoring.py::eval_retrieval`
     — đúng hàm BTC chạy — trên một submission.zip dựng từ dev.

Chế độ 2 là phép kiểm duy nhất phủ được TOÀN BỘ đường ống kể cả khâu đóng gói.
Mọi con số trước đó (`p4_fuse`, `p4_paired_test`) đo trên dict trong RAM; chúng
không biết gì về việc `make_submission` khử trùng lặp ra sao, cắt ở đâu, ép kiểu
thế nào. Nếu dev đi trọn đường ống mà ra đúng con số đã đo, thì đường ống đúng, và
bài nộp public chạy qua cùng đường ống đó cũng đúng.

    # dựng submission dev từ ranking dev đã có
    python -m scripts.p4_to_preds --ranking outputs/p4_calib/dev_rrf_syl_w06.json \\
        --questions data/dev.json --out outputs/v0.4_submit/dev_preds.json
    python -m src.make_submission --preds outputs/v0.4_submit/dev_preds.json \\
        --questions data/dev.json --corpus data/corpus_clean.jsonl \\
        --out outputs/v0.4_submit/dev_submission.zip

    # chấm bằng mã BTC — phải ra ĐÚNG 0.8733
    python -m scripts.p4_check_submission --zip outputs/v0.4_submit/dev_submission.zip \\
        --questions data/dev.json --gold data/dev.json

    # rồi kiểm cấu trúc bài nộp thật
    python -m scripts.p4_check_submission --zip outputs/v0.4_submit/submission.zip \\
        --questions data/public-official.json --corpus data/corpus_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--questions", required=True,
                    help="tập câu hỏi bài nộp nhắm tới, để đối chiếu qid")
    ap.add_argument("--gold", default=None,
                    help="tập có nhãn (dev.json / holdout.json) → chấm bằng mã BTC")
    ap.add_argument("--corpus", default=None,
                    help="corpus_clean.jsonl → bắt doc_id không tồn tại")
    a = ap.parse_args()

    fail: list[str] = []
    warn: list[str] = []

    # ── 1. Zip phải chứa ĐÚNG một submission.json ─────────────────────────
    with zipfile.ZipFile(a.zip) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        print(f"zip chứa: {names}")
        if names != ["submission.json"]:
            fail.append(
                f"zip phải chứa duy nhất 'submission.json' ở gốc, đang có {names}. "
                f"Thư mục lồng hay file thừa đều làm BTC đọc trượt.")
        sub = json.loads(z.read("submission.json").decode("utf-8"))

    # ── 2. Tập qid phải khớp CHÍNH XÁC ────────────────────────────────────
    q = json.load(open(a.questions, encoding="utf-8"))
    want, got = {str(k) for k in q}, {str(k) for k in sub}
    if want != got:
        fail.append(
            f"tập qid lệch: thiếu {len(want - got)}, thừa {len(got - want)}. "
            f"Mã BTC raise ngay → toàn bài Failed, mất một lượt nộp.")
    print(f"qid: {len(got)} câu, khớp tập câu hỏi: {want == got}")

    # ── 3. Ràng buộc 5 doc + kiểu dữ liệu ─────────────────────────────────
    from collections import Counter
    dist, n_dup, n_nonstr = Counter(), 0, 0
    for qid, v in sub.items():
        ans = v.get("answer") if isinstance(v, dict) else v
        if not isinstance(ans, list):
            fail.append(f"qid {qid}: 'answer' không phải list")
            continue
        dist[len(ans)] += 1
        if len(ans) != len({*map(str, ans)}):
            n_dup += 1
        if any(not isinstance(x, str) for x in ans):
            n_nonstr += 1
    print(f"phân bố số doc/câu: {dict(sorted(dist.items()))}")

    over = sum(c for k, c in dist.items() if k > 5)
    if over:
        fail.append(
            f"{over} câu có HƠN 5 doc. Theo mã chấm, mỗi câu như vậy bị gán "
            f"Recall=0 VÀ Precision=0.")
    if dist.get(0):
        warn.append(f"{dist[0]} câu rỗng → Recall=0 và Precision=0 cho các câu đó.")
    if n_dup:
        warn.append(f"{n_dup} câu có doc_id trùng lặp — `set()` của mã chấm nuốt bản "
                    f"sao nhưng MẪU SỐ Precision vẫn là len(list) → tự bóp điểm.")
    if n_nonstr:
        fail.append(f"{n_nonstr} câu có doc_id không phải str. `set(int) & set(str)` "
                    f"luôn rỗng → 0 điểm, im lặng.")

    # ── 4. doc_id có thật không ───────────────────────────────────────────
    if a.corpus:
        real = {str(json.loads(l)["doc_id"]) for l in open(a.corpus, encoding="utf-8") if l.strip()}
        bad = {str(d) for v in sub.values()
               for d in (v.get("answer") if isinstance(v, dict) else v)} - real
        if bad:
            fail.append(f"{len(bad)} doc_id không có trong corpus, vd {sorted(bad)[:5]}")
        else:
            print(f"doc_id: tất cả đều có trong corpus ({len(real)} văn bản)")

    # ── 5. Chấm bằng mã BTC ───────────────────────────────────────────────
    if a.gold:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "btc_scoring"))
        from scoring import eval_retrieval  # type: ignore

        g = json.load(open(a.gold, encoding="utf-8"))
        truth = {str(k): [str(x) for x in v["answer"]] for k, v in g.items()
                 if v.get("answer")}
        if len(truth) != len(g):
            fail.append(f"--gold có {len(g) - len(truth)} câu không nhãn, không chấm được")
        else:
            s = eval_retrieval(sub, truth)
            print(f"\n>>> MÃ CHẤM BTC: Recall = {s['recall']:.4f}  ·  "
                  f"Precision = {s['precision']:.4f}")
            print("    Đối chiếu với con số p4_fuse/p4_paired_test đã in. "
                  "Lệch một chữ số nào cũng là đường ống hỏng, không phải làm tròn.")

    print()
    for w in warn:
        print(f"⚠️  {w}")
    for f in fail:
        print(f"❌ {f}")
    if fail:
        print("\nKHÔNG NỘP cho tới khi hết ❌.")
        return 1
    print("✅ Cấu trúc hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
