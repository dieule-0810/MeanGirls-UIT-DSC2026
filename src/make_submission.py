"""
Sinh submission.zip. Đây là chốt chặn cuối cùng trước CodaLab.

⚠️ CHỦ SỞ HỮU: P1. Không sửa trực tiếp — báo P1.

Triết lý: FAIL LOUD. Mọi thứ mã chấm của BTC xử lý im lặng (hoặc crash) đều bị chặn ở đây,
tại chỗ, với thông báo nói rõ phải sửa gì. Xem docs/scoring_behaviour.md.

Dùng:
    python -m src.make_submission \
        --preds outputs/v0.1_bm25/predictions.json \
        --questions data/public-official.json \
        --out outputs/v0.1_bm25/submission.zip \
        --corpus data/corpus_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

MAX_ANSWERS = 5
SUBMISSION_FILENAME = "submission.json"  # tên do BTC quy định, không đổi


def dedupe_keep_order(items: list) -> list[str]:
    """Khử trùng lặp, GIỮ NGUYÊN thứ tự xếp hạng. Ép str tại đây."""
    seen, out = set(), []
    for d in items:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_submission(
    preds: dict[str, list],
    expected_qids: set[str],
    corpus_ids: set[str] | None = None,
    fill_missing: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    preds         : {qid: [doc_id, ...]} — đã xếp hạng, chưa cần dedupe/cắt
    expected_qids : tập qid của file câu hỏi. Phải khớp CHÍNH XÁC.
    """
    errors: list[str] = []
    warnings: list[str] = []

    preds = {str(k): v for k, v in preds.items()}

    missing = expected_qids - set(preds)
    extra = set(preds) - expected_qids
    if missing:
        if fill_missing:
            warnings.append(
                f"⚠️  {len(missing)} câu thiếu dự đoán, đã điền list rỗng. "
                f"Các câu này chắc chắn 0 điểm — chỉ dùng khi debug, KHÔNG dùng để nộp thật."
            )
            for q in missing:
                preds[q] = []
        else:
            errors.append(
                f"THIẾU {len(missing)} câu hỏi (vd {sorted(missing)[:3]}). "
                f"BTC sẽ raise Exception → submission FAILED, mất một lượt nộp. "
                f"Dùng --fill-missing nếu đang debug."
            )
    if extra:
        errors.append(
            f"THỪA {len(extra)} qid không có trong file câu hỏi (vd {sorted(extra)[:3]}). "
            f"BTC sẽ raise TypeError → submission FAILED."
        )
    if errors:
        raise SystemExit("❌ KHÔNG THỂ TẠO SUBMISSION:\n  - " + "\n  - ".join(errors))

    out: dict[str, dict[str, list[str]]] = {}
    n_truncated = n_deduped = n_empty = 0
    unknown_ids: set[str] = set()

    for qid in expected_qids:
        docs = preds[qid]
        clean = dedupe_keep_order(docs)
        if len(clean) < len(docs):
            n_deduped += 1
        if len(clean) > MAX_ANSWERS:
            n_truncated += 1
            clean = clean[:MAX_ANSWERS]
        if not clean:
            n_empty += 1
        if corpus_ids is not None:
            unknown_ids |= {d for d in clean if d not in corpus_ids}
        out[qid] = {"answer": clean}

    # ── Assert cuối, sau khi đã dựng xong. Không tin bước nào ở trên. ──
    assert len(out) == len(expected_qids), "Số câu hỏi không khớp sau khi dựng"
    for qid, v in out.items():
        assert isinstance(qid, str), f"qid {qid!r} không phải str"
        a = v["answer"]
        assert isinstance(a, list), f"{qid}: answer không phải list"
        assert len(a) <= MAX_ANSWERS, f"{qid}: {len(a)} doc > {MAX_ANSWERS}"
        assert len(a) == len(set(a)), f"{qid}: còn trùng lặp"
        for d in a:
            assert isinstance(d, str), (
                f"{qid}: doc_id {d!r} kiểu {type(d).__name__}, phải là str. "
                f"BTC sẽ cho 0 điểm IM LẶNG."
            )

    if n_deduped:
        warnings.append(f"⚠️  {n_deduped} câu có doc_id trùng, đã khử.")
    if n_truncated:
        warnings.append(f"⚠️  {n_truncated} câu bị cắt còn {MAX_ANSWERS} doc.")
    if n_empty:
        warnings.append(f"🔴 {n_empty} câu trả về RỖNG → chắc chắn 0 điểm cho các câu đó.")
    if unknown_ids:
        warnings.append(
            f"🔴 {len(unknown_ids)} doc_id không tồn tại trong corpus "
            f"(vd {sorted(unknown_ids)[:3]}) → nhiều khả năng có bug."
        )
    for w in warnings:
        print(w)

    return out


def write_zip(submission: dict, out_path: str | Path) -> Path:
    """Ghi submission.json ở GỐC zip. Không thư mục con."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(submission, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SUBMISSION_FILENAME, payload)

    # Đọc ngược lại từ zip để xác minh, không tin biến trong RAM
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        assert names == [SUBMISSION_FILENAME], (
            f"Zip phải chứa DUY NHẤT {SUBMISSION_FILENAME}, thực tế: {names}"
        )
        back = json.loads(zf.read(SUBMISSION_FILENAME).decode("utf-8"))
    assert back == submission, "Nội dung zip không khớp sau khi ghi"
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo submission.zip an toàn cho CodaLab.")
    ap.add_argument("--preds", required=True, help="{qid: [doc_id]} đã xếp hạng")
    ap.add_argument("--questions", required=True, help="public-official.json / private-official.json")
    ap.add_argument("--out", required=True, help="đường dẫn submission.zip")
    ap.add_argument("--corpus", default=None, help="corpus_clean.jsonl để kiểm doc_id lạ")
    ap.add_argument("--fill-missing", action="store_true", help="CHỈ dùng khi debug")
    args = ap.parse_args()

    preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    if preds and isinstance(next(iter(preds.values())), dict):
        preds = {k: v["answer"] for k, v in preds.items()}

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    expected = {str(k) for k in questions}

    corpus_ids = None
    if args.corpus:
        corpus_ids = set()

        corpus_path = Path(args.corpus)

        with corpus_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue

                try:
                    corpus_ids.add(json.loads(line)["doc_id"])
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"JSONL lỗi tại {corpus_path}, dòng {line_no}: {e}"
                    ) from e

    sub = build_submission(preds, expected, corpus_ids, args.fill_missing)
    path = write_zip(sub, args.out)

    sizes = [len(v["answer"]) for v in sub.values()]
    print(f"\n✅ {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print(f"   {len(sub)} câu hỏi, trung bình {sum(sizes)/len(sizes):.2f} doc/câu")
    print(f"   Phân bố: {({i: sizes.count(i) for i in sorted(set(sizes))})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
