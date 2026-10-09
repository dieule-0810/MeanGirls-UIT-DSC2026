"""Sinh `submission.zip` — chốt chặn cuối cùng trước CodaLab.

⚠️ CHỦ SỞ HỮU: P1 (file KHOÁ). Triết lý FAIL LOUD: mọi thứ mã chấm BTC xử lý im lặng (hoặc
crash) đều bị chặn ở đây, tại chỗ, với thông báo nói rõ phải sửa gì (docs/scoring_behaviour.md).

Typical usage example:

    python -m src.make_submission \\
        --preds outputs/v0.8_hybrid_rrf/predictions.json \\
        --questions data/private-official.json \\
        --out outputs/v0.8_hybrid_rrf/submission.zip \\
        --corpus data/corpus_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.common.io import load_corpus_ids

MAX_ANSWERS = 5
SUBMISSION_FILENAME = "submission.json"  # tên do BTC quy định, không đổi


@dataclass
class _CleanStats:
    """Đếm những gì đã phải sửa khi làm sạch dự đoán — để in cảnh báo."""

    n_deduped: int = 0
    n_truncated: int = 0
    n_empty: int = 0
    unknown_ids: set[str] = field(default_factory=set)

    def warnings(self) -> list[str]:
        """Các dòng cảnh báo cần in."""
        out = []
        if self.n_deduped:
            out.append(f"⚠️  {self.n_deduped} câu có doc_id trùng, đã khử.")
        if self.n_truncated:
            out.append(f"⚠️  {self.n_truncated} câu bị cắt còn {MAX_ANSWERS} doc.")
        if self.n_empty:
            out.append(f"🔴 {self.n_empty} câu trả về RỖNG → chắc chắn 0 điểm cho các câu đó.")
        if self.unknown_ids:
            out.append(
                f"🔴 {len(self.unknown_ids)} doc_id không tồn tại trong corpus "
                f"(vd {sorted(self.unknown_ids)[:3]}) → nhiều khả năng có bug."
            )
        return out


def dedupe_keep_order(items: list) -> list[str]:
    """Khử trùng lặp, GIỮ NGUYÊN thứ tự xếp hạng, ép `str` tại đây.

    Mã chấm không khử trùng: mẫu số precision và giới hạn 5 đều đếm theo list.
    """
    seen, out = set(), []
    for d in items:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _check_qids(preds: dict[str, list], expected_qids: set[str], fill_missing: bool) -> list[str]:
    """Đối chiếu tập qid; điền list rỗng cho câu thiếu nếu `fill_missing` (sửa `preds` tại chỗ).

    Returns:
        Cảnh báo cần in.

    Raises:
        SystemExit: Thiếu qid (khi không `fill_missing`) hoặc thừa qid — ở BTC là crash.
    """
    errors, warnings = [], []
    missing = expected_qids - set(preds)
    extra = set(preds) - expected_qids
    if missing and fill_missing:
        warnings.append(
            f"⚠️  {len(missing)} câu thiếu dự đoán, đã điền list rỗng. "
            f"Các câu này chắc chắn 0 điểm — chỉ dùng khi debug, KHÔNG dùng để nộp thật."
        )
        for q in missing:
            preds[q] = []
    elif missing:
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
    return warnings


def _clean_answers(
    preds: dict[str, list], expected_qids: set[str], corpus_ids: set[str] | None
) -> tuple[dict[str, dict[str, list[str]]], _CleanStats]:
    """Dedupe giữ thứ tự, cắt 5, bọc `{"answer": [...]}` cho từng câu."""
    out: dict[str, dict[str, list[str]]] = {}
    stats = _CleanStats()
    for qid in expected_qids:
        docs = preds[qid]
        clean = dedupe_keep_order(docs)
        if len(clean) < len(docs):
            stats.n_deduped += 1
        if len(clean) > MAX_ANSWERS:
            stats.n_truncated += 1
            clean = clean[:MAX_ANSWERS]
        if not clean:
            stats.n_empty += 1
        if corpus_ids is not None:
            stats.unknown_ids |= {d for d in clean if d not in corpus_ids}
        out[qid] = {"answer": clean}
    return out, stats


def _assert_submission(out: dict, expected_qids: set[str]) -> None:
    """Assert cuối, sau khi đã dựng xong — không tin bước nào ở trên."""
    assert len(out) == len(expected_qids), "Số câu hỏi không khớp sau khi dựng"
    for qid, v in out.items():
        assert isinstance(qid, str), f"qid {qid!r} không phải str"
        a = v["answer"]
        assert isinstance(a, list), f"{qid}: answer không phải list"
        assert len(a) <= MAX_ANSWERS, f"{qid}: {len(a)} doc > {MAX_ANSWERS}"
        assert len(a) == len(set(a)), f"{qid}: còn trùng lặp"
        for d in a:
            assert isinstance(d, str), (
                f"{qid}: doc_id {d!r} kiểu {type(d).__name__}, phải là str. BTC sẽ cho 0 điểm IM LẶNG."
            )


def build_submission(
    preds: dict[str, list],
    expected_qids: set[str],
    corpus_ids: set[str] | None = None,
    fill_missing: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Dựng nội dung `submission.json` đã kiểm mọi bẫy của mã chấm.

    Args:
        preds: `{qid: [doc_id, ...]}` đã xếp hạng, chưa cần dedupe/cắt.
        expected_qids: Tập qid của file câu hỏi. Phải khớp CHÍNH XÁC.
        corpus_ids: Tập doc_id của corpus để cảnh báo id lạ; None = bỏ qua.
        fill_missing: Điền list rỗng cho câu thiếu — CHỈ dùng khi debug.

    Returns:
        `{qid: {"answer": [doc_id, ...]}}`, mỗi câu tối đa 5 doc, không trùng, toàn str.

    Raises:
        SystemExit: Tập qid không khớp.
        AssertionError: Kết quả vi phạm bất biến (lưới an toàn cuối).
    """
    preds = {str(k): v for k, v in preds.items()}
    warnings = _check_qids(preds, expected_qids, fill_missing)
    out, stats = _clean_answers(preds, expected_qids, corpus_ids)
    _assert_submission(out, expected_qids)
    for w in warnings + stats.warnings():
        print(w)
    return out


def write_zip(submission: dict, out_path: str | Path) -> Path:
    """Ghi `submission.json` ở GỐC zip (không thư mục con) rồi đọc ngược lại để xác minh.

    Args:
        submission: Kết quả `build_submission`.
        out_path: Đường dẫn file zip.

    Returns:
        Đường dẫn đã ghi.

    Raises:
        AssertionError: Zip chứa thứ khác ngoài `submission.json`, hoặc nội dung đọc lại lệch.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(submission, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SUBMISSION_FILENAME, payload)

    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        assert names == [SUBMISSION_FILENAME], f"Zip phải chứa DUY NHẤT {SUBMISSION_FILENAME}, thực tế: {names}"
        back = json.loads(zf.read(SUBMISSION_FILENAME).decode("utf-8"))
    assert back == submission, "Nội dung zip không khớp sau khi ghi"
    return out_path


def main() -> int:
    """CLI: `predictions.json` → `submission.zip` an toàn."""
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
    expected = {str(k) for k in json.loads(Path(args.questions).read_text(encoding="utf-8"))}
    corpus_ids = load_corpus_ids(args.corpus) if args.corpus else None

    sub = build_submission(preds, expected, corpus_ids, args.fill_missing)
    path = write_zip(sub, args.out)
    sizes = [len(v["answer"]) for v in sub.values()]
    print(f"\n✅ {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print(f"   {len(sub)} câu hỏi, trung bình {sum(sizes)/len(sizes):.2f} doc/câu")
    print(f"   Phân bố: {({i: sizes.count(i) for i in sorted(set(sizes))})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
