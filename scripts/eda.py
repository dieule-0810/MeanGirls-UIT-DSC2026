"""
scripts/eda.py — Khám phá dữ liệu trước khi build (xem plan.md mục 0.5).

Chạy trên corpus + train.json GỐC, trước khi chunker/parser bị chỉnh lần cuối.
Không phụ thuộc corpus_clean.jsonl / chunks.jsonl vì mục đích là phát hiện vấn đề
TRƯỚC bước tiền xử lý, không phải kiểm tra lại sau đó.

Cách chạy:
    python scripts/eda.py \
        --corpus-dir data/selected-contexts \
        --train data/train.json \
        --out docs/eda_notes.md

Output: in tóm tắt ra stdout + ghi báo cáo markdown vào --out.

TODO cho P2: các hàm dưới đây là khung — điền phần tokenize tiếng Việt
(underthesea/pyvi) nếu muốn số liệu "từ" chính xác hơn whitespace-split.
Whitespace-split hiện tại đủ để ra thứ tự độ lớn (order of magnitude),
đừng chặn tiến độ chỉ vì thiếu word-segmenter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

DIEU_PATTERN = re.compile(r"\bĐiều\s+\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: Path) -> list[dict]:
    """Đọc toàn bộ context_*.json. Không ép str(id) ở đây — mục đích EDA là
    NHÌN THẤY dữ liệu thô, kể cả bẫy kiểu dữ liệu, không phải sửa nó."""
    docs = []
    files = sorted(corpus_dir.glob("context_*.json"))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        raw["_source_file"] = fp.name
        docs.append(raw)
    return docs


def load_train(train_path: Path) -> dict[str, dict]:
    """train.json / public-official.json thật là dict {qid: {"question": ..., "answer": [...]}}
    — ĐÃ kiểm tra trực tiếp trên file thật, không phải list như nhiều format QA khác."""
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"{train_path} không phải dict như kỳ vọng — cấu trúc train.json đã đổi? Kiểm tra lại."
    )
    return data


# ---------------------------------------------------------------------------
# 1. Phân bố độ dài văn bản
# ---------------------------------------------------------------------------

def doc_length_stats(docs: list[dict]) -> dict:
    lengths_words = []
    lengths_chars = []
    for d in docs:
        text = d.get("passage") or ""
        lengths_chars.append(len(text))
        lengths_words.append(len(text.split()))

    def pct(vals, p):
        if not vals:
            return 0
        vals_sorted = sorted(vals)
        idx = min(len(vals_sorted) - 1, int(len(vals_sorted) * p))
        return vals_sorted[idx]

    return {
        "n_docs": len(docs),
        "words_mean": round(statistics.mean(lengths_words), 1) if lengths_words else 0,
        "words_median": statistics.median(lengths_words) if lengths_words else 0,
        "words_p90": pct(lengths_words, 0.90),
        "words_p99": pct(lengths_words, 0.99),
        "words_max": max(lengths_words) if lengths_words else 0,
        "chars_mean": round(statistics.mean(lengths_chars), 1) if lengths_chars else 0,
    }


# ---------------------------------------------------------------------------
# 2. Tỉ lệ có cấu trúc "Điều N" vs cần fallback
# ---------------------------------------------------------------------------

def dieu_structure_stats(docs: list[dict]) -> dict:
    n_with_dieu = 0
    dieu_counts = []
    for d in docs:
        text = d.get("passage") or ""
        matches = DIEU_PATTERN.findall(text)
        if matches:
            n_with_dieu += 1
        dieu_counts.append(len(matches))

    n = len(docs) or 1
    return {
        "pct_with_dieu": round(100 * n_with_dieu / n, 1),
        "pct_needs_fallback": round(100 * (n - n_with_dieu) / n, 1),
        "avg_dieu_per_doc_when_present": (
            round(sum(c for c in dieu_counts if c > 0) / max(n_with_dieu, 1), 1)
        ),
    }


# ---------------------------------------------------------------------------
# 3. Trường thiếu (name / link)
# ---------------------------------------------------------------------------

def missing_field_stats(docs: list[dict]) -> dict:
    n = len(docs) or 1
    missing_name = sum(1 for d in docs if not d.get("name"))
    missing_link = sum(1 for d in docs if not d.get("link"))
    missing_text = sum(1 for d in docs if not d.get("passage"))
    return {
        "pct_missing_name": round(100 * missing_name / n, 1),
        "pct_missing_link": round(100 * missing_link / n, 1),
        "pct_missing_text": round(100 * missing_text / n, 1),
        "missing_name_files": [
            d["_source_file"] for d in docs if not d.get("name")
        ][:20],  # cắt bớt nếu dài, chỉ để soi nhanh
    }


# ---------------------------------------------------------------------------
# 4. Phân bố số đáp án đúng / câu hỏi (toàn bộ train.json)
# ---------------------------------------------------------------------------

def answer_count_stats(train: dict[str, dict]) -> dict:
    counts = []
    empty_questions = 0
    for qid, item in train.items():
        answers = item.get("answer") or []
        counts.append(len(answers))
        if not (item.get("question") or "").strip():
            empty_questions += 1

    n = len(counts) or 1
    dist = Counter(counts)
    return {
        "n_questions": len(train),
        "pct_exactly_1_answer": round(100 * dist.get(1, 0) / n, 1),
        "distribution": dict(sorted(dist.items())),
        "n_empty_question_text": empty_questions,
    }


# ---------------------------------------------------------------------------
# 5. Trùng / gần trùng văn bản (kiểm tra thô bằng hash sau khi chuẩn hoá)
# ---------------------------------------------------------------------------

def near_duplicate_stats(docs: list[dict]) -> dict:
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    hash_to_ids = {}
    for d in docs:
        h = hashlib.sha256(normalize(d.get("passage", "")).encode("utf-8")).hexdigest()
        hash_to_ids.setdefault(h, []).append(d.get("id", d["_source_file"]))

    exact_dup_groups = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    return {
        "n_exact_duplicate_groups": len(exact_dup_groups),
        "example_groups": list(exact_dup_groups.values())[:5],
        "note": (
            "Đây chỉ là trùng CHÍNH XÁC sau normalize whitespace. "
            "Gần trùng (near-duplicate thật, khác vài từ) cần MinHash/SimHash "
            "nếu số exact-dup thấp mà vẫn nghi ngờ — chưa làm ở bản khung này."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Độ dài câu hỏi / câu hỏi lỗi
# ---------------------------------------------------------------------------

def question_length_stats(train: dict[str, dict]) -> dict:
    items = list(train.values())
    lengths = [len((item.get("question") or "").split()) for item in items]
    lengths = [l for l in lengths if l > 0]
    suspicious = [
        item.get("question", "")[:80]
        for item in items
        if len((item.get("question") or "").split()) <= 2
    ][:10]
    return {
        "words_mean": round(statistics.mean(lengths), 1) if lengths else 0,
        "words_min": min(lengths) if lengths else 0,
        "words_max": max(lengths) if lengths else 0,
        "n_suspiciously_short": len(suspicious),
        "examples_suspiciously_short": suspicious,
    }


# ---------------------------------------------------------------------------
# 7. doc_id trong train.json có phủ hết corpus không / có id mồ côi không
# ---------------------------------------------------------------------------

def doc_id_coverage_stats(docs: list[dict], train: dict[str, dict]) -> dict:
    corpus_ids = {str(d.get("id")) for d in docs}
    train_ids = set()
    for item in train.values():
        answers = item.get("answer") or []
        train_ids.update(str(a) for a in answers)

    orphan_ids = train_ids - corpus_ids  # xuất hiện trong train nhưng KHÔNG có trong corpus
    unused_ids = corpus_ids - train_ids  # có trong corpus nhưng chưa từng là đáp án train

    return {
        "n_corpus_ids": len(corpus_ids),
        "n_train_referenced_ids": len(train_ids),
        "n_orphan_ids_in_train": len(orphan_ids),  # phải bằng 0, nếu không → BUG NGHIÊM TRỌNG
        "orphan_examples": sorted(orphan_ids)[:10],
        "n_corpus_ids_never_answer_in_train": len(unused_ids),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_markdown(results: dict) -> str:
    lines = ["# EDA Notes — chạy trước khi chunker/parser bị chỉnh lần cuối", ""]
    lines.append("> Sinh bởi `scripts/eda.py`. Điền thủ công phần nhận xét sau mỗi mục.")
    lines.append("")
    for section, data in results.items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("**Nhận xét:** _(điền)_")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/selected-contexts"))
    ap.add_argument("--train", type=Path, default=Path("data/train.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/eda_notes.md"))
    args = ap.parse_args()

    print(f"Đang đọc corpus từ {args.corpus_dir} ...")
    docs = load_corpus(args.corpus_dir)
    print(f"  → {len(docs)} văn bản")

    print(f"Đang đọc {args.train} ...")
    train = load_train(args.train)
    print(f"  → {len(train)} câu hỏi")

    results = {
        "1. Phân bố độ dài văn bản": doc_length_stats(docs),
        "2. Cấu trúc Điều N vs fallback": dieu_structure_stats(docs),
        "3. Trường thiếu (name/link/text)": missing_field_stats(docs),
        "4. Phân bố số đáp án / câu hỏi": answer_count_stats(train),
        "5. Trùng / gần trùng văn bản": near_duplicate_stats(docs),
        "6. Độ dài câu hỏi": question_length_stats(train),
        "7. Độ phủ doc_id (train vs corpus)": doc_id_coverage_stats(docs, train),
    }

    print("\n=== TÓM TẮT ===")
    for section, data in results.items():
        print(f"\n{section}")
        for k, v in data.items():
            if isinstance(v, list):
                continue  # danh sách ví dụ chỉ in trong file markdown, không spam stdout
            print(f"  {k}: {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(results), encoding="utf-8")
    print(f"\nĐã ghi báo cáo: {args.out}")

    n_orphan = results["7. Độ phủ doc_id (train vs corpus)"]["n_orphan_ids_in_train"]
    if n_orphan > 0:
        print(
            f"\n⚠️  CẢNH BÁO: {n_orphan} doc_id trong train.json không tồn tại trong corpus. "
            "Kiểm tra ngay — có thể là bẫy str/int (xem INTERFACES.md mục 0) hoặc thiếu file corpus."
        )


if __name__ == "__main__":
    main()
