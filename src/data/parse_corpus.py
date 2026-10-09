"""Gộp và làm sạch kho văn bản thô `context_*.json` → `corpus_clean.jsonl`. CHỦ SỞ HỮU: P2.

Đọc 8.532 file thô theo thứ tự tên file, loại 25 văn bản đã chốt (20 rỗng + 5 trùng-dư, danh
sách trong `docs/exclusion_decisions.json` do `scripts/eda.py` sinh), làm sạch rồi ghi 8.507
dòng JSONL đúng hợp đồng INTERFACES.md §1, in SHA-256 và chạy bộ thẩm định ở cuối.

Các bẫy dữ liệu đã xử lý:

* `id` trong file thô là int, nhãn BTC là str ⇒ ép `str` NGAY tại điểm đọc.
* 13,2% văn bản thiếu `name` ⇒ quy về `""` (không bịa nội dung).
* Văn bản rỗng nằm trong danh sách loại; rỗng MỚI phát sinh thì giữ `""`, không chế chữ.
* Rác crawler ("quý khách vui lòng đăng nhập", ...) bị chặt từ vị trí xuất hiện trở đi.
* Mọi khoảng trắng/xuống dòng (`\\r\\n\\n`) kéo phẳng thành một dấu cách; Unicode NFC.
* `link` đưa về chữ thường.
* Thứ tự dòng theo thứ tự tên file (`context_100` trước `context_2`) — cố định để SHA-256 tái lập.

Typical usage example:

    python scripts/eda.py                       # sinh docs/exclusion_decisions.json (nếu chưa có)
    python -m src.data.parse_corpus --corpus-dir data/selected-contexts --out data/corpus_clean.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

CRAWLER_JUNK_PATTERNS = [
    "quý khách vui lòng đăng nhập",
    "đăng nhập để xem",
    "rò rỉ mật khẩu",
    "vui lòng đăng nhập để",
    "đăng nhập để tiếp tục",
]
# Chặt toàn bộ nội dung từ vị trí dính rác trở đi.
CLEAN_JUNK_REGEX = re.compile(r"(" + "|".join(CRAWLER_JUNK_PATTERNS) + r").*$", re.IGNORECASE)
DEFAULT_DECISIONS = Path("docs/exclusion_decisions.json")
REQUIRED_KEYS = ("doc_id", "name", "link", "text")


# ─────────────────────────────────────────────────────────────────────────────
# Làm sạch một văn bản
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(raw_text: str) -> str:
    """NFC, kéo phẳng mọi khoảng trắng thành một dấu cách, chặt đuôi rác crawler.

    Args:
        raw_text: Nội dung thô.

    Returns:
        Văn bản sạch; rỗng nếu đầu vào rỗng.
    """
    if not raw_text:
        return ""
    cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw_text))
    return CLEAN_JUNK_REGEX.sub("", cleaned).strip()


def clean_record(doc: dict) -> dict:
    """Một bản ghi thô (đã có `id`) → bản ghi đúng INTERFACES.md §1.

    Args:
        doc: Nội dung một `context_*.json`.

    Returns:
        `{"doc_id": str, "name": str, "link": str, "text": str}`.
    """
    name = doc.get("name")
    passage = doc.get("passage")
    if passage is None:
        passage = doc.get("text") or ""
    return {
        "doc_id": str(doc["id"]),
        "name": "" if name is None or not str(name).strip() else str(name).strip(),
        "link": str(doc.get("link") or "").strip().lower(),
        "text": clean_text(str(passage)),
    }


def calculate_sha256(file_path: Path) -> str:
    """SHA-256 của file, đọc theo khối 64 KB để không nạp cả file vào RAM."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_decisions(path: Path) -> dict:
    """Đọc quyết định loại trừ (`docs_exclude_from_corpus`, `n_corpus_after_exclusion`, ...).

    Raises:
        FileNotFoundError: Chưa có file — chạy `python scripts/eda.py` trước.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. File này do `python scripts/eda.py` sinh (bị .gitignore chặn), "
            f"chạy nó trước khi parse."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Thẩm định corpus sạch
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusScan:
    """Kết quả quét `corpus_clean.jsonl`."""

    n_lines: int = 0
    keys_ok: bool = True
    ids_str: bool = True
    empty_text: int = 0
    empty_name: int = 0
    raw_newlines: int = 0
    upper_links: int = 0
    junk_leaks: list[str] = field(default_factory=list)
    suspicious_login: list[str] = field(default_factory=list)

    def passed(self, expected_lines: int) -> bool:
        """Đạt mọi điều kiện (số dòng, khoá, kiểu, rác)."""
        return (
            self.n_lines == expected_lines and self.keys_ok and self.ids_str
            and self.raw_newlines == 0 and self.upper_links == 0
            and self.empty_text == 0 and not self.junk_leaks
        )


def _scan_doc(scan: CorpusScan, idx: int, doc: dict) -> None:
    """Cập nhật `scan` với một bản ghi corpus."""
    for key in REQUIRED_KEYS:
        if key not in doc:
            print(f"  Dòng {idx + 1} (ID: {doc.get('doc_id')}) bị khuyết khóa: {key}")
            scan.keys_ok = False
    doc_id = doc.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        scan.ids_str = False
    scan.empty_name += doc.get("name") == ""
    passage = doc.get("text")
    scan.empty_text += passage == ""
    if "\r" in passage or "\n\n" in passage:
        scan.raw_newlines += 1
    if any(c.isupper() for c in doc.get("link", "") if c.isalpha()):
        scan.upper_links += 1
    lowered = unicodedata.normalize("NFC", passage).lower()
    if any(unicodedata.normalize("NFC", p).lower() in lowered for p in CRAWLER_JUNK_PATTERNS):
        scan.junk_leaks.append(doc_id)
    elif "đăng nhập" in lowered and len(lowered.split()) < 300:
        # Văn bản ngắn mà có "đăng nhập" rất dễ là file rác cụt đầu — soi tay.
        scan.suspicious_login.append(doc_id)


def scan_corpus(path: Path) -> CorpusScan:
    """Quét `corpus_clean.jsonl`, đếm mọi vi phạm hợp đồng và dấu vết rác."""
    scan = CorpusScan()
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            scan.n_lines += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Dòng {idx + 1} không phải JSON hợp lệ: {e}")
                scan.keys_ok = False
                continue
            _scan_doc(scan, idx, doc)
    return scan


def _print_corpus_scan(scan: CorpusScan, expected_lines: int) -> None:
    """In kết quả quét corpus."""
    ok = "đạt"
    print(f"  - Số dòng: {scan.n_lines} (kỳ vọng {expected_lines})")
    print(f"  - Khoá {', '.join(REQUIRED_KEYS)}: " + (ok if scan.keys_ok else "có bản ghi thiếu khoá"))
    print("  - doc_id là str: " + (ok if scan.ids_str else "có id kiểu số (int)!"))
    print("  - Không còn \\r / \\n\\n: " + (ok if not scan.raw_newlines else f"còn {scan.raw_newlines} dòng"))
    print("  - link chữ thường: " + (ok if not scan.upper_links else f"còn {scan.upper_links} link viết hoa"))
    print(f"  - name rỗng: {scan.empty_name} · text rỗng: {scan.empty_text}")
    print("  - Rác crawler: " + (ok if not scan.junk_leaks else f"còn {len(scan.junk_leaks)} văn bản: {scan.junk_leaks}"))
    if scan.suspicious_login:
        print(f"  - Nghi ngờ (soi tay): {len(scan.suspicious_login)} văn bản ngắn chứa 'đăng nhập' {scan.suspicious_login[:10]}")


def verify_processed_corpus(corpus_clean_path: Path, expected_lines: int) -> bool:
    """Thẩm định `corpus_clean.jsonl` theo INTERFACES.md §1 và các phát hiện EDA.

    Args:
        corpus_clean_path: File vừa sinh.
        expected_lines: Số dòng kỳ vọng (8.507).

    Returns:
        True nếu đạt mọi điều kiện.
    """
    if not corpus_clean_path.exists():
        print(f"Bộ QA thất bại: Không tìm thấy file {corpus_clean_path}")
        return False
    print("\n" + "=" * 80 + "\nTHẨM ĐỊNH corpus_clean.jsonl\n" + "=" * 80)
    scan = scan_corpus(corpus_clean_path)
    _print_corpus_scan(scan, expected_lines)
    passed = scan.passed(expected_lines)
    print("\nKẾT LUẬN: " + ("corpus sạch theo ràng buộc." if passed else "dữ liệu chưa sạch, kiểm tra lại parser."))
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Chạy cả corpus
# ─────────────────────────────────────────────────────────────────────────────
def _iter_raw_docs(corpus_dir: Path):
    """Duyệt `context_*.json` theo thứ tự tên file; bỏ qua (có báo) file hỏng hoặc thiếu `id`."""
    for fp in sorted(corpus_dir.glob("context_*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:  # ValueError gồm cả JSONDecodeError và lỗi giải mã UTF-8
            print(f"Lỗi đọc file {fp.name}: {e}")
            continue
        if doc.get("id") is None:
            print(f"File {fp.name} bị lỗi: Khuyết trường 'id'!")
            continue
        yield doc


def parse_corpus(corpus_dir: Path, out_path: Path, decisions_path: Path = DEFAULT_DECISIONS) -> int:
    """Gộp, làm sạch, loại trừ, ghi `corpus_clean.jsonl`; in SHA-256 rồi thẩm định.

    Args:
        corpus_dir: Thư mục chứa `context_*.json`.
        out_path: File JSONL đích.
        decisions_path: File quyết định loại trừ.

    Returns:
        Số văn bản đã ghi.

    Raises:
        FileNotFoundError: Thiếu thư mục dữ liệu thô hoặc file quyết định.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu thô tại: {corpus_dir}")
    decisions = load_decisions(decisions_path)
    excluded = set(decisions["docs_exclude_from_corpus"])
    expected = decisions["n_corpus_after_exclusion"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for doc in _iter_raw_docs(corpus_dir):
            if str(doc["id"]) in excluded:
                continue
            out_f.write(json.dumps(clean_record(doc), ensure_ascii=False) + "\n")
            count += 1

    checksum = calculate_sha256(out_path)
    print(f"Đã ghi {count} / {expected} văn bản vào {out_path}")
    if count != expected:
        print(f"  CẢNH BÁO: số dòng ({count}) khác kỳ vọng ({expected}) — kiểm tra danh sách loại hoặc dữ liệu nguồn.")
    print(f"SHA-256: {checksum}   (đối chiếu README mục 'Con số kỳ vọng')")
    verify_processed_corpus(out_path, expected)
    return count


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description="Gộp và làm sạch kho văn bản thô (P2)")
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/selected-contexts"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus_clean.jsonl"))
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS, help="sinh bởi scripts/eda.py")
    args = ap.parse_args()
    parse_corpus(args.corpus_dir, args.out, args.decisions)


if __name__ == "__main__":
    main()
