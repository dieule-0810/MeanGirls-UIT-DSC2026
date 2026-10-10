"""Chunker lai cho văn bản pháp luật: cắt theo "Điều N", fallback cửa sổ trượt.

Đọc `data/corpus_clean.jsonl`, ghi `chunks.jsonl` đúng hợp đồng INTERFACES.md §2
(`chunk_id = f"{doc_id}::{position:04d}"`), rồi tự chạy bộ thẩm định ở cuối.

Hai chiến lược, khác nhau đúng ở cách nhận ranh giới Điều:

* `loose` (v1, mặc định của module này) — khớp `Điều N` ở BẤT KỲ đâu, kể cả trích dẫn chéo
  giữa câu ("theo Điều 5 Luật ..."), nên cắt vụn hơn. Sinh kho 524.422 chunk mà mọi số
  v0.1–v0.6 đã đo.
* `strict` (v2, `python -m src.data.chunker_dieu`) — chỉ khớp tiêu đề `Điều N. `, và trừ hao
  độ dài nhãn "[Điều N. - tiếp theo]" vào ngân sách cửa sổ. Sinh kho 432.142 chunk mà v0.8
  (pipeline nộp bài hiện tại) dùng, và là kho embedding dense đã encode trên đó.

Luật chung của cả hai: Điều ngắn hơn `chunk_size` từ giữ nguyên một chunk; Điều dài hơn thì
cắt cửa sổ trượt LỒNG trong Điều, các mảnh sau gắn nhãn "[Điều N - tiếp theo]"; văn bản không
có Điều nào (TCVN/QCVN, ~8,7%) thì cắt cửa sổ trượt toàn văn bản. Bỏ nhãn đi thì mọi chunk là
chuỗi con nguyên văn của văn bản gốc — không phải dữ liệu ngoài, không phải augmentation.

Typical usage example:

    python -m src.data.chunker --input data/corpus_clean.jsonl --out data/chunks.jsonl
    python -m src.data.chunker --strategy strict --out data/chunks.jsonl     # kho của v0.8
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ChunkStrategy:
    """Cách nhận ranh giới Điều.

    Attributes:
        name: `loose` hoặc `strict`.
        dieu_pattern: Regex nhận một ranh giới Điều.
        reserve_tag_budget: Trừ số từ của nhãn "tiếp theo" khỏi ngân sách cửa sổ, để chunk có
            nhãn không vượt `chunk_size`.
        default_out: File ra mặc định của CLI.
    """

    name: str
    dieu_pattern: re.Pattern
    reserve_tag_budget: bool
    default_out: Path


STRATEGIES: dict[str, ChunkStrategy] = {
    "loose": ChunkStrategy("loose", re.compile(r"\bĐiều\s+\d+\b", re.IGNORECASE), False, Path("data/chunks.jsonl")),
    "strict": ChunkStrategy("strict", re.compile(r"\bĐiều\s+\d+\.\s", re.IGNORECASE), True, Path("data/chunks_dieu.jsonl")),
}


@dataclass(frozen=True)
class ChunkerConfig:
    """Tham số chunking.

    Attributes:
        chunk_size: Số từ tối đa mỗi chunk.
        overlap: Số từ gối đầu giữa hai cửa sổ liên tiếp.
        strategy: Khoá trong `STRATEGIES`.
    """

    chunk_size: int = 256
    overlap: int = 64
    strategy: str = "loose"

    def __post_init__(self) -> None:
        """Kiểm `strategy` có trong `STRATEGIES`."""
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy '{self.strategy}' không có. Dùng: {', '.join(STRATEGIES)}")

    @property
    def rule(self) -> ChunkStrategy:
        """Chiến lược ứng với `strategy`."""
        return STRATEGIES[self.strategy]


# ─────────────────────────────────────────────────────────────────────────────
# Cắt một văn bản
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window(text: str, chunk_size: int = 256, overlap: int = 64) -> list[str]:
    """Cắt văn bản theo số từ, các cửa sổ gối đầu nhau `overlap` từ.

    Args:
        text: Văn bản.
        chunk_size: Số từ mỗi cửa sổ.
        overlap: Số từ gối đầu; `overlap >= chunk_size` thì coi như không gối.

    Returns:
        Các cửa sổ; văn bản ngắn hơn `chunk_size` giữ nguyên làm một chunk; rỗng → [].
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # phòng chia cho 0 / lặp vô hạn
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks


def split_by_dieu(text: str, pattern: re.Pattern) -> list[str]:
    """Tách văn bản tại mỗi ranh giới Điều, giữ tiêu đề Điều ở đầu đoạn và giữ phần mở đầu.

    Args:
        text: Văn bản.
        pattern: Regex ranh giới Điều của chiến lược đang dùng.

    Returns:
        `[phần mở đầu (nếu có), Điều 1 ..., Điều 2 ..., ...]`; không có Điều nào → [].
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    parts = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        parts.append(preamble)
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[match.start() : end].strip()
        if content:
            parts.append(content)
    return parts


def window_with_dieu_tag(section: str, cfg: ChunkerConfig) -> list[str]:
    """Cắt cửa sổ trượt bên trong một Điều quá dài; mảnh thứ 2 trở đi gắn nhãn "tiếp theo".

    Args:
        section: Nội dung một Điều (bắt đầu bằng tiêu đề Điều).
        cfg: Tham số chunking.

    Returns:
        Các mảnh, đã gắn nhãn nếu cần.
    """
    m = cfg.rule.dieu_pattern.search(section)
    label = m.group(0).strip() if m else ""
    tag = f"[{label} - tiếp theo] " if label else ""
    budget = cfg.chunk_size
    if tag and cfg.rule.reserve_tag_budget:
        budget = max(cfg.chunk_size - len(tag.split()), 1)
    tagged = []
    for i, sub in enumerate(sliding_window(section, budget, cfg.overlap)):
        if i == 0 or not label or sub.strip().startswith(label):
            tagged.append(sub)
        else:
            tagged.append(f"{tag}{sub}")
    return tagged


def chunk_document(text: str, cfg: ChunkerConfig | None = None) -> list[str]:
    """Chia một văn bản theo chiến lược lai.

    Args:
        text: Văn bản đã làm sạch.
        cfg: Tham số chunking; None = mặc định (`loose`, 256/64).

    Returns:
        Các chunk theo thứ tự; văn bản rỗng → [] (phòng doc rỗng mới phát sinh).
    """
    cfg = cfg or ChunkerConfig()
    if not text or not text.strip():
        return []
    sections = split_by_dieu(text, cfg.rule.dieu_pattern)
    if not sections:
        return sliding_window(text, cfg.chunk_size, cfg.overlap)
    chunks: list[str] = []
    for section in sections:
        if len(section.split()) <= cfg.chunk_size:
            chunks.append(section)
        else:
            chunks.extend(window_with_dieu_tag(section, cfg))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Thẩm định file chunk
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChunkScan:
    """Kết quả quét một file chunk."""

    total: int = 0
    keys_ok: bool = True
    ids_ok: bool = True
    doc_ids_str: bool = True
    empty_text: int = 0
    per_doc: Counter = field(default_factory=Counter)

    @property
    def passed(self) -> bool:
        """Đạt mọi điều kiện cấu trúc."""
        return self.total > 0 and self.keys_ok and self.ids_ok and self.doc_ids_str and self.empty_text == 0


def _scan_record(scan: ChunkScan, idx: int, chunk: dict) -> None:
    """Cập nhật `scan` với một bản ghi chunk."""
    for key in ("chunk_id", "doc_id", "position", "text"):
        if key not in chunk:
            print(f"  Dòng {idx + 1} (Chunk ID: {chunk.get('chunk_id')}) bị khuyết khóa: {key}")
            scan.keys_ok = False
    doc_id = chunk.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        scan.doc_ids_str = False
    parts = chunk.get("chunk_id", "").split("::")
    if len(parts) != 2 or parts[0] != doc_id or not parts[1].isdigit():
        scan.ids_ok = False
    if not chunk.get("text", "").strip():
        scan.empty_text += 1
    if doc_id:
        scan.per_doc[doc_id] += 1


def scan_chunks_file(chunks_path: Path) -> ChunkScan:
    """Quét file chunk, kiểm 4 khoá bắt buộc, định dạng `chunk_id`, kiểu `doc_id`, text rỗng."""
    scan = ChunkScan()
    with open(chunks_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            scan.total += 1
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Dòng {idx + 1} không phải JSON hợp lệ: {e}")
                scan.keys_ok = False
                continue
            _scan_record(scan, idx, chunk)
    return scan


def _print_scan(scan: ChunkScan, expected_docs: int | None) -> None:
    """In kết quả quét file chunk."""
    counts = list(scan.per_doc.values())
    n_docs = len(counts)
    print(f"  - Tổng số chunk: {scan.total}")
    if expected_docs is not None:
        print(f"  - Số văn bản có chunk: {n_docs} / {expected_docs}")
        if n_docs != expected_docs:
            print(f"  CẢNH BÁO: {expected_docs - n_docs} văn bản không sinh ra chunk nào.")
    else:
        print(f"  - Số văn bản có chunk: {n_docs}")
    print("  - Cấu trúc khóa: " + ("đủ (chunk_id, doc_id, position, text)" if scan.keys_ok else "có chunk thiếu khóa"))
    print("  - Định dạng chunk_id `doc_id::NNNN`: " + ("đúng" if scan.ids_ok else "SAI quy cách"))
    print("  - Kiểu doc_id: " + ("100% str" if scan.doc_ids_str else "có doc_id kiểu số (int)!"))
    print("  - Chunk rỗng: " + (str(scan.empty_text) if scan.empty_text else "0"))
    if counts:
        print(f"  - Chunk/văn bản: trung bình {sum(counts) / n_docs:.2f} · max {max(counts)} · min {min(counts)}")


def verify_chunks_file(chunks_path: Path, expected_docs: int | None = None) -> bool:
    """Thẩm định file chunk vừa sinh theo INTERFACES.md §2 và in thống kê.

    Args:
        chunks_path: File chunk.
        expected_docs: Số văn bản kỳ vọng (để cảnh báo văn bản không sinh chunk nào).

    Returns:
        True nếu đạt mọi điều kiện cấu trúc.
    """
    if not chunks_path.exists():
        print(f"❌ Bộ QA thất bại: Không tìm thấy file {chunks_path}")
        return False
    print("\n" + "=" * 80 + "\nTHẨM ĐỊNH FILE CHUNK\n" + "=" * 80)
    scan = scan_chunks_file(chunks_path)
    _print_scan(scan, expected_docs)
    print("\nKẾT LUẬN: " + ("kho chunk ĐẠT." if scan.passed else "phát hiện lỗi cấu trúc, kiểm tra lại chunker."))
    return scan.passed


# ─────────────────────────────────────────────────────────────────────────────
# Chạy cả corpus
# ─────────────────────────────────────────────────────────────────────────────
def run_chunker(input_path: Path, out_path: Path, cfg: ChunkerConfig) -> int:
    """Chunk toàn bộ `corpus_clean.jsonl`, ghi file chunk rồi thẩm định.

    Args:
        input_path: `corpus_clean.jsonl`.
        out_path: File chunk đích.
        cfg: Tham số chunking.

    Returns:
        Số chunk đã ghi.

    Raises:
        FileNotFoundError: Không có file corpus.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file corpus đã làm sạch tại: {input_path}")
    print(f"Chunking {input_path} · strategy={cfg.strategy} · size={cfg.chunk_size} · overlap={cfg.overlap}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_chunks = n_fallback = 0
    with open(input_path, encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            n_docs += 1
            doc = json.loads(line)
            text = doc.get("text", "")
            if text.strip() and not cfg.rule.dieu_pattern.search(text):
                n_fallback += 1
            for pos, chunk_text in enumerate(chunk_document(text, cfg)):
                record = {"chunk_id": f"{doc['doc_id']}::{pos:04d}", "doc_id": doc["doc_id"], "position": pos, "text": chunk_text}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"Đã ghi {n_chunks} chunk từ {n_docs} văn bản vào {out_path}")
    print(f"Văn bản fallback cửa sổ trượt: {n_fallback} / {n_docs} (~{n_fallback / max(n_docs, 1) * 100:.1f}%)")
    verify_chunks_file(out_path, expected_docs=n_docs)
    return n_chunks


def main(default_strategy: str = "loose") -> None:
    """CLI. `src.data.chunker_dieu` gọi hàm này với `default_strategy="strict"`."""
    ap = argparse.ArgumentParser(description="Chunker lai cho văn bản pháp luật (P2)")
    ap.add_argument("--input", type=Path, default=Path("data/corpus_clean.jsonl"))
    ap.add_argument("--out", type=Path, default=None, help="mặc định theo strategy")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default=default_strategy)
    ap.add_argument("--chunk-size", type=int, default=256, help="số từ tối đa mỗi chunk")
    ap.add_argument("--overlap", type=int, default=64, help="số từ gối đầu giữa hai cửa sổ")
    args = ap.parse_args()
    cfg = ChunkerConfig(args.chunk_size, args.overlap, args.strategy)
    run_chunker(args.input, args.out or cfg.rule.default_out, cfg)


if __name__ == "__main__":
    main()
