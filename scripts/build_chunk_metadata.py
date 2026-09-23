import json
import re
import bisect
from pathlib import Path

CHUONG_PATTERN = re.compile(r"\bChương\s+[IVXLCDM\d]+\b", re.IGNORECASE)
MUC_PATTERN = re.compile(r"\bMục\s+\d+\b", re.IGNORECASE)

def load_corpus_texts(corpus_path: Path) -> dict:
    """Đọc corpus_clean.jsonl vào dict: doc_id -> text gốc."""
    texts = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            texts[doc["doc_id"]] = doc.get("text", "")
    return texts


def find_structure_offsets(text: str) -> list:
    """
    Trả về danh sách (vị_trí, nhãn) cho mọi lần xuất hiện Chương/Mục trong text,
    đã sắp theo đúng thứ tự xuất hiện (vì finditer trả về theo thứ tự vị trí).
    """
    hits = []
    for m in CHUONG_PATTERN.finditer(text):
        hits.append((m.start(), m.group(0)))
    for m in MUC_PATTERN.finditer(text):
        hits.append((m.start(), m.group(0)))
    hits.sort(key=lambda x: x[0])
    return hits

TAG_PREFIX_PATTERN = re.compile(r"^\[.*?\]\s*")


def strip_continuation_tag(chunk_text: str) -> str:
    """Bỏ tiền tố '[Điều N. - tiếp theo] ' nếu có, để tìm lại đúng trong văn bản gốc."""
    return TAG_PREFIX_PATTERN.sub("", chunk_text, count=1)


def locate_chunks_in_doc(doc_text: str, chunks_of_doc: list) -> list:
    """
    chunks_of_doc: list các dict {"chunk_id", "text", "position"}, ĐÃ sắp theo position tăng dần.
    Trả về list song song: offset bắt đầu của từng chunk trong doc_text (None nếu không tìm thấy).
    """
    offsets = []
    search_from = 0
    for c in chunks_of_doc:
        snippet = strip_continuation_tag(c["text"])
        # Lấy 1 đoạn đầu đủ dài để tìm cho chắc, tránh snippet quá ngắn dễ trùng nhầm
        probe = snippet[:80] if len(snippet) >= 80 else snippet
        idx = doc_text.find(probe, search_from)
        if idx == -1:
            offsets.append(None)
        else:
            offsets.append(idx)
            search_from = idx + len(probe)
    return offsets

def match_nearest_structure(chunk_offset: int, structure_offsets: list) -> dict:
    """
    Tìm Chương/Mục gần nhất PHÍA TRƯỚC chunk_offset.
    structure_offsets: list (vị_trí, nhãn) đã sắp theo vị_trí tăng dần (từ find_structure_offsets).
    Trả về {"chuong": ..., "muc": ...} — None nếu không tìm thấy.
    """
    result = {"chuong": None, "muc": None}
    if chunk_offset is None:
        return result

    positions = [x[0] for x in structure_offsets]
    # bisect_right: vị trí chèn chunk_offset vào positions mà vẫn giữ thứ tự
    # -> mọi phần tử TRƯỚC idx đều có vị trí <= chunk_offset
    idx = bisect.bisect_right(positions, chunk_offset)

    # Duyệt ngược từ idx-1 về đầu, tìm Chương gần nhất và Mục gần nhất (có thể khác chỗ nhau)
    for i in range(idx - 1, -1, -1):
        pos, label = structure_offsets[i]
        if label.lower().startswith("chương") and result["chuong"] is None:
            result["chuong"] = label
        elif label.lower().startswith("mục") and result["muc"] is None:
            result["muc"] = label
        if result["chuong"] is not None and result["muc"] is not None:
            break

    return result

def build_metadata(corpus_path: Path, chunks_path: Path, out_path: Path) -> None:
    print(f"Đang nạp corpus gốc từ: {corpus_path}")
    doc_texts = load_corpus_texts(corpus_path)
    print(f"Đã nạp {len(doc_texts)} văn bản.")

    print(f"Đang nạp chunks từ: {chunks_path}")
    chunks_by_doc = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_doc.setdefault(c["doc_id"], []).append(c)
    # Đảm bảo mỗi nhóm chunk trong 1 doc được sắp đúng theo position tăng dần
    for doc_id in chunks_by_doc:
        chunks_by_doc[doc_id].sort(key=lambda c: c["position"])
    print(f"Đã nạp chunk cho {len(chunks_by_doc)} văn bản.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    unmatched_offset = 0
    has_chuong = 0
    has_muc = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for doc_id, chunks_of_doc in chunks_by_doc.items():
            doc_text = doc_texts.get(doc_id, "")
            structure_offsets = find_structure_offsets(doc_text)
            offsets = locate_chunks_in_doc(doc_text, chunks_of_doc)

            for c, offset in zip(chunks_of_doc, offsets):
                total_chunks += 1
                if offset is None:
                    unmatched_offset += 1
                meta = match_nearest_structure(offset, structure_offsets)
                if meta["chuong"] is not None:
                    has_chuong += 1
                if meta["muc"] is not None:
                    has_muc += 1
                record = {
                    "chunk_id": c["chunk_id"],
                    "chuong": meta["chuong"],
                    "muc": meta["muc"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nĐã ghi {total_chunks} dòng metadata vào: {out_path}")
    print(f"Số chunk KHÔNG tìm được vị trí trong văn bản gốc: {unmatched_offset} ({unmatched_offset/total_chunks*100:.2f}%)")
    print(f"Số chunk có gán Chương: {has_chuong} ({has_chuong/total_chunks*100:.2f}%)")
    print(f"Số chunk có gán Mục: {has_muc} ({has_muc/total_chunks*100:.2f}%)")


if __name__ == "__main__":
    build_metadata(
        corpus_path=Path("data/corpus_clean.jsonl"),
        chunks_path=Path("data/chunks_dieu.jsonl"),
        out_path=Path("data/chunk_metadata.jsonl"),
    )