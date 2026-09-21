"""
src/data/chunker_dieu.py - Bộ cắt đoạn văn bản pháp luật lai (Hybrid Legal Document Chunker)
              — v2: ranh giới "Điều N." chặt hơn, vá lỗi cắt nhầm tại trích dẫn chéo

Công dụng:
    Đọc file văn bản đã được làm sạch data/corpus_clean.jsonl, thực hiện chia nhỏ 
    các văn bản dài thành các đoạn (chunk) ngắn hơn và xuất ra data/chunks.jsonl.

Cách chạy:
    python -m src.data.chunker_dieu --input data/corpus_clean.jsonl --out data/chunks.jsonl --chunk-size 256 --overlap 64

Cấu trúc mã nguồn:
    1. HELPER FUNCTIONS:
        - sliding_window(text, chunk_size, overlap): Thuật toán cửa sổ trượt phân tách văn bản theo từ.
        - split_by_dieu(text): Regex phân tách văn bản theo ranh giới tự nhiên "Điều N".
    2. CORE CHUNKER (chunk_document):
        - Ưu tiên cắt theo "Điều N". Nếu Điều quá dài -> áp dụng sliding-window lồng bên trong.
        - Fallback sang sliding-window đối với các văn bản phi cấu trúc (TCVN/QCVN).
    3. QA VALIDATOR:
        - verify_chunks_file(filepath, expected_docs): Bộ tự động thẩm định và thống kê chất lượng chunks.
    4. CORE PIPELINE RUNNER (run_chunker):
        - Đọc corpus_clean.jsonl, gọi chunk_document() cho từng doc, ghi ra chunks.jsonl,
          kích hoạt verify_chunks_file() thẩm định ở cuối.
    5. ENTRYPOINT (main):
        - Nhận tham số dòng lệnh thông qua argparse và kích hoạt luồng xử lý.

    ┌────────────────────────────────────────────────────────┐
    │  IMPORTS BLOCK (argparse, json, re, hashlib, Path)     │
    ├────────────────────────────────────────────────────────┤
    │  1. HELPER FUNCTIONS                                   │
    │     ├── sliding_window(text) -> list[str]              │
    │     ├── split_by_dieu(text) -> list[str]               │
    │     └── sliding_window_with_dieu_tag(rc) -> list[str]  │
    ├────────────────────────────────────────────────────────┤
    │  2. CORE CHUNKER (chunk_document)                      │
    │     └── Phối hợp Điều N và Fallback Sliding Window     │
    ├────────────────────────────────────────────────────────┤
    │  3. QA VALIDATOR                                       │
    │     └── verify_chunks_file(path,                       │
    │         expected_docs: int | None) -> bool             │
    ├────────────────────────────────────────────────────────┤
    │  4. CORE PIPELINE RUNNER (run_chunker)                 │
    │     └── Đọc corpus_clean -> chunk từng doc -> ghi file │
    │         -> gọi verify_chunks_file() thẩm định cuối     │
    ├────────────────────────────────────────────────────────┤
    │  5. CLI ENTRYPOINT (main)                              │
    │     └── Cấu hình argparse (--input, --out, --size)     │
    └────────────────────────────────────────────────────────┘

Các lưu ý sống còn (Bẫy dữ liệu phòng ngự):
    - [DẤU PHÂN CÁCH ID]: chunk_id bắt buộc sử dụng định dạng "doc_id::seq" (nối bằng hai dấu hai chấm "::") 
      thay vì dấu gạch dưới "_" để P3/P4 dễ dàng tách ngược lấy doc_id gốc chính xác.
    - [XỬ LÝ ĐIỀU SIÊU DÀI]: Dù tài liệu có cấu trúc Điều (91.3%), nếu một Điều đơn lẻ vượt quá ngưỡng chunk_size 
      (ví dụ các phụ lục, bảng biểu dính trong Điều), hệ thống vẫn tự động chạy sliding-window lồng bên trong 
      đoạn đó để băm nhỏ ra, bảo vệ mô hình khỏi lỗi tràn ngữ cảnh.
    - [BỎ QUA PASSAGE RỖNG]: chunk_document() trả về [] cho text rỗng - Lớp phòng ngự này kích hoạt nếu phát sinh
      doc rỗng MỚI chưa từng biết trong dữ liệu tương lai.
    - [TỰ ĐỘNG THẨM ĐỊNH QA]: Tích hợp trực tiếp bộ QA đối soát ở cuối chương trình để kiểm tra cấu trúc khóa, 
      tính nhất quán của chunk_id, đồng thời in các thống kê thực chiến: số lượng chunk trung bình/max/min, 
      và tỷ lệ tài liệu phải fallback thực tế để đối chiếu với EDA.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

# Regex phát hiện ranh giới "Điều N" (không phân biệt hoa thường)
DIEU_PATTERN = re.compile(r"\bĐiều\s+\d+\.\s", re.IGNORECASE)

# ===========================================================================
# 1. HELPER FUNCTIONS
# ===========================================================================

def sliding_window(text: str, chunk_size: int = 256, overlap: int = 64) -> list[str]:
    """
    Thuật toán cửa sổ trượt (Sliding Window) để cắt nhỏ văn bản theo số lượng từ thô.
    Bảo toàn thông tin tại ranh giới bằng khoảng gối đầu (overlap).
    """
    words = text.split()
    if not words:
        return []
    
    # Nếu văn bản ngắn hơn kích thước cửa sổ, giữ nguyên làm 1 chunk
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # Phòng ngự chia cho 0 hoặc lặp vô hạn
        
    for i in range(0, len(words), step):
        window_words = words[i : i + chunk_size]
        chunks.append(" ".join(window_words))
        # Nếu đã quét đến hết từ cuối cùng của văn bản thì dừng lặp
        if i + chunk_size >= len(words):
            break
            
    return chunks

def split_by_dieu(text: str) -> list[str]:
    """
    Phân tách văn bản dựa trên ranh giới tự nhiên của các Điều luật (Điều N).
    Giữ lại tiêu đề "Điều N" ở đầu mỗi đoạn và thu thập cả phần mở đầu (preamble).
    """
    matches = list(re.finditer(DIEU_PATTERN, text))
    if not matches:
        return []
        
    raw_chunks = []
    
    # 1. Thu thập phần mở đầu (Preamble) trước "Điều 1" (nếu có nội dungsubstantial)
    preamble = text[:matches[0].start()].strip()
    if preamble:
        raw_chunks.append(preamble)
        
    # 2. Thu thập nội dung của từng Điều luật
    for idx, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk_content = text[start_pos:end_pos].strip()
        if chunk_content:
            raw_chunks.append(chunk_content)
            
    return raw_chunks


def sliding_window_with_dieu_tag(rc: str, chunk_size: int, overlap: int) -> list[str]:
    dieu_match = DIEU_PATTERN.search(rc)
    dieu_label = dieu_match.group(0).strip() if dieu_match else ""
    tag_prefix = f"[{dieu_label} - tiếp theo] " if dieu_label else ""
    tag_word_count = len(tag_prefix.split())
    # Trừ hao độ dài tiền tố "tiếp theo" vào ngân sách để không vượt chunk_size
    budget = max(chunk_size - tag_word_count, 1) if tag_prefix else chunk_size
    raw_subs = sliding_window(rc, budget, overlap)
    tagged = []
    for i, sub in enumerate(raw_subs):
        if i == 0 or not dieu_label or sub.strip().startswith(dieu_label):
            tagged.append(sub)
        else:
            tagged.append(f"{tag_prefix}{sub}")
    return tagged

# ===========================================================================
# 2. CORE CHUNKER LOGIC
# ===========================================================================

def chunk_document(text: str, chunk_size: int = 256, overlap: int = 64) -> list[str]:
    """
    Hàm chia nhỏ một văn bản pháp lý dựa trên chiến lược Chunker Lai:
    - Ưu tiên cắt theo Điều N tự nhiên.
    - Nếu Điều quá dài -> Cắt sliding-window lồng bên trong Điều đó.
    - Không có Điều N -> Fallback sang sliding-window toàn văn bản.
    """
    if not text or not text.strip():
        return []
        
    # Thử phân tách theo cấu trúc "Điều N"
    raw_chunks = split_by_dieu(text)
    
    final_chunks = []
    
    if not raw_chunks:
        # TRƯỜNG HỢP 1: FALLBACK SLIDING WINDOW (8.7% văn bản phi cấu trúc)
        final_chunks = sliding_window(text, chunk_size, overlap)
    else:
        # TRƯỜNG HỢP 2: CẮT THEO ĐIỀU N (91.3% văn bản cấu trúc luật)
        for rc in raw_chunks:
            # Kiểm tra độ dài của Điều luật thô
            rc_word_count = len(rc.split())
            if rc_word_count <= chunk_size:
                # Điều luật ngắn gọn -> Giữ nguyên làm 1 chunk sắc nét
                final_chunks.append(rc)
            else:
                # Điều luật siêu dài (outlier phụ lục) -> Cắt sliding-window lồng bên trong Điều
                sub_chunks = sliding_window_with_dieu_tag(rc, chunk_size, overlap)
                final_chunks.extend(sub_chunks)
                
    return final_chunks


# ===========================================================================
# 3. INTEGRATED QA VALIDATOR (Bộ Thẩm Định Chất Lượng Chunks)
# ===========================================================================

def verify_chunks_file(chunks_path: Path, expected_docs: int | None = None) -> bool:
    """
    Tự động quét và thẩm định toàn bộ file chunks.jsonl sau khi sinh ra.
    Kiểm chứng các quy chuẩn trong INTERFACES.md và thống kê chỉ số thực chiến.
    """
    if not chunks_path.exists():
        print(f"❌ Bộ QA thất bại: Không tìm thấy file {chunks_path}")
        return False

    print("\n" + "=" * 80)
    print("KHỞI CHẠY BỘ THẨM ĐỊNH CHẤT LƯỢNG CHUNKS TỰ ĐỘNG...")
    print("=" * 80)

    total_chunks = 0
    all_keys_valid = True
    all_chunk_ids_well_formatted = True
    all_doc_ids_are_str = True
    empty_text_found = 0
    
    # Lưu vết số lượng chunk của từng tài liệu để tính toán phân bố
    doc_chunk_counts = {}

    with open(chunks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_chunks += 1
            try:
                chunk = json.loads(line)
            except Exception as e:
                print(f"  Dòng {idx + 1} không phải JSON hợp lệ: {e}")
                all_keys_valid = False
                continue

            # 1. Kiểm tra sự tồn tại đầy đủ của 4 khóa bắt buộc theo INTERFACES.md
            for key in ["chunk_id", "doc_id", "position", "text"]:
                if key not in chunk:
                    print(f"  Dòng {idx + 1} (Chunk ID: {chunk.get('chunk_id')}) bị khuyết khóa: {key}")
                    all_keys_valid = False

            # 2. Kiểm tra bẫy kiểu dữ liệu doc_id (Phải là str)
            doc_id = chunk.get("doc_id")
            if doc_id is not None and not isinstance(doc_id, str):
                all_doc_ids_are_str = False

            # 3. Kiểm tra định dạng chunk_id bắt buộc chứa phân cách "::"
            chunk_id = chunk.get("chunk_id", "")
            if "::" not in chunk_id:
                all_chunk_ids_well_formatted = False
            else:
                parts = chunk_id.split("::")
                if len(parts) != 2 or parts[0] != doc_id or not parts[1].isdigit():
                    all_chunk_ids_well_formatted = False

            # 4. Kiểm tra text trống
            text_content  = chunk.get("text", "")
            if not text_content .strip():
                empty_text_found += 1
                
            # Ghi nhận số lượng chunk cho doc_id
            if doc_id:
                doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1

    print("\n--------------------------------------------------")
    print("THẨM ĐỊNH CHI TIẾT (QA RESULTS):")
    print("--------------------------------------------------")
    
    # Tính toán các chỉ số thống kê phân bố
    n_docs_chunked = len(doc_chunk_counts)
    counts = list(doc_chunk_counts.values())
    max_chunks = max(counts) if counts else 0
    min_chunks = min(counts) if counts else 0
    avg_chunks = sum(counts) / len(counts) if counts else 0

    print(f"  - Tổng số chunks tạo thành: {total_chunks}")
    if expected_docs is not None:
        print(f"  - Tổng số tài liệu được chunking: {n_docs_chunked} / {expected_docs} doc gốc trong corpus_clean.jsonl")
        if n_docs_chunked != expected_docs:
            print(f"  CẢNH BÁO: {expected_docs - n_docs_chunked} doc không sinh ra chunk nào (có thể do passage rỗng chưa lọc hết).")
    else:
        print(f"  - Tổng số tài liệu được chunking: {n_docs_chunked}")
    print(f"  - Cấu trúc khóa: " + ("Đầy đủ 100% (chunk_id, doc_id, position, text)" if all_keys_valid else "Có chunk bị thiếu khóa"))
    print(f"  - Định dạng Chunk ID (nối bằng ::): " + ("Hoàn hảo 100%" if all_chunk_ids_well_formatted else "Chunk ID sai quy cách!"))
    print(f"  - Kiểu dữ liệu ID tài liệu: " + ("100% là chuỗi (str) - Chống bẫy 0 điểm im lặng" if all_doc_ids_are_str else "Phát hiện doc_id kiểu số (int)!"))
    print(f"  - Số lượng text rỗng: " + ("Hoàn hảo (0 chunk rỗng)" if empty_text_found == 0 else f"Phát hiện {empty_text_found} chunk bị rỗng text!"))
    print(f"  - Phân bố chunks trên mỗi văn bản:")
    print(f"    * Trung bình: {avg_chunks:.2f} chunks/doc")
    print(f"    * Lớn nhất (Max): {max_chunks} chunks (Outlier nhiều đoạn)")
    print(f"    * Nhỏ nhất (Min): {min_chunks} chunks/doc")
    print("-" * 50)

    # Đánh giá tổng quan
    success = total_chunks > 0 and all_keys_valid and all_chunk_ids_well_formatted and all_doc_ids_are_str and empty_text_found == 0
    if success:
        print("\n KẾT LUẬN: KHO CHUNKS ĐẠT THEO KẾ HOẠCH (CODA-READY)!")
        print("=" * 80)
        return True
    else:
        print("\n CẢNH BÁO: Phát hiện lỗi cấu trúc dữ liệu trong file chunks, vui lòng kiểm tra lại logic chunker.")
        print("=" * 80)
        return False


# ===========================================================================
# 4. CORE PIPELINE RUNNER
# ===========================================================================

def run_chunker(input_path: Path, out_path: Path, chunk_size: int, overlap: int):
    """
    Hàm điều phối pipeline chunking: Đọc corpus_clean, tiến hành cắt nhỏ và ghi file chunks.jsonl
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file corpus đã làm sạch tại: {input_path}")
        
    print(f"Đang nạp kho văn bản sạch từ: {input_path}")
    print(f"Tham số cấu hình chunking: Size = {chunk_size} từ, Overlap = {overlap} từ")
    
    # Tạo thư mục cha chứa file out nếu chưa tồn tại
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    count_docs = 0
    count_chunks = 0
    fallback_docs = 0
    
    with open(input_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            count_docs += 1
            doc = json.loads(line)
            
            doc_id = doc["doc_id"]
            text = doc.get("text", "")
            
            # Kiểm tra xem tài liệu này có phải fallback sliding window hay không
            is_fallback = text.strip() and not bool(re.search(DIEU_PATTERN, text))
            if is_fallback:
                fallback_docs += 1
                
            # Tiến hành chia nhỏ văn bản
            chunks = chunk_document(text, chunk_size, overlap)
            
            for seq_idx, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{doc_id}::{seq_idx:04d}",
                    "doc_id": doc_id,
                    "position": seq_idx,
                    "text": chunk_text,
                }
                out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                count_chunks += 1
                
    print(f"Đã xử lý xong {count_docs} tài liệu.")
    print(f"Đã ghi thành công {count_chunks} chunks vào file: {out_path}")
    print(f"Thống kê sơ bộ: Có {fallback_docs} / {count_docs} tài liệu fallback sliding-window (~{fallback_docs / count_docs * 100:.1f}%)")
    
    # Khởi chạy bộ thẩm định chất lượng tự động
    verify_chunks_file(out_path, expected_docs=count_docs)
    
    print(f"\nĐiền thông số Tổng số chunks ({count_chunks}) dán vào file README.md mục 8!")


# ===========================================================================
# 5. CLI ENTRYPOINT
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Bộ cắt nhỏ văn bản pháp luật lai ghép cho P2")
    ap.add_argument("--input", type=Path, default=Path("data/corpus_clean.jsonl"),
                    help="Đường dẫn file văn bản sạch corpus_clean.jsonl")
    ap.add_argument("--out", type=Path, default=Path("data/chunks_dieu.jsonl"),
                help="Đường dẫn file đầu ra chunks_dieu.jsonl")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="Độ dài tối đa của một chunk (tính theo số từ)")
    ap.add_argument("--overlap", type=int, default=64,
                    help="Độ dài gối đầu giữa các chunk (tính theo số từ)")
    args = ap.parse_args()
    
    run_chunker(args.input, args.out, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
