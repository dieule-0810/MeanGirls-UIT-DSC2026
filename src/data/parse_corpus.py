"""
src/data/parse_corpus.py - Bộ tiền xử lý và gộp kho văn bản pháp luật thô (Corpus Preprocessing)

Công dụng:
    Đọc toàn bộ 8.532 file context_*.json từ thư mục dữ liệu thô, tiến hành dọn dẹp rác xuống dòng,
    lọc sạch thông báo bảo mật (crawler pollution), ép kiểu dữ liệu an toàn, tự động kiểm tra
    chất lượng (QA) và xuất ra một file JSON Lines (JSONL) duy nhất tại data/corpus_clean.jsonl.

Cách chạy:
    python -m src.data.parse_corpus --corpus-dir data/selected-contexts --out data/corpus_clean.jsonl

Cấu trúc mã nguồn:
    1. HELPER FUNCTIONS:
        - clean_text(text): Dọn rác \r\n\n, khoảng trắng thừa và lọc sạch đuôi thông báo bảo mật bằng Regex.
        - calculate_sha256(filepath): Tính toán dấu vân tay kỹ thuật số SHA-256 của file output.
    2. QA VALIDATOR:
        - verify_processed_corpus(filepath): Bộ tự động thẩm định chất lượng dữ liệu sạch.
    3. CORE PARSER (parse_corpus):
        - Quét tuần tự 8.532 file thô theo thứ tự bảng chữ cái để đảm bảo tính tuần tự không đổi.
        - Áp dụng các chốt chặn an toàn cho từng dòng dữ liệu (ép kiểu, gán giá trị mặc định, lọc rác).
        - Ghi stream từng dòng JSON vào file đích và kích hoạt bộ QA validator ở cuối.
    4. ENTRYPOINT (main):
        - Nhận tham số dòng lệnh thông qua argparse và kích hoạt luồng xử lý.

    ┌────────────────────────────────────────────────────────┐
    │  1. IMPORTS BLOCK (argparse, json, re, hashlib, Path)  │
    ├────────────────────────────────────────────────────────┤
    │  2. HELPER FUNCTIONS                                   │
    │     ├── clean_text(raw_text) -> str                    │
    │     └── calculate_sha256(file_path) -> str             │
    ├────────────────────────────────────────────────────────┤
    │  3. QA VALIDATOR                                       │
    │     └── verify_processed_corpus(file_path) -> bool     │
    ├────────────────────────────────────────────────────────┤
    │  4. CORE PARSER (parse_corpus)                         │
    │     ├── Quét 8.532 file thô bằng .glob()               │
    │     ├── Đọc, ép str(id), sửa khuyết name/passage       │
    │     ├── Dọn rác crawler (security popup) bằng Regex    │
    │     └── Ghi tuần tự từng dòng vào file .jsonl          │
    ├────────────────────────────────────────────────────────┤
    │  5. CLI ENTRYPOINT (main)                              │
    │     └── Cấu hình argparse (--corpus-dir, --out)        │
    └────────────────────────────────────────────────────────┘

Các lưu ý sống còn (Bẫy dữ liệu phòng ngự):
    - [BẪY KIỂU DỮ LIỆU]: id của file thô là int (177504) nhưng nhãn BTC dùng str ("177504").
      Bắt buộc ép str(doc_id) ngay tại điểm đọc để tránh lỗi "0 điểm im lặng" trên Leaderboard.
    - [BẪY KHUYẾT TRƯỜNG]: 13.2% tài liệu khuyết trường 'name' -> dùng .get("name", "Không có tiêu đề").
    - [BẪY RỖNG PASSAGE]: 0.2% file rỗng passage -> giữ nguyên "", TUYỆT ĐỐI không bịa nội dung rác 
      để tránh làm lệch không gian vector của mô hình.
    - [CHỮ THƯỜNG LINK]: Đổi link về dạng lowercase() để đồng bộ hóa các trường hợp trùng lặp.
    - [XÁC MINH CHECKSUM]: Bắt buộc tự in số dòng (phải đúng 8.532) và SHA-256 Checksum sau khi ghi xong.
    - [THỨ TỰ DÒNG / ID]: Dữ liệu được ghi theo thứ tự bảng chữ cái (lexicographical) của tên file thô 
      (ví dụ: ID 100 đứng trước ID 2). Đảm bảo tính nhất quán tuyệt đối giúp mã băm SHA-256 trùng khớp 
      100%, hoàn toàn không ảnh hưởng đến hiệu năng lập chỉ mục hay điểm số truy hồi của BTC.
    - [LỌC RÁC CRAWLER]: Tự động nhận diện và xóa bỏ 100% thông báo bảo mật ("Quý khách vui lòng đăng nhập",
      "đăng nhập để xem", "tránh rò rỉ mật khẩu",...)
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


# ===========================================================================
# CONSTANTS & REGEX FOR CRAWLER POLLUTION
# ===========================================================================
# Danh sách các cụm rác thực tế bám đuôi từ website nguồn
CRAWLER_JUNK_PATTERNS = [
    "quý khách vui lòng đăng nhập",
    "đăng nhập để xem",
    "rò rỉ mật khẩu",
    "vui lòng đăng nhập để",
    "đăng nhập để tiếp tục"
]

# Regex mẫu dùng chung để xóa rác (chặt toàn bộ nội dung từ vị trí dính rác trở đi)
CLEAN_JUNK_REGEX = re.compile(
    r"(" + "|".join(CRAWLER_JUNK_PATTERNS) + r").*$",
    re.IGNORECASE
)

# ===========================================================================
# 1. HELPER FUNCTIONS
# ===========================================================================

def clean_text(raw_text: str) -> str:
    """
    Sử dụng Regex dọn dẹp sạch sẽ ký tự xuống dòng rác, khoảng trắng thừa,
    chuẩn hóa Unicode NFC và dọn sạch thông báo bảo mật (crawler junk).
    """
    if not raw_text:
        return ""

    # 1. Ép về chuẩn Unicode NFC để xử lý triệt để bẫy NFD
    raw_text = unicodedata.normalize("NFC", raw_text)

    # Kéo phẳng văn bản bằng Regex (\s+ đại diện cho mọi ký tự khoảng trắng/xuống dòng)
    cleaned = re.sub(r"\s+", " ", raw_text)
    
    # 2. Loại bỏ rác bằng hằng số chung (chặt toàn bộ đoạn rác trở đi)
    cleaned = CLEAN_JUNK_REGEX.sub("", cleaned)
    
    return cleaned.strip()


def calculate_sha256(file_path: Path) -> str:
    """
    Tính mã hash SHA-256 (vân tay kỹ thuật số) của file output theo từng block 64KB
    để chống tràn bộ nhớ RAM đối với file dung lượng lớn.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ===========================================================================
# 2. INTEGRATED QA VALIDATOR (Bộ Thẩm Định Chất Lượng)
# ===========================================================================

def verify_processed_corpus(corpus_clean_path: Path) -> bool:
    """
    Tự động quét kiểm tra file corpus_clean.jsonl đã sinh ra.
    Xác minh tất cả các điều kiện ràng buộc của INTERFACES.md và thống kê từ EDA.
    """
    if not corpus_clean_path.exists():
        print(f"Bộ QA thất bại: Không tìm thấy file {corpus_clean_path}")
        return False

    print("\n" + "="*80)
    print("THẨM ĐỊNH CHẤT LƯỢNG DỮ LIỆU TỰ ĐỘNG (QA)...")
    print("="*80)

    n_lines = 0
    all_keys_valid = True
    all_ids_are_str = True
    empty_passages = 0
    empty_names = 0
    raw_newlines_found = 0
    links_uppercase = 0

    # Bộ kiểm định rác độc lập (Broad check)
    leaked_security_popups = []  # Chứa các file dính "rò rỉ mật khẩu"
    potential_false_positives = []  # Chứa các file dính "đăng nhập" hợp lệ

    with open(corpus_clean_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            n_lines += 1
            try:
                doc = json.loads(line)
            except Exception as e:
                print(f"Dòng {idx+1} không phải JSON hợp lệ: {e}")
                all_keys_valid = False
                continue

            # 1. Kiểm tra sự tồn tại đầy đủ của 4 trường bắt buộc theo INTERFACES.md
            for key in ["id", "name", "link", "passage"]:
                if key not in doc:
                    print(f"  Dòng {idx + 1} (ID: {doc.get('id')}) bị khuyết khóa: {key}")
                    all_keys_valid = False

            # 2. Kiểm tra bẫy kiểu dữ liệu ID (Bắt buộc phải là str)
            doc_id = doc.get("id")
            if doc_id is not None and not isinstance(doc_id, str):
                all_ids_are_str = False

            # 3. Kiểm tra khuyết name (đã quy về chuỗi rỗng "" chưa)
            name = doc.get("name")
            if name == "":
                empty_names += 1

            # 4. Kiểm tra rỗng passage (phải giữ rỗng chứ không được chế chữ rác)
            passage = doc.get("passage")
            if passage == "":
                empty_passages += 1

            # 5. Kiểm tra xem Regex đã dọn sạch các ký tự xuống dòng rác (\r\n\n) chưa
            if "\r" in passage or "\n\n" in passage:
                raw_newlines_found += 1

            # 6. Kiểm tra URL đã đưa về viết thường (lowercase) chưa
            link = doc.get("link", "")
            if any(c.isupper() for c in link if c.isalpha()):
                links_uppercase += 1

            # 7. Kiểm tra xem còn dính từ khóa bảo mật rác nào sót lại không
            # Thẩm định bằng Unicode NFC chuẩn hóa
            passage_nfc = unicodedata.normalize("NFC", passage).lower()
            
            has_junk_leak = False
            for pattern in CRAWLER_JUNK_PATTERNS:
                pattern_nfc = unicodedata.normalize("NFC", pattern).lower()
                if pattern_nfc in passage_nfc:
                    has_junk_leak = True
                    break

            if has_junk_leak:
                # Nếu dính bất kỳ cụm rác chính xác nào
                leaked_security_popups.append(doc_id)
                
            # Quét cảnh báo False-Positive lành tính cho từ đơn lẻ "đăng nhập" ngoài cụm rác thực tế
            elif "đăng nhập" in passage_nfc and len(passage_nfc.split()) < 300: 
                # Nếu văn bản quá ngắn mà dính "đăng nhập", rất dễ là file rác cụt đầu
                potential_false_positives.append(doc_id)

    print("\n--------------------------------------------------")
    print(" THẨM ĐỊNH CHI TIẾT:")
    print("--------------------------------------------------")
    
    # Check các điều kiện
    passed_lines = n_lines == 8532
    passed_keys = all_keys_valid
    passed_ids = all_ids_are_str
    passed_newlines = raw_newlines_found == 0
    passed_links = links_uppercase == 0
    passed_empty_names = empty_names == 1125
    passed_empty_passages = empty_passages == 20
    passed_security_junk = len(leaked_security_popups) == 0

    print(f"  - Tổng số dòng: {n_lines} " + ("(Chuẩn 8532 dòng)" if n_lines == 8532 else "(Sai số lượng dòng!)"))
    print(f"  - Cấu trúc khóa: " + ("Đầy đủ 100% (id, name, link, passage)" if all_keys_valid else "Có file bị thiếu khóa"))
    print(f"  - Kiểu dữ liệu ID: " + ("100% là chuỗi (str) - Chống bẫy 0 điểm im lặng" if all_ids_are_str else "Phát hiện ID kiểu số (int)!"))
    print(f"  - Làm phẳng văn bản (\\r\\n\\n): " + ("Hoàn hảo 100%" if raw_newlines_found == 0 else f"Còn {raw_newlines_found} dòng bị dính ký tự rác"))
    print(f"  - Đồng bộ URL (lowercase): " + ("Đã chuyển viết thường hoàn chỉnh" if links_uppercase == 0 else f"Còn {links_uppercase} link chứa chữ viết hoa"))
    print(f"  - Thống kê file khuyết name: {empty_names} file " + ("(Khớp đúng thống kê EDA ~13.2% khuyết)" if empty_names == 1125 else "Lệch số lượng khuyết name!"))
    print(f"  - Thống kê file rỗng passage: {empty_passages} file " + ("(Khớp đúng 20 file siêu lỗi gốc)" if empty_passages == 20 else "Lệch số lượng rỗng!"))
    print(f"  - Quét sạch rác Crawler bảo mật: " + ("Sạch 100% (0/8532 file còn dính cụm rác đã biết)" if passed_security_junk else f"Còn sót {len(leaked_security_popups)} file dính rác gốc: {leaked_security_popups}"))
    print("-" * 50)

    if len(potential_false_positives) > 0:
        print(f"  - Nghi ngờ False-Positive (Soi tay): Có {len(potential_false_positives)} file chứa từ 'đăng nhập' ({potential_false_positives[:10]})")
    print("-" * 50)

    # Đánh giá tổng quan
    success = passed_lines and passed_keys and passed_ids and passed_newlines and passed_links and passed_empty_names and passed_empty_passages and passed_security_junk
    if success:
        print("\nKẾT LUẬN: clean_corpus SẠCH THEO KẾ HOẠCH RÀNG BUỘC (CODA-READY)!")
        print("="*80)
        return True
    else:
        print("\nCẢNH BÁO: Dữ liệu chưa hoàn toàn sạch, vui lòng kiểm tra lại parser.")
        print("="*80)
        return False


# ===========================================================================
# 3. CORE PARSER
# ===========================================================================

def parse_corpus(corpus_dir: Path, out_path: Path):
    """
    Hàm xử lý cốt lõi quét qua toàn bộ file context thô và gộp thành file .jsonl sạch
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu thô tại: {corpus_dir}")
        
    print(f"Đang quét các file thô trong thư mục: {corpus_dir}")
    files = sorted(corpus_dir.glob("context_*.json"))
    total_files = len(files)
    print(f"Phát hiện {total_files} file context thô.")
    
    # Tạo thư mục cha chứa file out nếu chưa tồn tại
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as in_f:
                try:
                    doc = json.load(in_f)
                except Exception as e:
                    print(f"Lỗi đọc file {fp.name}: {e}")
                    continue
                
                # 1. Ép kiểu str cho id ngay tại điểm đọc (Phòng bẫy 0 điểm im lặng)
                raw_id = doc.get("id")
                if raw_id is None:
                    print(f"File {fp.name} bị lỗi: Khuyết trường 'id'!")
                    continue
                doc_id = str(raw_id)
                
                # 2. Xử lý khuyết name (Khuyết 13.2%: quy về chuỗi rỗng "", tránh "ô nhiễm từ vựng")
                name = doc.get("name")
                if name is None or not str(name).strip():
                    clean_name = ""
                else:
                    clean_name = str(name).strip()
                
                # 3. Xử lý khuyết passage/text (0.2% khuyết)
                passage = doc.get("passage")
                if passage is None:
                    passage = doc.get("text") or ""
                
                # Làm sạch văn bản bằng Regex
                clean_passage = clean_text(str(passage))
                
                # 4. Chuẩn hóa link về chữ thường
                link = str(doc.get("link") or "").strip().lower()
                
                # Khởi tạo bản ghi sạch đúng theo INTERFACES.md
                clean_record = {
                    "id": doc_id,
                    "name": clean_name,
                    "link": link,
                    "passage": clean_passage
                }
                
                # Ghi stream từng dòng JSONL
                out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                count += 1
                
    print(f" Đã ghi thành công {count} dòng vào file: {out_path}")
    
    # 5. Xác minh bắt buộc (SHA-256 Checksum)
    print(" Đang tính toán mã vân tay SHA-256...")
    checksum = calculate_sha256(out_path)
    print("\n" + "=" * 80)
    print(f" BÁO CÁO NGHIỆM THU TIỀN XỬ LÝ (P2) CODA-READY:")
    print(f"  - Tổng số dòng ghi được: {count} / 8532 dòng")
    print(f"  - SHA-256 Checksum: {checksum}")
    print("=" * 80)
    
    # 6. Kích hoạt bộ QA kiểm định tự động chuyên sâu
    verify_processed_corpus(out_path)
    
    print("Đưa 2 thông số (Số dòng và mã SHA-256 Checksum) dán vào file README.md mục 8!")


# ===========================================================================
# 4. CLI ENTRYPOINT
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Bộ tiền xử lý và gộp kho văn bản thô cho P2")
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/selected-contexts"),
                    help="Thư mục chứa các file context_*.json gốc")
    ap.add_argument("--out", type=Path, default=Path("data/corpus_clean.jsonl"),
                    help="Đường dẫn file đầu ra sau khi làm sạch")
    args = ap.parse_args()
    
    parse_corpus(args.corpus_dir, args.out)


if __name__ == "__main__":
    main()
