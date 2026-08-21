"""
scripts/final_broad_pollution_scan.py
Quét rộng LẦN CUỐI trên corpus_clean.jsonl trước khi tag release (đổi tên
file output theo version hiện hành, không hard-code "v0.1").
Không giới hạn độ dài văn bản khi soi (khác các lượt theo dõi thường xuyên trước đó, vốn
giới hạn <300 từ để lọc bớt false-positive). Ở đây mục đích khác: tìm biến thể rác CHƯA
từng biết, nên quét toàn bộ, không lọc độ dài, để không bỏ sót file dài mà vẫn dính rác.

Lưu ý: corpus_clean.jsonl tại thời điểm quét đã qua bước loại trừ 25 doc_id (20 rỗng +
5 trùng-dư, xem docs/exclusion_decisions.json) - n_scanned kỳ vọng là 8.507, không phải
8.532 như bản quét trước 20/8.
"""

import json
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

# Thêm project root vào sys.path để import được src.data.parse_corpus
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.parse_corpus import CRAWLER_JUNK_PATTERNS as KNOWN_PATTERNS

# Từ khóa rộng - CỐ Ý rộng hơn CRAWLER_JUNK_PATTERNS trong parse_corpus.py
# để dò biến thể rác chưa biết, không phải để confirm lại pattern đã biết
BROAD_KEYWORDS = ["đăng nhập", "mật khẩu", "quý khách", "rời quầy", "tài khoản của bạn"]


def already_known(passage_nfc: str) -> bool:
    return any(p in passage_nfc for p in KNOWN_PATTERNS)


def main():
    corpus_path = Path("data/corpus_clean.jsonl")
    out_path = Path("docs/final_pollution_scan_v0.1.json")

    hits_by_keyword = defaultdict(list)
    records = []
    n_scanned = 0

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            n_scanned += 1
            doc = json.loads(line)
            passage_nfc = unicodedata.normalize("NFC", doc["text"]).lower()

            if not passage_nfc:
                continue

            for kw in BROAD_KEYWORDS:
                if kw in passage_nfc:
                    idx = passage_nfc.find(kw)
                    context = doc["text"][max(0, idx - 60): idx + 150]
                    is_known = already_known(passage_nfc)
                    hits_by_keyword[kw].append(doc["doc_id"])
                    records.append({
                        "id": doc["doc_id"],
                        "keyword": kw,
                        "known_pattern": is_known,
                        "context": context,
                    })
                    break

    n_unknown = sum(1 for r in records if not r["known_pattern"])

    report = {
        "n_scanned": n_scanned,
        "n_hits_total": len(records),
        "n_hits_by_keyword": {k: len(v) for k, v in hits_by_keyword.items()},
        "n_unknown_candidates": n_unknown,
        "records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)

    print(f"Đã quét {n_scanned} dòng.")
    print(f"Số file trúng ít nhất 1 từ khóa rộng: {len(records)}")
    print(f"Chi tiết từng keyword: {report['n_hits_by_keyword']}")
    print(f"Số ứng viên MỚI (known_pattern=False, chưa thuộc 5 pattern đã biết): {n_unknown}")
    print(f"Ghi chi tiết vào: {out_path}")
    if n_unknown > 0:
        print("\n=> Còn ứng viên mới, cần soi tay để xác nhận rác thật hay false positive.")
    else:
        print("\n=> 0 ứng viên mới - không phát hiện biến thể rác nào chưa biết.")


if __name__ == "__main__":
    main()