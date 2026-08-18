"""
scripts/final_broad_pollution_scan.py
Quét rộng LẦN CUỐI trên corpus_clean.jsonl trước khi tag v0.1 release.
Không giới hạn <300 từ như ketquanghivan.txt (giới hạn đó chỉ để lọc bớt false-positive
khi soi tay theo dõi thường xuyên). Ở đây mục đích khác: tìm biến thể rác CHƯA từng biết,
nên quét toàn bộ, không lọc độ dài, để không bỏ sót file dài mà vẫn dính rác.
"""

import json
import unicodedata
from pathlib import Path
from collections import defaultdict

# Từ khóa rộng - CỐ Ý rộng hơn CRAWLER_JUNK_PATTERNS trong parse_corpus.py
# để dò biến thể rác chưa biết, không phải để confirm lại pattern đã biết
BROAD_KEYWORDS = ["đăng nhập", "mật khẩu", "quý khách", "rời quầy", "tài khoản của bạn"]

# 5 cụm đã biết + đã xử lý trong parse_corpus.py - dùng để loại các match đã confirm sạch
KNOWN_PATTERNS = [
    "quý khách vui lòng đăng nhập",
    "đăng nhập để xem",
    "rò rỉ mật khẩu",
    "vui lòng đăng nhập để",
    "đăng nhập để tiếp tục",
]

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
            passage_nfc = unicodedata.normalize("NFC", doc["passage"]).lower()

            if not passage_nfc:
                continue

            for kw in BROAD_KEYWORDS:
                if kw in passage_nfc:
                    idx = passage_nfc.find(kw)
                    context = doc["passage"][max(0, idx - 60): idx + 150]
                    is_known = already_known(passage_nfc)
                    hits_by_keyword[kw].append(doc["id"])
                    records.append({
                        "id": doc["id"],
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
        "n_unknown_candidates": n_unknown,  # cần soi tay: pattern rác KHÔNG thuộc 5 cụm đã biết
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
        print("   Lọc nhanh bằng: [r for r in report['records'] if not r['known_pattern']]")
    else:
        print("\n=> 0 ứng viên mới - không phát hiện biến thể rác nào chưa biết.")


if __name__ == "__main__":
    main()
