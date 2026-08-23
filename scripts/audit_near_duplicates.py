"""
scripts/audit_near_duplicates.py — mở rộng phép kiểm nhất quán nhãn.

Vì sao cần: audit_labels.py --scan chỉ khớp CHÍNH XÁC nên chỉ soi được 26/7000
câu (0,37%). Trên phần soi được, 31% cặp bất đồng gold. Câu hỏi tiếp theo là:
tỉ lệ đó giữ nguyên khi mở rộng ra các cặp GẦN trùng không?

Mỗi cặp câu hỏi gần trùng là một phép kiểm miễn phí: nếu hai câu hỏi hỏi cùng
một thứ mà gold khác nhau, ít nhất một nhãn có vấn đề — kết luận này không cần
ai đọc văn bản luật, nên không nhiễm ý kiến chủ quan của người gán nhãn.

Thuần Python, không phụ thuộc sklearn/numpy (Python 3.14 chưa đủ wheel).

    python scripts/audit_near_duplicates.py --threshold 0.6
    python scripts/audit_near_duplicates.py --threshold 0.5 --export

Kết quả: outputs/label_audit/near_duplicates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train.json"
OUT = REPO / "outputs" / "label_audit"

# Từ chức năng — xuất hiện ở hầu hết câu hỏi pháp luật nên không mang thông tin
# phân biệt. Dùng để CHẶN khối ứng viên, không dùng khi tính độ tương đồng.
STOP = {
    "là", "gì", "nào", "được", "của", "và", "có", "cho", "trong", "khi", "thì",
    "các", "những", "về", "với", "theo", "quy", "định", "như", "thế", "ai",
    "bao", "lâu", "nhiêu", "phải", "không", "này", "đó", "một", "người",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) > 1}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def load_train() -> dict[str, dict]:
    if not TRAIN.exists():
        sys.exit(f"Không có {TRAIN}")
    raw = json.loads(TRAIN.read_text(encoding="utf-8"))
    return {
        str(k): {"question": v["question"], "gold": [str(a) for a in v["answer"]]}
        for k, v in raw.items()
    }


def build_blocks(train: dict, toks: dict) -> dict[str, list[str]]:
    """Chặn khối bằng chỉ mục ngược trên token HIẾM.

    So mọi cặp là 24,5 triệu phép so — không cần. Hai câu hỏi gần trùng gần như
    chắc chắn chia sẻ ít nhất một token hiếm, nên chỉ so trong khối là đủ.
    """
    df = Counter(t for s in toks.values() for t in s)
    n = len(train)
    # bỏ token xuất hiện ở >2% câu hỏi: chúng tạo khối khổng lồ mà không lọc được gì
    rare = {t for t, c in df.items() if c <= n * 0.02 and t not in STOP}

    idx = defaultdict(list)
    for qid, s in toks.items():
        for t in s & rare:
            idx[t].append(qid)
    return idx


def find_pairs(train: dict, threshold: float) -> list[dict]:
    toks = {qid: tokens(v["question"]) for qid, v in train.items()}
    idx = build_blocks(train, toks)

    seen: set[tuple[str, str]] = set()
    pairs = []
    for qids in idx.values():
        if len(qids) > 200:   # khối quá lớn = token không đủ hiếm, bỏ qua
            continue
        for a, b in combinations(sorted(qids), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sim = jaccard(toks[a], toks[b])
            if sim < threshold:
                continue
            ga, gb = set(train[a]["gold"]), set(train[b]["gold"])
            if ga == gb:
                status = "AGREE"
            elif ga & gb:
                status = "NESTED"      # lồng/giao nhau → nhãn THIẾU ở một bên
            else:
                status = "DISJOINT"    # rời nhau → mâu thuẫn trực tiếp
            pairs.append({
                "qid_a": a, "qid_b": b, "sim": round(sim, 3), "status": status,
                "question_a": train[a]["question"],
                "question_b": train[b]["question"],
                "gold_a": sorted(ga), "gold_b": sorted(gb),
            })
    return sorted(pairs, key=lambda p: -p["sim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="ngưỡng Jaccard. 1.0 = trùng hệt. Thử 0.5/0.6/0.7 và so.")
    ap.add_argument("--export", action="store_true", help="ghi file JSON đầy đủ")
    ap.add_argument("--show", type=int, default=10, help="in bao nhiêu ví dụ bất đồng")
    a = ap.parse_args()

    train = load_train()
    pairs = find_pairs(train, a.threshold)
    if not pairs:
        print(f"Không có cặp nào ≥ {a.threshold}. Hạ ngưỡng xuống thử.")
        return

    c = Counter(p["status"] for p in pairs)
    n = len(pairs)
    disagree = c["DISJOINT"] + c["NESTED"]

    print(f"\nNgưỡng Jaccard ≥ {a.threshold}")
    print(f"Số cặp câu hỏi gần trùng kiểm được: {n:,}")
    print(f"  (khớp chính xác chỉ cho 16 cặp — mở rộng {n/16:.0f}× phạm vi soi)\n")
    print(f"  AGREE     {c['AGREE']:>5}   {c['AGREE']/n*100:5.1f}%   gold giống hệt")
    print(f"  NESTED    {c['NESTED']:>5}   {c['NESTED']/n*100:5.1f}%   gold giao nhau → NHÃN THIẾU")
    print(f"  DISJOINT  {c['DISJOINT']:>5}   {c['DISJOINT']/n*100:5.1f}%   gold rời nhau → MÂU THUẪN")
    print(f"\n  Tổng bất đồng: {disagree:,}/{n:,} = {disagree/n*100:.1f}%")

    print("\n⚠️ Đây là tỉ lệ TRÊN CÁC CẶP GẦN TRÙNG, không phải trên toàn tập 7.000 câu.")
    print("   Câu hỏi gần trùng có thể nhiễu hơn mức trung bình (gán bởi nhiều người,")
    print("   nhiều thời điểm). Dùng để chứng minh hiện tượng TỒN TẠI và ước lượng")
    print("   độ lớn, KHÔNG dùng làm tỉ lệ nhiễu toàn cục.")

    print(f"\n{'─'*72}\nVÍ DỤ BẤT ĐỒNG (sim cao nhất):\n")
    shown = 0
    for p in pairs:
        if p["status"] == "AGREE" or shown >= a.show:
            continue
        shown += 1
        print(f"[{p['status']}] sim={p['sim']}")
        print(f"  {p['qid_a']}: {p['question_a'][:88]}")
        print(f"     gold {p['gold_a']}")
        print(f"  {p['qid_b']}: {p['question_b'][:88]}")
        print(f"     gold {p['gold_b']}")
        if p["status"] == "NESTED":
            extra = set(p["gold_b"]) ^ set(p["gold_a"])
            print(f"  → doc {sorted(extra)} được xác nhận hợp lệ ở một bên, THIẾU ở bên kia")
        print()

    if a.export:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"near_duplicates_t{a.threshold}.json"
        path.write_text(json.dumps({
            "threshold": a.threshold,
            "n_pairs": n,
            "counts": dict(c),
            "pct_disagree": round(disagree / n * 100, 2),
            "pairs": pairs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã ghi {path}")

    print("\nBƯỚC TIẾP: chạy lại với --threshold 0.5 và 0.7.")
    print("Nếu tỉ lệ bất đồng ỔN ĐỊNH qua các ngưỡng → hiện tượng thật, không phải")
    print("hiện vật của một ngưỡng cụ thể. Đó là số liệu đủ chắc để đưa vào bài báo.")


if __name__ == "__main__":
    main()
