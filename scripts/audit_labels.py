"""
scripts/audit_labels.py — dò nhiễu nhãn trực tiếp trên train.json.

Chủ sở hữu: P4. KHÔNG phụ thuộc mô hình, chạy được ngay từ Tuần 1.

Vì sao quan trọng: nếu X% nhãn có vấn đề thì Recall trần thật là (1 − X%), và
mọi nỗ lực đẩy Recall vượt mốc đó là lãng phí. Không ai khác trong team đo được
con số này, và nó là phát hiện mạnh nhất P4 mang vào bài báo.

    python scripts/audit_labels.py --scan          # dò tự động, in tổng quan
    python scripts/audit_labels.py --sample 30     # xuất 30 câu để đọc tay
    python scripts/audit_labels.py --report        # tổng hợp sau khi đọc

Kết quả: outputs/label_audit/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train.json"
CORPUS = REPO / "data" / "corpus_clean.jsonl"
OUT = REPO / "outputs" / "label_audit"


def norm(s: str) -> str:
    """Chuẩn hoá để so câu hỏi gần trùng: NFC, thường, bỏ dấu câu, gộp khoảng trắng."""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_train() -> dict[str, dict]:
    if not TRAIN.exists():
        sys.exit(f"Không có {TRAIN}")
    raw = json.loads(TRAIN.read_text(encoding="utf-8"))
    return {
        str(k): {"question": v["question"], "gold": [str(a) for a in v["answer"]]}
        for k, v in raw.items()
    }


def load_corpus() -> dict[str, dict]:
    """Tuỳ chọn — nếu P2 chưa sinh corpus_clean thì bỏ qua các kiểm tra cần text."""
    if not CORPUS.exists():
        print(f"⚠️  Chưa có {CORPUS} — bỏ qua kiểm tra cần nội dung văn bản.\n")
        return {}
    out = {}
    for line in CORPUS.open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            out[str(d.get("doc_id", d.get("id")))] = d
    return out


# ─────────────────────────────────────────────────────────────── các phép dò ──
def find_conflicts(train: dict) -> list[dict]:
    """Câu hỏi TRÙNG NHAU (sau chuẩn hoá) nhưng gold KHÁC NHAU.

    Đây là bằng chứng cứng nhất về nhiễu nhãn: cùng một câu hỏi không thể có hai
    đáp án đúng khác nhau. Ít nhất một trong hai sai — hoặc cả hai đều đúng, và
    khi đó nhãn đang thiếu.
    """
    groups = defaultdict(list)
    for qid, v in train.items():
        groups[norm(v["question"])].append(qid)

    out = []
    for key, qids in groups.items():
        if len(qids) < 2:
            continue
        golds = {qid: frozenset(train[qid]["gold"]) for qid in qids}
        if len(set(golds.values())) > 1:
            out.append({
                "kind": "CONFLICT",
                "question": train[qids[0]]["question"],
                "qids": qids,
                "golds": {q: sorted(golds[q]) for q in qids},
                "disjoint": len(set.intersection(*[set(g) for g in golds.values()])) == 0,
            })
    return out


def find_dup_consistent(train: dict, conflicts: list[dict]) -> int:
    """Câu hỏi trùng nhưng gold GIỐNG nhau (sau khi đã loại trừ các qid thuộc nhóm xung đột)."""
    conflicted = {q for c in conflicts for q in c["qids"]}
    groups = defaultdict(list)
    for qid, v in train.items():
        if qid not in conflicted:
            groups[norm(v["question"])].append(qid)
    return sum(len(q) - 1 for q in groups.values() if len(q) > 1)


def find_broken_gold(train: dict, corpus: dict) -> list[dict]:
    """Gold doc không có trong corpus, hoặc có nhưng rỗng/gần rỗng."""
    if not corpus:
        return []
    out = []
    for qid, v in train.items():
        for g in v["gold"]:
            doc = corpus.get(g)
            if doc is None:
                out.append({"kind": "MISSING_DOC", "qid": qid, "doc_id": g})
                continue
            text = (doc.get("passage") or doc.get("text") or "").strip()
            if len(text) < 200:
                out.append({"kind": "EMPTY_DOC", "qid": qid, "doc_id": g, "n_chars": len(text)})
    return out


def find_overloaded_docs(train: dict) -> list[tuple[str, int]]:
    """Doc được gán làm gold cho rất nhiều câu hỏi khác nhau.

    Một văn bản trả lời được 40 câu hỏi thì hoặc nó là văn bản khung rất rộng,
    hoặc nhãn đang bị gán ẩu về một doc mặc định. Cả hai đều đáng nhìn.
    """
    c = Counter(g for v in train.values() for g in v["gold"])
    return [(d, n) for d, n in c.most_common(20) if n >= 10]


# ────────────────────────────────────────────────────────────────── lệnh ──
def cmd_scan(train, corpus):
    n = len(train)
    print(f"train.json: {n:,} câu hỏi\n")

    conflicts = find_conflicts(train)
    n_q_conflict = sum(len(c["qids"]) for c in conflicts)
    print(f"1. XUNG ĐỘT NHÃN (câu hỏi trùng, gold khác)")
    print(f"   {len(conflicts):,} nhóm, liên quan {n_q_conflict:,} câu ({n_q_conflict/n*100:.2f}%)")
    n_disjoint = sum(1 for c in conflicts if c["disjoint"])
    print(f"   Trong đó {n_disjoint:,} nhóm có gold RỜI NHAU HOÀN TOÀN — mâu thuẫn trực tiếp.")

    dup = find_dup_consistent(train, conflicts)
    print(f"\n2. CÂU HỎI LẶP (gold giống nhau): {dup:,} bản sao thừa ({dup/n*100:.2f}%)")
    print(f"   → cỡ tập huấn luyện thực tế chỉ khoảng {n-dup:,} câu độc lập.")

    broken = find_broken_gold(train, corpus)
    if corpus:
        bc = Counter(b["kind"] for b in broken)
        print(f"\n3. GOLD DOC HỎNG")
        print(f"   Không có trong corpus : {bc.get('MISSING_DOC', 0):,}")
        print(f"   Rỗng / <200 ký tự     : {bc.get('EMPTY_DOC', 0):,}")

    over = find_overloaded_docs(train)
    if over:
        print(f"\n4. DOC BỊ GÁN GOLD QUÁ NHIỀU (≥10 câu)")
        for d, k in over[:10]:
            name = (corpus.get(d, {}).get("name") or "?")[:55] if corpus else "?"
            print(f"   [{d}] {k:>3} câu   {name}")

    print("\n" + "─" * 70)
    lower = n_q_conflict / n
    print(f"CẬN DƯỚI tỉ lệ nhãn có vấn đề: {lower*100:.2f}%")
    print("Đây chỉ là phần dò được TỰ ĐỘNG. Phần lớn nhiễu nhãn (gold không trả lời")
    print("được câu hỏi, hoặc có doc khác cũng trả lời được) chỉ phát hiện bằng mắt.")
    print(f"→ chạy --sample 30 để ước lượng phần còn lại.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scan.json").write_text(
        json.dumps({
            "n_questions": n,
            "n_conflict_groups": len(conflicts),
            "n_questions_in_conflict": n_q_conflict,
            "pct_conflict": round(n_q_conflict / n * 100, 3),
            "n_disjoint_groups": n_disjoint,
            "n_duplicate_questions": dup,
            "broken_gold": Counter(b["kind"] for b in broken) if corpus else None,
            "conflicts": conflicts[:50],
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nChi tiết: {OUT/'scan.json'}")


FIELDS = ["qid", "verdict", "confidence", "note", "question",
          "gold_ids", "gold_names", "gold_links"]

HOLDOUT = TRAIN.parent / "holdout.json"   # KHÔNG BAO GIỜ lấy mẫu từ đây


def already_read() -> set[str]:
    """QID đã đọc ở các vòng trước (manual_audit_r*.csv).

    Các vòng phải CỘNG DỒN, không xáo lại: đổi cỡ mẫu hay đổi pool đều làm
    random.sample cho kết quả hoàn toàn khác, và công đọc vòng trước mất trắng.
    """
    seen: set[str] = set()
    for f in sorted(OUT.glob("manual_audit_r*.csv")):
        with f.open(encoding="utf-8") as fh:
            seen |= {r["qid"] for r in csv.DictReader(fh) if r.get("qid")}
    return seen


def holdout_qids() -> set[str]:
    """Tập câu phải loại khỏi mọi vòng đọc tay.

    error_taxonomy.md: chỉ đọc error_pool, KHÔNG đọc holdout. Nhưng cmd_sample
    lấy mẫu từ train.json — vốn chứa CẢ holdout. Vòng 30 câu đầu đã dính 2 câu
    (62912, 119006) vì thiếu bộ lọc này. Ở cỡ mẫu 150 sẽ là ~21 câu.
    """
    if not HOLDOUT.exists():
        sys.exit(f"Không có {HOLDOUT} — không thể bảo đảm mẫu sạch, dừng.")
    return set(json.loads(HOLDOUT.read_text(encoding="utf-8")))

VERDICTS = {
    "OK":        "gold trả lời trực tiếp câu hỏi, không thấy vấn đề",
    "N-ALTOK":   "gold đúng, NHƯNG có văn bản khác cũng trả lời được",
    "N-CONSOLID":"gold trùng nội dung với một văn bản HỢP NHẤT cũng có trong kho",
    "N-VERSION": "nội dung gold ĐÃ BỊ SỬA bởi văn bản khác; bản sửa đúng hơn gold",
    "N-TRUNC":   "gold đúng, nhưng nội dung nằm ở file đính kèm / phụ lục không có trong corpus",
    "N-WRONG":   "gold KHÔNG trả lời được câu hỏi → nhãn sai",
    "N-AMBIG":   "câu hỏi mơ hồ, không xác định được doc nào đúng",
}
# N-WRONG là khẳng định về BTC; ba nhãn kia là về bài toán/dữ liệu. Tách khi báo cáo.
CEILING = ("N-ALTOK", "N-CONSOLID", "N-VERSION", "N-TRUNC", "N-AMBIG")


def cmd_sample(train, corpus, k: int, seed: int):
    """Xuất k câu NGẪU NHIÊN (không phải k câu xung đột) để đọc tay.

    Phải ngẫu nhiên trên toàn tập, nếu không thì tỉ lệ ước lượng được sẽ vô nghĩa:
    lấy mẫu từ nhóm xung đột chỉ cho biết nhóm xung đột trông thế nào.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manual_audit.csv"
    if path.exists():
        sys.exit(f"{path} đã tồn tại — xoá nếu muốn lấy mẫu lại (sẽ mất nhãn đã ghi).")

    hold, seen = holdout_qids(), already_read()
    pool = sorted(set(train) - hold - seen)
    if len(pool) < k:
        sys.exit(f"Chỉ còn {len(pool)} câu ngoài holdout, không đủ {k}.")
    rng = random.Random(seed)
    qids = rng.sample(pool, k)
    assert not (set(qids) & hold), "Rò rỉ holdout — dừng."

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for qid in qids:
            v = train[qid]
            names = " ‖ ".join(
                (corpus.get(g, {}).get("name") or "?")[:60] for g in v["gold"]
            ) if corpus else ""
            links = " | ".join(
                (corpus.get(g, {}).get("link") or "?") for g in v["gold"]
            ) if corpus else ""
            row = [qid, "", "", "", v["question"],
                   "|".join(v["gold"]), names, links]
            # fail-loud: thêm cột vào FIELDS mà quên thêm ô ở đây sẽ lệch
            # toàn bộ CSV một cách âm thầm — đúng loại lỗi đã xảy ra một lần.
            assert len(row) == len(FIELDS), f"{len(row)} ô nhưng FIELDS có {len(FIELDS)}"
            w.writerow(row)

    print(f"Đã xuất {k} câu → {path}")
    print(f"(pool {len(pool):,} câu — đã loại {len(hold):,} holdout"
          + (f" và {len(seen):,} câu đã đọc vòng trước" if seen else "") + ")\n")
    print("Mở bằng Excel/LibreOffice. Với MỖI câu, mở văn bản gold (link trong")
    print("corpus) và tự trả lời: văn bản này có thật sự trả lời được câu hỏi không?\n")
    print("Điền cột verdict một trong các giá trị:")
    for name, desc in VERDICTS.items():
        print(f"  {name:<10} — {desc}")
    print("\nCột confidence: high / medium / low. Quá 3 phút chưa quyết được → low.")
    print("Cột note BẮT BUỘC với mọi verdict khác OK — đó là bằng chứng.")
    print("Đọc 30 câu mất khoảng 90 phút. Đừng vội, con số này sẽ vào bài báo.")


def cmd_report():
    path = OUT / "manual_audit.csv"
    if not path.exists():
        sys.exit(f"Chưa có {path} — chạy --sample trước.")
    all_rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in all_rows if r.get("verdict", "").strip()]
    if not rows:
        sys.exit("Chưa điền cột verdict.")

    # fail-loud: verdict lạ là lỗi gõ, đừng đếm im lặng
    bad_v = sorted({r["verdict"].strip().upper() for r in rows} - set(VERDICTS))
    if bad_v:
        sys.exit(f"Verdict không hợp lệ: {bad_v}\nHợp lệ: {sorted(VERDICTS)}")

    n, n_all = len(rows), len(all_rows)
    c = Counter(r["verdict"].strip().upper() for r in rows)
    print(f"\nĐã đọc {n}/{n_all} câu.\n")
    for k in VERDICTS:
        if c.get(k):
            print(f"  {k:<10} {c[k]:>3}   {c[k]/n*100:5.1f}%   {VERDICTS[k]}")

    # confidence — quy tắc 20% của error_taxonomy.md
    conf = Counter((r.get("confidence") or "").strip().lower() for r in rows)
    low = conf.get("low", 0)
    if low:
        print(f"\nconfidence=low: {low}/{n} ({low/n*100:.1f}%)")
        if low / n > 0.20:
            print("  🔴 Quá 20% → taxonomy chưa đủ tốt. Sửa taxonomy TRƯỚC khi đọc tiếp.")
    if conf.get("", 0):
        print(f"\n⚠️ {conf['']} câu chưa điền confidence.")

    bad = n - c.get("OK", 0)
    p = bad / n
    se = (p * (1 - p) / n) ** 0.5
    lo, hi = max(0, p - 1.96 * se), min(1, p + 1.96 * se)
    print(f"\nTỉ lệ nhãn có vấn đề: {p*100:.1f}%  (KTC 95%: {lo*100:.1f}% – {hi*100:.1f}%)")
    print(f"→ Recall trần ước lượng: khoảng {(1-p)*100:.0f}%")

    n_ceil = sum(c.get(k, 0) for k in CEILING)
    n_wrong = c.get("N-WRONG", 0)
    print(f"\n  Trần bài toán ({len(CEILING)} nhãn không sửa được): {n_ceil} ({n_ceil/n*100:.1f}%)")
    print(f"    → gold đúng, nhưng bài toán mơ hồ hoặc dữ liệu cụt. Mô hình không sửa được.")
    print(f"  Nhãn sai thật sự (N-WRONG):                 {n_wrong} ({n_wrong/n*100:.1f}%)")
    print(f"    → khẳng định về nhãn BTC. Con số này cần chuẩn nhất trong bài báo.")

    print(f"\n⚠️ Cỡ mẫu {n} rất nhỏ, khoảng tin cậy rộng. Dùng để định hướng, KHÔNG")
    print("   dùng làm con số chốt trong bài báo. Muốn hẹp khoảng thì cần ~150 câu.")

    if n_wrong:
        print(f"\n🔴 {n_wrong} câu có nhãn SAI. Đây là khẳng định mạnh — kiểm tra lại")
        print("   từng câu cùng một người thứ hai trước khi đưa vào báo cáo.")


ATTACH_MARKERS = ("FILE ĐƯỢC ĐÍNH KÈM", "FILE ĐÍNH KÈM")
SHORT_CHARS = 2500


def cmd_scan_truncated(train, corpus):
    """Câu hỏi có gold bị cắt cụt: nội dung thật nằm ở file đính kèm, hoặc
    văn bản ngắn bất thường. Không mô hình nào giải được → đè trần Recall,
    nhưng KHÁC N-WRONG: nhãn đúng, dữ liệu thiếu. Phát hiện ở câu 10 của
    vòng đọc tay, sau đó tổng quát hoá thành phép quét này."""
    rows = []
    for qid, item in train.items():
        for g in item["gold"]:
            d = corpus.get(g)
            if not d:
                continue
            t = d.get("text") or d.get("passage") or ""
            up = t.upper()
            flags = []
            if len(t) < SHORT_CHARS:
                flags.append("SHORT")
            if any(a in up for a in ATTACH_MARKERS):
                flags.append("ATTACH")
            if flags:
                rows.append((qid, g, len(t), "+".join(flags), item["question"][:60]))

    n_q = len({r[0] for r in rows})
    n_short = len({r[0] for r in rows if "SHORT" in r[3]})
    print(f"\n{len(train):,} câu hỏi, {len(corpus):,} văn bản\n")
    print(f"Câu hỏi có gold bị gắn cờ: {n_q} ({n_q/len(train)*100:.2f}%)")
    print(f"  trong đó gold < {SHORT_CHARS:,} ký tự: {n_short} — gần như chắc chắn không giải được\n")

    for r in sorted(rows, key=lambda x: x[2])[:40]:
        print(f"  qid {r[0]:<8} doc {r[1]:<8} {r[2]:>7,} ký tự  {r[3]:<12} {r[4]}")
    if len(rows) > 40:
        print(f"  … còn {len(rows)-40} dòng")

    print("\nATTACH đơn thuần chỉ là CẬN TRÊN của rủi ro — nhiều văn bản có phụ lục")
    print("đính kèm nhưng phần thân vẫn trả lời được. SHORT mới là tín hiệu mạnh.")
    print("Chuyển danh sách SHORT cho P2: parse_corpus.py hiện chỉ lọc passage rỗng.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--scan-truncated", action="store_true", dest="scan_truncated")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.report:
        return cmd_report()
    train, corpus = load_train(), load_corpus()
    if a.scan_truncated:
        return cmd_scan_truncated(train, corpus)
    if a.scan:
        return cmd_scan(train, corpus)
    if a.sample:
        return cmd_sample(train, corpus, a.sample, a.seed)
    ap.print_help()


if __name__ == "__main__":
    main()