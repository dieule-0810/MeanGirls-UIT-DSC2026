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


def cmd_sample(train, corpus, k: int, seed: int):
    """Xuất k câu NGẪU NHIÊN (không phải k câu xung đột) để đọc tay.

    Phải ngẫu nhiên trên toàn tập, nếu không thì tỉ lệ ước lượng được sẽ vô nghĩa:
    lấy mẫu từ nhóm xung đột chỉ cho biết nhóm xung đột trông thế nào.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manual_audit.csv"
    if path.exists():
        sys.exit(f"{path} đã tồn tại — xoá nếu muốn lấy mẫu lại (sẽ mất nhãn đã ghi).")

    rng = random.Random(seed)
    qids = rng.sample(sorted(train), k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "verdict", "note", "question", "gold_ids", "gold_names"])
        for qid in qids:
            v = train[qid]
            names = " ‖ ".join(
                (corpus.get(g, {}).get("name") or "?")[:60] for g in v["gold"]
            ) if corpus else ""
            w.writerow([qid, "", "", v["question"], "|".join(v["gold"]), names])

    print(f"Đã xuất {k} câu → {path}\n")
    print("Mở bằng Excel/LibreOffice. Với MỖI câu, mở văn bản gold (link trong")
    print("corpus) và tự trả lời: văn bản này có thật sự trả lời được câu hỏi không?\n")
    print("Điền cột verdict một trong bốn giá trị:")
    print("  OK       — gold trả lời được, không thấy vấn đề")
    print("  N-ALTOK  — gold đúng, NHƯNG có văn bản khác cũng trả lời được")
    print("  N-WRONG  — gold KHÔNG trả lời được câu hỏi → nhãn sai")
    print("  N-AMBIG  — câu hỏi mơ hồ, không xác định được doc nào đúng")
    print("\nCột note BẮT BUỘC với mọi verdict khác OK — đó là bằng chứng.")
    print("Đọc 30 câu mất khoảng 90 phút. Đừng vội, con số này sẽ vào bài báo.")


def cmd_report():
    path = OUT / "manual_audit.csv"
    if not path.exists():
        sys.exit(f"Chưa có {path} — chạy --sample trước.")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["verdict"].strip()]
    if not rows:
        sys.exit("Chưa điền cột verdict.")

    n = len(rows)
    c = Counter(r["verdict"].strip().upper() for r in rows)
    print(f"\nĐã đọc {n} câu.\n")
    for k, v in c.most_common():
        print(f"  {k:<10} {v:>3}   {v/n*100:5.1f}%")

    bad = n - c.get("OK", 0)
    p = bad / n
    # sai số chuẩn nhị thức — cỡ mẫu 30 rất nhỏ, phải nói rõ khoảng
    se = (p * (1 - p) / n) ** 0.5
    lo, hi = max(0, p - 1.96 * se), min(1, p + 1.96 * se)

    print(f"\nTỉ lệ nhãn có vấn đề: {p*100:.1f}%  (KTC 95%: {lo*100:.1f}% – {hi*100:.1f}%)")
    print(f"→ Recall trần ước lượng: khoảng {(1-p)*100:.0f}%")
    print(f"\n⚠️ Cỡ mẫu {n} rất nhỏ, khoảng tin cậy rộng. Dùng để định hướng, KHÔNG")
    print("   dùng làm con số chốt trong bài báo. Muốn hẹp khoảng thì cần ~150 câu.")

    if c.get("N-WRONG", 0):
        print(f"\n🔴 {c['N-WRONG']} câu có nhãn SAI. Đây là khẳng định mạnh — kiểm tra lại")
        print("   từng câu cùng một người thứ hai trước khi đưa vào báo cáo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.report:
        return cmd_report()
    train, corpus = load_train(), load_corpus()
    if a.scan:
        return cmd_scan(train, corpus)
    if a.sample:
        return cmd_sample(train, corpus, a.sample, a.seed)
    ap.print_help()


if __name__ == "__main__":
    main()