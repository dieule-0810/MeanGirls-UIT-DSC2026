"""Điểm thưởng theo tiêu đề văn bản — xếp lại top-k ĐÃ CÓ, không index lại.

Vì sao: corpus_clean.jsonl mang trường `name` là slug URL KHÔNG DẤU
("Thong-tu-17-2022-TT-BGTVT-..."), thường chứa số hiệu văn bản. Trường này hiện
không được dùng ở bất kỳ đâu trong đường truy hồi: chunks.jsonl không mang nó, và
chunker.py lẫn bm25.py không đọc nó lần nào.

Hai thí nghiệm trước đều KHÔNG bác bỏ được giá trị của nó, dù trông như vậy:

  - A/B `--prepend-name` ở tầng RERANK cho R@5 giống hệt (0,8206). Nhưng cross-encoder
    đọc slug viết tắt không dấu như ngôn ngữ tự nhiên, với nó gần như là nhiễu. Kết
    luận đó chỉ áp cho tầng nơ-ron.
  - Không tokenizer nào trong lưới cứu được: `fold_tone` trong tokenizers.py chỉ quy
    chuẩn VỊ TRÍ dấu thanh (hoà↔hòa), KHÔNG tước dấu. Nên câu hỏi có dấu và slug không
    dấu là bất khả khớp ở tầng BM25, bất kể P3 chọn tokenizer nào.

Nên cách duy nhất để trường này lên tiếng là chuẩn hoá cả hai phía về CÙNG một dạng
(tước dấu) rồi cộng thưởng ở tầng tính điểm:

    điểm mới = điểm cũ × (1 + λ · |token(câu hỏi) ∩ token(name)| / |token(name)|)

Mẫu số là số token của name, không phải của câu hỏi: ta muốn thưởng cho việc name được
câu hỏi PHỦ HẾT, chứ không thưởng cho name dài ăn may trùng vài chữ.

Thao tác này chỉ xếp lại chính tập ứng viên đã có nên R@50 bất biến — toàn bộ mức tăng
nằm ở phần nông. Đó cũng là trần của nó: không cứu được câu mà BM25 đã bỏ sót.

RỦI RO cần ghi vào phần hạn chế: tín hiệu này thưởng cho câu hỏi có trích số hiệu văn
bản. 12,9% văn bản đã rỗng `name`. Nếu private-test có tỷ lệ rỗng cao hơn hoặc slug
định dạng khác, khoản lời co lại. Chạy --breakdown để xem mức tăng phân bố thế nào.

    # quét λ trên dev
    python -m scripts.p4_name_prior --ranking outputs/p4_bm25_ranking/dev_top50_mean_top2.json \
        --questions data/dev.json --sweep

    # kiểm chéo 2 nửa: λ có phải hiện vật của việc chỉnh trên chính tập đo không
    python -m scripts.p4_name_prior --ranking ... --questions data/dev.json --cv

    # chốt λ rồi xuất ranking mới
    python -m scripts.p4_name_prior --ranking ... --questions data/dev.json \
        --lam 0.20 --out outputs/p4_name_prior/dev_namebonus.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

_WORD = re.compile(r"\w+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Tước HẲN dấu tiếng Việt. KHÁC với fold_tone_placement trong tokenizers.py,
    thứ chỉ quy chuẩn vị trí dấu thanh chứ không bỏ dấu."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "d")


def toks(text: str, min_len: int = 3) -> set[str]:
    """Bỏ token ngắn: 'tt', 'qd', 'va' xuất hiện ở hầu hết slug nên không phân biệt được gì."""
    return {t for t in _WORD.findall(strip_diacritics(text)) if len(t) >= min_len}


def load_names(corpus: str) -> dict[str, set[str]]:
    out = {}
    for line in open(corpus, encoding="utf-8"):
        d = json.loads(line)
        out[str(d["doc_id"])] = toks((d.get("name") or "").replace("-", " "))
    return out


def rescore(ranking: dict, questions: dict, names: dict, lam: float) -> dict:
    """Trả ranking mới, giữ nguyên cấu trúc phần tử để các script khác đọc được."""
    out = {}
    for qid in questions:
        qid = str(qid)
        qt = toks(questions[qid]["question"])
        rows = []
        for item in ranking[qid]:
            doc, score = str(item[0]), float(item[1])
            nm = names.get(doc, set())
            bonus = len(qt & nm) / len(nm) if nm else 0.0
            rows.append([doc, score * (1 + lam * bonus), *item[2:]])
        rows.sort(key=lambda r: -r[1])
        out[qid] = rows
    return out


def recall_at(ranking: dict, questions: dict, k: int, keys=None) -> float:
    keys = list(keys or questions)
    tot = 0.0
    for qid in keys:
        gold = {str(a) for a in questions[str(qid)]["answer"]}
        got = {str(r[0]) for r in ranking[str(qid)][:k]}
        tot += len(gold & got) / len(gold)
    return tot / len(keys)


GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--lam", type=float, default=0.20, help="hệ số điểm thưởng λ")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="quét λ và in bảng")
    ap.add_argument("--cv", action="store_true", help="kiểm chéo 2 nửa")
    ap.add_argument("--breakdown", action="store_true", help="tách mức tăng theo câu có/không trích số hiệu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if "holdout" in a.questions and (a.sweep or a.cv):
        raise SystemExit(
            "Không quét λ trên holdout. Chốt λ trên dev rồi mới chạy --lam trên holdout MỘT LẦN."
        )

    q = json.load(open(a.questions, encoding="utf-8"))
    ranking = json.load(open(a.ranking, encoding="utf-8"))
    names = load_names(a.corpus)
    n_empty = sum(1 for v in names.values() if not v)
    print(f"Corpus : {len(names):,} văn bản, {n_empty:,} ({100*n_empty/len(names):.1f}%) không có name dùng được")
    print(f"Tập đo : {a.questions} (n={len(q)})\n")

    base = rescore(ranking, q, names, 0.0)
    r0 = recall_at(base, q, a.k)

    if a.sweep:
        print(f"{'λ':>6} {'R@'+str(a.k):>9} {'Δ':>9}")
        print("─" * 26)
        for lam in GRID:
            r = recall_at(rescore(ranking, q, names, lam), q, a.k)
            print(f"{lam:>6.2f} {r:>9.4f} {r-r0:>+9.4f}")
        print()

    if a.cv:
        keys = sorted(q)
        halves = [(keys[::2], keys[1::2], "A→B"), (keys[1::2], keys[::2], "B→A")]
        print("Kiểm chéo 2 nửa — λ chọn trên nửa huấn luyện, đo trên nửa kia:")
        for tr, te, label in halves:
            cached = {lam: rescore(ranking, q, names, lam) for lam in GRID}
            best = max(GRID, key=lambda l: recall_at(cached[l], q, a.k, tr))
            b0 = recall_at(base, q, a.k, te)
            b1 = recall_at(cached[best], q, a.k, te)
            print(f"  {label}: λ*={best:.2f} | nửa kiểm tra {b0:.4f} → {b1:.4f} ({b1-b0:+.4f})")
        print()

    new = rescore(ranking, q, names, a.lam)
    r1 = recall_at(new, q, a.k)
    print(f"λ={a.lam:.2f}:  R@{a.k} {r0:.4f} → {r1:.4f}  ({r1-r0:+.4f})")
    for k in (1, 20, 50):
        if k == a.k:
            continue
        print(f"          R@{k:<2} {recall_at(base,q,k):.4f} → {recall_at(new,q,k):.4f}")
    print("  (R@50 bất biến là đúng — đây là xếp lại hạng, không mở rộng ứng viên)")

    if a.breakdown:
        pat = re.compile(r"\d+\s*/\s*\d{4}|\d{2,}/[A-ZĐ]{2,}")
        has = [k for k in q if pat.search(q[k]["question"])]
        hasnt = [k for k in q if k not in set(has)]
        print("\nPhân bố mức tăng:")
        for label, ks in (("câu CÓ trích số hiệu", has), ("câu KHÔNG trích", hasnt)):
            if not ks:
                continue
            b, n_ = recall_at(base, q, a.k, ks), recall_at(new, q, a.k, ks)
            print(f"  {label:<22} n={len(ks):>4}  {b:.4f} → {n_:.4f}  ({n_-b:+.4f})")
        print("  Nếu mức tăng dồn hết vào nhóm CÓ trích, đây là khai thác đặc thù định dạng")
        print("  chứ không phải cải tiến truy hồi tổng quát — phải nói rõ trong bài báo.")

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ {p}")


if __name__ == "__main__":
    main()
