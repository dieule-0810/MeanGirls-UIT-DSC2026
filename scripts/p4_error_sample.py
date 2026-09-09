"""Trích mẫu câu TRUY VẤN SAI để đọc tay — task Tuần 1 của P4.

KHÁC với `audit_labels.py`: script kia hỏi "gold BTC gán có đúng không?" (mã `N-*`),
script này hỏi "vì sao hệ thống trượt câu này?" (mã `R-*`). Hai việc khác nhau, hai
người nhận kết quả khác nhau.

TÁCH HAI LOẠI THẤT BẠI — quan trọng, đừng trộn:
    MISS50  không gold nào trong top-50  → thất bại BAO PHỦ  → của P3 (trần R@50)
    MISS5   có trong top-50, ngoài top-5 → thất bại XẾP HẠNG → của P4 (rerank/hợp nhất)
Trộn chung rồi báo cáo một tỉ lệ là làm cả hai người không biết phần nào của mình.

⚠️ LẤY MẪU VƯỢT TỈ LỆ. Trên dev n=1000 chỉ có 38 câu MISS50 so với 193 MISS5. Lấy đúng
tỉ lệ thì mẫu 50 câu chỉ còn 8 câu MISS50 — không đủ nói gì với P3. Nên script lấy
vượt và IN RA hệ số quy đổi. Mọi % trong báo cáo phải nhân lại hệ số đó, nếu không sẽ
thổi phồng tỉ lệ lỗi bao phủ.

Sinh 2 file: CSV để điền verdict, HTML để đọc (có tên văn bản, link, và đoạn khớp).

    python -m scripts.p4_error_sample \\
        --ranking outputs/p4_bm25_ranking/dev_top50.json --questions data/dev.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CODES = ["R-TERM", "R-SEG", "R-NUM", "R-SHORT", "R-DUP", "R-ENTITY", "N-*", "khac"]


def tier(f: int) -> str:
    if f == 0:
        return "freq=0"
    if f <= 2:
        return "freq=1-2"
    if f <= 10:
        return "freq=3-10"
    return "freq>=11"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--chunks-file", default="data/chunks.jsonl")
    ap.add_argument("--train-split", default="data/train_split.json")
    ap.add_argument("--out-dir", default="outputs/error_audit")
    ap.add_argument("--n-miss5", type=int, default=30, help="mẫu lỗi XẾP HẠNG (P4)")
    ap.add_argument("--n-miss50", type=int, default=20, help="mẫu lỗi BAO PHỦ (P3)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    rank = json.load(open(a.ranking, encoding="utf-8"))
    freq = Counter(
        str(x)
        for v in json.load(open(a.train_split, encoding="utf-8")).values()
        for x in v["answer"]
    )
    meta = {}
    for line in open(a.corpus, encoding="utf-8"):
        d = json.loads(line)
        meta[str(d["doc_id"])] = ((d.get("name") or "").strip(), d.get("link", ""))

    # phân loại
    modes: dict[str, list[str]] = defaultdict(list)
    for qid, v in q.items():
        gold = {str(x) for x in v["answer"]}
        t5 = {r[0] for r in rank[str(qid)][:5]}
        t50 = {r[0] for r in rank[str(qid)][:50]}
        if gold <= t5:
            modes["OK"].append(str(qid))
        elif not (gold & t50):
            modes["MISS50"].append(str(qid))
        else:
            modes["MISS5"].append(str(qid))

    n = len(q)
    print(f"{n} câu: OK {len(modes['OK'])} · MISS5 {len(modes['MISS5'])} "
          f"· MISS50 {len(modes['MISS50'])}")

    rng = random.Random(a.seed)
    picked: list[tuple[str, str]] = []
    for mode, want in (("MISS5", a.n_miss5), ("MISS50", a.n_miss50)):
        pool = sorted(modes[mode])
        rng.shuffle(pool)
        take = pool[:want]
        picked += [(qid, mode) for qid in take]
        share_pop = len(modes[mode]) / n
        share_sam = len(take) / (a.n_miss5 + a.n_miss50)
        print(f"  {mode}: lấy {len(take)}/{len(modes[mode])}  "
              f"tỉ lệ thật {share_pop:.1%} → trong mẫu {share_sam:.1%}  "
              f"HỆ SỐ QUY ĐỔI ×{share_pop/share_sam:.3f}")

    # đoạn văn bản của gold và của top-5, để đọc được vì sao trượt
    need = set()
    for qid, _ in picked:
        for r in rank[qid][:5]:
            need.add(r[2])
        for r in rank[qid]:
            if r[0] in {str(x) for x in q[qid]["answer"]}:
                need.add(r[2])
    texts = {}
    for line in open(a.chunks_file, encoding="utf-8"):
        d = json.loads(line)
        if d["chunk_id"] in need:
            texts[d["chunk_id"]] = d["text"]
            if len(texts) == len(need):
                break

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for qid, mode in picked:
        gold = [str(x) for x in q[qid]["answer"]]
        pos = {r[0]: i + 1 for i, r in enumerate(rank[qid])}
        rows.append({
            "qid": qid,
            "mode": mode,
            "tier": tier(min(freq.get(g, 0) for g in gold)),
            "gold_rank": ";".join(str(pos.get(g, ">50")) for g in gold),
            "code": "",
            "confidence": "",
            "note": "",
            "question": q[qid]["question"],
            "gold_ids": ";".join(gold),
            "gold_names": " | ".join(meta.get(g, ("", ""))[0] for g in gold),
            "gold_links": " | ".join(meta.get(g, ("", ""))[1] for g in gold),
        })
    csv_p = out / "error_audit_r1.csv"
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # HTML để đọc
    def esc(s: str) -> str:
        return html.escape(s or "")

    parts = ["""<meta charset="utf-8"><style>
body{font:15px/1.6 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}
.q{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.5rem 0}
.hd{font-weight:700;font-size:1.05rem}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.8rem;
 font-weight:700;color:#fff}
.m5{background:#b45309}.m50{background:#b91c1c}
.gold{background:#ecfdf5;border-left:4px solid #059669;padding:.5rem;margin:.4rem 0}
.pred{background:#f8fafc;border-left:4px solid #94a3b8;padding:.5rem;margin:.4rem 0}
.txt{color:#444;font-size:.88rem;white-space:pre-wrap}
.meta{color:#666;font-size:.85rem}code{background:#eee;padding:0 .3rem}
</style><h1>Mẫu lỗi truy vấn — vòng 1</h1>
<p><b>MISS5</b> = gold có trong top-50 nhưng ngoài top-5 → lỗi XẾP HẠNG (P4).<br>
<b>MISS50</b> = không gold nào trong top-50 → lỗi BAO PHỦ (P3).</p>
<p>Mã: <code>R-TERM</code> từ thường dân vs thuật ngữ · <code>R-SEG</code> tách từ sai
· <code>R-NUM</code> số hiệu không được ưu tiên · <code>R-SHORT</code> câu quá chung
· <code>R-DUP</code> trả về bản gần trùng · <code>R-ENTITY</code> đúng khung điều luật
sai lĩnh vực · <code>N-*</code> nghi nhãn sai, chuyển sang audit nhãn.</p>
<p><b>3 phút/câu.</b> Quá thì ghi <code>confidence=low</code> và đi tiếp. Không bỏ câu
khó — bỏ là tự lọc mẫu.</p>"""]

    for qid, mode in picked:
        gold = [str(x) for x in q[qid]["answer"]]
        pos = {r[0]: i + 1 for i, r in enumerate(rank[qid])}
        cls = "m5" if mode == "MISS5" else "m50"
        parts.append(f'<div class="q"><div class="hd">'
                     f'<span class="badge {cls}">{mode}</span> '
                     f'qid {esc(qid)} — {esc(q[qid]["question"])}</div>')
        for g in gold:
            nm, lk = meta.get(g, ("", ""))
            p = pos.get(g)
            ct = ""
            for r in rank[qid]:
                if r[0] == g:
                    ct = texts.get(r[2], "")[:600]
                    break
            parts.append(
                f'<div class="gold"><b>GOLD {esc(g)}</b> — hạng '
                f'{p if p else "&gt;50"}<br><span class="meta">{esc(nm)}</span>'
                + (f' · <a href="{esc(lk)}">nguồn</a>' if lk else "")
                + (f'<div class="txt">{esc(ct)}</div>' if ct else
                   '<div class="txt">(không lọt top-50, không có đoạn)</div>')
                + "</div>")
        parts.append("<div class=meta><b>Top-5 hệ thống trả về:</b></div>")
        for i, r in enumerate(rank[qid][:5]):
            nm, lk = meta.get(r[0], ("", ""))
            mark = " ✅" if r[0] in gold else ""
            parts.append(
                f'<div class="pred">{i+1}. <b>{esc(r[0])}</b>{mark} '
                f'<span class="meta">{esc(nm)}</span>'
                f'<div class="txt">{esc(texts.get(r[2], "")[:400])}</div></div>')
        parts.append("</div>")

    html_p = out / "error_packet.html"
    html_p.write_text("\n".join(parts), encoding="utf-8")

    print(f"\n✅ {csv_p}  ({len(rows)} câu — điền cột code, confidence, note)")
    print(f"✅ {html_p}  (mở bằng trình duyệt để đọc)")
    print("\n⚠️ Mọi % tính từ CSV này là % TRONG TỪNG NHÓM. Khi báo cáo tỉ lệ trên toàn\n"
          "   tập, nhân lại hệ số quy đổi in ở trên — nếu không sẽ thổi phồng lỗi bao phủ.")


if __name__ == "__main__":
    main()
