"""
scripts/build_audit_packet.py — sinh gói đọc cho vòng audit nhãn thủ công.

Vấn đề: văn bản luật dài 20–50k ký tự. Đọc đủ 30 văn bản mất ~6 tiếng.
Giải: trích các đoạn có mật độ từ khoá của câu hỏi cao nhất, kèm đoạn đầu
(quốc hiệu/tiêu đề để biết đây là văn bản gì), tô đậm từ khớp.

    python scripts/build_audit_packet.py --n 30 --seed 42

Seed và n PHẢI trùng với audit_labels.py --sample, nếu không CSV và gói đọc
lệch nhau.

Nguồn văn bản: ưu tiên data/corpus_clean.jsonl (nếu P2 đã sinh), nếu chưa thì
đọc thẳng selected-contexts của BTC.

Kết quả: outputs/label_audit/packet.html — mở bằng trình duyệt.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train.json"
CORPUS = REPO / "data" / "corpus_clean.jsonl"
RAW_DIRS = [
    REPO / "data" / "raw" / "selected-contexts",
    REPO / "data" / "selected-contexts",
    REPO / "data" / "raw",
]
OUT = REPO / "outputs" / "label_audit"

STOP = {
    "là", "gì", "nào", "được", "của", "và", "có", "cho", "trong", "khi", "thì",
    "các", "những", "về", "với", "theo", "như", "thế", "ai", "bao", "lâu",
    "nhiêu", "phải", "không", "này", "đó", "một", "người", "ra", "sao", "hay",
    "bị", "đến", "từ", "sẽ", "mà", "nếu", "hoặc", "cũng", "còn", "đã", "tại",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def keywords(q: str) -> list[str]:
    """Từ nội dung của câu hỏi, dài trước — từ dài thường mang nhiều thông tin hơn."""
    ks = [t for t in norm(q).split() if len(t) > 1 and t not in STOP]
    return sorted(set(ks), key=len, reverse=True)


# ────────────────────────────────────────────────────────────── nạp dữ liệu ──
def load_docs() -> dict[str, dict]:
    if CORPUS.exists():
        print(f"Đọc {CORPUS}")
        out = {}
        for line in CORPUS.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                out[str(d.get("doc_id", d.get("id")))] = d
        return out

    for base in RAW_DIRS:
        files = sorted(base.glob("context_*.json")) if base.exists() else []
        if files:
            print(f"Chưa có corpus_clean.jsonl — đọc thẳng {len(files)} file từ {base}")
            out = {}
            for f in files:
                data = json.loads(f.read_text(encoding="utf-8"))
                for d in (data if isinstance(data, list) else [data]):
                    out[str(d["id"])] = d
            return out

    sys.exit(
        "Không tìm thấy văn bản.\n"
        "Đặt selected-contexts đã giải nén vào data/raw/selected-contexts/,\n"
        "hoặc chờ P2 sinh data/corpus_clean.jsonl, rồi chạy lại."
    )


def load_train() -> dict[str, dict]:
    raw = json.loads(TRAIN.read_text(encoding="utf-8"))
    return {
        str(k): {"question": v["question"], "gold": [str(a) for a in v["answer"]]}
        for k, v in raw.items()
    }


# ──────────────────────────────────────────────────────────── trích đoạn ──
def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r", "")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # gộp đoạn quá ngắn vào đoạn sau để không vụn
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip() if len(buf) < 80 else buf
        if len(buf) >= 80:
            out.append(buf)
            buf = ""
        elif buf == "":
            buf = p
    if buf:
        out.append(buf)
    return out or [text]


def score_para(p: str, ks: list[str]) -> int:
    np_ = norm(p)
    return sum(np_.count(k) * len(k) for k in ks)


def excerpts(text: str, ks: list[str], top: int = 4) -> tuple[str, list[tuple[int, str]]]:
    paras = split_paragraphs(text)
    head = paras[0][:600]
    scored = sorted(
        ((score_para(p, ks), i, p) for i, p in enumerate(paras)),
        key=lambda x: -x[0],
    )
    picked = [(i, p) for s, i, p in scored[:top] if s > 0]
    picked.sort()
    return head, picked


def highlight(text: str, ks: list[str]) -> str:
    esc = html.escape(text)
    for k in ks[:12]:
        esc = re.sub(f"({re.escape(k)})", r"<mark>\1</mark>", esc, flags=re.IGNORECASE)
    return esc


# ────────────────────────────────────────────────────────────────── HTML ──
CSS = """
body{font:16px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;
margin:0 auto;padding:24px;color:#1a1a1a}
.q{border:1px solid #ddd;border-radius:8px;padding:20px;margin:32px 0;background:#fff}
.qh{font-size:13px;color:#666;letter-spacing:.04em;text-transform:uppercase}
.qt{font-size:19px;font-weight:600;margin:8px 0 20px}
.doc{border-left:3px solid #4a7;padding-left:14px;margin:18px 0}
.dn{font-weight:600;font-size:14px;color:#245}
.meta{font-size:12px;color:#777;margin:4px 0 10px}
.head{background:#f7f7f7;padding:10px;font-size:13px;color:#555;white-space:pre-wrap;
border-radius:4px;margin-bottom:12px}
.ex{background:#fbfbf5;padding:12px;border-radius:4px;margin:10px 0;
white-space:pre-wrap;font-size:14.5px}
.ei{font-size:11px;color:#999;margin-bottom:4px}
mark{background:#ffe9a8;padding:0 2px}
.none{color:#a33;font-weight:600;padding:10px;background:#fee;border-radius:4px}
.v{margin-top:16px;padding:12px;background:#f0f4f8;border-radius:4px;font-size:14px}
.rub{background:#fffbe8;border:1px solid #e8d98a;padding:16px 20px;border-radius:8px}
code{background:#eee;padding:1px 5px;border-radius:3px;font-size:13px}
"""

RUBRIC = """
<div class="rub">
<h2 style="margin-top:0">Cách chấm — đọc trước, đừng vừa đọc vừa nghĩ tiêu chí</h2>
<p><b>Câu hỏi duy nhất cần trả lời cho mỗi mục:</b><br>
<i>Văn bản gold này có chứa một đoạn trả lời trực tiếp câu hỏi không?</i></p>
<table style="font-size:14.5px;border-collapse:collapse">
<tr><td style="padding:6px 12px 6px 0;vertical-align:top"><code>OK</code></td>
<td style="padding:6px 0">Tìm thấy đoạn trả lời trực tiếp. Đây là mặc định khi không có lý do phản đối.</td></tr>
<tr><td style="padding:6px 12px 6px 0;vertical-align:top"><code>N-WRONG</code></td>
<td style="padding:6px 0">Đã đọc kỹ, văn bản <b>không</b> chứa câu trả lời. Gồm cả trường hợp đúng chủ đề
nhưng không có điều khoản cụ thể được hỏi.</td></tr>
<tr><td style="padding:6px 12px 6px 0;vertical-align:top"><code>N-VERSION</code></td>
<td style="padding:6px 0">Văn bản trả lời được, nhưng nó là <b>một đời luật</b> trong khi câu hỏi không nêu
thời điểm — và tồn tại đời luật khác cũng trả lời được. Dấu hiệu: câu hỏi dạng
"X là gì?", "mức hưởng X là bao nhiêu?" về khái niệm đã bị sửa đổi qua các năm.</td></tr>
<tr><td style="padding:6px 12px 6px 0;vertical-align:top"><code>N-AMBIG</code></td>
<td style="padding:6px 0">Câu hỏi thiếu thông tin tới mức không xác định được văn bản nào <b>nên</b> đúng —
thiếu ngành/đối tượng/phạm vi, chứ không phải thiếu mốc thời gian.</td></tr>
</table>
<p style="margin-bottom:0"><b>Bốn quy tắc giữ cho số liệu dùng được:</b></p>
<ol style="margin-top:6px">
<li><b>Giới hạn 3 phút mỗi câu.</b> Quá 3 phút mà chưa quyết được → ghi verdict tốt nhất
kèm <code>confidence=low</code>, đi tiếp. Đọc lâu không làm phán đoán chính xác hơn, chỉ
làm bạn tự thuyết phục mình.</li>
<li><b>Đọc theo đúng thứ tự, không bỏ câu khó.</b> Bỏ câu khó là tự lọc mẫu, và mẫu đã
lọc thì tỉ lệ tính ra vô nghĩa.</li>
<li><b>Trích đoạn có thể bỏ sót.</b> Công cụ neo theo từ khoá; nếu văn bản diễn đạt hoàn
toàn khác câu hỏi, đoạn đúng có thể không được trích. Trước khi ghi <code>N-WRONG</code>,
mở link đầy đủ và Ctrl-F thêm một lần.</li>
<li><b>Ghi chú bắt buộc với mọi verdict khác OK.</b> Đó là bằng chứng, và nếu không viết
được ghi chú thuyết phục thì verdict đó chưa chắc.</li>
</ol>
</div>
"""


def build(train, docs, qids) -> str:
    out = [f"<html><head><meta charset='utf-8'><title>Audit nhãn</title>",
           f"<style>{CSS}</style></head><body>",
           f"<h1>Gói đọc audit nhãn — {len(qids)} câu</h1>", RUBRIC]

    for n, qid in enumerate(qids, 1):
        item = train[qid]
        ks = keywords(item["question"])
        out.append("<div class='q'>")
        out.append(f"<div class='qh'>{n} / {len(qids)} &nbsp;·&nbsp; qid {qid}</div>")
        out.append(f"<div class='qt'>{html.escape(item['question'])}</div>")

        for g in item["gold"]:
            doc = docs.get(g)
            out.append("<div class='doc'>")
            if doc is None:
                out.append(f"<div class='none'>Gold {g} KHÔNG có trong kho văn bản — báo P2</div></div>")
                continue
            name = doc.get("name") or "(không tiêu đề)"
            link = doc.get("link", "")
            text = doc.get("passage") or doc.get("text") or ""
            out.append(f"<div class='dn'>[{g}] {html.escape(name)}</div>")
            out.append(f"<div class='meta'>{len(text):,} ký tự"
                       + (f" · <a href='{html.escape(link)}' target='_blank'>mở bản đầy đủ</a>" if link else "")
                       + "</div>")
            head, picked = excerpts(text, ks)
            out.append(f"<div class='head'>{html.escape(head)}</div>")
            if not picked:
                out.append("<div class='none'>Không đoạn nào chứa từ khoá của câu hỏi "
                           "— dấu hiệu mạnh của N-WRONG, nhưng hãy mở bản đầy đủ kiểm lại</div>")
            for i, p in picked:
                out.append(f"<div class='ex'><div class='ei'>đoạn #{i}</div>"
                           f"{highlight(p[:2200], ks)}</div>")
            out.append("</div>")

        out.append("<div class='v'>Ghi vào manual_audit.csv &nbsp;·&nbsp; "
                   "verdict: <code>OK</code> / <code>N-WRONG</code> / "
                   "<code>N-VERSION</code> / <code>N-AMBIG</code></div>")
        out.append("</div>")

    out.append("</body></html>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train, docs = load_train(), load_docs()
    print(f"{len(docs):,} văn bản, {len(train):,} câu hỏi")

    # PHẢI khớp cách lấy mẫu của audit_labels.py --sample
    qids = random.Random(a.seed).sample(sorted(train), a.n)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "packet.html"
    path.write_text(build(train, docs, qids), encoding="utf-8")
    print(f"\nĐã sinh {path}")
    print(f"Mở bằng trình duyệt. Ghi verdict vào {OUT/'manual_audit.csv'}.")


if __name__ == "__main__":
    main()
