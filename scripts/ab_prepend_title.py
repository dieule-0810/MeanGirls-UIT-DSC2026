#!/usr/bin/env python3
"""
A/B: gắn tiêu đề văn bản vào mọi chunk (giả thuyết H6). CHỦ SỞ HỮU: P3.

Index HAI lần trên cùng corpus, cùng tokenizer, cùng pooling — khác đúng một thứ: chunk có được
ghép "<số hiệu> <LOẠI> <tên>" ở đầu hay không.

Script đo CẢ HAI chiều của giả thuyết, vì chúng đối nghịch nhau:

  (+) chunk giữa văn bản lấy lại được danh tính văn bản (trung vị 36 chunk/văn bản)
  (−) cùng một chuỗi thêm vào 36 chunk ⇒ df tăng ⇒ IDF sụp; và khớp cấp CHỦ ĐỀ lấn át
      khớp cấp ĐIỀU KHOẢN

Nên bảng kết quả có cột `IDF trung bình của term tiêu đề` và `avgdl` — nếu recall tăng mà IDF
không sụp thì cơ chế (+) thắng; nếu recall giảm, hai cột đó nói cho ta biết vì sao.

    python -u scripts/ab_prepend_title.py --config configs/v0.3_bm25_best.yaml \\
        --questions data/dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.common.io import load_chunks, load_questions  # noqa: E402
from src.evaluate import eval_official, load_truth, recall_at_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.enrich import prepend_titles, title_map  # noqa: E402


def run(chunks: list[dict], spec: dict, texts: list[str], qids: list[str], top_k: int, titles: dict | None):
    r = BM25Retriever(**spec)
    r.index(chunks)
    t0 = time.perf_counter()
    ranked = r.search(texts, top_k)
    took = time.perf_counter() - t0
    preds = {q: [d for d, _ in res] for q, res in zip(qids, ranked)}

    # IDF trung bình của các term xuất hiện trong tiêu đề — thước đo trực tiếp của "IDF sụp".
    idf_title = None
    if titles:
        terms = set()
        for t in list(titles.values())[:2000]:
            terms.update(r.tokenizer(t))
        vals = [float(r.idf[r.vocab[t]]) for t in terms if t in r.vocab]
        idf_title = round(float(np.mean(vals)), 4) if vals else None
    return preds, {
        "vocab": len(r.vocab),
        "avgdl": round(r.avgdl, 1),
        "nnz": int(r.matrix.nnz),
        "idf_title_terms": idf_title,
        "search_s": round(took, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v0.3_bm25_best.yaml")
    ap.add_argument("--questions", default=None, help="mặc định paths.dev")
    ap.add_argument("--max-words", type=int, default=30, help="độ dài tối đa của tiêu đề gắn vào")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--report", default="docs/ab_prepend_title.md")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    q_path = args.questions or cfg["paths"].get("dev")
    if "holdout" in Path(q_path).name:
        raise SystemExit("❌ holdout là tập đo lần cuối — A/B phải chạy trên dev.")

    spec = dict(cfg["retrieval"]["bm25"])
    if cfg["paths"].get("cache_dir"):
        spec.setdefault("cache_dir", cfg["paths"]["cache_dir"])
    spec["verbose"] = True

    chunks = load_chunks(REPO / cfg["paths"]["chunks"])
    qids, texts = load_questions(REPO / q_path)
    gold = load_truth(REPO / q_path)

    print(f"{len(chunks)} chunk · {len(qids)} câu · {q_path}\n")
    print("── trích tiêu đề ──")
    titles = title_map(REPO / cfg["paths"]["corpus_clean"], args.max_words)
    enriched, st = prepend_titles(chunks, titles)
    print(json.dumps(st, ensure_ascii=False))
    if st["pct_prepended"] < 0.5:
        print("⚠️  gắn được cho dưới 50% chunk — kết quả A/B sẽ bị pha loãng, kiểm lại regex trích tiêu đề")

    print("\n── A: KHÔNG gắn tiêu đề ──")
    pa, sa = run(chunks, spec, texts, qids, args.top_k, titles)
    print("\n── B: CÓ gắn tiêu đề ──")
    pb, sb = run(enriched, spec, texts, qids, args.top_k, titles)

    ks = [1, 3, 5, 20, args.top_k]
    rows = []
    print(f"\n{'':<22}" + "".join(f"{'R@'+str(k):>9}" for k in ks))
    for name, p in (("A không tiêu đề", pa), ("B có tiêu đề", pb)):
        vals = [recall_at_k(p, gold, k) for k in ks]
        rows.append((name, vals))
        print(f"{name:<22}" + "".join(f"{v:>9.4f}" for v in vals))
    print(f"{'Δ (B − A)':<22}" + "".join(f"{b-a:>+9.4f}" for a, b in zip(rows[0][1], rows[1][1])))

    print(f"\n{'':<22}{'vocab':>12}{'avgdl':>9}{'nnz':>13}{'IDF tiêu đề':>13}")
    for name, s in (("A không tiêu đề", sa), ("B có tiêu đề", sb)):
        print(f"{name:<22}{s['vocab']:>12}{s['avgdl']:>9}{s['nnz']:>13}{str(s['idf_title_terms']):>13}")

    # Thắng/thua theo từng câu ở K=5: tổng recall che mất chuyện đổi chác giữa hai nhóm câu.
    win = lose = 0
    for q, g in gold.items():
        a = bool(set(pa.get(q, [])[:5]) & set(g))
        b = bool(set(pb.get(q, [])[:5]) & set(g))
        win += b and not a
        lose += a and not b
    print(f"\n@5: B cứu được {win} câu · B làm hỏng {lose} câu · ròng {win - lose:+d}/{len(gold)}")
    if win + lose:
        print(f"    (McNemar: chỉ {win + lose} câu bất đồng — chạy scripts/p4_paired_test.py nếu Δ nhỏ)")

    md = [
        "# A/B — gắn tiêu đề văn bản vào chunk (H6)", "",
        f"> `{args.config}` · `{q_path}` n={len(qids)} · top-{args.top_k} · {date.today().isoformat()}",
        "", "```json", json.dumps(st, ensure_ascii=False, indent=2), "```", "",
        "| | " + " | ".join(f"R@{k}" for k in ks) + " | vocab | avgdl | IDF tiêu đề |",
        "|---|" + "---:|" * (len(ks) + 3),
    ]
    for (name, vals), s in ((rows[0], sa), (rows[1], sb)):
        md.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in vals) +
                  f" | {s['vocab']} | {s['avgdl']} | {s['idf_title_terms']} |")
    md += ["", f"**@5: cứu {win} câu · hỏng {lose} câu · ròng {win-lose:+d}**", "",
           "## Nhận xét (điền tay)", "",
           "- Recall đổi theo chiều nào, và IDF term tiêu đề sụp bao nhiêu? ",
           "- Câu được cứu có phải loại hỏi theo CHỦ ĐỀ, câu bị hỏng có phải loại hỏi theo ĐIỀU KHOẢN? ",
           "- Có đáng tốn thêm một mẻ encode 2,5 giờ GPU để thử điều này cho dense không? ",
           ]
    out = REPO / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n✅ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
