#!/usr/bin/env python3
"""
Đồ thị trích dẫn / sửa đổi giữa các văn bản pháp luật. CHỦ SỞ HỮU: P3.

KHÔNG phải GraphRAG. GraphRAG dùng LLM trích thực thể rồi tóm tắt cụm — với 8.532 văn bản ×
~8.535 từ (≈73 triệu từ) thì không khả thi trong ngân sách còn lại, lại cần một model sinh
chưa được BTC duyệt, và bản tóm tắt do LLM sinh nằm ở vùng xám của luật "không augmentation".

Bản này khai thác một đồ thị đã có sẵn và TƯỜNG MINH trong chính văn bản: văn bản pháp luật VN
trích dẫn nhau bằng số hiệu ("Nghị định 15/2022/NĐ-CP"), và 88% văn bản tự khai số hiệu của mình
ở đầu ("Số: 17/2022/TT-BGTVT"). Trích bằng regex ⇒ **0 tham số, 0 model, 0 đăng ký, không đụng
luật augmentation**, và rơi đúng ô `chien_luoc_du_lieu` của ma trận thực nghiệm (plan.md 0.6).

Nhắm vào hai nhóm lỗi đã có tên trong docs/error_taxonomy.md:
  R-DUP       trả về bản GẦN TRÙNG của gold (bản sửa đổi, bản hợp nhất)
  N-CONSOLID  gold trùng nội dung với một văn bản hợp nhất (VBHN) cũng nằm trong kho

Hai lệnh, cố ý tách rời — không ai được cắm đồ thị vào pipeline trước khi có số của `probe`:

    python scripts/build_citation_graph.py build --config configs/v0.3_bm25_best.yaml
    python scripts/build_citation_graph.py probe --ranking outputs/<exp>/ranking_full.json \\
        --questions data/dev.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.common.io import read_jsonl  # noqa: E402

# "Số: 17/2022/TT-BGTVT" hoặc "Số: 569/QĐ-TTg" — phần đầu văn bản, sau quốc hiệu.
SELF_RE = re.compile(
    r"Số:\s*([0-9]{1,5}[A-ZĐ]*\s*/\s*(?:[0-9]{4}\s*/\s*)?[A-ZĐ]{2,}[a-z]*(?:\s*-\s*[A-ZĐ]+[a-z]*)*)"
)  # [a-z]* cho hậu tố QĐ-TTg, NQ-HĐND
# Trích dẫn trong thân văn bản: luôn có năm ⇒ ít dương tính giả hơn dạng không năm.
CITE_RE = re.compile(r"\b([0-9]{1,5}[A-ZĐ]*/[0-9]{4}/[A-ZĐ]{2,}[a-z]*(?:-[A-ZĐ]+[a-z]*)*)\b")
# Số hiệu nằm trong slug của `name`: "Thong-tu-17-2022-TT-BGTVT-sua-doi-..." → 17/2022/TT-BGTVT
SLUG_RE = re.compile(r"(?:^|-)(\d{1,5})-(\d{4})-([A-Za-zĐđ]{2,}(?:-[A-Za-zĐđ]+)*?)(?=-[a-zđ]{2,}-|$)")
# Quan hệ "đời văn bản" — cạnh đáng giá nhất, vì nó chính là R-DUP / N-CONSOLID.
AMEND_RE = re.compile(r"(sửa đổi|bổ sung|thay thế|bãi bỏ|hết hiệu lực|hợp nhất)", re.I)
AMEND_WINDOW = 120  # ký tự trước một trích dẫn, đủ để bắt "được sửa đổi bởi <số hiệu>"


def norm_code(code: str) -> str:
    """Chuẩn hoá số hiệu về một dạng để tra bảng: bỏ khoảng trắng, NFC, hoa."""
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", code)).upper()


def self_code(doc: dict) -> str | None:
    """Số hiệu của chính văn bản: ưu tiên 'Số:' trong text, fallback slug của `name`."""
    m = SELF_RE.search(doc.get("text", "")[:1500])
    if m:
        return norm_code(m.group(1))
    m = SLUG_RE.search(doc.get("name") or "")
    if m:
        return norm_code(f"{m.group(1)}/{m.group(2)}/{m.group(3)}")
    return None


def cmd_build(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8")) if args.config else {}
    corpus_path = REPO / (args.corpus or cfg.get("paths", {}).get("corpus_clean", "data/corpus_clean.jsonl"))

    docs = [
        {"doc_id": str(d.get("doc_id", d.get("id"))), "name": d.get("name") or "",
         "text": d.get("text", d.get("passage", "")) or ""}
        for d in read_jsonl(corpus_path)
    ]
    print(f"{len(docs)} văn bản từ {corpus_path.name}")

    # ── bảng tra số hiệu → doc_id ────────────────────────────────────────────
    by_code: dict[str, str] = {}
    collisions = 0
    for d in docs:
        c = self_code(d)
        if not c:
            continue
        if c in by_code and by_code[c] != d["doc_id"]:
            # Hai văn bản cùng số hiệu = trùng lặp trong kho (EDA mục 5). Giữ cái đầu, ghi nhận.
            collisions += 1
            continue
        by_code[c] = d["doc_id"]
    print(f"bảng tra: {len(by_code)} số hiệu ({len(by_code)/len(docs):.0%} văn bản) · {collisions} va chạm")

    # ── cạnh ─────────────────────────────────────────────────────────────────
    cites: dict[str, dict[str, str]] = {d["doc_id"]: {} for d in docs}  # src → {dst: loại}
    n_raw = n_res = 0
    for d in docs:
        own = self_code(d)
        text = d["text"]
        seen: dict[str, str] = {}
        for m in CITE_RE.finditer(text):
            code = norm_code(m.group(1))
            n_raw += 1
            if code == own:
                continue
            dst = by_code.get(code)
            if dst is None or dst == d["doc_id"]:
                continue  # trích dẫn tới văn bản KHÔNG có trong kho 8.532 — bỏ, không bịa node
            n_res += 1
            kind = "amend" if AMEND_RE.search(text[max(0, m.start() - AMEND_WINDOW) : m.start()]) else "cite"
            # amend "thắng" cite: một cặp vừa trích dẫn vừa sửa đổi thì quan hệ đời là quan hệ mạnh hơn
            if seen.get(dst) != "amend":
                seen[dst] = kind
        cites[d["doc_id"]] = seen

    cited_by: dict[str, dict[str, str]] = {d["doc_id"]: {} for d in docs}
    for src, dsts in cites.items():
        for dst, kind in dsts.items():
            cited_by[dst][src] = kind

    deg = [len(v) for v in cites.values()]
    n_edges = sum(deg)
    n_amend = sum(1 for v in cites.values() for k in v.values() if k == "amend")
    iso = sum(1 for d in docs if not cites[d["doc_id"]] and not cited_by[d["doc_id"]])
    stats = {
        "n_docs": len(docs),
        "n_codes": len(by_code),
        "code_coverage": round(len(by_code) / len(docs), 4),
        "n_citations_raw": n_raw,
        "n_citations_resolved": n_res,
        "resolve_rate": round(n_res / max(n_raw, 1), 4),
        "n_edges": n_edges,
        "n_edges_amend": n_amend,
        "mean_out_degree": round(n_edges / len(docs), 2),
        "max_out_degree": max(deg) if deg else 0,
        "n_isolated_docs": iso,
        "pct_isolated": round(iso / len(docs), 4),
        "corpus": str(corpus_path.relative_to(REPO)),
        "ngay": date.today().isoformat(),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"stats": stats, "cites": cites, "cited_by": cited_by}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n✅ {out}")
    if stats["pct_isolated"] > 0.5:
        print(
            f"⚠️  {stats['pct_isolated']:.0%} văn bản cô lập — đồ thị thưa tới mức mở rộng ứng viên "
            f"chỉ chạm được một phần nhỏ corpus. Chạy `probe` rồi hãy quyết."
        )
    return 0


def neighbours(graph: dict, doc_id: str, kinds: set[str], hops: int = 1) -> set[str]:
    """Láng giềng vô hướng trong `hops` bước, lọc theo loại cạnh."""
    frontier, seen = {doc_id}, {doc_id}
    for _ in range(hops):
        nxt: set[str] = set()
        for d in frontier:
            for side in ("cites", "cited_by"):
                for n, kind in graph[side].get(d, {}).items():
                    if kind in kinds and n not in seen:
                        nxt.add(n)
        seen |= nxt
        frontier = nxt
    return seen - {doc_id}


def cmd_probe(args: argparse.Namespace) -> int:
    """
    TRẦN TRÊN của ý tưởng, đo trước khi viết một dòng code cắm vào pipeline.

    Câu hỏi: trong những câu mà gold KHÔNG nằm trong top-5, có bao nhiêu câu mà gold nằm cách
    top-5 đúng một bước trên đồ thị? Con số đó là tất cả những gì mở rộng theo đồ thị có thể
    giành được. Nhỏ hơn ~1% thì bỏ ý tưởng, đừng cắm.
    """
    graph = json.loads((REPO / args.graph).read_text(encoding="utf-8"))
    ranking = json.loads((REPO / args.ranking).read_text(encoding="utf-8"))
    qs = json.loads((REPO / args.questions).read_text(encoding="utf-8"))
    gold = {str(q): [str(a) for a in v["answer"]] for q, v in qs.items()}
    kinds = set(args.edge_kinds.split(","))

    k = args.top_k
    n = hit = reach_any = reach_top1 = 0
    added: list[int] = []
    for qid, g in gold.items():
        row = ranking.get(qid)
        if row is None:
            continue
        docs = [str(r[0]) if isinstance(r, (list, tuple)) else str(r) for r in row]
        n += 1
        top = docs[:k]
        if set(top) & set(g):
            hit += 1
            continue
        nb_all: set[str] = set()
        for d in top:
            nb_all |= neighbours(graph, d, kinds, args.hops)
        nb_top1 = neighbours(graph, top[0], kinds, args.hops) if top else set()
        added.append(len(nb_all - set(top)))
        if nb_all & set(g):
            reach_any += 1
        if nb_top1 & set(g):
            reach_top1 += 1

    miss = n - hit
    med = sorted(added)[len(added) // 2] if added else 0
    print(
        f"Câu đo        : {n}\n"
        f"Đã trúng @{k}  : {hit} ({hit/max(n,1):.4f})\n"
        f"Trượt @{k}     : {miss}\n"
        f"── trần trên của mở rộng theo đồ thị (cạnh: {sorted(kinds)}, {args.hops} hop) ──\n"
        f"gold nằm cạnh top-{k} : {reach_any}/{miss} câu trượt  ⇒  +{reach_any/max(n,1):.4f} Recall tối đa\n"
        f"gold nằm cạnh top-1   : {reach_top1}/{miss} câu trượt  ⇒  +{reach_top1/max(n,1):.4f}\n"
        f"số doc thêm vào (trung vị, nếu bung hết): {med}"
    )
    if reach_any / max(n, 1) < 0.01:
        print(
            "\n❌ Dưới 1% — KHÔNG cắm vào pipeline. Đồ thị trích dẫn không chứa gold đang thiếu, "
            "nên mở rộng chỉ thêm nhiễu và hạ Precision (tie-break của giải)."
        )
    else:
        print(
            f"\n✅ Còn {reach_any/max(n,1):.2%} Recall nằm trong tầm với. Bước tiếp: chỉ bung khi "
            f"top-5 còn chỗ trống (92,1% câu chỉ 1 đáp án ⇒ thường thừa 4 ô) và khi margin thấp — "
            f"gắn vào src/rerank/calibrate.py, KHÔNG bung vô điều kiện."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="trích cạnh từ corpus → data/citation_graph.json")
    b.add_argument("--config", default=None)
    b.add_argument("--corpus", default=None)
    b.add_argument("--out", default="data/citation_graph.json")
    b.set_defaults(func=cmd_build)

    p = sub.add_parser("probe", help="đo TRẦN TRÊN trước khi cắm vào pipeline")
    p.add_argument("--graph", default="data/citation_graph.json")
    p.add_argument("--ranking", required=True, help="ranking_full.json của một lần chạy")
    p.add_argument("--questions", required=True, help="file có nhãn (dev.json / error_pool.json)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--edge-kinds", default="amend,cite", help="amend | cite | cả hai")
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
