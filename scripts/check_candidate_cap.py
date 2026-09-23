#!/usr/bin/env python3
"""
Kiểm tương đương `candidate_chunks: 2000` vs `null`. CHỦ SỞ HỮU: P3.

VÌ SAO CẦN LẠI: `configs/v0.1_bm25_cap2000.yaml` đã kiểm điều này, nhưng chỉ trên
`tokenizer=regex` + `pool=max`. Hai thứ đó đều đổi ở v0.2:

  - `syllable_bigram` sinh gấp đôi số term (âm tiết + bigram) ⇒ phân bố điểm chunk khác hẳn,
    đuôi dài hơn, nên "chunk thứ 2000 không bao giờ ảnh hưởng top-50 doc" phải chứng minh lại.
  - `max` chỉ cần MỘT chunk tốt nhất mỗi doc, còn `mean_topN`/`logsumexp` cộng dồn NHIỀU chunk
    ⇒ cắt ứng viên có thể đổi điểm doc ngay cả khi chunk tốt nhất vẫn còn. Đây mới là rủi ro thật.

Nếu không kiểm, cả lưới bench có thể chỉ là hiện vật của việc cắt ứng viên chứ không phải
tính chất của tokenizer.

Đo trên `error_pool.json` (300 câu, tập của P4 dùng cho phân tích lỗi — KHÔNG phải dev, để dev
giữ nguyên vai trò tập đo), so khớp CHÍNH XÁC danh sách top-K doc của hai chế độ.

    python scripts/check_candidate_cap.py --config configs/v0.2_bm25_tokenizer.yaml
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

import yaml  # noqa: E402

from src.common.io import load_chunks, load_questions  # noqa: E402
from src.evaluate import recall_at_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.tokenizers import available_tokenizers  # noqa: E402

NO_CAP = 10**9  # đúng giá trị BaseRetriever dùng khi candidate_chunks: null


def top_docs(r: BM25Retriever, cands, pool: str, top_k: int) -> list[list[str]]:
    return [[d for d, _ in r.pool_candidates(idx, sc, top_k, pool=pool)] for idx, sc in cands]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v0.2_bm25_tokenizer.yaml")
    ap.add_argument("--questions", default=None, help="mặc định: data/error_pool.json")
    ap.add_argument("--top-k", type=int, default=50, help="độ sâu phải giống hệt nhau")
    ap.add_argument("--tokenizers", default=None)
    ap.add_argument("--pools", default=None)
    ap.add_argument(
        "--out",
        default="docs/candidate_cap_check.json",
        help="docs/ chứ không phải outputs/: .gitignore chặn outputs/* nhưng chừa !docs/*.json",
    )
    ap.add_argument("--report", default="docs/candidate_cap_check.md")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    bench = cfg.get("bench", {})
    tokenizers = args.tokenizers.split(",") if args.tokenizers else bench.get("tokenizers", ["regex"])
    pools = args.pools.split(",") if args.pools else bench.get("pools", ["max"])
    cap = cfg["retrieval"]["bm25"].get("candidate_chunks") or NO_CAP

    q_path = args.questions or "data/error_pool.json"
    chunks = load_chunks(REPO / cfg["paths"]["chunks"])
    qids, texts = load_questions(REPO / q_path)
    truth = {q: [str(d) for d in v["answer"]] for q, v in json.loads((REPO / q_path).read_text(encoding="utf-8")).items()}

    bm25_opts = dict(cfg["retrieval"].get("bm25", {}))
    bm25_opts.pop("tokenizer", None)
    bm25_opts.pop("candidate_chunks", None)
    if cfg.get("paths", {}).get("cache_dir"):
        bm25_opts.setdefault("cache_dir", cfg["paths"]["cache_dir"])

    print(f"{len(chunks)} chunk · {len(qids)} câu ({q_path}) · cap={cap} vs null · top-{args.top_k}")
    have = available_tokenizers()
    rows: list[dict] = []

    for tok in tokenizers:
        if not have.get(tok, False):
            print(f"\n⏭  bỏ qua '{tok}': chưa cài")
            continue
        print(f"\n══ tokenizer: {tok} ══")
        opts = dict(bm25_opts)
        opts["tokenizer"] = tok          # thiếu dòng này = chạy regex N lần mà bảng vẫn ghi N tên
        r = BM25Retriever(candidate_chunks=NO_CAP, **opts)
        r.index(chunks)
        # Fail loud: lần chạy đầu 11/09 trượt đúng lỗi này và chỉ lộ ra vì 5 khối số liệu
        # giống hệt nhau tới từng chữ số.
        assert r.tokenizer.name == tok, f"yêu cầu {tok!r} nhưng retriever dùng {r.tokenizer.name!r}"

        t0 = time.perf_counter()
        cands_full = r.candidates(texts, NO_CAP)
        s_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        cands_cap = r.candidates(texts, cap)
        s_cap = time.perf_counter() - t0
        n_cand_med = sorted(len(i) for i, _ in cands_full)[len(cands_full) // 2]
        print(
            f"  null: {s_full / len(qids) * 1000:.0f} ms/câu · cap: {s_cap / len(qids) * 1000:.0f} ms/câu"
            f" · trung vị ứng viên khi không cắt: {n_cand_med}"
        )

        for pool in pools:
            full = top_docs(r, cands_full, pool, args.top_k)
            capped = top_docs(r, cands_cap, pool, args.top_k)
            n_diff = sum(1 for a, b in zip(full, capped) if a != b)
            # Danh sách lệch mà TẬP vẫn bằng nhau = chỉ đổi thứ tự; mã chấm không dùng thứ tự
            # nên vẫn là tương đương về điểm, nhưng phải báo tách bạch chứ không gộp làm một.
            n_set_diff = sum(1 for a, b in zip(full, capped) if set(a) != set(b))
            row = {
                "tokenizer": r.tokenizer.key,
                "pool": pool,
                "n_questions": len(qids),
                "n_list_diff": n_diff,
                "n_set_diff": n_set_diff,
                "recall@5_null": round(recall_at_k(dict(zip(qids, full)), truth, 5), 4),
                "recall@5_cap": round(recall_at_k(dict(zip(qids, capped)), truth, 5), 4),
                f"recall@{args.top_k}_null": round(recall_at_k(dict(zip(qids, full)), truth, args.top_k), 4),
                f"recall@{args.top_k}_cap": round(recall_at_k(dict(zip(qids, capped)), truth, args.top_k), 4),
                "vocab": len(r.vocab),
                "ms_per_query_null": round(s_full / len(qids) * 1000, 1),
                "ms_per_query_cap": round(s_cap / len(qids) * 1000, 1),
                "median_candidates_uncapped": n_cand_med,
            }
            row["tuong_duong"] = row["n_set_diff"] == 0
            rows.append(row)
            flag = "✅" if row["tuong_duong"] else "❌"
            print(
                f"  {flag} pool={pool:<11} lệch danh sách {n_diff}/{len(qids)} · "
                f"lệch TẬP {n_set_diff}/{len(qids)} · "
                f"R@{args.top_k} {row[f'recall@{args.top_k}_null']} vs {row[f'recall@{args.top_k}_cap']}"
            )

    bad = [r for r in rows if not r["tuong_duong"]]
    meta = {
        "config": args.config,
        "questions": q_path,
        "cap": cap,
        "top_k": args.top_k,
        "n_chunks": len(chunks),
        "ngay": date.today().isoformat(),
        "ket_luan": "TƯƠNG ĐƯƠNG" if not bad else f"KHÔNG tương đương ở {len(bad)}/{len(rows)} ô",
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Kiểm tương đương `candidate_chunks: {cap}` vs `null`",
        "",
        "> Sinh bởi `scripts/check_candidate_cap.py`. Chủ sở hữu: P3.",
        f"> Cấu hình `{args.config}` · `{q_path}` n={len(qids)} · corpus {len(chunks)} chunk · top-{args.top_k}.",
        "",
        "```json",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "```",
        "",
        f"| tokenizer | pool | lệch danh sách | lệch TẬP | R@{args.top_k} null | R@{args.top_k} cap | ms/câu null | ms/câu cap | ứng viên (trung vị) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tokenizer']} | {r['pool']} | {r['n_list_diff']}/{r['n_questions']} | "
            f"{r['n_set_diff']}/{r['n_questions']} | {r[f'recall@{args.top_k}_null']} | "
            f"{r[f'recall@{args.top_k}_cap']} | {r['ms_per_query_null']} | {r['ms_per_query_cap']} | "
            f"{r['median_candidates_uncapped']} |"
        )
    lines += [
        "",
        "**Đọc bảng:** *lệch TẬP* = 0 nghĩa là hai chế độ trả về đúng cùng một tập doc "
        f"ở độ sâu {args.top_k} ⇒ mọi con số Recall đều không đổi (mã chấm chỉ dùng phép giao tập hợp). "
        "*lệch danh sách* > 0 mà *lệch TẬP* = 0 chỉ là khác thứ tự trong cùng tập.",
        "",
        f"**Kết luận:** {meta['ket_luan']}.",
        "",
        "## Nhận xét (điền tay)",
        "",
        "- Ô nào lệch, và lệch ở pooling cộng dồn (`mean_topN`/`logsumexp`) hay ở `max`? ",
        "- Nếu có ô lệch: cap phải nâng lên bao nhiêu, hay tokenizer đó buộc phải chạy `null`? ",
    ]
    report = REPO / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{'✅' if not bad else '❌'} {meta['ket_luan']}")
    print(f"✅ {out}\n✅ {report}")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
