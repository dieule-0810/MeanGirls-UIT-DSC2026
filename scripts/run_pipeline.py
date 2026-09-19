#!/usr/bin/env python3
"""
Pipeline end-to-end: chunks.jsonl → submission.zip. CHỦ SỞ HỮU: P3 (nhánh p3/pipeline-e2e).

Thay cho `scripts/run_v0.1.py` (đang hỏng: gọi `src.data.split_holdout` không tồn tại và truyền
`--config` cho module P2 vốn không nhận). Bản này KHÔNG chạy lại parse/chunk — `chunks.jsonl` là
đầu vào, đúng phạm vi được giao.

Bốn tầng, mỗi tầng bật/tắt độc lập bằng YAML để đo đóng góp riêng (plan.md mục 2 — ablation của
bài báo cần đúng điều này):

    retrieve  →  rerank (tuỳ chọn)  →  calibrate (tuỳ chọn)  →  submission

Hợp nhất BM25+dense KHÔNG phải một tầng ở đây: theo INTERFACES §3, gộp là việc NỘI BỘ của
retriever ⇒ nó sẽ là `type: hybrid` trong `src/retrieval/hybrid.py`, và file này không cần biết.

    python scripts/run_pipeline.py --config configs/v0.4_dense.yaml --demo
    python scripts/run_pipeline.py --config configs/v0.3_bm25_best.yaml \\
        --questions data/dev.json --eval
    python scripts/run_pipeline.py --config configs/v0.3_bm25_best.yaml \\
        --questions data/public-official.json --submission
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.common.io import load_chunks, load_questions, write_predictions  # noqa: E402
from src.retrieval.base import build_retriever  # noqa: E402


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Corpus giả cho --demo (không cần data/, chạy được trên máy sạch)
# ─────────────────────────────────────────────────────────────────────────────
def demo_data() -> tuple[list[dict], dict]:
    docs = {
        "740": ["Điều 1. Cơ quan thuế huỷ bỏ hoá đơn điện tử đã lập sai.",
                "Điều 2. Việc huỷ hoá đơn điện tử phải lập biên bản theo Thông tư 78/2021/TT-BTC."],
        "812": ["Điều 1. Cơ sở dữ liệu quốc gia về dân cư do Bộ Công an quản lý.",
                "Điều 2. Quan hệ lao động giữa người sử dụng lao động và người lao động."],
        "915": ["Điều 1. Mức đóng bảo hiểm xã hội bắt buộc của người lao động.",
                "Điều 2. Thời gian hưởng chế độ thai sản theo Luật Bảo hiểm xã hội."],
    }
    chunks = [
        {"chunk_id": f"{d}::{i:04d}", "doc_id": d, "position": i, "text": t}
        for d, texts in docs.items()
        for i, t in enumerate(texts)
    ]
    questions = {
        "1": {"question": "Huỷ hoá đơn điện tử lập sai thì làm thế nào?", "answer": ["740"]},
        "2": {"question": "Cơ sở dữ liệu quốc gia về dân cư do ai quản lý?", "answer": ["812"]},
        "3": {"question": "Thời gian hưởng chế độ thai sản là bao lâu?", "answer": ["915"]},
    }
    return chunks, questions


# ─────────────────────────────────────────────────────────────────────────────
# Các tầng
# ─────────────────────────────────────────────────────────────────────────────
def stage_retrieve(cfg: dict, chunks: list[dict], texts: list[str], top_k: int, demo: bool):
    """Dựng retriever từ config rồi truy hồi kèm chunk đại diện (INTERFACES §3b)."""
    pipe = cfg.get("pipeline", {})
    kind = pipe.get("retriever") or _infer_retriever_kind(cfg)
    spec = dict(cfg["retrieval"][kind])
    spec["type"] = kind
    if demo:
        # corpus giả 6 chunk: cache token và embedding encode sẵn đều vô nghĩa
        spec.pop("cache_dir", None)
        spec.pop("embeddings_path", None)
    elif kind == "dense" and cfg["paths"].get("embeddings"):
        spec.setdefault("embeddings_path", cfg["paths"]["embeddings"])
    elif kind == "bm25" and cfg["paths"].get("cache_dir"):
        spec.setdefault("cache_dir", cfg["paths"]["cache_dir"])

    r = build_retriever(spec)
    print(f"── retrieve: {kind} ──")
    t0 = time.perf_counter()
    r.index(chunks)
    ranked = r.search_with_anchor(texts, top_k)
    print(f"   {len(texts)} câu trong {time.perf_counter() - t0:.1f}s")
    return r, ranked


def _infer_retriever_kind(cfg: dict) -> str:
    """Config chỉ có một khối retriever thì không bắt người dùng khai lại tên nó."""
    kinds = [k for k in cfg.get("retrieval", {}) if isinstance(cfg["retrieval"][k], dict)]
    if len(kinds) != 1:
        raise SystemExit(
            f"❌ `retrieval` có {len(kinds)} khối ({kinds}). Khai rõ `pipeline.retriever: <tên>` "
            f"để không ai phải đoán bản chạy dùng cái nào."
        )
    return kinds[0]


def stage_rerank(cfg: dict, chunks: list[dict], qids, texts, ranked, top_k_submit: int):
    """
    Xếp hạng lại top-N bằng cross-encoder.

    ⚠️ MẶC ĐỊNH LÀ `rrf`, KHÔNG PHẢI `replace`. P4 đo trên dev_sub300: dùng reranker để THAY THẾ
    thứ hạng làm TỆ R@5 ở cả ba model (bge-m3 −0,0183, mmarco −0,0683, ViRanker −0,0906) dù probe
    gold-vs-bừa đạt 95%. Giỏi theo CẶP không kéo theo giỏi theo DANH SÁCH — bài toán thật là phân
    biệt gold với BỐN văn bản khó nhất. Hợp nhất thứ hạng giữ bề rộng BM25 và lấy phần đỉnh
    reranker: +0,0300 R@5. Muốn `replace` thì phải khai tường minh trong YAML.
    """
    rcfg = cfg["pipeline"]["rerank"]
    mode = rcfg.get("mode", "rrf")
    depth = int(rcfg.get("depth", 20))
    k_rrf = int(rcfg.get("rrf_k", 60))
    w = float(rcfg.get("weight_retriever", 0.6))

    from src.rerank.cross_encoder import CrossEncoderReranker

    text_of = {str(c["chunk_id"]): c["text"] for c in chunks}
    rr = CrossEncoderReranker(
        name=rcfg["model"],
        device=rcfg.get("device"),
        max_length=int(rcfg.get("max_length", 512)),
        batch_size=int(rcfg.get("batch_size", 16)),
    )
    print(f"── rerank: {rcfg['model']} · mode={mode} · depth={depth} ──")

    out = []
    for q, res in zip(texts, ranked):
        head, tail = res[:depth], res[depth:]
        scores = rr.score([(q, text_of.get(cid, "")) for _, _, cid in head], quiet=True)
        by_rr = sorted(zip(head, scores), key=lambda t: -t[1])
        if mode == "replace":
            new_head = [h for h, _ in by_rr]
        else:
            rank_ret = {h[0]: i + 1 for i, h in enumerate(head)}
            rank_rr = {h[0]: i + 1 for i, (h, _) in enumerate(by_rr)}
            fused = sorted(
                head,
                key=lambda h: (
                    -(w / (k_rrf + rank_ret[h[0]]) + (1 - w) / (k_rrf + rank_rr[h[0]])),
                    rank_ret[h[0]],
                ),
            )
            new_head = fused
        out.append(new_head + tail)
    return out


def stage_calibrate(cfg: dict, ranked, top_k_submit: int) -> dict:
    """Bộ quyết định số lượng doc. Chưa có `src/rerank/calibrate.py` thì cắt cứng top-k."""
    ccfg = cfg.get("pipeline", {}).get("calibrate")
    if not ccfg:
        return {}
    try:
        from src.rerank.calibrate import decide_counts
    except ImportError:
        print("   ⚠️  chưa có src/rerank/calibrate.py — bỏ qua tầng calibrate, cắt cứng top-k.")
        return {}
    return {"counts": decide_counts(ranked, **ccfg)}


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", default=None, help="mặc định: paths.dev")
    ap.add_argument("--demo", action="store_true", help="corpus giả, không cần data/")
    ap.add_argument("--dry-run", action="store_true", help="in kế hoạch rồi thoát")
    ap.add_argument("--eval", action="store_true", help="file câu hỏi có nhãn → in Recall@k")
    ap.add_argument("--submission", action="store_true", help="sinh submission.zip")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    pipe = cfg.get("pipeline", {})
    top_k_submit = int(cfg["retrieval"].get("top_k_submit", 5))
    top_k = max(int(cfg["retrieval"].get("ranking_full_k", 50)), top_k_submit)
    out_dir = Path(args.out_dir or cfg["paths"].get("out_dir", f"outputs/{cfg.get('exp_id','run')}"))
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    stages = ["retrieve"] + [s for s in ("rerank", "calibrate") if pipe.get(s)]
    print(
        f"exp_id : {cfg.get('exp_id')}\ncommit : {git_commit()}\n"
        f"tầng   : {' → '.join(stages)}{' → submission' if args.submission else ''}\n"
        f"ra     : {out_dir}"
    )

    if args.demo:
        chunks, raw_q = demo_data()
        qids = list(raw_q)
        texts = [raw_q[q]["question"] for q in qids]
        truth = {q: [str(d) for d in raw_q[q]["answer"]] for q in qids}
        q_path = "demo"
    else:
        q_path = args.questions or cfg["paths"].get("dev") or cfg["paths"]["questions"]
        if "holdout" in Path(q_path).name and not args.allow_holdout:
            raise SystemExit(
                f"❌ {q_path} là tập ĐO LẦN CUỐI, chạm đúng một lần. Dùng data/dev.json, "
                f"hoặc --allow-holdout nếu cả nhóm đã chốt đây LÀ lần đo cuối."
            )
        chunks = load_chunks(REPO / cfg["paths"]["chunks"])
        qids, texts = load_questions(REPO / q_path if not Path(q_path).is_absolute() else q_path)
        truth = None

    print(f"câu hỏi: {len(qids)} ({q_path}) · chunk: {len(chunks)} · top_k={top_k}")
    if args.dry_run:
        print("\n--dry-run: dừng ở đây, chưa nạp model, chưa index.")
        return 0

    retriever, ranked = stage_retrieve(cfg, chunks, texts, top_k, args.demo)
    if pipe.get("rerank"):
        ranked = stage_rerank(cfg, chunks, qids, texts, ranked, top_k_submit)
    extra = stage_calibrate(cfg, ranked, top_k_submit)

    counts = extra.get("counts") or [top_k_submit] * len(qids)
    preds_full = {q: [d for d, _, _ in res] for q, res in zip(qids, ranked)}
    preds = {q: preds_full[q][: counts[i]] for i, q in enumerate(qids)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking_full.json").write_text(
        json.dumps(
            {q: [[d, round(float(s), 4), c] for d, s, c in res] for q, res in zip(qids, ranked)},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    write_predictions(preds, out_dir / "predictions.json")
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "exp_id": cfg.get("exp_id"), "config": args.config, "commit": git_commit(),
                "questions": str(q_path), "n_questions": len(qids), "n_chunks": len(chunks),
                "stages": stages, "top_k": top_k, "top_k_submit": top_k_submit,
                "retriever": retriever.stats(), "ngay": date.today().isoformat(),
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n✅ {out_dir/'predictions.json'}\n✅ {out_dir/'ranking_full.json'}\n✅ {out_dir/'run_meta.json'}")

    if args.eval:
        from src.evaluate import eval_official, load_truth, recall_at_k

        gold = truth if truth is not None else load_truth(REPO / q_path)
        for k in sorted({k for k in (5, 20, 50, 100) if k <= top_k} | {top_k}):
            print(f"   Recall@{k:<4}: {recall_at_k(preds_full, gold, k):.4f}")
        s = eval_official(preds, gold)
        print(f"   Chấm như BTC: recall={s['recall']:.4f} precision={s['precision']:.4f}")

    if args.submission:
        from src.make_submission import build_submission, write_zip

        corpus_ids = None
        if not args.demo and cfg["paths"].get("corpus_clean"):
            from src.common.io import load_corpus_ids

            corpus_ids = load_corpus_ids(REPO / cfg["paths"]["corpus_clean"])
        sub = build_submission(preds, set(qids), corpus_ids=corpus_ids)
        path = write_zip(sub, out_dir / "submission.zip")
        print(f"✅ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
