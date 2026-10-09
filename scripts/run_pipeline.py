#!/usr/bin/env python3
"""Pipeline end-to-end: `chunks.jsonl` → `predictions.json` → `submission.zip`. CHỦ SỞ HỮU: P3.

Bốn tầng, mỗi tầng bật/tắt độc lập bằng YAML để đo đóng góp riêng (ablation cho bài báo):

    retrieve  →  rerank (tuỳ chọn)  →  calibrate (tuỳ chọn)  →  submission

Hợp nhất BM25 + dense KHÔNG phải một tầng ở đây: gộp là việc NỘI BỘ của retriever
(INTERFACES.md §3), nên hybrid chỉ là `pipeline.retriever: hybrid` trong YAML. File này không
chạy parse/chunk — `chunks.jsonl` là đầu vào (xem docs/reproduce.md mục 1).

Typical usage example:

    python scripts/run_pipeline.py --config configs/v0.8_hybrid_demo.yaml --demo --eval
    python scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \\
        --questions data/dev.json --eval
    python scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml \\
        --questions data/private-official.json --submission
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import load_config, repo_path, resolve_out_dir  # noqa: E402
from src.common.demo import tiny_corpus  # noqa: E402
from src.common.io import load_chunks, load_questions, write_predictions  # noqa: E402
from src.common.runinfo import git_commit, guard_holdout  # noqa: E402
from src.retrieval.base import BaseRetriever, build_retriever, retriever_spec  # noqa: E402

Ranked = list[list[tuple[str, float, str]]]  # mỗi câu: (doc_id, score, chunk_id đại diện)


@dataclass
class RunInputs:
    """Đầu vào của một lần chạy.

    Attributes:
        chunks: Kho chunk.
        qids: Mã câu hỏi, cùng thứ tự với `texts`.
        texts: Nội dung câu hỏi.
        truth: Nhãn khi chạy `--demo`; None = đọc từ file câu hỏi lúc `--eval`.
        q_path: Đường dẫn file câu hỏi (hoặc `demo`).
    """

    chunks: list[dict]
    qids: list[str]
    texts: list[str]
    truth: dict[str, list[str]] | None
    q_path: str


# ─────────────────────────────────────────────────────────────────────────────
# Các tầng
# ─────────────────────────────────────────────────────────────────────────────
def stage_retrieve(cfg: dict, chunks: list[dict], texts: list[str], top_k: int, demo: bool) -> tuple[BaseRetriever, Ranked]:
    """Dựng retriever từ config, index rồi truy hồi kèm chunk đại diện (INTERFACES.md §3b).

    Args:
        cfg: Config đã đọc.
        chunks: Kho chunk.
        texts: Câu hỏi.
        top_k: Số doc giữ lại mỗi câu.
        demo: Chạy trên corpus giả (gỡ cache/embedding khỏi spec).

    Returns:
        `(retriever, ranked)`.
    """
    try:
        spec = retriever_spec(cfg, demo=demo)
    except ValueError as e:
        raise SystemExit(f"❌ {e}") from e
    r = build_retriever(spec)
    print(f"── retrieve: {spec['type']} ──")
    t0 = time.perf_counter()
    r.index(chunks)
    ranked = r.search_with_anchor(texts, top_k)
    print(f"   {len(texts)} câu trong {time.perf_counter() - t0:.1f}s")
    return r, ranked


def _rrf_merge(head: list, by_reranker: list, w: float, k: int) -> list:
    """Hợp nhất thứ hạng retriever và reranker trên cùng `head`; hoà thì giữ thứ tự retriever."""
    rank_ret = {h[0]: i + 1 for i, h in enumerate(head)}
    rank_rr = {h[0]: i + 1 for i, (h, _) in enumerate(by_reranker)}
    return sorted(
        head,
        key=lambda h: (-(w / (k + rank_ret[h[0]]) + (1 - w) / (k + rank_rr[h[0]])), rank_ret[h[0]]),
    )


def stage_rerank(rcfg: dict, chunks: list[dict], texts: list[str], ranked: Ranked) -> Ranked:
    """Xếp hạng lại `depth` doc đầu bằng cross-encoder.

    Mặc định `mode: rrf`, KHÔNG phải `replace`: P4 đo được dùng reranker để THAY THẾ thứ hạng
    làm TỆ R@5 ở cả ba model dù probe gold-vs-bừa đạt 95% — giỏi theo cặp không kéo theo giỏi
    theo danh sách. Muốn `replace` thì khai tường minh trong YAML.

    Args:
        rcfg: Khối `pipeline.rerank` (`model`, `mode`, `depth`, `rrf_k`, `weight_retriever`, ...).
        chunks: Kho chunk, để tra văn bản của chunk đại diện.
        texts: Câu hỏi.
        ranked: Kết quả tầng retrieve.

    Returns:
        Ranking mới cùng định dạng.
    """
    from src.rerank.cross_encoder import CrossEncoderReranker

    mode = rcfg.get("mode", "rrf")
    depth = int(rcfg.get("depth", 20))
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
            new_head = _rrf_merge(head, by_rr, float(rcfg.get("weight_retriever", 0.6)), int(rcfg.get("rrf_k", 60)))
        out.append(new_head + tail)
    return out


def stage_calibrate(ccfg: dict | None, ranked: Ranked, n_questions: int, top_k_submit: int) -> list[int]:
    """Số doc trả về cho từng câu; không có khối `pipeline.calibrate` thì cắt cứng `top_k_submit`."""
    if not ccfg:
        return [top_k_submit] * n_questions
    from src.rerank.calibrate import decide_counts

    return decide_counts(ranked, **ccfg)


# ─────────────────────────────────────────────────────────────────────────────
# Vào / ra
# ─────────────────────────────────────────────────────────────────────────────
def load_inputs(cfg: dict, args: argparse.Namespace) -> RunInputs:
    """Nạp kho chunk và câu hỏi (corpus giả nếu `--demo`); chặn holdout chưa được phép."""
    if args.demo:
        chunks, raw_q = tiny_corpus()
        qids = list(raw_q)
        truth = {q: [str(d) for d in raw_q[q]["answer"]] for q in qids}
        return RunInputs(chunks, qids, [raw_q[q]["question"] for q in qids], truth, "demo")
    q_path = args.questions or cfg["paths"].get("dev") or cfg["paths"]["questions"]
    guard_holdout(q_path, args.allow_holdout)
    qids, texts = load_questions(repo_path(q_path))
    return RunInputs(load_chunks(repo_path(cfg["paths"]["chunks"])), qids, texts, None, str(q_path))


def write_outputs(out_dir: Path, ranked: Ranked, preds: dict[str, list[str]], meta: dict) -> None:
    """Ghi `ranking_full.json`, `predictions.json`, `run_meta.json` vào `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking = {q: [[d, round(float(s), 4), c] for d, s, c in res] for q, res in zip(meta["qids"], ranked)}
    (out_dir / "ranking_full.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=1), encoding="utf-8")
    write_predictions(preds, out_dir / "predictions.json")
    meta = {k: v for k, v in meta.items() if k != "qids"}
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    for name in ("predictions.json", "ranking_full.json", "run_meta.json"):
        print(f"✅ {out_dir / name}")


def print_eval(preds_full: dict, preds: dict, gold: dict, top_k: int) -> None:
    """In Recall@k của ranking đầy đủ và điểm chấm như BTC của dự đoán đã cắt."""
    from src.evaluate import eval_official, recall_at_k

    for k in sorted({k for k in (5, 20, 50, 100) if k <= top_k} | {top_k}):
        print(f"   Recall@{k:<4}: {recall_at_k(preds_full, gold, k):.4f}")
    s = eval_official(preds, gold)
    print(f"   Chấm như BTC: recall={s['recall']:.4f} precision={s['precision']:.4f}")


def write_submission(cfg: dict, preds: dict, qids: list[str], out_dir: Path, demo: bool) -> None:
    """Đóng gói `submission.zip` qua `make_submission` (kiểm doc_id lạ nếu có corpus)."""
    from src.make_submission import build_submission, write_zip

    corpus_ids = None
    if not demo and cfg["paths"].get("corpus_clean"):
        from src.common.io import load_corpus_ids

        corpus_ids = load_corpus_ids(repo_path(cfg["paths"]["corpus_clean"]))
    print(f"✅ {write_zip(build_submission(preds, set(qids), corpus_ids=corpus_ids), out_dir / 'submission.zip')}")


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", default=None, help="mặc định: paths.dev")
    ap.add_argument("--demo", action="store_true", help="corpus giả, không cần data/")
    ap.add_argument("--dry-run", action="store_true", help="in kế hoạch rồi thoát")
    ap.add_argument("--eval", action="store_true", help="file câu hỏi có nhãn → in Recall@k")
    ap.add_argument("--submission", action="store_true", help="sinh submission.zip")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-holdout", action="store_true")
    return ap.parse_args()


def main() -> int:
    """Điểm vào CLI: chạy các tầng đã bật trong YAML."""
    args = parse_args()
    cfg = load_config(args.config)
    pipe = cfg.get("pipeline", {})
    top_k_submit = int(cfg["retrieval"].get("top_k_submit", 5))
    top_k = max(int(cfg["retrieval"].get("ranking_full_k", 50)), top_k_submit)
    out_dir = resolve_out_dir(cfg, args.out_dir)
    stages = ["retrieve"] + [s for s in ("rerank", "calibrate") if pipe.get(s)]
    print(
        f"exp_id : {cfg.get('exp_id')}\ncommit : {git_commit()}\n"
        f"tầng   : {' → '.join(stages)}{' → submission' if args.submission else ''}\nra     : {out_dir}"
    )

    inp = load_inputs(cfg, args)
    print(f"câu hỏi: {len(inp.qids)} ({inp.q_path}) · chunk: {len(inp.chunks)} · top_k={top_k}")
    if args.dry_run:
        print("\n--dry-run: dừng ở đây, chưa nạp model, chưa index.")
        return 0

    retriever, ranked = stage_retrieve(cfg, inp.chunks, inp.texts, top_k, args.demo)
    if pipe.get("rerank"):
        ranked = stage_rerank(pipe["rerank"], inp.chunks, inp.texts, ranked)
    counts = stage_calibrate(pipe.get("calibrate"), ranked, len(inp.qids), top_k_submit)

    preds_full = {q: [d for d, _, _ in res] for q, res in zip(inp.qids, ranked)}
    preds = {q: preds_full[q][: counts[i]] for i, q in enumerate(inp.qids)}
    write_outputs(out_dir, ranked, preds, {
        "exp_id": cfg.get("exp_id"), "config": args.config, "commit": git_commit(),
        "questions": inp.q_path, "n_questions": len(inp.qids), "n_chunks": len(inp.chunks),
        "stages": stages, "top_k": top_k, "top_k_submit": top_k_submit,
        "retriever": retriever.stats(), "ngay": date.today().isoformat(), "qids": inp.qids,
    })
    if args.eval:
        from src.evaluate import load_truth

        print_eval(preds_full, preds, inp.truth or load_truth(repo_path(inp.q_path)), top_k)
    if args.submission:
        write_submission(cfg, preds, inp.qids, out_dir, args.demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
