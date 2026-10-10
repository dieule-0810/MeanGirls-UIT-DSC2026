#!/usr/bin/env python3
"""Lưới benchmark BM25 tầng 1: tokenizer × chiến lược gộp chunk→doc.

Chấm điểm chunk MỘT lần cho mỗi tokenizer rồi thử mọi chiến lược gộp trên cùng tập ứng viên —
lưới 5×5 nhưng chỉ 5 lần index. Trả lời H1/H1b/H2 (configs/v0.2_bm25_tokenizer.yaml); kết quả:
syllable_bigram + mean_top2 thắng. Metric chính là **Recall@50**.

Typical usage example:

    python scripts/bench_retrieval.py --demo
    python scripts/bench_retrieval.py --config configs/v0.2_bm25_tokenizer.yaml --questions data/dev.json
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
from src.common.demo import synthetic_corpus  # noqa: E402
from src.common.io import load_chunks, load_questions  # noqa: E402
from src.common.runinfo import append_experiment_rows, git_commit, guard_holdout  # noqa: E402
from src.evaluate import eval_official, load_truth, recall_at_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.tokenizers import available_tokenizers  # noqa: E402


@dataclass(frozen=True)
class BenchGrid:
    """Lưới cần chạy.

    Attributes:
        tokenizers: Tên tokenizer.
        pools: Chiến lược gộp.
        recall_ks: Các mốc Recall@k.
        top_k: Độ sâu ranking.
        bm25_opts: Spec BM25 dùng chung (không có `tokenizer`).
    """

    tokenizers: list[str]
    pools: list[str]
    recall_ks: list[int]
    top_k: int
    bm25_opts: dict


@dataclass
class BenchData:
    """Kho chunk và tập câu hỏi có nhãn."""

    chunks: list[dict]
    qids: list[str]
    texts: list[str]
    gold: dict[str, list[str]]
    source: str


def _pool_row(r: BM25Retriever, cands: list, data: BenchData, pool: str, grid: BenchGrid) -> dict:
    """Một ô (tokenizer, pool): gộp ứng viên đã chấm sẵn rồi tính Recall@k + điểm BTC top-5."""
    t0 = time.perf_counter()
    preds = {q: [d for d, _ in r.pool_candidates(idx, sc, grid.top_k, pool=pool)] for q, (idx, sc) in zip(data.qids, cands)}
    row = {"tokenizer": r.tokenizer.key, "pool": pool, "vocab": len(r.vocab),
           "index_s": round(r.stats()["index_seconds"], 1), "pool_s": round(time.perf_counter() - t0, 2)}
    for k in grid.recall_ks:
        row[f"recall@{k}"] = round(recall_at_k(preds, data.gold, k), 4)
    btc = eval_official({q: v[:5] for q, v in preds.items()}, data.gold)
    row["btc_recall@5"] = round(btc["recall"], 4)
    row["btc_precision@5"] = round(btc["precision"], 4)
    return row


def _bench_tokenizer(tok_name: str, data: BenchData, grid: BenchGrid) -> list[dict]:
    """Index một lần với `tok_name`, chấm ứng viên một lần, thử mọi pool."""
    print(f"\n══ tokenizer: {tok_name} ══")
    r = BM25Retriever.from_spec({**grid.bm25_opts, "tokenizer": tok_name})
    r.index(data.chunks)
    t0 = time.perf_counter()
    cands = r.candidates(data.texts, max(r.candidate_chunks, grid.top_k))
    score_s = round(time.perf_counter() - t0, 1)
    n_empty = sum(1 for idx, _ in cands if len(idx) == 0)
    print(f"  chấm điểm {len(data.texts)} câu hỏi: {score_s}s, {n_empty} câu không khớp term nào")
    rows = []
    for pool in grid.pools:
        row = {**_pool_row(r, cands, data, pool, grid), "score_s": score_s, "n_empty": n_empty}
        rows.append(row)
        print(f"  pool={pool:<12} " + "  ".join(f"R@{k}={row[f'recall@{k}']:.4f}" for k in grid.recall_ks)
              + f"  P@5={row['btc_precision@5']:.4f}")
    return rows


def run_grid(data: BenchData, grid: BenchGrid) -> list[dict]:
    """Chạy cả lưới; tokenizer chưa cài thì bỏ qua (có báo) thay vì chết giữa lưới.

    Returns:
        Mỗi ô một dict chỉ số.
    """
    have = available_tokenizers()
    rows: list[dict] = []
    for tok_name in grid.tokenizers:
        if not have.get(tok_name, False):
            print(f"\n⏭  Bỏ qua tokenizer '{tok_name}': chưa cài. cài pyvi/underthesea (requirements.txt) nếu cần ô này.")
            continue
        rows.extend(_bench_tokenizer(tok_name, data, grid))
    return rows


def render_report(rows: list[dict], recall_ks: list[int], meta: dict) -> str:
    """Bảng kết quả → markdown, sắp theo Recall@50. Phần **Nhận xét** để điền tay."""
    kpi = f"recall@{50 if 50 in recall_ks else recall_ks[-1]}"
    best = max(rows, key=lambda r: r[kpi]) if rows else None
    cols = ["tokenizer", "pool"] + [f"recall@{k}" for k in recall_ks] + [
        "btc_recall@5", "btc_precision@5", "vocab", "index_s", "score_s"]
    lines = [
        "# Bench tầng 1 — tokenizer × gộp chunk→doc", "",
        "> Sinh bởi `scripts/bench_retrieval.py`. Phần **Nhận xét** điền tay — đó mới là thứ",
        "> đi vào bài báo (plan.md mục 0.6: mỗi phương pháp phải nói rõ yếu ở đâu).", "",
        "```json", json.dumps(meta, ensure_ascii=False, indent=2), "```", "",
        "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in sorted(rows, key=lambda r: -r[kpi]):
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", f"**Sắp xếp theo {kpi} — KPI của P3.**", ""]
    if best:
        lines += [f"Cấu hình tốt nhất: `tokenizer={best['tokenizer']}`, `pool={best['pool']}` "
                  f"→ {kpi} = {best[kpi]:.4f}, Precision@5 = {best['btc_precision@5']:.4f}.", ""]
    lines += ["## Nhận xét (điền tay)", "",
              "- H1 — word-segment hơn âm tiết thuần? ",
              "- H1b — bigram âm tiết lấy lại được bao nhiêu phần của word-segment? ",
              "- H2 — chiến lược gộp nào thắng, và vì sao (văn bản dài có bị `sum` thiên vị không)? ",
              "- Câu `n_empty` (không khớp term nào) rơi vào nhóm văn bản nào? ",
              "- Việc tiếp theo để nâng Recall@50: "]
    return "\n".join(lines) + "\n"


def experiment_rows(rows: list[dict], cfg_path: str, nguoi_chay: str, tap_do: str) -> list[dict]:
    """Mỗi ô lưới một dòng `experiments.csv`."""
    commit = git_commit()
    return [{
        "exp_id": f"v0.2_bm25_{r['tokenizer']}_{r['pool']}", "ngay": date.today().isoformat(),
        "nguoi_chay": nguoi_chay, "commit": commit, "config": cfg_path, "tap_do": tap_do,
        "recall": r.get("btc_recall@5", ""), "precision": r.get("btc_precision@5", ""),
        "recall_lb": "", "precision_lb": "",
        "ghi_chu": " ".join(f"{k}={v}" for k, v in r.items() if k.startswith("recall@")) + f" vocab={r['vocab']}",
        "nhom_so_sanh": "co_dien",
        "gia_thuyet_lien_quan": "H1 word-segment > âm tiết thuần; H2 gộp chunk→doc",
        "diem_yeu_khac_phuc_tu_exp_truoc": (
            "v0.1 dùng regex \\w+ (âm tiết rời) + pool=max: term phổ thông khớp nhiễu, "
            "và văn bản khớp nhiều điều khoản không được cộng điểm"
        ),
    } for r in rows]


def build_grid(cfg: dict, args: argparse.Namespace) -> BenchGrid:
    """Lưới từ khối `bench:` của YAML, ghi đè bằng CLI."""
    bench = cfg.get("bench", {})
    recall_ks = bench.get("recall_at", [5, 20, 50, 100])
    bm25_opts = dict(cfg["retrieval"].get("bm25", {}))
    bm25_opts.pop("tokenizer", None)
    if cfg.get("paths", {}).get("cache_dir") and not args.demo:
        bm25_opts.setdefault("cache_dir", cfg["paths"]["cache_dir"])
    return BenchGrid(
        tokenizers=args.tokenizers.split(",") if args.tokenizers else bench.get("tokenizers", ["regex"]),
        pools=args.pools.split(",") if args.pools else bench.get("pools", ["max"]),
        recall_ks=recall_ks,
        top_k=max(recall_ks + [cfg["retrieval"].get("top_k_retrieve", 100)]),
        bm25_opts=bm25_opts,
    )


def load_data(cfg: dict, args: argparse.Namespace) -> BenchData:
    """Corpus giả (`--demo`) hoặc kho chunk + tập có nhãn (mặc định `paths.dev`, KHÔNG holdout)."""
    if args.demo:
        print("⚠️  CHẾ ĐỘ DEMO — corpus giả, con số KHÔNG dùng để kết luận gì.")
        chunks, questions, gold = synthetic_corpus()
        qids = list(questions)
        data = BenchData(chunks, qids, [questions[q] for q in qids], gold, "demo")
    else:
        q_path = args.questions or cfg["paths"].get("dev") or cfg["paths"]["holdout"]
        guard_holdout(q_path, args.allow_holdout)
        qids, texts = load_questions(repo_path(q_path))
        data = BenchData(load_chunks(repo_path(args.chunks or cfg["paths"]["chunks"])), qids, texts,
                         load_truth(repo_path(q_path)), str(q_path))
    n_q = args.n_questions or cfg.get("bench", {}).get("n_questions")
    if n_q:
        data.qids, data.texts = data.qids[:n_q], data.texts[:n_q]
        data.gold = {q: data.gold[q] for q in data.qids}
    return data


def write_reports(cfg: dict, args: argparse.Namespace, rows: list[dict], report: str, meta: dict) -> None:
    """Ghi `bench_results.json` và `bench_report.md` (mặc định vào `paths.out_dir`)."""
    out_dir = resolve_out_dir(cfg, fallback="bench")
    results_path = repo_path(args.results) if args.results else out_dir / "bench_results.json"
    report_path = repo_path(args.report) if args.report else out_dir / "bench_report.md"
    for p in (results_path, report_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report + f"✅ {report_path}\n✅ {results_path}")


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v0.2_bm25_tokenizer.yaml")
    ap.add_argument("--demo", action="store_true", help="chạy trên corpus giả, không cần data/")
    ap.add_argument("--chunks", default=None, help="ghi đè paths.chunks")
    ap.add_argument("--questions", default=None, help="ghi đè paths.dev (file có nhãn)")
    ap.add_argument("--allow-holdout", action="store_true", help="chỉ khi cả nhóm đã chốt đây là lần đo cuối")
    ap.add_argument("--tokenizers", default=None, help="danh sách phẩy, ghi đè bench.tokenizers")
    ap.add_argument("--pools", default=None, help="danh sách phẩy, ghi đè bench.pools")
    ap.add_argument("--n-questions", type=int, default=None, help="chỉ lấy N câu đầu (chạy nhanh)")
    ap.add_argument("--report", default=None, help="file markdown xuất ra")
    ap.add_argument("--results", default=None,
                    help="file JSON xuất ra (mặc định out_dir/bench_results.json)")
    ap.add_argument("--log-experiments", action="store_true", help="thêm dòng vào experiments.csv")
    ap.add_argument("--nguoi-chay", default="P3")
    return ap.parse_args()


def main() -> int:
    """Điểm vào CLI: chạy lưới, ghi báo cáo, (tuỳ chọn) ghi experiments.csv."""
    args = parse_args()
    cfg = load_config(args.config)
    grid = build_grid(cfg, args)
    data = load_data(cfg, args)
    print(f"{len(data.chunks)} chunk, {len(data.qids)} câu hỏi, top_k={grid.top_k}")
    rows = run_grid(data, grid)
    if not rows:
        print("❌ Không ô nào chạy được — kiểm tra danh sách tokenizer.")
        return 1
    meta = {
        "exp_id": cfg.get("exp_id"), "config": args.config, "commit": git_commit(), "questions": data.source,
        "demo": args.demo, "n_chunks": len(data.chunks), "n_questions": len(data.qids), "top_k": grid.top_k,
        "bm25": {k: v for k, v in grid.bm25_opts.items() if k not in ("cache_dir", "verbose")},
        "ngay": date.today().isoformat(),
    }
    write_reports(cfg, args, rows, render_report(rows, grid.recall_ks, meta), meta)
    if args.log_experiments:
        if args.demo:
            print("⏭  Bỏ qua experiments.csv: số liệu demo không phải thí nghiệm thật.")
        else:
            n = append_experiment_rows(experiment_rows(rows, args.config, args.nguoi_chay, Path(data.source).stem))
            print(f"📝 Đã thêm {n} dòng vào experiments.csv — nhớ `git pull --rebase` trước khi commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
