#!/usr/bin/env python3
"""
Quét (w, rrf_k, fuse_level) cho `src/retrieval/hybrid.py`. CHỦ SỞ HỮU: P3.

ĐIỂM YẾU ĐANG SỬA. Prototype `scripts/p4_fuse.py` quét w trên **chính tập nó báo cáo** rồi in
"w tốt nhất = 0,6 → R@5 = ...". Con số đó lạc quan theo một lượng không đo được, và chính file
đó cũng tự cảnh báo ("w chọn trên chính tập này nên con số LẠC QUAN"). Script này tách đôi:

    chọn siêu tham số trên  --fit   (mặc định train_split.json, 4.689 câu)
    báo cáo MỘT lần trên    --eval  (mặc định dev.json, 1.000 câu)

và in thêm **độ lạc quan** — khoảng cách giữa "ô tốt nhất trên tập đo" và "ô đã chọn trên tập
fit, áp lên tập đo". Đó là cái giá của việc chọn trên tập đo, tính bằng số, cho bài báo.

CHI PHÍ. Mỗi nguồn được chấm điểm ĐÚNG MỘT LẦN cho toàn bộ câu hỏi (fit + eval gộp chung một
lượt), rồi cả lưới chạy trong RAM trên kết quả đó. Không có nó thì lưới 11×3×2 = 66 ô sẽ là 66
lần index 524.422 chunk cho cùng một kết quả.

    python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --demo
    python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --limit-fit 1500
    python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml \\
        --fit data/train_split.json --eval data/dev.json --log-csv --nguoi-chay P3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.common.io import load_chunks  # noqa: E402
from src.evaluate import eval_official, recall_at_k  # noqa: E402
from src.retrieval.base import build_retriever, retriever_spec  # noqa: E402
from src.retrieval.hybrid import HybridRetriever, rrf_from_ranks, sweep_weights  # noqa: E402

EXPERIMENTS_CSV = REPO / "experiments.csv"


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def parse_grid(spec: str) -> list[float]:
    """'0:1:0.1' → [0,0.1,...,1]  ·  '0,0.5,1' → [0,0.5,1]."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step))
        return [round(lo + i * step, 6) for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


def load_labelled(path: Path) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Đọc tập CÓ NHÃN → (qids, texts, gold). Tập test (answer=null) bị từ chối tại đây."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    qids, texts, gold = [], [], {}
    for q in raw:
        v = raw[q]
        if not isinstance(v, dict) or not v.get("answer"):
            raise SystemExit(
                f"❌ {path}: qid {q} không có nhãn. Quét siêu tham số cần nhãn ở CẢ HAI tập; "
                f"tập thi (public/private) không dùng được ở đây."
            )
        qids.append(str(q))
        texts.append(v["question"])
        gold[str(q)] = [str(a) for a in v["answer"]]
    return qids, texts, gold


# ─────────────────────────────────────────────────────────────────────────────
# Một ô lưới
# ─────────────────────────────────────────────────────────────────────────────
def rank_one(retriever, per_source_q, doc_ranks_q, level, weights, rrf_k, top_k) -> list[str]:
    """Thứ hạng doc của MỘT câu hỏi ở một ô lưới."""
    if level == "doc":
        # Thứ hạng doc của từng nguồn KHÔNG phụ thuộc (w, k) — đã tính sẵn một lần. Phần còn
        # lại dùng đúng `rrf_from_ranks` mà `_fuse_doc_level()` gọi, không phải bản sao.
        fused = rrf_from_ranks(doc_ranks_q, weights, rrf_k, retriever.absent_rank)
        return sorted(fused, key=lambda d: (-fused[d], d))[:top_k]
    idx, sc = retriever.fuse_query(per_source_q, weights, rrf_k, "chunk")
    return [d for d, _ in retriever.pool_candidates(idx, sc, top_k)]


def rank_all(retriever, per_source, doc_ranks, qids, level, weights, rrf_k, top_k) -> dict:
    return {
        qid: rank_one(
            retriever,
            {n: per_source[n][i] for n in per_source},
            {n: doc_ranks[n][i] for n in doc_ranks} if level == "doc" else None,
            level, weights, rrf_k, top_k,
        )
        for i, qid in enumerate(qids)
    }


def score_cell(ranked: dict, gold: dict, recall_ks: list[int], top_k_submit: int) -> dict:
    out = {f"recall@{k}": round(recall_at_k(ranked, gold, k), 4) for k in recall_ks}
    cut = {q: v[:top_k_submit] for q, v in ranked.items()}
    btc = eval_official(cut, gold)
    out["btc_recall"] = round(btc["recall"], 4)
    out["btc_precision"] = round(btc["precision"], 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fit", default=None, help="tập CHỌN siêu tham số; mặc định paths.train_split")
    ap.add_argument("--eval", dest="eval_path", default=None, help="tập BÁO CÁO; mặc định paths.dev")
    # Mặc định = None ⇒ lấy từ khối `tune_rrf:` của YAML, rồi mới đến giá trị dựng sẵn.
    # Quy ước repo (AGENTS.md §6): mỗi thí nghiệm là một file YAML, không hằng số trong code.
    ap.add_argument("--grid-w", default=None, help="trọng số của nguồn ĐẦU TIÊN trong config")
    ap.add_argument("--grid-k", default=None)
    ap.add_argument("--levels", default=None, help="mức hợp nhất đem so")
    ap.add_argument("--select-on", default=None, help="chỉ số dùng để CHỌN ô trên tập fit")
    ap.add_argument("--recall-at", default=None)
    ap.add_argument("--limit-fit", type=int, default=1500,
                    help="cắt bớt tập fit cho nhanh. 0 = dùng hết. Cắt là ĐÁNH ĐỔI: ô thắng "
                         "trên 1.500 câu có thể không phải ô thắng trên 4.689 câu.")
    ap.add_argument("--demo", action="store_true", help="corpus giả, không cần data/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-csv", action="store_true", help="thêm một dòng vào experiments.csv")
    ap.add_argument("--nguoi-chay", default="P3")
    a = ap.parse_args()

    cfg = yaml.safe_load((REPO / a.config).read_text(encoding="utf-8"))
    spec = retriever_spec(cfg, kind="hybrid", demo=a.demo)

    tcfg = cfg.get("tune_rrf") or {}

    def opt(name: str, fallback: str) -> str:
        """CLI > khối `tune_rrf:` của YAML > giá trị dựng sẵn."""
        return str(getattr(a, name) or tcfg.get(name) or fallback)

    top_k_submit = int(cfg["retrieval"].get("top_k_submit", 5))
    recall_ks = [int(x) for x in opt("recall_at", "5,20,50").split(",")]
    top_k = max(recall_ks + [top_k_submit])
    ws = parse_grid(opt("grid_w", "0:1:0.1"))
    ks = [int(x) for x in parse_grid(opt("grid_k", "20,60,100"))]
    levels = [x.strip() for x in opt("levels", "chunk,doc").split(",") if x.strip()]
    select_on = opt("select_on", "recall@5")
    out_dir = Path(a.out_dir or cfg["paths"].get("out_dir", f"outputs/{cfg.get('exp_id','tune_rrf')}"))
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    # ── hai tập, và các chốt chặn ────────────────────────────────────────────
    if a.demo:
        from scripts.bench_retrieval import demo_data

        chunks, questions, gold_all = demo_data(n_docs=80)
        ids = list(questions)
        half = len(ids) // 2
        fit_ids, eval_ids = ids[:half], ids[half:]
        fit = (fit_ids, [questions[q] for q in fit_ids], {q: gold_all[q] for q in fit_ids})
        ev = (eval_ids, [questions[q] for q in eval_ids], {q: gold_all[q] for q in eval_ids})
        fit_path = eval_path = "demo"
    else:
        fit_path = Path(a.fit or cfg["paths"]["train_split"])
        eval_path = Path(a.eval_path or cfg["paths"]["dev"])
        if fit_path.resolve() == eval_path.resolve():
            raise SystemExit(
                f"❌ --fit và --eval trỏ cùng một file ({fit_path}). Chọn siêu tham số rồi báo "
                f"cáo trên cùng tập chính là điểm yếu mà script này sinh ra để sửa."
            )
        if "holdout" in eval_path.name and not a.allow_holdout:
            raise SystemExit(
                f"❌ {eval_path} là tập ĐO LẦN CUỐI, chạm đúng một lần. Dùng data/dev.json, "
                f"hoặc --allow-holdout nếu cả nhóm đã chốt đây LÀ lần đo cuối."
            )
        if "holdout" in Path(fit_path).name:
            raise SystemExit(f"❌ Không được fit trên {fit_path} — đó là tập đo lần cuối.")
        chunks = load_chunks(REPO / cfg["paths"]["chunks"])
        fit = load_labelled(fit_path if fit_path.is_absolute() else REPO / fit_path)
        ev = load_labelled(eval_path if eval_path.is_absolute() else REPO / eval_path)
        if a.limit_fit and len(fit[0]) > a.limit_fit:
            fit = (fit[0][: a.limit_fit], fit[1][: a.limit_fit],
                   {q: fit[2][q] for q in fit[0][: a.limit_fit]})

    fit_ids, fit_texts, fit_gold = fit
    ev_ids, ev_texts, ev_gold = ev
    n_cells = len(ws) * len(ks) * len(levels)
    print(
        f"exp_id : {cfg.get('exp_id')}\ncommit : {git_commit()}\n"
        f"fit    : {fit_path} · {len(fit_ids)} câu\n"
        f"eval   : {eval_path} · {len(ev_ids)} câu\n"
        f"lưới   : w={ws}\n         k={ks} × mức={levels}  →  {n_cells} ô\n"
        f"chọn   : {select_on}\nra     : {out_dir}"
    )
    if a.dry_run:
        print("\n--dry-run: dừng ở đây, chưa index.")
        return 0

    # ── index MỘT lần, chấm điểm MỘT lần cho cả fit lẫn eval ─────────────────
    r = build_retriever(spec)
    if not isinstance(r, HybridRetriever):
        raise SystemExit(f"❌ config dựng ra {type(r).__name__}, script này chỉ quét `type: hybrid`.")
    names = list(r.sources)
    if select_on not in [f"recall@{k}" for k in recall_ks] + ["btc_recall", "btc_precision"]:
        raise SystemExit(f"❌ --select-on '{select_on}' không nằm trong các chỉ số sẽ tính.")

    t0 = time.perf_counter()
    r.index(chunks)
    print(f"\nindex {len(chunks)} chunk trong {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    all_texts = fit_texts + ev_texts
    per_source_all = r.source_candidates(all_texts)
    n_fit = len(fit_texts)
    per_fit = {n: per_source_all[n][:n_fit] for n in names}
    per_ev = {n: per_source_all[n][n_fit:] for n in names}
    print(f"chấm {len(all_texts)} câu × {len(names)} nguồn trong {time.perf_counter()-t0:.1f}s")

    # Thứ hạng doc của từng nguồn không phụ thuộc (w, k) → tính sẵn, cả lưới dùng lại.
    doc_fit = doc_ev = {}
    if "doc" in levels:
        t0 = time.perf_counter()
        doc_fit = {
            n: [[d for d, _ in r.sources[n].pool_candidates(i, s, r.doc_depth)] for i, s in per_fit[n]]
            for n in names
        }
        doc_ev = {
            n: [[d for d, _ in r.sources[n].pool_candidates(i, s, r.doc_depth)] for i, s in per_ev[n]]
            for n in names
        }
        print(f"gộp chunk→doc cho mức 'doc' trong {time.perf_counter()-t0:.1f}s")

    # ── lưới trên tập FIT ────────────────────────────────────────────────────
    rows: list[dict] = []
    t0 = time.perf_counter()
    for level in levels:
        for k in ks:
            for w in ws:
                weights = sweep_weights(names, w, r.weights)
                ranked = rank_all(r, per_fit, doc_fit, fit_ids, level, weights, k, top_k)
                rows.append(
                    {"level": level, "rrf_k": k, "w": w, "weights": {n: round(v, 4) for n, v in weights.items()},
                     **score_cell(ranked, fit_gold, recall_ks, top_k_submit)}
                )
                print(f"  fit · {level:5s} k={k:<4} w={w:<5} {select_on}={rows[-1][select_on]:.4f}",
                      flush=True)
    print(f"lưới {len(rows)} ô trong {time.perf_counter()-t0:.1f}s")

    # Hoà thì ưu tiên ô ĐƠN GIẢN hơn: w gần 1 (nghiêng về nguồn đầu, thường là BM25 —
    # rẻ, không GPU), rồi k nhỏ. Không để thứ tự duyệt lưới quyết định thay mình.
    best = max(rows, key=lambda x: (x[select_on], x["w"], -x["rrf_k"]))
    print(
        f"\n★ CHỌN TRÊN TẬP FIT: mức={best['level']} · rrf_k={best['rrf_k']} · w={best['w']} "
        f"→ {select_on}={best[select_on]:.4f}"
    )

    # ── đo MỘT lần trên tập EVAL với ô đã chọn ───────────────────────────────
    chosen_w = sweep_weights(names, best["w"], r.weights)
    chosen = score_cell(
        rank_all(r, per_ev, doc_ev, ev_ids, best["level"], chosen_w, best["rrf_k"], top_k),
        ev_gold, recall_ks, top_k_submit,
    )
    print(f"\n── ĐO TRÊN {eval_path} (ô cố định, không quét) ──")
    for k in recall_ks:
        print(f"   Recall@{k:<4}: {chosen[f'recall@{k}']:.4f}")
    print(f"   Chấm như BTC: recall={chosen['btc_recall']:.4f} precision={chosen['btc_precision']:.4f}")

    # Đường cơ sở: từng nguồn chạy một mình, đo trên cùng tập eval, cùng đường code.
    baselines = {}
    for n in names:
        solo = {x: (1.0 if x == n else 0.0) for x in names}
        baselines[n] = score_cell(
            rank_all(r, per_ev, doc_ev, ev_ids, best["level"], solo, best["rrf_k"], top_k),
            ev_gold, recall_ks, top_k_submit,
        )
        print(f"   [cơ sở] chỉ '{n}': recall@5={baselines[n]['recall@5']:.4f}")
    gain = chosen["recall@5"] - max(b["recall@5"] for b in baselines.values())
    print(f"   Hợp nhất so với nguồn đơn tốt nhất: {gain:+.4f} @5")

    # ── độ lạc quan: cái giá của việc chọn trên chính tập đo ─────────────────
    ev_rows = []
    for level in levels:
        for k in ks:
            for w in ws:
                wts = sweep_weights(names, w, r.weights)
                ev_rows.append(
                    {"level": level, "rrf_k": k, "w": w,
                     **score_cell(rank_all(r, per_ev, doc_ev, ev_ids, level, wts, k, top_k),
                                  ev_gold, recall_ks, top_k_submit)}
                )
    best_on_eval = max(ev_rows, key=lambda x: x[select_on])
    optimism = best_on_eval[select_on] - chosen[select_on]
    print(
        f"\n── Độ lạc quan (KHÔNG được dùng để chọn) ──\n"
        f"   ô tốt nhất NẾU quét trên chính tập đo: mức={best_on_eval['level']} "
        f"k={best_on_eval['rrf_k']} w={best_on_eval['w']} → {select_on}="
        f"{best_on_eval[select_on]:.4f}\n"
        f"   ô đã chọn trên tập fit, áp lên tập đo : {select_on}={chosen[select_on]:.4f}\n"
        f"   → chênh {optimism:+.4f}. Đây là lượng mà `p4_fuse.py` đã cộng thêm vào con số\n"
        f"     nó báo cáo, và là lượng phải trừ khi đọc mọi bảng quét-trên-tập-đo."
    )

    # ── ghi kết quả ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "exp_id": cfg.get("exp_id"), "config": a.config, "commit": git_commit(),
        "ngay": date.today().isoformat(),
        "fit": {"path": str(fit_path), "n": len(fit_ids)},
        "eval": {"path": str(eval_path), "n": len(ev_ids)},
        "select_on": select_on, "grid": {"w": ws, "rrf_k": ks, "levels": levels},
        "sources": {n: r.sources[n].stats() for n in names},
        "fit_grid": rows,
        "chosen": {"level": best["level"], "rrf_k": best["rrf_k"], "w": best["w"],
                   "weights": chosen_w, "fit": {m: best[m] for m in best if "@" in m or m.startswith("btc")}},
        "eval_result": chosen,
        "eval_baselines": baselines,
        "eval_grid": ev_rows,
        "optimism": round(optimism, 4),
    }
    (out_dir / "tune_rrf.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    (out_dir / "tune_rrf.md").write_text(render_report(result, recall_ks), encoding="utf-8")
    print(f"\n✅ {out_dir/'tune_rrf.json'}\n✅ {out_dir/'tune_rrf.md'}")

    if a.log_csv:
        log_experiment(cfg, a, result, best, chosen, gain, optimism)
        print(f"✅ thêm 1 dòng vào {EXPERIMENTS_CSV}")
    return 0


def render_report(res: dict, recall_ks: list[int]) -> str:
    L = [
        f"# tune_rrf — {res['exp_id']}",
        "",
        f"- config `{res['config']}` · commit `{res['commit']}` · {res['ngay']}",
        f"- **fit** `{res['fit']['path']}` (n={res['fit']['n']}) — chọn siêu tham số",
        f"- **eval** `{res['eval']['path']}` (n={res['eval']['n']}) — báo cáo, đo một lần",
        f"- chọn theo `{res['select_on']}`",
        "",
        "## Ô đã chọn",
        "",
        f"`fuse_level={res['chosen']['level']}` · `rrf_k={res['chosen']['rrf_k']}` · "
        f"`w={res['chosen']['w']}` → " + " · ".join(f"{k}={v}" for k, v in res['chosen']['weights'].items()),
        "",
        "| tập | " + " | ".join(f"R@{k}" for k in recall_ks) + " | BTC recall | BTC precision |",
        "|---|" + "---|" * (len(recall_ks) + 2),
    ]
    for tag, d in (("fit", res["chosen"]["fit"]), ("eval", res["eval_result"])):
        L.append(
            f"| {tag} | " + " | ".join(f"{d.get(f'recall@{k}', '—')}" for k in recall_ks)
            + f" | {d.get('btc_recall','—')} | {d.get('btc_precision','—')} |"
        )
    L += ["", "## Nguồn đơn lẻ trên tập eval (cùng đường code)", "",
          "| nguồn | " + " | ".join(f"R@{k}" for k in recall_ks) + " |",
          "|---|" + "---|" * len(recall_ks)]
    for n, d in res["eval_baselines"].items():
        L.append(f"| {n} | " + " | ".join(f"{d[f'recall@{k}']}" for k in recall_ks) + " |")
    L += [
        "",
        f"## Độ lạc quan: **{res['optimism']:+.4f}**",
        "",
        "Chênh lệch giữa ô tốt nhất nếu quét trên chính tập đo và ô chọn trên tập fit rồi áp",
        "lên tập đo. Mọi bảng quét-trên-tập-đo (kể cả `scripts/p4_fuse.py`) phải trừ lượng này",
        "trước khi so với một phương pháp không quét.",
        "",
        "## Lưới trên tập fit",
        "",
        "| mức | rrf_k | w | " + " | ".join(f"R@{k}" for k in recall_ks) + " | BTC P |",
        "|---|---:|---:|" + "---:|" * (len(recall_ks) + 1),
    ]
    for r in res["fit_grid"]:
        L.append(
            f"| {r['level']} | {r['rrf_k']} | {r['w']} | "
            + " | ".join(f"{r[f'recall@{k}']}" for k in recall_ks)
            + f" | {r['btc_precision']} |"
        )
    return "\n".join(L) + "\n"


def log_experiment(cfg, a, res, best, chosen, gain, optimism) -> None:
    raw = EXPERIMENTS_CSV.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with EXPERIMENTS_CSV.open("ab") as fh:
            fh.write(b"\n")
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header).writerow({
            "exp_id": f"{cfg.get('exp_id')}_{best['level']}_w{best['w']}_k{best['rrf_k']}",
            "ngay": date.today().isoformat(),
            "nguoi_chay": a.nguoi_chay,
            "commit": res["commit"],
            "config": a.config,
            "tap_do": Path(res["eval"]["path"]).name,
            "recall": chosen["btc_recall"],
            "precision": chosen["btc_precision"],
            "recall_lb": "", "precision_lb": "",
            "ghi_chu": (
                f"hybrid RRF {list(res['sources'])} · fuse_level={best['level']} rrf_k={best['rrf_k']} "
                f"w={best['w']} · siêu tham số chọn trên {Path(res['fit']['path']).name} "
                f"(n={res['fit']['n']}), đo trên {Path(res['eval']['path']).name} "
                f"(n={res['eval']['n']}) · hơn nguồn đơn tốt nhất {gain:+.4f} @5 · "
                f"độ lạc quan nếu quét trên tập đo {optimism:+.4f}"
            ),
            "nhom_so_sanh": "chien_luoc_du_lieu",
            "gia_thuyet_lien_quan": (
                "H4: hợp nhất THỨ HẠNG giữ bề rộng của nguồn từ vựng và lấy phần đỉnh của nguồn "
                "ngữ nghĩa; cộng ĐIỂM thì không được vì hai thang không so được. Kèm câu hỏi phụ: "
                "hợp nhất ở mức chunk hay mức doc thì tốt hơn."
            ),
            "diem_yeu_khac_phuc_tu_exp_truoc": (
                "scripts/p4_fuse.py quét w trên chính tập nó báo cáo nên con số lạc quan một lượng "
                f"không đo được; ở đây w/k chọn trên tập fit tách rời và lượng lạc quan đó được đo "
                f"thành số ({optimism:+.4f})."
            ),
        })


if __name__ == "__main__":
    raise SystemExit(main())
