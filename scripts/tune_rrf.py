#!/usr/bin/env python3
"""Quét (w, rrf_k, fuse_level) cho `src/retrieval/hybrid.py` — chọn trên tập fit, báo cáo một lần. P3.

ĐIỂM YẾU ĐANG SỬA. Prototype `scripts/p4_fuse.py` quét w trên CHÍNH tập nó báo cáo, nên con số
lạc quan một lượng không đo được. Script này tách đôi:

* chọn siêu tham số trên `--fit` (mặc định `paths.train_split`, 4.689 câu);
* báo cáo MỘT lần trên `--eval` (mặc định `paths.dev`, 1.000 câu);

và in thêm **độ lạc quan** — chênh giữa "ô tốt nhất nếu quét trên tập đo" và "ô chọn trên tập
fit, áp lên tập đo". Đó là cái giá của việc chọn trên tập đo, tính thành số cho bài báo.

CHI PHÍ. Mỗi nguồn được chấm điểm ĐÚNG MỘT LẦN cho fit + eval gộp chung, rồi cả lưới chạy trong
RAM — không index lại 432.142 chunk ở mỗi ô.

Typical usage example:

    python scripts/tune_rrf.py --config configs/v0.8_hybrid_demo.yaml --demo
    python scripts/tune_rrf.py --config configs/v0.8_hybrid_rrf.yaml --limit-fit 0 --log-csv
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
from src.common.io import load_chunks, load_labelled  # noqa: E402
from src.common.runinfo import append_experiment_rows, git_commit, guard_holdout  # noqa: E402
from src.evaluate import eval_official, recall_at_k  # noqa: E402
from src.retrieval.base import build_retriever, retriever_spec  # noqa: E402
from src.retrieval.hybrid import HybridRetriever, rrf_from_ranks, sweep_weights  # noqa: E402


def parse_grid(spec: str) -> list[float]:
    """Chuỗi lưới → danh sách giá trị: `'0:1:0.1'` → [0, 0.1, ..., 1]; `'0,0.5,1'` → [0, 0.5, 1]."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step))
        return [round(lo + i * step, 6) for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


@dataclass(frozen=True)
class GridSettings:
    """Lưới và cách chấm, đã giải từ CLI > khối `tune_rrf:` của YAML > mặc định."""

    ws: list[float]
    ks: list[int]
    levels: list[str]
    select_on: str
    recall_ks: list[int]
    top_k_submit: int

    @property
    def top_k(self) -> int:
        """Độ sâu ranking cần để tính mọi chỉ số."""
        return max(self.recall_ks + [self.top_k_submit])

    @property
    def n_cells(self) -> int:
        """Số ô của lưới."""
        return len(self.ws) * len(self.ks) * len(self.levels)

    def cells(self):
        """Duyệt mọi ô `(level, rrf_k, w)` theo đúng thứ tự in báo cáo."""
        for level in self.levels:
            for k in self.ks:
                for w in self.ws:
                    yield level, k, w


def grid_settings(cfg: dict, args: argparse.Namespace) -> GridSettings:
    """Giải lưới: CLI > khối `tune_rrf:` > giá trị dựng sẵn (không hằng số trong code)."""
    tcfg = cfg.get("tune_rrf") or {}

    def opt(name: str, fallback: str) -> str:
        return str(getattr(args, name) or tcfg.get(name) or fallback)

    return GridSettings(
        ws=parse_grid(opt("grid_w", "0:1:0.1")),
        ks=[int(x) for x in parse_grid(opt("grid_k", "20,60,100"))],
        levels=[x.strip() for x in opt("levels", "chunk,doc").split(",") if x.strip()],
        select_on=opt("select_on", "recall@5"),
        recall_ks=[int(x) for x in opt("recall_at", "5,20,50").split(",")],
        top_k_submit=int(cfg["retrieval"].get("top_k_submit", 5)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hai tập câu hỏi
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LabelledSet:
    """Một tập có nhãn: `path` chỉ để ghi log."""

    path: str
    ids: list[str]
    texts: list[str]
    gold: dict[str, list[str]]


def _demo_sets() -> tuple[list[dict], LabelledSet, LabelledSet]:
    """Chia corpus giả làm đôi thành tập fit và tập eval."""
    chunks, questions, gold = synthetic_corpus(n_docs=80)
    ids = list(questions)
    half = len(ids) // 2

    def make(part: list[str]) -> LabelledSet:
        return LabelledSet("demo", part, [questions[q] for q in part], {q: gold[q] for q in part})

    return chunks, make(ids[:half]), make(ids[half:])


def load_sets(cfg: dict, args: argparse.Namespace) -> tuple[list[dict], LabelledSet, LabelledSet]:
    """Nạp kho chunk, tập fit, tập eval — với mọi chốt chặn chống chọn tham số trên tập đo.

    Raises:
        SystemExit: fit và eval là một file, hoặc chạm holdout chưa được phép.
    """
    if args.demo:
        return _demo_sets()
    fit_path = Path(args.fit or cfg["paths"]["train_split"])
    eval_path = Path(args.eval_path or cfg["paths"]["dev"])
    if repo_path(fit_path).resolve() == repo_path(eval_path).resolve():
        raise SystemExit(
            f"❌ --fit và --eval trỏ cùng một file ({fit_path}). Chọn siêu tham số rồi báo cáo "
            f"trên cùng tập chính là điểm yếu mà script này sinh ra để sửa."
        )
    guard_holdout(eval_path, args.allow_holdout)
    guard_holdout(fit_path, False, purpose="fit")
    fit = LabelledSet(str(fit_path), *load_labelled(repo_path(fit_path)))
    ev = LabelledSet(str(eval_path), *load_labelled(repo_path(eval_path)))
    if args.limit_fit and len(fit.ids) > args.limit_fit:
        keep = fit.ids[: args.limit_fit]
        fit = LabelledSet(fit.path, keep, fit.texts[: args.limit_fit], {q: fit.gold[q] for q in keep})
    return load_chunks(repo_path(cfg["paths"]["chunks"])), fit, ev


# ─────────────────────────────────────────────────────────────────────────────
# Chấm lưới
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScoredSplit:
    """Ứng viên đã chấm sẵn của một tập — không phụ thuộc (w, k), cả lưới dùng lại.

    Attributes:
        data: Tập câu hỏi.
        per_source: `{nguồn: [(chỉ số chunk, điểm) mỗi câu]}`.
        doc_ranks: `{nguồn: [[doc_id xếp hạng] mỗi câu]}` cho mức `doc`; rỗng nếu không quét mức doc.
    """

    data: LabelledSet
    per_source: dict
    doc_ranks: dict


class GridScorer:
    """Xếp hạng và chấm một ô lưới, dùng ĐÚNG các hàm production gọi (`fuse_query`, `rrf_from_ranks`)."""

    def __init__(self, retriever: HybridRetriever, settings: GridSettings) -> None:
        self.r = retriever
        self.s = settings
        self.names = list(retriever.sources)

    def prepare(self, data: LabelledSet, per_source: dict, with_doc: bool) -> ScoredSplit:
        """Gộp chunk→doc cho từng nguồn một lần (mức doc), đóng gói thành `ScoredSplit`."""
        doc_ranks = {}
        if with_doc:
            doc_ranks = {
                n: [[d for d, _ in self.r.sources[n].pool_candidates(i, s, self.r.doc_depth)] for i, s in per_source[n]]
                for n in self.names
            }
        return ScoredSplit(data, per_source, doc_ranks)

    def _rank_one(self, split: ScoredSplit, qi: int, level: str, weights: dict, rrf_k: int) -> list[str]:
        """Thứ hạng doc của câu thứ `qi` ở một ô lưới."""
        if level == "doc":
            fused = rrf_from_ranks({n: split.doc_ranks[n][qi] for n in self.names}, weights, rrf_k, self.r.absent_rank)
            return sorted(fused, key=lambda d: (-fused[d], d))[: self.s.top_k]
        one = {n: split.per_source[n][qi] for n in self.names}
        idx, sc = self.r.fuse_query(one, weights, rrf_k, "chunk")
        return [d for d, _ in self.r.pool_candidates(idx, sc, self.s.top_k)]

    def rank(self, split: ScoredSplit, level: str, weights: dict, rrf_k: int) -> dict[str, list[str]]:
        """Thứ hạng doc của mọi câu trong tập ở một ô lưới."""
        return {qid: self._rank_one(split, i, level, weights, rrf_k) for i, qid in enumerate(split.data.ids)}

    def score(self, split: ScoredSplit, level: str, weights: dict, rrf_k: int) -> dict:
        """Recall@k của ranking và điểm BTC của top-`top_k_submit`."""
        ranked = self.rank(split, level, weights, rrf_k)
        gold = split.data.gold
        out = {f"recall@{k}": round(recall_at_k(ranked, gold, k), 4) for k in self.s.recall_ks}
        btc = eval_official({q: v[: self.s.top_k_submit] for q, v in ranked.items()}, gold)
        out["btc_recall"] = round(btc["recall"], 4)
        out["btc_precision"] = round(btc["precision"], 4)
        return out

    def sweep(self, split: ScoredSplit, tag: str) -> list[dict]:
        """Chấm mọi ô lưới trên một tập."""
        rows = []
        for level, k, w in self.s.cells():
            weights = sweep_weights(self.names, w, self.r.weights)
            rows.append({"level": level, "rrf_k": k, "w": w,
                         "weights": {n: round(v, 4) for n, v in weights.items()},
                         **self.score(split, level, weights, k)})
            if tag:
                print(f"  {tag} · {level:5s} k={k:<4} w={w:<5} {self.s.select_on}={rows[-1][self.s.select_on]:.4f}", flush=True)
        return rows

    def baselines(self, split: ScoredSplit, level: str, rrf_k: int) -> dict[str, dict]:
        """Từng nguồn chạy một mình trên cùng tập, cùng đường code."""
        return {n: self.score(split, level, {x: float(x == n) for x in self.names}, rrf_k) for n in self.names}


# ─────────────────────────────────────────────────────────────────────────────
# Báo cáo
# ─────────────────────────────────────────────────────────────────────────────
def render_report(res: dict, recall_ks: list[int]) -> str:
    """Kết quả quét → markdown (`tune_rrf.md`)."""
    r_cols = " | ".join(f"R@{k}" for k in recall_ks)
    lines = [
        f"# tune_rrf — {res['exp_id']}", "",
        f"- config `{res['config']}` · commit `{res['commit']}` · {res['ngay']}",
        f"- **fit** `{res['fit']['path']}` (n={res['fit']['n']}) — chọn siêu tham số",
        f"- **eval** `{res['eval']['path']}` (n={res['eval']['n']}) — báo cáo, đo một lần",
        f"- chọn theo `{res['select_on']}`", "", "## Ô đã chọn", "",
        f"`fuse_level={res['chosen']['level']}` · `rrf_k={res['chosen']['rrf_k']}` · `w={res['chosen']['w']}` → "
        + " · ".join(f"{k}={v}" for k, v in res["chosen"]["weights"].items()),
        "", f"| tập | {r_cols} | BTC recall | BTC precision |", "|---|" + "---|" * (len(recall_ks) + 2),
    ]
    for tag, d in (("fit", res["chosen"]["fit"]), ("eval", res["eval_result"])):
        lines.append(f"| {tag} | " + " | ".join(f"{d.get(f'recall@{k}', '—')}" for k in recall_ks)
                     + f" | {d.get('btc_recall', '—')} | {d.get('btc_precision', '—')} |")
    lines += ["", "## Nguồn đơn lẻ trên tập eval (cùng đường code)", "", f"| nguồn | {r_cols} |",
              "|---|" + "---|" * len(recall_ks)]
    for n, d in res["eval_baselines"].items():
        lines.append(f"| {n} | " + " | ".join(f"{d[f'recall@{k}']}" for k in recall_ks) + " |")
    lines += [
        "", f"## Độ lạc quan: **{res['optimism']:+.4f}**", "",
        "Chênh lệch giữa ô tốt nhất nếu quét trên chính tập đo và ô chọn trên tập fit rồi áp",
        "lên tập đo. Mọi bảng quét-trên-tập-đo (kể cả `scripts/p4_fuse.py`) phải trừ lượng này",
        "trước khi so với một phương pháp không quét.", "", "## Lưới trên tập fit", "",
        f"| mức | rrf_k | w | {r_cols} | BTC P |", "|---|---:|---:|" + "---:|" * (len(recall_ks) + 1),
    ]
    for r in res["fit_grid"]:
        lines.append(f"| {r['level']} | {r['rrf_k']} | {r['w']} | "
                     + " | ".join(f"{r[f'recall@{k}']}" for k in recall_ks) + f" | {r['btc_precision']} |")
    return "\n".join(lines) + "\n"


def experiment_row(res: dict, args: argparse.Namespace, gain: float) -> dict:
    """Một dòng `experiments.csv` cho ô đã chọn (3 cột cuối điền cụ thể, không chung chung)."""
    best, chosen = res["chosen"], res["eval_result"]
    fit_name, ev_name = Path(res["fit"]["path"]).name, Path(res["eval"]["path"]).name
    return {
        "exp_id": f"{res['exp_id']}_{best['level']}_w{best['w']}_k{best['rrf_k']}",
        "ngay": date.today().isoformat(), "nguoi_chay": args.nguoi_chay, "commit": res["commit"],
        "config": args.config, "tap_do": ev_name,
        "recall": chosen["btc_recall"], "precision": chosen["btc_precision"], "recall_lb": "", "precision_lb": "",
        "ghi_chu": (
            f"hybrid RRF {list(res['sources'])} · fuse_level={best['level']} rrf_k={best['rrf_k']} w={best['w']} · "
            f"siêu tham số chọn trên {fit_name} (n={res['fit']['n']}), đo trên {ev_name} (n={res['eval']['n']}) · "
            f"hơn nguồn đơn tốt nhất {gain:+.4f} @5 · độ lạc quan nếu quét trên tập đo {res['optimism']:+.4f}"
        ),
        "nhom_so_sanh": "chien_luoc_du_lieu",
        "gia_thuyet_lien_quan": (
            "H4: hợp nhất THỨ HẠNG giữ bề rộng của nguồn từ vựng và lấy phần đỉnh của nguồn ngữ nghĩa; "
            "cộng ĐIỂM thì không được vì hai thang không so được. Kèm câu hỏi phụ: mức chunk hay mức doc."
        ),
        "diem_yeu_khac_phuc_tu_exp_truoc": (
            "scripts/p4_fuse.py quét w trên chính tập nó báo cáo nên con số lạc quan một lượng không đo được; "
            f"ở đây w/k chọn trên tập fit tách rời và lượng lạc quan đó được đo thành số ({res['optimism']:+.4f})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_and_score(cfg: dict, args: argparse.Namespace, chunks: list[dict], fit: LabelledSet, ev: LabelledSet,
                    settings: GridSettings) -> tuple[GridScorer, ScoredSplit, ScoredSplit]:
    """Index MỘT lần, chấm mọi nguồn MỘT lần cho cả fit lẫn eval.

    Raises:
        SystemExit: Config không dựng ra `HybridRetriever`, hoặc `select_on` không nằm trong chỉ số sẽ tính.
    """
    r = build_retriever(retriever_spec(cfg, kind="hybrid", demo=args.demo))
    if not isinstance(r, HybridRetriever):
        raise SystemExit(f"❌ config dựng ra {type(r).__name__}, script này chỉ quét `type: hybrid`.")
    if settings.select_on not in [f"recall@{k}" for k in settings.recall_ks] + ["btc_recall", "btc_precision"]:
        raise SystemExit(f"❌ --select-on '{settings.select_on}' không nằm trong các chỉ số sẽ tính.")
    t0 = time.perf_counter()
    r.index(chunks)
    print(f"\nindex {len(chunks)} chunk trong {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    per_all = r.source_candidates(fit.texts + ev.texts)
    n_fit = len(fit.texts)
    print(f"chấm {len(fit.texts) + len(ev.texts)} câu × {len(r.sources)} nguồn trong {time.perf_counter() - t0:.1f}s")
    scorer = GridScorer(r, settings)
    with_doc = "doc" in settings.levels
    split_fit = scorer.prepare(fit, {n: v[:n_fit] for n, v in per_all.items()}, with_doc)
    split_ev = scorer.prepare(ev, {n: v[n_fit:] for n, v in per_all.items()}, with_doc)
    return scorer, split_fit, split_ev


def run_tuning(scorer: GridScorer, split_fit: ScoredSplit, split_ev: ScoredSplit) -> dict:
    """Quét trên fit, chọn ô, đo một lần trên eval, đo nguồn đơn lẻ và độ lạc quan."""
    s = scorer.s
    rows = scorer.sweep(split_fit, "fit")
    # Hoà thì ưu tiên ô ĐƠN GIẢN hơn: w gần 1 (nghiêng về nguồn đầu — rẻ, không GPU), rồi k nhỏ.
    best = max(rows, key=lambda x: (x[s.select_on], x["w"], -x["rrf_k"]))
    print(f"\n★ CHỌN TRÊN TẬP FIT: mức={best['level']} · rrf_k={best['rrf_k']} · w={best['w']} → {s.select_on}={best[s.select_on]:.4f}")
    chosen_w = sweep_weights(scorer.names, best["w"], scorer.r.weights)
    chosen = scorer.score(split_ev, best["level"], chosen_w, best["rrf_k"])
    baselines = scorer.baselines(split_ev, best["level"], best["rrf_k"])
    ev_rows = scorer.sweep(split_ev, "")
    best_on_eval = max(ev_rows, key=lambda x: x[s.select_on])
    return {
        "fit_grid": rows, "eval_grid": ev_rows, "eval_result": chosen, "eval_baselines": baselines,
        "chosen": {"level": best["level"], "rrf_k": best["rrf_k"], "w": best["w"], "weights": chosen_w,
                   "fit": {m: best[m] for m in best if "@" in m or m.startswith("btc")}},
        "best_on_eval": best_on_eval,
        "optimism": round(best_on_eval[s.select_on] - chosen[s.select_on], 4),
    }


def print_eval_summary(res: dict, settings: GridSettings, eval_path: str) -> float:
    """In kết quả trên tập eval; trả mức hơn nguồn đơn tốt nhất ở @5."""
    chosen, sel = res["eval_result"], settings.select_on
    print(f"\n── ĐO TRÊN {eval_path} (ô cố định, không quét) ──")
    for k in settings.recall_ks:
        print(f"   Recall@{k:<4}: {chosen[f'recall@{k}']:.4f}")
    print(f"   Chấm như BTC: recall={chosen['btc_recall']:.4f} precision={chosen['btc_precision']:.4f}")
    for n, b in res["eval_baselines"].items():
        print(f"   [cơ sở] chỉ '{n}': recall@5={b['recall@5']:.4f}")
    gain = chosen["recall@5"] - max(b["recall@5"] for b in res["eval_baselines"].values())
    print(f"   Hợp nhất so với nguồn đơn tốt nhất: {gain:+.4f} @5")
    b = res["best_on_eval"]
    print(
        f"\n── Độ lạc quan (KHÔNG được dùng để chọn) ──\n"
        f"   ô tốt nhất NẾU quét trên chính tập đo: mức={b['level']} k={b['rrf_k']} w={b['w']} → {sel}={b[sel]:.4f}\n"
        f"   ô đã chọn trên tập fit, áp lên tập đo : {sel}={chosen[sel]:.4f}\n"
        f"   → chênh {res['optimism']:+.4f} — lượng phải trừ khi đọc mọi bảng quét-trên-tập-đo."
    )
    return gain


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fit", default=None, help="tập CHỌN siêu tham số; mặc định paths.train_split")
    ap.add_argument("--eval", dest="eval_path", default=None, help="tập BÁO CÁO; mặc định paths.dev")
    ap.add_argument("--grid-w", default=None, help="trọng số của nguồn ĐẦU TIÊN trong config")
    ap.add_argument("--grid-k", default=None)
    ap.add_argument("--levels", default=None, help="mức hợp nhất đem so")
    ap.add_argument("--select-on", default=None, help="chỉ số dùng để CHỌN ô trên tập fit")
    ap.add_argument("--recall-at", default=None)
    ap.add_argument("--limit-fit", type=int, default=1500,
                    help="cắt bớt tập fit cho nhanh; 0 = dùng hết. Cắt là ĐÁNH ĐỔI: ô thắng trên 1.500 câu "
                         "có thể không phải ô thắng trên 4.689 câu.")
    ap.add_argument("--demo", action="store_true", help="corpus giả, không cần data/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-csv", action="store_true", help="thêm một dòng vào experiments.csv")
    ap.add_argument("--nguoi-chay", default="P3")
    return ap.parse_args()


def main() -> int:
    """Điểm vào CLI: quét, chọn, báo cáo, ghi kết quả."""
    args = parse_args()
    cfg = load_config(args.config)
    settings = grid_settings(cfg, args)
    out_dir = resolve_out_dir(cfg, args.out_dir, fallback="tune_rrf")
    chunks, fit, ev = load_sets(cfg, args)
    print(
        f"exp_id : {cfg.get('exp_id')}\ncommit : {git_commit()}\n"
        f"fit    : {fit.path} · {len(fit.ids)} câu\neval   : {ev.path} · {len(ev.ids)} câu\n"
        f"lưới   : w={settings.ws}\n         k={settings.ks} × mức={settings.levels}  →  {settings.n_cells} ô\n"
        f"chọn   : {settings.select_on}\nra     : {out_dir}"
    )
    if args.dry_run:
        print("\n--dry-run: dừng ở đây, chưa index.")
        return 0

    scorer, split_fit, split_ev = build_and_score(cfg, args, chunks, fit, ev, settings)
    res = run_tuning(scorer, split_fit, split_ev)
    gain = print_eval_summary(res, settings, ev.path)
    res.update({
        "exp_id": cfg.get("exp_id"), "config": args.config, "commit": git_commit(), "ngay": date.today().isoformat(),
        "fit": {"path": fit.path, "n": len(fit.ids)}, "eval": {"path": ev.path, "n": len(ev.ids)},
        "select_on": settings.select_on,
        "grid": {"w": settings.ws, "rrf_k": settings.ks, "levels": settings.levels},
        "sources": {n: scorer.r.sources[n].stats() for n in scorer.names},
    })
    res.pop("best_on_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tune_rrf.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out_dir / "tune_rrf.md").write_text(render_report(res, settings.recall_ks), encoding="utf-8")
    print(f"\n✅ {out_dir / 'tune_rrf.json'}\n✅ {out_dir / 'tune_rrf.md'}")
    if args.log_csv:
        append_experiment_rows([experiment_row(res, args, gain)])
        print("✅ thêm 1 dòng vào experiments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
