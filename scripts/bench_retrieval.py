"""
Lưới benchmark tầng 1: tokenizer × chiến lược gộp chunk→doc. CHỦ SỞ HỮU: P3.

    python scripts/bench_retrieval.py --config configs/v0.2_bm25_tokenizer.yaml
    python scripts/bench_retrieval.py --demo          # corpus giả, không cần data/

Trả lời H1/H2 (docs/retrieval.md mục 2) bằng số trên cùng một held-out: chấm điểm chunk
MỘT lần cho mỗi tokenizer rồi thử mọi chiến lược gộp trên cùng kết quả — lưới 5×4 nhưng chỉ
5 lần index/truy vấn, và bốn chiến lược gộp so trên đúng cùng tập ứng viên.

Metric chính là **Recall@50**, KPI của P3. Recall/Precision top-5 in kèm để đối chiếu leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.common.io import load_chunks, load_questions  # noqa: E402
from src.evaluate import eval_official, load_truth, recall_at_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.tokenizers import available_tokenizers  # noqa: E402

EXPERIMENTS_CSV = REPO / "experiments.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Dữ liệu giả cho --demo
# ─────────────────────────────────────────────────────────────────────────────
def demo_data(n_docs: int = 60, seed: int = 42) -> tuple[list[dict], dict, dict]:
    """
    Corpus giả để chạy thử khung khi `data/chunks.jsonl` chưa tồn tại — KHÔNG phải benchmark,
    chỉ chứng minh đường ống chạy đúng định dạng. Câu hỏi đặt dấu thanh khác văn bản
    ("uỷ ban" ↔ "ủy ban") để phần chuẩn hoá dấu có việc thật mà làm.
    """
    rng = random.Random(seed)
    linh_vuc = [
        ("giấy phép lái xe", "sở giao thông vận tải", "cấp đổi"),
        ("hoá đơn điện tử", "cơ quan thuế", "huỷ bỏ"),
        ("giấy chứng nhận quyền sử dụng đất", "uỷ ban nhân dân cấp huyện", "thu hồi"),
        ("đăng ký kinh doanh", "phòng đăng ký kinh doanh", "đình chỉ"),
        ("an toàn thực phẩm", "bộ y tế", "kiểm tra"),
        ("bảo hiểm xã hội bắt buộc", "cơ quan bảo hiểm xã hội", "truy thu"),
        ("phòng cháy chữa cháy", "cơ quan công an", "thẩm duyệt"),
        ("xử phạt vi phạm hành chính", "chủ tịch uỷ ban nhân dân", "ra quyết định"),
        ("đấu thầu qua mạng", "bên mời thầu", "huỷ thầu"),
        ("chứng chỉ hành nghề xây dựng", "sở xây dựng", "cấp lại"),
    ]
    boilerplate = [
        "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015",
        "Thông tư này quy định chi tiết một số điều của Nghị định số 15/2020/NĐ-CP",
        "Các quy định trước đây trái với Thông tư này đều bị bãi bỏ",
        "Trong quá trình thực hiện, nếu có vướng mắc, đề nghị phản ánh về Bộ để xem xét",
        "Thông tư này có hiệu lực thi hành kể từ ngày ký ban hành",
    ]

    chunks: list[dict] = []
    truth: dict[str, dict] = {}
    for d in range(n_docs):
        doc_id = str(100000 + d * 7)
        chu_de, co_quan, hanh_vi = linh_vuc[d % len(linh_vuc)]
        bien_the = f"{chu_de} thuộc nhóm {d % 5 + 1}"
        n_dieu = rng.randint(2, 6)
        for pos in range(n_dieu):
            body = rng.choice(boilerplate)
            if pos == 0:
                body = (
                    f"{co_quan} có thẩm quyền {hanh_vi} đối với {bien_the} "
                    f"theo quy định tại Điều {pos + 1}. {body}"
                )
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::{pos:04d}",
                    "doc_id": doc_id,
                    "position": pos,
                    "text": f"Điều {pos + 1}. Quy định về {chu_de}. {body}",
                }
            )
        # Đặt dấu thanh kiểu khác văn bản → ép phần chuẩn hoá làm việc thật
        q = (
            f"Cơ quan nào có thẩm quyền {hanh_vi.replace('uỷ', 'ủy')} đối với "
            f"{bien_the.replace('uỷ', 'ủy').replace('hoá', 'hóa')}?"
        )
        truth[f"demo_{d:03d}"] = {"question": q, "answer": [doc_id]}

    # Văn bản nhiễu: dài, chứa đúng âm tiết câu hỏi nhưng ở từ ghép khác ("cơ sở"+"quan hệ"
    # ≠ "cơ quan") — kịch bản word-segment/bigram thắng âm tiết thuần, và pool=sum thua vì
    # thiên vị văn bản dài. Không có nhiễu này thì lưới bench chỉ toàn số 1.000.
    nhieu = [
        "cơ sở dữ liệu; quan hệ lao động; giấy tờ tuỳ thân; phép đo lường",
        "thẩm định giá; quyền sở hữu; hoá chất công nghiệp; đơn vị sự nghiệp",
        "lái tàu đường sắt; xe máy chuyên dùng; điện lực; tử tuất",
        "uỷ thác đầu tư; ban quản lý dự án; nhân sự; dân sinh",
        "thu nhập cá nhân; hồi tố; kinh tế tập thể; doanh trại",
    ]
    for j in range(n_docs // 3):
        doc_id = str(900000 + j)
        for pos in range(rng.randint(8, 14)):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::{pos:04d}",
                    "doc_id": doc_id,
                    "position": pos,
                    "text": f"Điều {pos + 1}. {rng.choice(nhieu)}. {rng.choice(boilerplate)}",
                }
            )

    questions = {q: v["question"] for q, v in truth.items()}
    gold = {q: [str(a) for a in v["answer"]] for q, v in truth.items()}
    return chunks, questions, gold


# ─────────────────────────────────────────────────────────────────────────────
# Lưới
# ─────────────────────────────────────────────────────────────────────────────
def run_grid(
    chunks: list[dict],
    qids: list[str],
    texts: list[str],
    gold: dict[str, list[str]],
    bm25_opts: dict,
    tokenizers: list[str],
    pools: list[str],
    recall_ks: list[int],
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    have = available_tokenizers()

    for tok_name in tokenizers:
        if not have.get(tok_name, False):
            print(
                f"\n⏭  Bỏ qua tokenizer '{tok_name}': chưa cài. "
                f"`pip install -r requirements-p3.txt` nếu muốn có ô này trong bảng ablation."
            )
            continue

        print(f"\n══ tokenizer: {tok_name} ══")
        opts = dict(bm25_opts)
        opts["tokenizer"] = tok_name
        r = BM25Retriever(**opts)
        r.index(chunks)

        t0 = time.perf_counter()
        cands = r.candidates(texts, max(r.candidate_chunks, top_k))
        score_s = time.perf_counter() - t0
        n_empty = sum(1 for idx, _ in cands if len(idx) == 0)
        print(f"  chấm điểm {len(texts)} câu hỏi: {score_s:.1f}s, {n_empty} câu không khớp term nào")

        for pool in pools:
            t0 = time.perf_counter()
            preds = {
                q: [d for d, _ in r.pool_candidates(idx, sc, top_k, pool=pool)]
                for q, (idx, sc) in zip(qids, cands)
            }
            pool_s = time.perf_counter() - t0

            row = {
                "tokenizer": r.tokenizer.key,
                "pool": pool,
                "vocab": len(r.vocab),
                "index_s": round(r.stats()["index_seconds"], 1),
                "score_s": round(score_s, 1),
                "pool_s": round(pool_s, 2),
                "n_empty": n_empty,
            }
            for k in recall_ks:
                row[f"recall@{k}"] = round(recall_at_k(preds, gold, k), 4)
            btc = eval_official({q: v[:5] for q, v in preds.items()}, gold)
            row["btc_recall@5"] = round(btc["recall"], 4)
            row["btc_precision@5"] = round(btc["precision"], 4)
            rows.append(row)
            print(
                f"  pool={pool:<12} "
                + "  ".join(f"R@{k}={row[f'recall@{k}']:.4f}" for k in recall_ks)
                + f"  P@5={row['btc_precision@5']:.4f}"
            )
    return rows


def render_report(rows: list[dict], recall_ks: list[int], meta: dict) -> str:
    kpi = f"recall@{50 if 50 in recall_ks else recall_ks[-1]}"
    best = max(rows, key=lambda r: r[kpi]) if rows else None
    cols = ["tokenizer", "pool"] + [f"recall@{k}" for k in recall_ks] + [
        "btc_recall@5",
        "btc_precision@5",
        "vocab",
        "index_s",
        "score_s",
    ]
    lines = [
        "# Bench tầng 1 — tokenizer × gộp chunk→doc",
        "",
        "> Sinh bởi `scripts/bench_retrieval.py`. Phần **Nhận xét** điền tay — đó mới là thứ",
        "> đi vào bài báo (plan.md mục 0.6: mỗi phương pháp phải nói rõ yếu ở đâu).",
        "",
        "```json",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "```",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in sorted(rows, key=lambda r: -r[kpi]):
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", f"**Sắp xếp theo {kpi} — KPI của P3.**", ""]
    if best:
        lines += [
            f"Cấu hình tốt nhất: `tokenizer={best['tokenizer']}`, `pool={best['pool']}` "
            f"→ {kpi} = {best[kpi]:.4f}, Precision@5 = {best['btc_precision@5']:.4f}.",
            "",
        ]
    lines += [
        "## Nhận xét (điền tay)",
        "",
        "- H1 — word-segment hơn âm tiết thuần? ",
        "- H1b — bigram âm tiết lấy lại được bao nhiêu phần của word-segment? ",
        "- H2 — chiến lược gộp nào thắng, và vì sao (văn bản dài có bị `sum` thiên vị không)? ",
        "- Câu `n_empty` (không khớp term nào) rơi vào nhóm văn bản nào? ",
        "- Việc tiếp theo để nâng Recall@50: ",
    ]
    return "\n".join(lines) + "\n"


def git_commit() -> str:
    """SHA ngắn lúc chạy — không có nó thì dòng experiments.csv không tái lập được."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO
        ).decode().strip()
        return f"{out}-dirty" if dirty else out
    except Exception:
        return "unknown"


def append_experiments(rows: list[dict], cfg_path: str, nguoi_chay: str, tap_do: str) -> None:
    commit = git_commit()
    # File thiếu newline cuối thì dòng mới bị dán vào dòng cuối của người khác — đã xảy ra
    # 11/09, làm hỏng dòng v0.2_fusion_rrf của P4 (27 cột thay vì 14).
    raw = EXPERIMENTS_CSV.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with EXPERIMENTS_CSV.open("ab") as fh:
            fh.write(b"\n")
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        for r in rows:
            w.writerow(
                {
                    "exp_id": f"v0.2_bm25_{r['tokenizer']}_{r['pool']}",
                    "ngay": date.today().isoformat(),
                    "nguoi_chay": nguoi_chay,
                    "commit": commit,
                    "config": cfg_path,
                    "tap_do": tap_do,
                    "recall": r.get("btc_recall@5", ""),
                    "precision": r.get("btc_precision@5", ""),
                    "recall_lb": "",
                    "precision_lb": "",
                    "ghi_chu": " ".join(
                        f"{k}={v}" for k, v in r.items() if k.startswith("recall@")
                    )
                    + f" vocab={r['vocab']}",
                    "nhom_so_sanh": "co_dien",
                    "gia_thuyet_lien_quan": "H1 word-segment > âm tiết thuần; H2 gộp chunk→doc",
                    "diem_yeu_khac_phuc_tu_exp_truoc": (
                        "v0.1 dùng regex \\w+ (âm tiết rời) + pool=max: term phổ thông khớp nhiễu, "
                        "và văn bản khớp nhiều điều khoản không được cộng điểm"
                    ),
                }
            )
    print(f"📝 Đã thêm {len(rows)} dòng vào {EXPERIMENTS_CSV.name} — nhớ `git pull --rebase` trước khi commit.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v0.2_bm25_tokenizer.yaml")
    ap.add_argument("--demo", action="store_true", help="Chạy trên corpus giả, không cần data/")
    ap.add_argument("--chunks", default=None, help="Ghi đè paths.chunks")
    ap.add_argument("--questions", default=None, help="Ghi đè paths.dev (file có nhãn)")
    ap.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Cho phép bench chạy trên holdout — chỉ dùng khi cả nhóm đã chốt đây là lần đo cuối",
    )
    ap.add_argument("--tokenizers", default=None, help="Danh sách phẩy, ghi đè bench.tokenizers")
    ap.add_argument("--pools", default=None, help="Danh sách phẩy, ghi đè bench.pools")
    ap.add_argument("--n-questions", type=int, default=None, help="Chỉ lấy N câu đầu (chạy nhanh)")
    ap.add_argument("--report", default=None, help="File markdown xuất ra")
    ap.add_argument(
        "--results",
        default=None,
        help="File JSON xuất ra (mặc định out_dir/bench_results.json). outputs/ bị .gitignore "
        "chặn — trỏ sang docs/ nếu đây là bản chốt cần commit",
    )
    ap.add_argument("--log-experiments", action="store_true", help="Thêm dòng vào experiments.csv")
    ap.add_argument("--nguoi-chay", default="P3")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    bench = cfg.get("bench", {})
    tokenizers = (
        args.tokenizers.split(",") if args.tokenizers else bench.get("tokenizers", ["regex"])
    )
    pools = args.pools.split(",") if args.pools else bench.get("pools", ["max"])
    recall_ks = bench.get("recall_at", [5, 20, 50, 100])
    top_k = max(recall_ks + [cfg["retrieval"].get("top_k_retrieve", 100)])
    bm25_opts = dict(cfg["retrieval"].get("bm25", {}))
    bm25_opts.pop("tokenizer", None)
    if cfg.get("paths", {}).get("cache_dir") and not args.demo:
        bm25_opts.setdefault("cache_dir", cfg["paths"]["cache_dir"])

    if args.demo:
        print("⚠️  CHẾ ĐỘ DEMO — corpus giả, con số KHÔNG dùng để kết luận gì.")
        chunks, questions, gold = demo_data()
        qids = list(questions)
        texts = [questions[q] for q in qids]
        bm25_opts.pop("cache_dir", None)
        q_path = "demo"
    else:
        chunks_path = args.chunks or cfg["paths"]["chunks"]
        # Mặc định là paths.dev, KHÔNG phải paths.holdout: lưới 5 tokenizer × 5 pooling là
        # 25 lần chạm tập đo. holdout chỉ được chạm đúng một lần (quy ước §1.1 của nhóm) và
        # đã chạm rồi. Quy ước bằng lời không đủ — chặn ở đây.
        q_path = args.questions or cfg["paths"].get("dev") or cfg["paths"]["holdout"]
        if "holdout" in Path(q_path).name and not args.allow_holdout:
            print(
                f"❌ Bench đang trỏ vào {q_path}.\n"
                f"   holdout là tập ĐO LẦN CUỐI, chạm một lần duy nhất — lưới này chạm nó "
                f"{len(tokenizers) * len(pools)} lần.\n"
                f"   Dùng data/dev.json (thêm `dev:` vào paths của config), hoặc "
                f"--allow-holdout nếu cả nhóm đã đồng ý đây LÀ lần đo cuối."
            )
            return 1
        chunks = load_chunks(REPO / chunks_path if not Path(chunks_path).is_absolute() else chunks_path)
        qids, texts = load_questions(q_path)
        gold = load_truth(q_path)

    n_q = args.n_questions or bench.get("n_questions")
    if n_q:
        qids, texts = qids[:n_q], texts[:n_q]
        gold = {q: gold[q] for q in qids}

    print(f"{len(chunks)} chunk, {len(qids)} câu hỏi, top_k={top_k}")
    rows = run_grid(chunks, qids, texts, gold, bm25_opts, tokenizers, pools, recall_ks, top_k)
    if not rows:
        print("❌ Không ô nào chạy được — kiểm tra danh sách tokenizer.")
        return 1

    meta = {
        "exp_id": cfg.get("exp_id"),
        "config": args.config,
        "commit": git_commit(),
        "questions": "demo" if args.demo else str(q_path),
        "demo": args.demo,
        "n_chunks": len(chunks),
        "n_questions": len(qids),
        "top_k": top_k,
        "bm25": {k: v for k, v in bm25_opts.items() if k not in ("cache_dir", "verbose")},
        "ngay": date.today().isoformat(),
    }
    report = render_report(rows, recall_ks, meta)

    out_dir = REPO / cfg["paths"].get("out_dir", "outputs/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results) if args.results else out_dir / "bench_results.json"
    if not results_path.is_absolute():
        results_path = REPO / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = Path(args.report) if args.report else out_dir / "bench_report.md"
    if not report_path.is_absolute():
        report_path = REPO / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"✅ {report_path}")
    print(f"✅ {results_path}")

    if args.log_experiments:
        if args.demo:
            print("⏭  Bỏ qua experiments.csv: số liệu demo không phải thí nghiệm thật.")
        else:
            append_experiments(rows, args.config, args.nguoi_chay, Path(q_path).stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
