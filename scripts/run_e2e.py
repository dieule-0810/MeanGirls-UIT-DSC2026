#!/usr/bin/env python3
"""Tái lập toàn bộ từ dữ liệu thô BTC đến `submission.zip` bằng MỘT lệnh (thay `run_v0.1.py`).

Các bước, chạy tuần tự, dừng ngay ở bước đầu tiên lỗi (fail loud):

1. `verify_env`  — Python 3.11.x và gói bắt buộc.
2. `eda`         — chỉ khi thiếu `docs/exclusion_decisions.json` (sinh danh sách loại trừ).
3. `parse`       — `context_*.json` → `data/corpus_clean.jsonl` (8.507 văn bản).
4. `chunk`       — chunker `strict` → `data/chunks.jsonl` (432.142 chunk).
5. `split`       — holdout / dev / error_pool / train_split.
6. `encode`      — `data/embeddings.npy`; CẦN GPU (~2,5 giờ T4), bỏ qua nếu file đã có.
7. `predict`     — `scripts/run_pipeline.py` với config chính thức → `submission.zip`.

Typical usage example:

    python scripts/run_e2e.py --questions data/private-official.json
    python scripts/run_e2e.py --steps predict --questions data/private-official.json
    python scripts/run_e2e.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import load_config, repo_path  # noqa: E402

STEPS = ("verify_env", "eda", "parse", "chunk", "split", "encode", "predict")
DEFAULT_CONFIG = "configs/v0.8_hybrid_rrf.yaml"
DECISIONS = REPO / "docs/exclusion_decisions.json"


def plan_commands(args: argparse.Namespace, cfg: dict) -> list[tuple[str, list[str]]]:
    """Danh sách `(bước, lệnh)` cần chạy, đã bỏ các bước không được chọn hoặc đã có đầu ra.

    Args:
        args: Tham số CLI.
        cfg: Config chính thức đã đọc.

    Returns:
        Các lệnh theo thứ tự.
    """
    py = sys.executable
    chunks = cfg["paths"]["chunks"]
    embeddings = repo_path(cfg["paths"]["embeddings"])
    encode_cfg = "configs/v0.4_dense.yaml"
    cmds = {
        "verify_env": [py, "-m", "src.verify_env"],
        "eda": [py, "scripts/eda.py", "--corpus-dir", args.corpus_dir, "--out", "outputs/eda/eda_notes.md"],
        "parse": [py, "-m", "src.data.parse_corpus", "--corpus-dir", args.corpus_dir, "--out", cfg["paths"]["corpus_clean"]],
        "chunk": [py, "-m", "src.data.chunker", "--strategy", "strict", "--input", cfg["paths"]["corpus_clean"], "--out", chunks],
        "split": [py, "-m", "src.data.split_data", "--corpus", cfg["paths"]["corpus_clean"], "--out-dir", "data"],
        "encode": [py, "-u", "scripts/encode_corpus.py", "--config", encode_cfg, "--resume"],
        "predict": [py, "-u", "scripts/run_pipeline.py", "--config", args.config, "--questions", args.questions, "--submission"],
    }
    wanted = args.steps.split(",") if args.steps else list(STEPS)
    unknown = sorted(set(wanted) - set(STEPS))
    if unknown:
        raise SystemExit(f"❌ bước không có: {unknown}. Có: {', '.join(STEPS)}")
    skip = {"eda": DECISIONS.exists(), "encode": embeddings.exists()}
    return [(s, cmds[s]) for s in STEPS if s in wanted and not (skip.get(s) and not args.steps)]


def run(cmds: list[tuple[str, list[str]]], dry_run: bool) -> int:
    """Chạy tuần tự; trả mã thoát của bước đầu tiên lỗi (0 nếu tất cả xong)."""
    for step, cmd in cmds:
        print(f"\n══ {step} ══\n$ {' '.join(cmd)}", flush=True)
        if dry_run:
            continue
        code = subprocess.run(cmd, cwd=REPO).returncode
        if code != 0:
            print(f"\n❌ Bước '{step}' thất bại (mã {code}). Dừng — các bước sau phụ thuộc bước này.")
            return code
    return 0


def main() -> int:
    """Điểm vào CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--questions", default="data/private-official.json")
    ap.add_argument("--corpus-dir", default="data/selected-contexts")
    ap.add_argument("--steps", default=None, help=f"danh sách phẩy trong {', '.join(STEPS)}; mặc định: tất cả")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in chuỗi lệnh")
    args = ap.parse_args()
    cmds = plan_commands(args, load_config(args.config))
    code = run(cmds, args.dry_run)
    if code == 0 and not args.dry_run:
        print(f"\n✅ Xong. submission.zip nằm trong paths.out_dir của {args.config}.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
