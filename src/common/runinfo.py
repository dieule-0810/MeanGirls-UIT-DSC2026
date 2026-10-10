"""Siêu dữ liệu của một lần chạy: commit, chốt chặn holdout, ghi `experiments.csv`.

Ba việc này trước đây mỗi script tự chép một bản (`git_commit` có ở 5 script, chốt holdout ở
7 script), và các bản lệch nhau đúng chỗ quan trọng — có bản không đánh dấu `-dirty`, nên một
dòng `experiments.csv` chạy từ cây làm việc bẩn trông như tái lập được trong khi không phải.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable

from src.common.config import REPO

EXPERIMENTS_CSV = REPO / "experiments.csv"


def git_commit(short: bool = True) -> str:
    """SHA của HEAD, thêm hậu tố `-dirty` nếu cây làm việc có thay đổi chưa commit.

    Args:
        short: True thì trả SHA rút gọn.

    Returns:
        Ví dụ `63fcf2b` hoặc `63fcf2b-dirty`; `unknown` nếu không chạy trong repo git.
    """
    try:
        rev = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        sha = subprocess.check_output(rev, cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def guard_holdout(path: str | Path, allow: bool, purpose: str = "đo") -> None:
    """Chặn mọi lần chạm `holdout.json` chưa được cả nhóm chốt.

    `holdout` là tập ĐO LẦN CUỐI, chỉ chạm một lần. Quy ước bằng lời không đủ — chặn bằng code.

    Args:
        path: File câu hỏi sắp dùng.
        allow: Giá trị cờ `--allow-holdout`.
        purpose: `đo` hoặc `fit`. Fit trên holdout bị cấm tuyệt đối, kể cả khi `allow`.

    Raises:
        SystemExit: Nếu `path` là holdout mà chưa được phép.
    """
    if "holdout" not in Path(path).name:
        return
    if purpose == "fit":
        raise SystemExit(f"❌ Không được fit trên {path} — đó là tập ĐO LẦN CUỐI.")
    if not allow:
        raise SystemExit(
            f"❌ {path} là tập ĐO LẦN CUỐI, chạm đúng một lần. Dùng data/dev.json, "
            f"hoặc --allow-holdout nếu cả nhóm đã chốt đây LÀ lần đo cuối."
        )


def append_experiment_rows(rows: Iterable[dict], csv_path: Path = EXPERIMENTS_CSV) -> int:
    """Thêm dòng vào `experiments.csv`, theo đúng header đang có trong file.

    File thiếu newline cuối thì dòng mới bị dán vào dòng cuối của người khác — đã xảy ra
    11/09 (dòng `v0.2_fusion_rrf` thành 27 cột). Hàm này vá newline trước khi ghi.

    Args:
        rows: Các dict có khoá trùng tên cột. Khoá lạ làm `csv.DictWriter` báo lỗi.
        csv_path: File đích, mặc định `experiments.csv` ở gốc repo.

    Returns:
        Số dòng đã ghi.
    """
    raw = csv_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with csv_path.open("ab") as fh:
            fh.write(b"\n")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    n = 0
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        for row in rows:
            writer.writerow(row)
            n += 1
    return n
