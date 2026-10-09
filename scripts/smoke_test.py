#!/usr/bin/env python3
"""Smoke test: kiểm môi trường rồi chạy toàn bộ pytest — CHẠY TRƯỚC MỌI THỨ.

Hỏng ở đây thì dừng: mọi con số chạy sau đó đều vô nghĩa.

Typical usage example:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str]) -> None:
    """Chạy một lệnh trong gốc repo, thoát ngay với cùng mã lỗi nếu lệnh hỏng.

    Raises:
        SystemExit: Lệnh trả mã khác 0.
    """
    print(f"\n> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"\n❌ Lỗi: Lệnh thất bại với mã {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    """Chạy `verify_env` rồi toàn bộ pytest."""
    print("══ Smoke test ══")
    run_command([sys.executable, "-m", "src.verify_env"])
    run_command([sys.executable, "-m", "pytest", "tests/", "-q"])
    print("\n✅ PASS — môi trường và logic chấm điểm đều đúng.")


if __name__ == "__main__":
    main()
