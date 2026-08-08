import subprocess
import sys


def run_command(cmd: list[str]):
  """Chạy lệnh và dừng ngay lập tức nếu có lỗi (tương đương set -euo pipefail)."""
  print(f"\n> {' '.join(cmd)}")
  result = subprocess.run(cmd)
  if result.returncode != 0:
    print(f"\n❌ Lỗi: Lệnh thất bại với mã {result.returncode}")
    sys.exit(result.returncode)


def main():
  print("══ Smoke test ══")

  # 1. Kiểm tra môi trường
  run_command([sys.executable, "-m", "src.verify_env"])

  # 2. Chạy unit test nhanh bằng pytest
  run_command([sys.executable, "-m", "pytest", "tests/", "-q"])

  print("\n✅ PASS — môi trường và logic chấm điểm đều đúng.")


if __name__ == "__main__":
  main()