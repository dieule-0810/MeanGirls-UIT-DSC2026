import argparse
from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str]):
  """Hỗ trợ chạy lệnh và dừng ngay lập tức nếu lệnh trả về lỗi (tương đương set -e)."""
  print(f"\n> {' '.join(cmd)}")
  result = subprocess.run(cmd)
  if result.returncode != 0:
    print(f"\n❌ Lệnh thất bại với mã lỗi {result.returncode}")
    sys.exit(result.returncode)


def main():
  parser = argparse.ArgumentParser(
      description="Chạy pipeline end-to-end v0.1 BM25"
  )
  parser.add_argument(
      "--config",
      default="configs/v0.1_bm25.yaml",
      help="Đường dẫn file config",
  )
  args = parser.parse_args()

  config = args.config
  out_dir = Path("outputs/v0.1_bm25")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("══ 0/5  Kiểm tra môi trường ══")
  run_command([sys.executable, "-m", "src.verify_env"])

  print("\n══ 1/5  Parse corpus ══")
  run_command([sys.executable, "-m", "src.data.parse_corpus", "--config", config])

  print("\n══ 2/5  Chunking ══")
  run_command([sys.executable, "-m", "src.data.chunker", "--config", config])

  print("\n══ 3/5  Tách held-out ══")
  run_command(
      [sys.executable, "-m", "src.data.split_holdout", "--config", config]
  )

  print("\n══ 4/5  BM25 trên held-out (đo chất lượng) ══")
  run_command([
      sys.executable,
      "-m",
      "src.retrieval.bm25",
      "--config",
      config,
      "--questions",
      "data/holdout.json",
      "--out",
      str(out_dir / "preds_holdout.json"),
  ])
  run_command([
      sys.executable,
      "-m",
      "src.evaluate",
      "--preds",
      str(out_dir / "preds_holdout.json"),
      "--truth",
      "data/holdout.json",
      "--corpus",
      "data/corpus_clean.jsonl",
  ])

  print("\n══ 5/5  BM25 trên public test → submission.zip ══")
  run_command([
      sys.executable,
      "-m",
      "src.retrieval.bm25",
      "--config",
      config,
      "--questions",
      "data/public-official.json",
      "--out",
      str(out_dir / "preds_public.json"),
  ])
  run_command([
      sys.executable,
      "-m",
      "src.make_submission",
      "--preds",
      str(out_dir / "preds_public.json"),
      "--questions",
      "data/public-official.json",
      "--corpus",
      "data/corpus_clean.jsonl",
      "--out",
      str(out_dir / "submission.zip"),
  ])

  print(f"\n✅ Xong. Nộp: {out_dir / 'submission.zip'}")

  # Lấy git commit SHA để ghi log
  try:
    git_sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    print(
        f"  Ghi kết quả vào experiments.csv kèm commit SHA: {git_sha}"
    )
  except Exception:
    print("  (Không lấy được Git commit SHA)")


if __name__ == "__main__":
  main()