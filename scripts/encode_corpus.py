#!/usr/bin/env python3
"""
Encode toàn bộ chunk → `data/embeddings.npy`. CHỦ SỞ HỮU: P3.

Tách khỏi `DenseRetriever.index()` vì đây là bước DÀI NHẤT của pipeline (524.422 chunk) và là
bước duy nhất thật sự cần GPU. Tách ra thì: encode một lần trên máy có GPU, mọi lần đo sau chỉ
đọc file; và chạy lại được sau khi đứt session mà không mất phần đã làm.

Ghi bằng memmap + sổ tiến độ: đứt giữa chừng thì `--resume` chạy tiếp từ chunk dở, không encode lại.

    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --dry-run   # ước tính, không nạp model
    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --limit 2000
    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --resume

⚠️ `--limit` ghi ra file RIÊNG (`embeddings_limit<N>.npy`), không bao giờ đè bản đầy đủ:
   một ma trận 2.000 hàng nằm ở chỗ của 524.422 hàng là cách hỏng im lặng nhất.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.common.io import load_chunks  # noqa: E402
from src.retrieval.dense import DenseRetriever, chunk_fingerprint, resolve_device  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="chỉ encode N chunk đầu, ghi ra file riêng")
    ap.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help="encode mảnh thứ K (đếm từ 0) trong N mảnh — để chạy song song nhiều tài khoản Kaggle. "
        "Ghi ra embeddings.shardKofN.npy, ghép lại bằng scripts/merge_embeddings.py",
    )
    ap.add_argument("--resume", action="store_true", help="chạy tiếp từ sổ tiến độ")
    ap.add_argument("--batch-size", type=int, default=None, help="ghi đè retrieval.dense.batch_size")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    ap.add_argument("--dry-run", action="store_true", help="in kế hoạch rồi thoát, không nạp model")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    spec = dict(cfg["retrieval"]["dense"])
    spec.pop("embeddings_path", None)
    if args.batch_size:
        spec["batch_size"] = args.batch_size
    if args.device:
        spec["device"] = args.device

    out_path = REPO / cfg["paths"].get("embeddings", "data/embeddings.npy")
    if args.limit and args.shard:
        raise SystemExit("❌ --limit và --shard loại trừ nhau: một cái cắt đầu, một cái chia đều.")
    if args.limit:
        out_path = out_path.with_name(f"{out_path.stem}_limit{args.limit}.npy")

    chunks = load_chunks(REPO / cfg["paths"]["chunks"])
    # Vân tay của TOÀN BỘ corpus, tính TRƯỚC khi cắt mảnh: mọi mảnh phải mang cùng một vân tay,
    # nếu không thì chúng được encode từ hai phiên bản chunks.jsonl khác nhau và ghép lại là rác.
    full_fp = chunk_fingerprint([str(c["chunk_id"]) for c in chunks])
    n_full = len(chunks)
    start, end = 0, n_full

    if args.shard:
        try:
            k, n_shards = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit("❌ --shard phải có dạng K/N, ví dụ 0/8")
        if not (0 <= k < n_shards):
            raise SystemExit(f"❌ --shard {args.shard}: K phải trong [0, {n_shards - 1}]")
        # Chia đều, phần dư rải vào các mảnh đầu — công thức này phải GIỐNG HỆT ở merge.
        base, rem = divmod(n_full, n_shards)
        start = k * base + min(k, rem)
        end = start + base + (1 if k < rem else 0)
        out_path = out_path.with_name(f"{out_path.stem}.shard{k}of{n_shards}.npy")
        chunks = chunks[start:end]

    # Đặt SAU khi out_path đã mang hậu tố mảnh. Trước đây hai dòng này nằm trên khối `if
    # args.shard` nên mọi mảnh đều ghi ra `embeddings.meta.json` và `embeddings.progress.json`
    # — không có `.shardKofN.`. Hậu quả: `merge_embeddings.py` tìm `*.shard*.meta.json` nên
    # không thấy gì và từ chối ghép, còn 8 mảnh chạy trên 8 phiên thì cùng đòi một tên file,
    # gom vào một thư mục là đè nhau. Sổ tiến độ cũng vậy: `--resume` của mảnh 3 đọc nhầm sổ
    # của mảnh 1 nếu hai mảnh chạy chung một thư mục làm việc.
    meta_path = out_path.with_suffix(".meta.json")
    prog_path = out_path.with_suffix(".progress.json")
    elif args.limit:
        chunks = chunks[: args.limit]

    texts = [c["text"] for c in chunks]
    fp = chunk_fingerprint([str(c["chunk_id"]) for c in chunks])

    r = DenseRetriever(**spec)
    bs = r.batch_size
    print(
        f"Model   : {r.repo}@{r.revision}\n"
        f"Mảnh    : {args.shard or 'toàn bộ'} → chunk [{start}, {end}) / {n_full}\n"
        f"Chunk   : {len(chunks)} · lô {bs} · {-(-len(chunks) // bs)} lô\n"
        f"Thiết bị: {args.device or spec.get('device', 'auto')}\n"
        f"Ra      : {out_path}\n"
        f"Vân tay : {fp}"
    )
    if args.dry_run:
        print("\n--dry-run: dừng ở đây, chưa nạp model, chưa đụng GPU.")
        return 0

    print(f"Thiết bị thật: {resolve_device(spec.get('device', 'auto'))}")

    # ── sổ tiến độ ───────────────────────────────────────────────────────────
    done = 0
    if args.resume and prog_path.exists() and out_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        if prog.get("chunk_fingerprint") != fp or prog.get("revision") != r.revision:
            raise SystemExit(
                "❌ Sổ tiến độ thuộc về bộ chunk/model KHÁC. Xoá file .progress.json và .npy rồi chạy lại."
            )
        done = int(prog.get("done", 0))
        print(f"↻ chạy tiếp từ chunk {done}")

    # Encode lô đầu để biết số chiều, rồi mở memmap đúng kích thước.
    t0 = time.perf_counter()
    first = r.encode(texts[done : done + bs], is_query=False, show_every=0)
    dim = int(first.shape[1])
    mm = np.lib.format.open_memmap(
        out_path, mode="r+" if done else "w+", dtype=np.float32, shape=(len(texts), dim)
    )
    mm[done : done + first.shape[0]] = first
    done += first.shape[0]

    try:
        while done < len(texts):
            block = r.encode(texts[done : done + bs], is_query=False, show_every=0)
            mm[done : done + block.shape[0]] = block
            done += block.shape[0]
            if (done // bs) % 50 == 0:
                mm.flush()
                prog_path.write_text(
                    json.dumps({"done": done, "chunk_fingerprint": fp, "revision": r.revision}),
                    encoding="utf-8",
                )
                rate = done / max(1e-9, time.perf_counter() - t0)
                print(f"  {done}/{len(texts)} ({rate:.0f} chunk/s, còn ~{(len(texts)-done)/max(rate,1e-9)/60:.1f} phút)")
    except KeyboardInterrupt:
        mm.flush()
        prog_path.write_text(
            json.dumps({"done": done, "chunk_fingerprint": fp, "revision": r.revision}), encoding="utf-8"
        )
        print(f"\n⏸  dừng ở chunk {done}. Chạy lại với --resume.")
        return 130

    mm.flush()
    meta_path.write_text(
        json.dumps(
            {
                "repo": r.repo,
                "revision": r.revision,
                "pooling": r.pooling,
                "normalize": r.normalize,
                "passage_prefix": r.passage_prefix,
                "max_length": r.max_length,
                "n_chunks": len(texts),
                "dim": dim,
                "chunk_fingerprint": fp,
                "full_chunk_fingerprint": full_fp,
                "n_chunks_full": n_full,
                "shard": args.shard,
                "start": start,
                "end": end,
                "chunks_file": cfg["paths"]["chunks"],
                "config": args.config,
                "commit": git_commit(),
                "ngay": date.today().isoformat(),
                "encode_seconds": round(time.perf_counter() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    prog_path.unlink(missing_ok=True)
    print(f"\n✅ {out_path}  ({len(texts)}×{dim}, {out_path.stat().st_size / 2**30:.2f} GB)")
    print(f"✅ {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
