#!/usr/bin/env python3
"""Encode toàn bộ kho chunk → `data/embeddings.npy` (+ `.meta.json`).

Tách khỏi `DenseRetriever.index()` vì đây là bước DÀI NHẤT và là bước duy nhất thật sự cần GPU:
encode một lần trên máy có GPU, mọi lần đo sau chỉ đọc file. Ghi bằng memmap + sổ tiến độ nên
đứt giữa chừng thì `--resume` chạy tiếp, không encode lại. `--shard K/N` chia kho cho nhiều tài
khoản Kaggle (`docs/kaggle_encode.ipynb`), ghép lại bằng `scripts/merge_embeddings.py`.

⚠️ `--limit` ghi ra file RIÊNG (`embeddings_limit<N>.npy`), không bao giờ đè bản đầy đủ.

Typical usage example:

    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --dry-run
    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --limit 2000
    python scripts/encode_corpus.py --config configs/v0.4_dense.yaml --shard 0/8 --resume
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

import numpy as np  # noqa: E402

from src.common.config import load_config, repo_path  # noqa: E402
from src.common.io import load_chunks  # noqa: E402
from src.common.runinfo import git_commit  # noqa: E402
from src.retrieval.dense import DenseRetriever, chunk_fingerprint, resolve_device  # noqa: E402

FLUSH_EVERY = 50  # lô


def shard_bounds(n_full: int, k: int, n_shards: int) -> tuple[int, int]:
    """Biên `[start, end)` của mảnh K/N: chia đều, phần dư rải vào các mảnh đầu.

    Công thức này phải GIỐNG HỆT ở `scripts/merge_embeddings.py`.
    """
    base, rem = divmod(n_full, n_shards)
    start = k * base + min(k, rem)
    return start, start + base + (1 if k < rem else 0)


def parse_shard(spec: str) -> tuple[int, int]:
    """`"K/N"` → (K, N).

    Raises:
        SystemExit: Sai định dạng hoặc K ngoài [0, N−1].
    """
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit("❌ --shard phải có dạng K/N, ví dụ 0/8") from None
    if not 0 <= k < n:
        raise SystemExit(f"❌ --shard {spec}: K phải trong [0, {n - 1}]")
    return k, n


@dataclass
class EncodePlan:
    """Phần kho cần encode và nơi ghi.

    Attributes:
        chunks: Chunk của mảnh này (đã cắt `--limit`).
        start: Chỉ số đầu trong toàn kho.
        end: Chỉ số cuối (không gồm) trong toàn kho.
        n_full: Kích thước toàn kho.
        full_fp: Vân tay TOÀN kho — mọi mảnh phải mang cùng một giá trị.
        fp: Vân tay của đúng phần đang encode.
        out_path: File `.npy`.
    """

    chunks: list[dict]
    start: int
    end: int
    n_full: int
    full_fp: str
    fp: str
    out_path: Path

    @property
    def meta_path(self) -> Path:
        """File meta đi kèm embedding."""
        return self.out_path.with_suffix(".meta.json")

    @property
    def prog_path(self) -> Path:
        # Theo tên file ĐÃ mang hậu tố mảnh: 8 mảnh chạy chung một thư mục không đè sổ của nhau.
        """Sổ tiến độ, đặt theo tên đã mang hậu tố mảnh."""
        return self.out_path.with_suffix(".progress.json")


def make_plan(cfg: dict, args: argparse.Namespace) -> EncodePlan:
    """Đọc kho chunk, cắt theo `--shard` / `--limit`, đặt tên file ra.

    Raises:
        SystemExit: Dùng cả `--limit` lẫn `--shard`.
    """
    if args.limit and args.shard:
        raise SystemExit("❌ --limit và --shard loại trừ nhau: một cái cắt đầu, một cái chia đều.")
    out_path = repo_path(cfg["paths"].get("embeddings", "data/embeddings.npy"))
    chunks = load_chunks(repo_path(cfg["paths"]["chunks"]))
    full_fp = chunk_fingerprint([str(c["chunk_id"]) for c in chunks])  # tính TRƯỚC khi cắt mảnh
    n_full = len(chunks)
    start, end = 0, n_full
    if args.limit:
        out_path = out_path.with_name(f"{out_path.stem}_limit{args.limit}.npy")
    if args.shard:
        k, n_shards = parse_shard(args.shard)
        start, end = shard_bounds(n_full, k, n_shards)
        out_path = out_path.with_name(f"{out_path.stem}.shard{k}of{n_shards}.npy")
        chunks = chunks[start:end]
    if args.limit:
        chunks = chunks[: args.limit]
    fp = chunk_fingerprint([str(c["chunk_id"]) for c in chunks])
    return EncodePlan(chunks, start, end, n_full, full_fp, fp, out_path)


def build_encoder(cfg: dict, args: argparse.Namespace) -> DenseRetriever:
    """DenseRetriever từ khối `retrieval.dense`, áp ghi đè `--batch-size` / `--device`."""
    spec = dict(cfg["retrieval"]["dense"])
    spec.pop("embeddings_path", None)
    r = DenseRetriever.from_spec(spec)
    overrides = {k: v for k, v in (("batch_size", args.batch_size), ("device", args.device)) if v}
    return r.with_encoder(**overrides) if overrides else r


def resume_point(plan: EncodePlan, revision: str, resume: bool) -> int:
    """Số chunk đã encode xong theo sổ tiến độ (0 nếu không `--resume`).

    Raises:
        SystemExit: Sổ tiến độ thuộc kho chunk hoặc model khác.
    """
    if not (resume and plan.prog_path.exists() and plan.out_path.exists()):
        return 0
    prog = json.loads(plan.prog_path.read_text(encoding="utf-8"))
    if prog.get("chunk_fingerprint") != plan.fp or prog.get("revision") != revision:
        raise SystemExit("❌ Sổ tiến độ thuộc về bộ chunk/model KHÁC. Xoá file .progress.json và .npy rồi chạy lại.")
    print(f"↻ chạy tiếp từ chunk {prog.get('done', 0)}")
    return int(prog.get("done", 0))


def _save_progress(plan: EncodePlan, done: int, revision: str) -> None:
    """Ghi sổ tiến độ để `--resume` chạy tiếp được."""
    plan.prog_path.write_text(
        json.dumps({"done": done, "chunk_fingerprint": plan.fp, "revision": revision}), encoding="utf-8"
    )


def encode_to_memmap(r: DenseRetriever, plan: EncodePlan, done: int) -> tuple[int, bool]:
    """Encode theo lô vào memmap, ghi sổ tiến độ mỗi `FLUSH_EVERY` lô.

    Lô đầu encode trước để biết số chiều rồi mới mở memmap đúng kích thước.

    Returns:
        `(số chiều, hoàn tất?)` — False nếu bị Ctrl-C (đã flush và ghi sổ để `--resume`).
    """
    texts = [c["text"] for c in plan.chunks]
    bs = r.batch_size
    t0 = time.perf_counter()
    first = r.encode(texts[done : done + bs], is_query=False, show_every=0)
    dim = int(first.shape[1])
    mm = np.lib.format.open_memmap(plan.out_path, mode="r+" if done else "w+", dtype=np.float32, shape=(len(texts), dim))
    mm[done : done + first.shape[0]] = first
    done += first.shape[0]
    try:
        while done < len(texts):
            block = r.encode(texts[done : done + bs], is_query=False, show_every=0)
            mm[done : done + block.shape[0]] = block
            done += block.shape[0]
            if (done // bs) % FLUSH_EVERY == 0:
                mm.flush()
                _save_progress(plan, done, r.revision)
                rate = done / max(1e-9, time.perf_counter() - t0)
                print(f"  {done}/{len(texts)} ({rate:.0f} chunk/s, còn ~{(len(texts) - done) / max(rate, 1e-9) / 60:.1f} phút)")
    except KeyboardInterrupt:
        mm.flush()
        _save_progress(plan, done, r.revision)
        print(f"\n⏸  dừng ở chunk {done}. Chạy lại với --resume.")
        return dim, False
    mm.flush()
    return dim, True


def write_meta(r: DenseRetriever, plan: EncodePlan, dim: int, args: argparse.Namespace, cfg: dict, seconds: float) -> None:
    """Ghi `.meta.json` — thứ `DenseRetriever` đối chiếu trước khi dùng embedding."""
    enc = r.encoder
    meta = {
        "repo": enc.repo, "revision": enc.revision, "pooling": enc.pooling, "normalize": enc.normalize,
        "passage_prefix": enc.passage_prefix, "max_length": enc.max_length,
        "n_chunks": len(plan.chunks), "dim": dim, "chunk_fingerprint": plan.fp,
        "full_chunk_fingerprint": plan.full_fp, "n_chunks_full": plan.n_full,
        "shard": args.shard, "start": plan.start, "end": plan.end,
        "chunks_file": cfg["paths"]["chunks"], "config": args.config, "commit": git_commit(),
        "ngay": date.today().isoformat(), "encode_seconds": round(seconds, 1),
    }
    plan.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="chỉ encode N chunk đầu, ghi ra file riêng")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="encode mảnh thứ K (đếm từ 0) trong N mảnh; ghép bằng scripts/merge_embeddings.py")
    ap.add_argument("--resume", action="store_true", help="chạy tiếp từ sổ tiến độ")
    ap.add_argument("--batch-size", type=int, default=None, help="ghi đè retrieval.dense.batch_size")
    ap.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    ap.add_argument("--dry-run", action="store_true", help="in kế hoạch rồi thoát, không nạp model")
    return ap.parse_args()


def main() -> int:
    """Điểm vào CLI: lập kế hoạch, encode, ghi meta."""
    args = parse_args()
    cfg = load_config(args.config)
    plan = make_plan(cfg, args)
    r = build_encoder(cfg, args)
    bs = r.batch_size
    print(
        f"Model   : {r.repo}@{r.revision}\n"
        f"Mảnh    : {args.shard or 'toàn bộ'} → chunk [{plan.start}, {plan.end}) / {plan.n_full}\n"
        f"Chunk   : {len(plan.chunks)} · lô {bs} · {-(-len(plan.chunks) // bs)} lô\n"
        f"Thiết bị: {r.encoder.device}\nRa      : {plan.out_path}\nVân tay : {plan.fp}"
    )
    if args.dry_run:
        print("\n--dry-run: dừng ở đây, chưa nạp model, chưa đụng GPU.")
        return 0
    print(f"Thiết bị thật: {resolve_device(r.encoder.device)}")

    t0 = time.perf_counter()
    dim, finished = encode_to_memmap(r, plan, resume_point(plan, r.revision, args.resume))
    if not finished:
        return 130
    write_meta(r, plan, dim, args, cfg, time.perf_counter() - t0)
    plan.prog_path.unlink(missing_ok=True)
    print(f"\n✅ {plan.out_path}  ({len(plan.chunks)}×{dim}, {plan.out_path.stat().st_size / 2**30:.2f} GB)")
    print(f"✅ {plan.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
