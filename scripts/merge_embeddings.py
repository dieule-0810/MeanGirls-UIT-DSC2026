#!/usr/bin/env python3
"""
Ghép các mảnh `embeddings.shardKofN.npy` (chạy song song trên nhiều tài khoản Kaggle)
thành một `data/embeddings.npy`. CHỦ SỞ HỮU: P3.

Ghép embedding là chỗ dễ hỏng im lặng nhất trong cả pipeline: hai mảnh encode bằng hai model
khác nhau, hoặc từ hai phiên bản `chunks.jsonl` khác nhau, hoặc thiếu một mảnh ở giữa — ma trận
vẫn có đúng số hàng, `np.load` vẫn chạy, recall chỉ tụt không rõ lý do. Nên file này kiểm TẤT CẢ
trước khi ghi, và từ chối ghi nếu có một điều kiện không thoả:

  1. mọi mảnh cùng `full_chunk_fingerprint` (cùng một bộ chunk gốc)
  2. mọi mảnh cùng `repo` + `revision` + `pooling` + `max_length` + `dim`
  3. các khoảng [start, end) LIỀN NHAU, phủ kín [0, n_chunks_full), không chồng lấn
  4. số hàng thật của mỗi .npy khớp đúng end - start

Ghi bằng memmap ⇒ không bao giờ giữ 2,15 GB × 2 trong RAM.

    python scripts/merge_embeddings.py --shards data/shards/ --out data/embeddings.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="thư mục chứa các file *.shardKofN.npy + .meta.json")
    ap.add_argument("--out", default="data/embeddings.npy")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="float16 giảm nửa dung lượng; truy vấn trên CPU thì float32 nhanh hơn")
    args = ap.parse_args()

    shard_dir = REPO / args.shards if not Path(args.shards).is_absolute() else Path(args.shards)
    metas = sorted(shard_dir.glob("*.shard*.meta.json"))
    if not metas:
        raise SystemExit(f"❌ Không thấy file *.shard*.meta.json nào trong {shard_dir}")

    parts = []
    for mp in metas:
        m = json.loads(mp.read_text(encoding="utf-8"))
        npy = mp.with_suffix("").with_suffix(".npy")
        if not npy.exists():
            raise SystemExit(f"❌ Có meta {mp.name} nhưng thiếu {npy.name}")
        parts.append((m, npy))
    parts.sort(key=lambda t: t[0]["start"])

    ref = parts[0][0]
    n_full = ref["n_chunks_full"]
    print(f"{len(parts)} mảnh · corpus {n_full} chunk · dim {ref['dim']} · {ref['repo']}@{ref['revision'][:12]}")

    # ── kiểm 1 + 2: mọi mảnh phải cùng nguồn gốc ─────────────────────────────
    for key in ("full_chunk_fingerprint", "repo", "revision", "pooling", "max_length", "dim", "n_chunks_full"):
        vals = {json.dumps(m.get(key)) for m, _ in parts}
        if len(vals) != 1:
            raise SystemExit(
                f"❌ Các mảnh KHÔNG cùng '{key}': {sorted(vals)}\n"
                f"   Ghép chúng lại là trộn hai không gian vector khác nhau — encode lại cho khớp."
            )

    # ── kiểm 3 + 4: phủ kín, liền nhau, đúng số hàng ─────────────────────────
    cursor = 0
    for m, npy in parts:
        if m["start"] != cursor:
            raise SystemExit(
                f"❌ Đứt quãng: mảnh {m.get('shard')} bắt đầu ở {m['start']} nhưng đang chờ {cursor}. "
                f"{'Thiếu mảnh ở giữa.' if m['start'] > cursor else 'Hai mảnh chồng lấn.'}"
            )
        rows = np.load(npy, mmap_mode="r").shape[0]
        if rows != m["end"] - m["start"]:
            raise SystemExit(
                f"❌ {npy.name} có {rows} hàng, meta khai {m['end'] - m['start']}. "
                f"Nhiều khả năng session Kaggle đứt giữa chừng — chạy lại mảnh đó với --resume."
            )
        cursor = m["end"]
    if cursor != n_full:
        raise SystemExit(f"❌ Mới phủ tới {cursor}/{n_full} chunk. Thiếu mảnh cuối.")

    # ── ghi ──────────────────────────────────────────────────────────────────
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(args.dtype)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=dt, shape=(n_full, ref["dim"]))
    for m, npy in parts:
        block = np.load(npy, mmap_mode="r")
        mm[m["start"] : m["end"]] = block.astype(dt, copy=False)
        print(f"  ghép {npy.name}: [{m['start']}, {m['end']})")
    mm.flush()

    meta = {
        k: ref[k] for k in ("repo", "revision", "pooling", "normalize", "passage_prefix", "max_length", "dim")
    }
    meta.update(
        n_chunks=n_full,
        # DenseRetriever kiểm khoá này — phải là vân tay của TOÀN BỘ corpus, không phải của mảnh.
        chunk_fingerprint=ref["full_chunk_fingerprint"],
        chunks_file=ref.get("chunks_file"),
        merged_from=[str(p.name) for _, p in parts],
        dtype=args.dtype,
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {out}  ({n_full}×{ref['dim']} {args.dtype}, {out.stat().st_size / 2**30:.2f} GB)")
    print(f"✅ {out.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
