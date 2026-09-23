#!/usr/bin/env python3
"""
Dựng lại `embeddings.shardKofN.meta.json` cho các mảnh đã encode nhưng mất meta.

VÌ SAO CẦN. `encode_corpus.py` từng tính `meta_path` TRƯỚC khi gắn hậu tố mảnh vào
`out_path`, nên mọi mảnh đều ghi ra đúng một tên `embeddings.meta.json`. Chạy 8 mảnh
tuần tự trong cùng một thư mục làm việc (một notebook Kaggle) ⇒ mảnh sau đè meta của
mảnh trước, cuối cùng chỉ còn một file. Lỗi đã sửa, nhưng các mảnh encode TRƯỚC khi sửa
thì meta không tự mọc lại. File này dựng lại chúng.

ĐÂY KHÔNG PHẢI TỰ KÝ GIẤY CHỨNG NHẬN. Meta chỉ có nghĩa nếu nội dung nó khai là thứ
kiểm chứng được, nên script dựng lại xong thì KIỂM, và từ chối ghi nếu không khớp:

  1. `chunk_fingerprint` của khoảng [start, end) tính lại từ chính file chunk gốc —
     nếu bạn có log Kaggle, `--logs` sẽ so nó với dòng "Vân tay :" mà lần chạy đó in ra.
     Trùng nghĩa là mảnh này thật sự encode từ đúng bộ chunk đó, không phải ta khai bừa.
  2. số hàng thật của `.npy` phải bằng `end - start` — công thức chia mảnh lấy nguyên
     từ `encode_corpus.py`, không chép tay lại.
  3. `dim` đọc từ chính `.npy`, không lấy từ config.
  4. đếm hàng toàn 0 ở cuối: `open_memmap` cấp phát đủ kích thước NGAY khi bắt đầu, nên
     một mảnh chết giữa chừng vẫn đủ dung lượng. Đây là chỗ duy nhất phát hiện ra nó.

Những trường còn lại (`repo`, `revision`, `pooling`, `max_length`, `normalize`,
`passage_prefix`) lấy từ config — chúng là ĐẦU VÀO của lần chạy, không phải kết quả đo,
nên phải truyền đúng config đã dùng lúc encode. Sai config ở đây thì kiểm 1 vẫn qua
(vân tay chỉ phụ thuộc chunk) nhưng `DenseRetriever` sẽ bắt được lúc nạp, vì nó so
`repo`/`revision` trong meta với config đang chạy.

    python scripts/rebuild_shard_meta.py --shards data/shards \\
        --config configs/v0.4_dense.yaml --chunks data/chunks_dieu.jsonl \\
        --logs data/shards/logs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.common.io import load_chunks  # noqa: E402
from src.retrieval.dense import chunk_fingerprint  # noqa: E402

SHARD_RE = re.compile(r"\.shard(\d+)of(\d+)\.npy$")
# Dòng encode_corpus.py in ra: "Vân tay : 63c8a431167a99ca6d25917c"
LOG_FP_RE = re.compile(r"V[âa]n tay\s*:\s*([0-9a-f]{8,})")


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def shard_range(n_full: int, k: int, n_shards: int) -> tuple[int, int]:
    """GIỐNG HỆT encode_corpus.py — phần dư rải vào các mảnh đầu."""
    base, rem = divmod(n_full, n_shards)
    start = k * base + min(k, rem)
    return start, start + base + (1 if k < rem else 0)


def fingerprint_in_log(log_dir: Path, k: int) -> str | None:
    """Vân tay mà chính lần chạy đó đã in ra. Không có log thì trả None."""
    for name in (f"shard{k}.log", f"shard{k}of*.log", f"*shard{k}*.log"):
        for p in sorted(log_dir.glob(name)):
            m = LOG_FP_RE.search(p.read_text(encoding="utf-8", errors="replace"))
            if m:
                return m.group(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="thư mục chứa embeddings.shardKofN.npy")
    ap.add_argument("--config", required=True, help="config ĐÃ DÙNG lúc encode (lấy repo/revision/pooling)")
    ap.add_argument("--chunks", default=None, help="file chunk gốc; mặc định lấy paths.chunks của config")
    ap.add_argument("--logs", default=None, help="thư mục chứa shardK.log của Kaggle — để đối chiếu vân tay")
    ap.add_argument("--force", action="store_true", help="ghi đè meta đã có")
    ap.add_argument("--dry-run", action="store_true", help="kiểm và in, không ghi file nào")
    a = ap.parse_args()

    cfg = yaml.safe_load(_abs(a.config).read_text(encoding="utf-8"))
    spec = cfg["retrieval"]["dense"]
    chunks_file = a.chunks or cfg["paths"]["chunks"]

    shard_dir = _abs(a.shards)
    npys = sorted(p for p in shard_dir.glob("*.shard*.npy") if SHARD_RE.search(p.name))
    if not npys:
        raise SystemExit(f"❌ Không thấy file *.shardKofN.npy nào trong {shard_dir}")

    chunks = load_chunks(_abs(chunks_file))
    n_full = len(chunks)
    chunk_ids = [str(c["chunk_id"]) for c in chunks]
    full_fp = chunk_fingerprint(chunk_ids)
    print(f"chunk gốc : {chunks_file} · {n_full} chunk · vân tay toàn bộ {full_fp}")
    print(f"model     : {spec['repo']}@{spec['revision']}\n")

    log_dir = _abs(a.logs) if a.logs else None
    planned, problems = [], []

    for npy in npys:
        m = SHARD_RE.search(npy.name)
        k, n_shards = int(m.group(1)), int(m.group(2))
        start, end = shard_range(n_full, k, n_shards)
        arr = np.load(npy, mmap_mode="r")
        rows, dim = int(arr.shape[0]), int(arr.shape[1])
        fp = chunk_fingerprint(chunk_ids[start:end])

        notes = []
        if rows != end - start:
            problems.append(
                f"{npy.name}: {rows} hàng nhưng khoảng [{start}, {end}) cần {end - start}. "
                f"Mảnh này encode từ bộ chunk khác, hoặc --shard lúc chạy khác {k}/{n_shards}."
            )
            continue

        # Hàng toàn 0 ở CUỐI = phần chưa kịp điền. open_memmap cấp phát trước nên
        # dung lượng file không nói lên điều gì; đây mới là chỗ thấy được.
        tail = 0
        for i in range(rows - 1, -1, -1):
            if np.abs(arr[i]).sum() > 0:
                break
            tail += 1
        if tail:
            problems.append(
                f"{npy.name}: {tail}/{rows} hàng cuối toàn 0 ⇒ mảnh CHƯA chạy xong. "
                f"Chạy lại mảnh {k}/{n_shards} (có --resume nếu còn .progress.json)."
            )
            continue

        if log_dir:
            got = fingerprint_in_log(log_dir, k)
            if got is None:
                notes.append("không thấy log để đối chiếu")
            elif not (fp.startswith(got) or got.startswith(fp) or got == fp):
                problems.append(
                    f"{npy.name}: log ghi vân tay {got}, tính lại ra {fp}. Mảnh này KHÔNG "
                    f"encode từ {chunks_file} — đừng dựng meta, sẽ khai sai nguồn gốc."
                )
                continue
            else:
                notes.append("vân tay khớp log ✓")

        meta = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "pooling": spec.get("pooling", "cls"),
            "normalize": spec.get("normalize", True),
            "passage_prefix": spec.get("passage_prefix", ""),
            "max_length": spec.get("max_length", 512),
            "n_chunks": rows,
            "dim": dim,
            "chunk_fingerprint": fp,
            "full_chunk_fingerprint": full_fp,
            "n_chunks_full": n_full,
            "shard": f"{k}/{n_shards}",
            "start": start,
            "end": end,
            "chunks_file": chunks_file,
            "config": a.config,
            "rebuilt_by": "scripts/rebuild_shard_meta.py",
            "rebuilt_note": (
                "Meta dựng lại sau sự cố đặt tên của encode_corpus.py (mọi mảnh ghi đè chung "
                "embeddings.meta.json). Vân tay và số hàng đã kiểm lại từ .npy và file chunk gốc; "
                "repo/revision/pooling lấy từ config, KHÔNG đo lại được từ ma trận."
            ),
            "ngay": date.today().isoformat(),
        }
        planned.append((npy.with_suffix(".meta.json"), meta))
        print(f"  {npy.name:<34} [{start:>7}, {end:>7})  {rows}×{dim}  "
              f"{fp[:16]}…  {' · '.join(notes) if notes else ''}")

    if problems:
        print("\n❌ Không ghi gì cả — có mảnh không kiểm được:")
        for p in problems:
            print(f"   • {p}")
        return 1

    # Phủ kín và liền nhau — kiểm ở đây luôn để không phải chạy merge mới biết thiếu mảnh.
    cursor = 0
    for _, meta in sorted(planned, key=lambda t: t[1]["start"]):
        if meta["start"] != cursor:
            raise SystemExit(
                f"❌ Đứt quãng ở chunk {cursor}: mảnh {meta['shard']} bắt đầu ở {meta['start']}. "
                f"{'Thiếu mảnh ở giữa' if meta['start'] > cursor else 'Hai mảnh chồng lấn'} — "
                f"tìm đủ mảnh rồi chạy lại."
            )
        cursor = meta["end"]
    if cursor != n_full:
        raise SystemExit(f"❌ Mới phủ tới {cursor}/{n_full} chunk. Thiếu mảnh cuối.")
    print(f"\n✓ {len(planned)} mảnh liền nhau, phủ kín [0, {n_full})")

    if a.dry_run:
        print("\n--dry-run: kiểm xong, chưa ghi file nào.")
        return 0

    written = 0
    for path, meta in planned:
        if path.exists() and not a.force:
            print(f"  ⏭  {path.name} đã có, bỏ qua (dùng --force để ghi đè)")
            continue
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    print(f"\n✅ ghi {written} file meta vào {shard_dir}")
    print(f"   bước tiếp: python scripts/merge_embeddings.py --shards {a.shards} --out data/embeddings.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
