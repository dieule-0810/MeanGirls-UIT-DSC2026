"""Chunker theo Điều, chiến lược `strict` (v2) — lối vào CLI giữ tương thích.

Toàn bộ logic nằm ở `src/data/chunker.py`; file này chỉ đổi chiến lược mặc định sang `strict`
và file ra mặc định sang `data/chunks_dieu.jsonl`.

Kho chunk của pipeline v0.8 dựng bằng:

    python -m src.data.chunker_dieu --out data/chunks.jsonl     # 432.142 chunk / 8.507 văn bản
"""
from src.data.chunker import main

if __name__ == "__main__":
    main(default_strategy="strict")
