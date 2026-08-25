"""
Đọc/ghi các file trung gian đã khoá trong INTERFACES.md mục 7. Dùng chung, không thuộc riêng ai.

Mọi nơi đọc `chunks.jsonl` / `corpus_clean.jsonl` / file câu hỏi đều báo lỗi giống nhau và
ép `str` giống nhau (INTERFACES.md mục 0: `doc_id` luôn là `str`), thay vì mỗi script tự viết
lại một kiểu rồi lệch nhau đúng chỗ quan trọng nhất.

`load_chunks`/`load_corpus_ids` chấp nhận cả `id`/`passage` lẫn `doc_id`/`text` — P2 từng ghi
tên trường khác hợp đồng, giờ đã sửa (main), giữ alias này chỉ để tương thích ngược với dữ
liệu cũ, không phải vá lỗi đang xảy ra.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

# Tên trường thay thế được chấp nhận → tên trong hợp đồng
DOC_ID_ALIASES = ("doc_id", "id")
TEXT_ALIASES = ("text", "passage")
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(msg)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Đọc JSONL, bỏ dòng trống, báo rõ số dòng khi JSON hỏng."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy {p}. File này do P2 sinh (INTERFACES.md mục 7) — chạy "
            f"`python -m src.data.parse_corpus` rồi `python -m src.data.chunker` trước."
        )
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL lỗi tại {p}, dòng {line_no}: {e}") from e


def _pick(record: dict, aliases: tuple[str, ...]) -> str | None:
    for k in aliases:
        if k in record:
            return k
    return None


def load_chunks(path: str | Path, skip_empty: bool = True) -> list[dict]:
    """
    Nạp `chunks.jsonl`, chuẩn hoá về `chunk_id`/`doc_id`/`position`/`text`, ép `str` ngay tại
    điểm đọc (INTERFACES.md mục 0). `skip_empty` bỏ chunk rỗng (EDA: 20 văn bản passage rỗng,
    index chúng chỉ phồng ma trận chứ không bao giờ khớp được gì).
    """
    chunks: list[dict] = []
    n_empty = 0
    for i, c in enumerate(read_jsonl(path)):
        doc_key = _pick(c, DOC_ID_ALIASES)
        text_key = _pick(c, TEXT_ALIASES)
        if doc_key is None or text_key is None:
            raise ValueError(
                f"{path} dòng {i + 1}: chunk thiếu trường bắt buộc. Cần một trong "
                f"{DOC_ID_ALIASES} và một trong {TEXT_ALIASES} (INTERFACES.md mục 2). "
                f"Thực tế có: {sorted(c)}"
            )
        if doc_key != "doc_id" or text_key != "text":
            _warn_once(
                "chunks_alias",
                f"  ⚠️  {path}: dùng '{doc_key}'/'{text_key}' thay vì 'doc_id'/'text' "
                f"(INTERFACES.md mục 2). Đã tự dịch tên trường — xem ghi chú đầu src/common/io.py.",
            )

        text = c[text_key] or ""
        if skip_empty and not str(text).strip():
            n_empty += 1
            continue

        chunk_id = c.get("chunk_id")
        position = c.get("position")
        if position is None and isinstance(chunk_id, str) and "::" in chunk_id:
            tail = chunk_id.rsplit("::", 1)[1]
            position = int(tail) if tail.isdigit() else i
        chunks.append(
            {
                **c,
                "doc_id": str(c[doc_key]),
                "text": str(text),
                "chunk_id": str(chunk_id) if chunk_id is not None else None,
                "position": int(position) if position is not None else i,
            }
        )
    if n_empty:
        print(f"  ⚠️  bỏ qua {n_empty} chunk rỗng (EDA mục 3: 20 văn bản passage rỗng)")
    if not chunks:
        raise ValueError(f"{path} rỗng — không có chunk nào để đánh chỉ mục.")
    return chunks


def load_corpus_ids(path: str | Path) -> set[str]:
    """Tập `doc_id` (str) của `corpus_clean.jsonl` — để kiểm doc_id lạ."""
    out: set[str] = set()
    for d in read_jsonl(path):
        key = _pick(d, DOC_ID_ALIASES)
        if key is None:
            raise ValueError(f"{path}: bản ghi thiếu doc_id/id — {sorted(d)}")
        out.add(str(d[key]))
    return out


def load_questions(path: str | Path) -> tuple[list[str], list[str]]:
    """
    Đọc file câu hỏi (`holdout.json`, `public-official.json`, `train.json`).

    Trả về `(qids, texts)` cùng thứ tự. `qid` luôn là `str`.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    qids = [str(q) for q in raw]
    texts = []
    for q in raw:
        v = raw[q]
        text = v["question"] if isinstance(v, dict) else v
        if not isinstance(text, str):
            raise ValueError(f"qid {q}: 'question' phải là str, nhận {type(text).__name__}")
        texts.append(text)
    return qids, texts


def write_predictions(preds: dict[str, list[str]], path: str | Path) -> Path:
    """
    Ghi `Predictions` (INTERFACES.md mục 4): `{qid: [doc_id, ...]}` — PHẲNG.

    Không bọc `{"answer": ...}` ở đây; cấu trúc đó chỉ xuất hiện trong `make_submission.py`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = {str(q): [str(d) for d in docs] for q, docs in preds.items()}
    out.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
