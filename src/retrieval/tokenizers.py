"""
Tokenizer tiếng Việt cho BM25 (plan.md mục 4) — CHỦ SỞ HỮU: P3.

5 backend cùng giao diện, cùng chuẩn hoá (NFC → tách từ → lowercase → gộp dấu thanh),
khác đúng một chỗ — cách cắt từ. Vì sao có cả word-segment lẫn âm tiết thuần, và giả thuyết
đo được gì: docs/p3_retrieval.md mục 2 (H1/H1b).

  regex              `\\w+`, mốc tham chiếu v0.1, đừng đổi.
  whitespace         giữ nguyên cụm dính dấu (số hiệu văn bản không vỡ vụn).
  syllable_bigram    âm tiết + bigram liền kề, xấp xỉ word-segment không cần thư viện.
  pyvi / underthesea word-segment thật, cần `pip install -r requirements-p3.txt`.

Segment TRƯỚC khi lowercase: pyvi/underthesea dùng chữ hoa làm tín hiệu tên riêng.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pickle
import sys
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence

WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_SPLIT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

BUILTIN_TOKENIZERS = ("regex", "whitespace", "syllable_bigram")
SEGMENTER_TOKENIZERS = ("pyvi", "underthesea")
TOKENIZER_NAMES = BUILTIN_TOKENIZERS + SEGMENTER_TOKENIZERS


# ─────────────────────────────────────────────────────────────────────────────
# Chuẩn hoá chính tả tiếng Việt
# ─────────────────────────────────────────────────────────────────────────────
def _build_tone_fold_map() -> dict[str, str]:
    """"hoà"↔"hòa" là cùng một từ, khác byte → hai term với BM25. Quy về một dạng cho oa/oe/uy."""
    tones = {
        "o": "òóỏõọ",
        "u": "ùúủũụ",
        "a": "àáảãạ",
        "e": "èéẻẽẹ",
        "y": "ỳýỷỹỵ",
    }
    m: dict[str, str] = {}
    for first, second in (("o", "a"), ("o", "e"), ("u", "y")):
        for i in range(5):
            m[first + tones[second][i]] = tones[first][i] + second
    return m


_TONE_FOLD_MAP = _build_tone_fold_map()
# (?<!q): "quý", "quỷ" đã đúng chuẩn — đừng đụng vào chúng.
_TONE_FOLD_RE = re.compile(r"(?<!q)u[ỳýỷỹỵ]|o[àáảãạèéẻẽẹ]")


def fold_tone_placement(text: str) -> str:
    """Quy vị trí dấu thanh của oa/oe/uy về một kiểu. Giả định đầu vào đã NFC + lowercase."""
    return _TONE_FOLD_RE.sub(lambda m: _TONE_FOLD_MAP[m.group(0)], text)


# ─────────────────────────────────────────────────────────────────────────────
# Backend tách từ (nạp lười, chỉ khi được gọi)
# ─────────────────────────────────────────────────────────────────────────────
def available_tokenizers() -> dict[str, bool]:
    """{tên: đã cài chưa} — để bench bỏ qua backend thiếu thay vì chết giữa lưới."""
    out = {name: True for name in BUILTIN_TOKENIZERS}
    for name in SEGMENTER_TOKENIZERS:
        out[name] = importlib.util.find_spec(name) is not None
    return out


@lru_cache(maxsize=None)
def _segmenter(name: str):
    """Nạp một lần cho mỗi tiến trình (quan trọng khi chạy multiprocessing)."""
    if name == "pyvi":
        try:
            from pyvi import ViTokenizer
        except ImportError as e:
            raise ImportError(
                "Cần `pip install -r requirements-p3.txt` để dùng tokenizer 'pyvi'. "
                "Không muốn cài thì dùng 'syllable_bigram' — xấp xỉ word-segment, không cần dependency."
            ) from e
        return lambda t: ViTokenizer.tokenize(t)
    if name == "underthesea":
        try:
            from underthesea import word_tokenize
        except ImportError as e:
            raise ImportError(
                "Cần `pip install -r requirements-p3.txt` để dùng tokenizer 'underthesea'. "
                "Không muốn cài thì dùng 'syllable_bigram'."
            ) from e
        return lambda t: word_tokenize(t, format="text")
    raise KeyError(name)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Tokenizer:
    """Pickle được (chỉ tên + tuỳ chọn, backend nạp lười) → dùng được với multiprocessing."""

    name: str = "regex"
    lowercase: bool = True
    fold_tone: bool = True
    min_len: int = 1
    keep_syllables: bool = True  # chỉ có nghĩa với syllable_bigram

    def __post_init__(self) -> None:
        if self.name not in TOKENIZER_NAMES:
            raise ValueError(
                f"Tokenizer '{self.name}' không tồn tại. Có: {', '.join(TOKENIZER_NAMES)}"
            )
        if self.min_len < 1:
            raise ValueError("min_len phải >= 1")

    # ── định danh: đi vào tên file cache và vào experiments.csv ──
    @property
    def key(self) -> str:
        bits = [self.name]
        if not self.lowercase:
            bits.append("nolower")
        if not self.fold_tone:
            bits.append("notonefold")
        if self.min_len > 1:
            bits.append(f"minlen{self.min_len}")
        if self.name == "syllable_bigram" and not self.keep_syllables:
            bits.append("bigramonly")
        return "-".join(bits)

    def __str__(self) -> str:  # pragma: no cover - tiện log
        return self.key

    # ── ba bước, dùng chung cho mọi backend ──
    def _segment(self, text: str) -> str:
        """NFC rồi tách từ nếu backend có — giữ nguyên chữ hoa cho segmenter."""
        text = unicodedata.normalize("NFC", text)
        if self.name in SEGMENTER_TOKENIZERS:
            return _segmenter(self.name)(text)
        return text

    def _normalize(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.fold_tone:
            text = fold_tone_placement(text)
        return text

    def _extract(self, text: str) -> list[str]:
        if self.name == "whitespace":
            # Tách theo khoảng trắng, chỉ gọt dấu câu hai đầu — giữ nguyên "03/2020/tt-btc".
            toks = []
            for raw in text.split():
                t = raw.strip(".,;:()[]{}\"'“”‘’…-–—/\\!?*")
                if t:
                    toks.append(t)
            return toks

        if self.name == "syllable_bigram":
            # Bigram không được bắc qua dấu câu: "…quy định. Điều 5…" không sinh "định_điều".
            toks: list[str] = []
            for seg in _PUNCT_SPLIT_RE.split(text):
                syl = WORD_RE.findall(seg)
                if self.keep_syllables:
                    toks.extend(syl)
                toks.extend(f"{a}_{b}" for a, b in zip(syl, syl[1:]))
            return toks

        # regex / pyvi / underthesea: '_' nằm trong \w nên "cơ_quan" giữ nguyên một token.
        return WORD_RE.findall(text)

    def __call__(self, text: str) -> list[str]:
        toks = self._extract(self._normalize(self._segment(text)))
        if self.min_len > 1:
            toks = [t for t in toks if len(t) >= self.min_len]
        # sys.intern: ~134 triệu token trên corpus thật → không intern là ~8 GB, intern
        # xuống còn ~1 GB con trỏ. Khác biệt giữa chạy được và OOM trên máy 16 GB.
        return [sys.intern(t) for t in toks]

    def batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self(t) for t in texts]


def get_tokenizer(name: str | Tokenizer = "regex", **opts) -> Tokenizer:
    """`get_tokenizer("pyvi", fold_tone=False)` hoặc truyền thẳng một Tokenizer để override."""
    if isinstance(name, Tokenizer):
        return replace(name, **opts) if opts else name
    return Tokenizer(name=name, **opts)


# ─────────────────────────────────────────────────────────────────────────────
# Tách từ hàng loạt: cache đĩa + đa tiến trình
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_TOKENIZER: Tokenizer | None = None


def _worker_init(tok: Tokenizer) -> None:  # pragma: no cover - chạy trong tiến trình con
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tok


def _worker_run(texts: list[str]) -> list[list[str]]:  # pragma: no cover
    assert _WORKER_TOKENIZER is not None
    return _WORKER_TOKENIZER.batch(texts)


def _cache_key(tok: Tokenizer, texts: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(tok.key.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()


def tokenize_many(
    texts: Sequence[str],
    tok: Tokenizer,
    *,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> list[list[str]]:
    """
    Tách từ cho cả corpus. Cache theo nội dung (P2 đổi chunker → cache tự hết hiệu lực),
    vì underthesea/pyvi tách 500k+ chunk mất hàng chục phút và bench chạy lại nhiều lần.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"tokens_{tok.key}_{_cache_key(tok, texts)}.pkl"
        if cache_path.exists():
            if verbose:
                print(f"  [tok] dùng cache {cache_path.name}")
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

    t0 = time.perf_counter()
    if n_jobs and n_jobs > 1 and len(texts) > 1000:
        from concurrent.futures import ProcessPoolExecutor

        block = max(200, len(texts) // (n_jobs * 8) + 1)
        blocks = [list(texts[i : i + block]) for i in range(0, len(texts), block)]
        out: list[list[str]] = []
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_worker_init, initargs=(tok,)
        ) as ex:
            for part in ex.map(_worker_run, blocks):
                out.extend(part)
    else:
        out = tok.batch(texts)
    dt = time.perf_counter() - t0

    if verbose:
        n_tok = sum(len(t) for t in out)
        print(
            f"  [tok] {tok.key}: {len(texts)} văn bản → {n_tok} token "
            f"({n_tok / max(1, len(texts)):.1f} token/văn bản) trong {dt:.1f}s"
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out
