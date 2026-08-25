"""
Khung `BaseRetriever` — hợp đồng của tầng truy hồi thứ nhất. CHỦ SỞ HỮU: P3.

Bản hợp đồng gốc nằm ở INTERFACES.md mục 3 và KHÔNG đổi:

    index(chunks) -> None
    search(queries, top_k) -> list[list[tuple[doc_id: str, score: float]]]
      · đã gộp chunk → document
      · đã loại trùng doc_id
      · đã sắp xếp score giảm dần
      · đã cắt còn top_k
      · len(kết quả) == len(queries)

File này thêm phần *thực thi* hợp đồng đó ở một chỗ duy nhất, thay vì mỗi retriever tự
lặp lại (và tự sai một kiểu). Subclass chỉ phải viết hai hàm:

    index(chunks)                      — dựng chỉ mục, gọi self._register_chunks(chunks)
    _score_chunks(queries, n)          — trả về, cho mỗi query, (chỉ số chunk, điểm) top-n

Còn lại — gộp chunk→doc, dedupe, sort, cắt, kiểm tra hợp đồng — khung lo.

VÌ SAO TÁCH `_score_chunks` RA:
plan.md mục 2 định hợp nhất BM25 + dense bằng RRF ở mức *chunk*, rồi mới gộp lên doc.
Nếu mỗi retriever chỉ phơi ra `search()` (đã gộp lên doc rồi) thì RRF chỉ còn hợp nhất
được ở mức doc — mất thông tin chunk nào khớp, và không so được các chiến lược gộp
(max / mean-top-3 / logsumexp) trên CÙNG một lần chấm điểm. `candidates()` giữ lại
tầng chunk cho P4 và cho `scripts/bench_retrieval.py`, mà không phá hợp đồng public.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

# Chiến lược gộp chunk → doc (plan.md mục 4, phần việc của P3)
POOL_KINDS = ("max", "sum", "mean_topN", "logsumexp")
_MEAN_TOP_RE = re.compile(r"^mean_top(\d+)$")


def parse_pool(strategy: str) -> tuple[str, int]:
    """'mean_top3' → ('mean_topN', 3). Các tên khác → (tên, 0). Sai tên thì fail ngay."""
    m = _MEAN_TOP_RE.match(strategy)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("mean_topN cần N >= 1")
        return "mean_topN", n
    if strategy in ("max", "sum", "logsumexp"):
        return strategy, 0
    raise ValueError(
        f"Chiến lược gộp '{strategy}' không có. Dùng: max | sum | mean_topN (vd mean_top3) | logsumexp"
    )


def pool_scores(
    doc_ids: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    strategy: str = "max",
    tau: float = 1.0,
) -> dict[str, float]:
    """
    Gộp điểm của nhiều chunk cùng một văn bản thành một điểm cho văn bản đó.

    max        — văn bản đáng lấy vì có MỘT điều khoản khớp. Mặc định, và là baseline v0.1.
    sum        — cộng dồn: thiên vị văn bản dài (nhiều chunk) → thường tệ với corpus luật,
                 nơi bộ luật vài trăm điều sẽ đè bẹp thông tư 3 điều. Giữ lại để CHỨNG MINH
                 điều đó bằng số, vì đây là lỗi trực giác hay gặp.
    mean_topN  — trung bình N chunk tốt nhất: thưởng cho văn bản khớp ở nhiều chỗ mà không
                 để độ dài quyết định. Văn bản có ít hơn N chunk khớp thì lấy trung bình
                 trên số chunk THẬT SỰ có (không đệm 0) — đệm 0 sẽ phạt oan văn bản ngắn.
    logsumexp  — cầu nối liên tục giữa max (tau → 0) và sum (tau lớn). Một tham số tau
                 tune được, thay vì chọn cứng một trong hai đầu.
    """
    kind, n_top = parse_pool(strategy)
    scores = np.asarray(scores, dtype=np.float64)
    if len(doc_ids) != len(scores):
        raise ValueError(f"doc_ids ({len(doc_ids)}) và scores ({len(scores)}) lệch độ dài")

    if kind == "max":
        out: dict[str, float] = {}
        for d, s in zip(doc_ids, scores):
            if s > out.get(d, -math.inf):
                out[d] = float(s)
        return out

    if kind == "sum":
        acc: dict[str, float] = {}
        for d, s in zip(doc_ids, scores):
            acc[d] = acc.get(d, 0.0) + float(s)
        return acc

    grouped: dict[str, list[float]] = {}
    for d, s in zip(doc_ids, scores):
        grouped.setdefault(d, []).append(float(s))

    if kind == "mean_topN":
        return {
            d: float(np.mean(sorted(v, reverse=True)[:n_top])) for d, v in grouped.items()
        }

    # logsumexp, ổn định số học: m + tau*log(sum(exp((s-m)/tau)))
    if tau <= 0:
        raise ValueError("logsumexp cần tau > 0 (tau → 0 chính là 'max', dùng 'max' cho rẻ)")
    out = {}
    for d, v in grouped.items():
        arr = np.asarray(v, dtype=np.float64)
        m = arr.max()
        out[d] = float(m + tau * np.log(np.exp((arr - m) / tau).sum()))
    return out


def check_contract(
    results: list[list[tuple[str, float]]],
    n_queries: int,
    top_k: int,
) -> None:
    """
    Kiểm tra kết quả đúng hợp đồng INTERFACES.md mục 3. Rẻ, chạy được cả trong production.

    Bốn lỗi bên dưới đều thuộc loại KHÔNG gây exception ở hạ nguồn mà chỉ làm tụt điểm
    im lặng (doc_id kiểu int → 0 điểm, trùng doc_id → tụt Precision, quá top_k → câu đó
    0 điểm). Bắt tại đây rẻ hơn nhiều so với bắt trên leaderboard.
    """
    if len(results) != n_queries:
        raise AssertionError(f"search() trả {len(results)} kết quả cho {n_queries} query")
    for i, res in enumerate(results):
        if len(res) > top_k:
            raise AssertionError(f"query {i}: {len(res)} doc > top_k={top_k}")
        ids = [d for d, _ in res]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"query {i}: doc_id trùng lặp — xem INTERFACES.md mục 0")
        for d, s in res:
            if not isinstance(d, str):
                raise AssertionError(
                    f"query {i}: doc_id {d!r} kiểu {type(d).__name__}, phải là str. "
                    f"BTC cho 0 điểm IM LẶNG với int."
                )
            if not isinstance(s, float):
                raise AssertionError(f"query {i}: score của {d} kiểu {type(s).__name__}, phải là float")
        vals = [s for _, s in res]
        if any(a < b for a, b in zip(vals, vals[1:])):
            raise AssertionError(f"query {i}: score chưa sắp xếp giảm dần")


class BaseRetriever(ABC):
    """
    Lớp cha của mọi retriever tầng 1 (BM25, dense, hybrid).

    KPI của P3 là Recall@50: doc đúng không lọt top-50 thì reranker của P4 không cứu được.
    Vì vậy mọi tham số ảnh hưởng tới trần đó (`candidate_chunks`, chiến lược gộp) đều nằm
    ở đây, khai báo trong YAML, không rải rác trong code.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        pool: str = "max",
        pool_tau: float = 1.0,
        candidate_chunks: int = 2000,
    ) -> None:
        parse_pool(pool)  # fail ngay lúc dựng, không phải sau 3 phút index
        self.pool = pool
        self.pool_tau = pool_tau
        self.candidate_chunks = candidate_chunks
        self.chunk_ids: list[str] = []
        self.chunk_doc_ids: list[str] = []
        self._doc_ids_arr: np.ndarray | None = None

    # ── hai hàm subclass phải viết ──────────────────────────────────────────
    @abstractmethod
    def index(self, chunks: list[dict]) -> None:
        """`chunks` đọc từ `chunks.jsonl` (INTERFACES.md mục 2). Gọi một lần."""

    @abstractmethod
    def _score_chunks(
        self, queries: list[str], n_candidates: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Cho mỗi query: `(chỉ số chunk, điểm)` của tối đa `n_candidates` chunk tốt nhất,
        đã sắp xếp giảm dần. Chỉ số trỏ vào `self.chunk_ids` / `self.chunk_doc_ids`.
        Query không khớp gì → hai mảng rỗng (KHÔNG phải None).
        """

    # ── phần khung lo ────────────────────────────────────────────────────────
    def _register_chunks(self, chunks: list[dict]) -> None:
        """Ghi nhận danh sách chunk và ép kiểu `str` — gọi ở đầu `index()`."""
        doc_ids, chunk_ids = [], []
        seen: set[str] = set()
        synthesized = 0
        for i, c in enumerate(chunks):
            doc_id = str(c["doc_id"])
            cid = c.get("chunk_id")
            if cid is None:
                cid = f"{doc_id}::{int(c.get('position', i)):04d}"
                synthesized += 1
            cid = str(cid)
            if cid in seen:
                raise ValueError(
                    f"chunk_id trùng: {cid!r}. INTERFACES.md mục 2 yêu cầu duy nhất toàn corpus "
                    f"— P4 sẽ dùng nó để tra ngược văn bản khi rerank."
                )
            seen.add(cid)
            doc_ids.append(doc_id)
            chunk_ids.append(cid)
        if synthesized:
            print(
                f"  ⚠️  {synthesized}/{len(chunks)} chunk thiếu 'chunk_id', đã tự sinh "
                f"'doc_id::position'. Báo P2 — INTERFACES.md mục 2 yêu cầu trường này."
            )
        self.chunk_doc_ids = doc_ids
        self.chunk_ids = chunk_ids
        self._doc_ids_arr = np.asarray(doc_ids, dtype=object)

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @property
    def n_docs(self) -> int:
        return len(set(self.chunk_doc_ids))

    def candidates(
        self, queries: list[str], n_candidates: int | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Tầng chunk, chưa gộp. Dùng cho RRF (plan.md mục 2) và cho lưới bench:
        chấm điểm MỘT lần rồi thử nhiều chiến lược gộp trên cùng kết quả đó.
        """
        if not self.chunk_ids:
            raise RuntimeError(f"{type(self).__name__}: gọi .index() trước khi truy vấn")
        n = n_candidates or self.candidate_chunks
        return self._score_chunks(list(queries), max(1, n))

    def pool_candidates(
        self,
        idx: np.ndarray,
        scores: np.ndarray,
        top_k: int,
        pool: str | None = None,
        tau: float | None = None,
    ) -> list[tuple[str, float]]:
        """Gộp kết quả chunk của MỘT query lên document, sort giảm dần, cắt top_k."""
        if len(idx) == 0:
            return []
        assert self._doc_ids_arr is not None
        docs = self._doc_ids_arr[idx]
        pooled = pool_scores(
            docs,
            scores,
            pool or self.pool,
            self.pool_tau if tau is None else tau,
        )
        # Sort phụ theo doc_id để hai lần chạy giống nhau ra cùng thứ tự khi điểm bằng nhau
        ranked = sorted(pooled.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(d, float(s)) for d, s in ranked[:top_k]]

    def search(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        """Hợp đồng INTERFACES.md mục 3. Không override ở subclass."""
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        n_cand = max(self.candidate_chunks, top_k)
        results = [
            self.pool_candidates(idx, sc, top_k)
            for idx, sc in self.candidates(list(queries), n_cand)
        ]
        check_contract(results, len(queries), top_k)
        return results

    def search_chunks(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        """
        Như `search()` nhưng trả `chunk_id` thay vì `doc_id`, chưa gộp.
        Dành cho P4 (rerank cần đúng đoạn văn bản) và cho hợp nhất RRF ở mức chunk.
        """
        out = []
        for idx, sc in self.candidates(list(queries), top_k):
            out.append([(self.chunk_ids[int(i)], float(s)) for i, s in zip(idx, sc)])
        return out

    def stats(self) -> dict:
        """Số liệu ghi vào log/experiments.csv. Subclass mở rộng thêm."""
        return {
            "retriever": self.name,
            "n_chunks": self.n_chunks,
            "n_docs": self.n_docs,
            "pool": self.pool,
            "candidate_chunks": self.candidate_chunks,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry: YAML gọi tên retriever, không import trực tiếp
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[BaseRetriever]] = {}


def register_retriever(name: str):
    def deco(cls: type[BaseRetriever]) -> type[BaseRetriever]:
        cu = _REGISTRY.get(name)
        # `python -m src.retrieval.bm25` nạp module hai lần (một lần qua package, một lần
        # dưới tên '__main__') → cùng một lớp đăng ký hai lần. Đó không phải xung đột.
        if cu is not None and cu.__qualname__ != cls.__qualname__:
            raise KeyError(f"Retriever '{name}' đã đăng ký bởi {cu.__qualname__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_retrievers() -> list[str]:
    return sorted(_REGISTRY)


def build_retriever(spec: dict) -> BaseRetriever:
    """
    `{"type": "bm25", "k1": 1.5, ...}` → instance.

    Cho phép YAML mô tả trọn vẹn một thí nghiệm (INTERFACES.md mục 6: không hằng số
    hard-code) và cho hybrid sau này dựng retriever con từ chính config của nó.
    """
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError("Thiếu khoá 'type' trong config retriever")
    if kind not in _REGISTRY:
        # Quy ước: retriever tên 'x' nằm ở src/retrieval/x.py. Nạp lười theo tên để
        # `src/retrieval/__init__.py` không phải import sẵn mọi backend (torch của dense
        # nặng và không phải lúc nào cũng cần).
        try:
            import importlib

            importlib.import_module(f"{__package__}.{kind}")
        except ImportError:
            pass
    if kind not in _REGISTRY:
        raise KeyError(f"Retriever '{kind}' chưa đăng ký. Có: {', '.join(available_retrievers())}")
    return _REGISTRY[kind](**spec)
