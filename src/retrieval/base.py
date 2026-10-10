"""Hợp đồng tầng truy hồi thứ nhất `BaseRetriever` (INTERFACES.md §3). CHỦ SỞ HỮU: P3.

Subclass chỉ viết hai hàm:

* `index(chunks)` — gọi `self._register_chunks(chunks)` ở dòng đầu;
* `_score_chunks(queries, n)` — top-n chunk mỗi query, đã sort giảm dần.

Khung lo phần còn lại: gộp chunk→doc, dedupe, sort, cắt `top_k`, `check_contract()`.
`_score_chunks` tách khỏi `search()` để giữ tầng chunk cho RRF mức chunk (`hybrid.py`) và
cho chunk đại diện mà reranker cần (INTERFACES.md §3b).

Retriever dựng từ YAML qua registry: `build_retriever({"type": "bm25", ...})` gọi
`from_spec()` của lớp đã đăng ký, tách spec phẳng thành các dataclass cấu hình.
"""
from __future__ import annotations

import importlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

# Chiến lược gộp chunk → doc. Vì sao có cả bốn: INTERFACES.md §3b.
POOL_KINDS = ("max", "sum", "mean_topN", "logsumexp")
_MEAN_TOP_RE = re.compile(r"^mean_top(\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Gộp chunk → doc
# ─────────────────────────────────────────────────────────────────────────────
def parse_pool(strategy: str) -> tuple[str, int]:
    """Tách tên chiến lược gộp thành (loại, N).

    Args:
        strategy: `max`, `sum`, `logsumexp` hoặc `mean_topN` (vd `mean_top2`).

    Returns:
        `("mean_topN", N)` cho mean_topN, `(tên, 0)` cho các loại còn lại.

    Raises:
        ValueError: Tên không hợp lệ hoặc N < 1 — fail ngay lúc dựng, không phải sau index.
    """
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


def _group_scores(doc_ids: Sequence[str], scores: np.ndarray) -> dict[str, list[float]]:
    """Gom điểm chunk theo `doc_id`."""
    grouped: dict[str, list[float]] = {}
    for d, s in zip(doc_ids, scores):
        grouped.setdefault(d, []).append(float(s))
    return grouped


def _logsumexp(values: list[float], tau: float) -> float:
    """`m + tau·log Σ exp((s−m)/tau)` — ổn định số học."""
    arr = np.asarray(values, dtype=np.float64)
    m = arr.max()
    return float(m + tau * np.log(np.exp((arr - m) / tau).sum()))


def pool_scores(
    doc_ids: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    strategy: str = "max",
    tau: float = 1.0,
) -> dict[str, float]:
    """Gộp điểm nhiều chunk cùng văn bản thành một điểm cho văn bản đó.

    `mean_topN` lấy trung bình trên số chunk THẬT SỰ có trong tập ứng viên, không đệm 0 —
    đệm 0 là phạt oan văn bản ngắn. Hệ quả: `candidate_chunks` KHÔNG trung tính với
    `mean_topN` (xem `v0.5_candnull_*` trong experiments.csv).

    Args:
        doc_ids: `doc_id` của từng chunk, cùng độ dài với `scores`.
        scores: Điểm của từng chunk.
        strategy: Chiến lược gộp, xem `parse_pool`.
        tau: Nhiệt độ của `logsumexp` (tau → 0 chính là `max`).

    Returns:
        `{doc_id: điểm đã gộp}`.

    Raises:
        ValueError: Hai dãy lệch độ dài, hoặc `tau <= 0` với `logsumexp`.
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

    grouped = _group_scores(doc_ids, scores)
    if kind == "mean_topN":
        return {d: float(np.mean(sorted(v, reverse=True)[:n_top])) for d, v in grouped.items()}
    if tau <= 0:
        raise ValueError("logsumexp cần tau > 0 (tau → 0 chính là 'max', dùng 'max' cho rẻ)")
    return {d: _logsumexp(v, tau) for d, v in grouped.items()}


def top_n_desc(
    scores: np.ndarray, n: int, candidates: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Chọn `n` chỉ số điểm cao nhất, sắp giảm dần.

    `argpartition` trước rồi mới sort phần đã chọn: chỉ cần biết top-n là ai, không cần sắp
    xếp 432.142 phần tử còn lại. Sort `stable` để hoà điểm giữ thứ tự chỉ số — kết quả xác
    định giữa hai lần chạy.

    Args:
        scores: Điểm của MỌI chunk (mảng 1 chiều).
        n: Số phần tử cần giữ.
        candidates: Chỉ chọn trong tập chỉ số này (vd chunk có điểm > 0). None = mọi chỉ số.

    Returns:
        `(chỉ số int64, điểm float32)` đã sắp giảm dần, dài tối đa `n`.
    """
    pool = np.arange(scores.size) if candidates is None else candidates
    if pool.size > n:
        pool = pool[np.argpartition(-scores[pool], n - 1)[:n]]
    pool = pool[np.argsort(-scores[pool], kind="stable")]
    return pool.astype(np.int64), scores[pool].astype(np.float32)


def check_contract(results: list[list[tuple[str, float]]], n_queries: int, top_k: int) -> None:
    """Kiểm hợp đồng INTERFACES.md §3. Rẻ, để chạy cả ở production.

    Bốn lỗi dưới đây BTC không báo, chỉ trừ điểm im lặng: doc_id kiểu int, trùng doc_id,
    quá top_k, điểm chưa sắp xếp.

    Args:
        results: Kết quả `search()` — mỗi query một danh sách `(doc_id, score)`.
        n_queries: Số query đã gửi.
        top_k: Độ dài tối đa cho phép của mỗi danh sách.

    Raises:
        AssertionError: Khi vi phạm bất kỳ điều khoản nào.
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


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PoolingConfig:
    """Cách một retriever gộp chunk lên doc.

    Attributes:
        pool: Chiến lược gộp (`max`, `sum`, `mean_topN`, `logsumexp`).
        pool_tau: Nhiệt độ của `logsumexp`.
        candidate_chunks: Số chunk tốt nhất giữ lại trước khi gộp. Là THAM SỐ PHƯƠNG PHÁP,
            không phải mẹo tăng tốc: với `mean_topN` nó đổi kết quả.
    """

    pool: str = "max"
    pool_tau: float = 1.0
    candidate_chunks: int = 2000

    def __post_init__(self) -> None:
        """Kiểm tên chiến lược gộp ngay lúc dựng."""
        parse_pool(self.pool)  # fail ngay lúc dựng, không phải sau 3 phút index


def split_spec(spec: dict, *configs: type) -> tuple[list[dict], dict]:
    """Chia một spec phẳng (từ YAML) thành kwargs cho từng dataclass cấu hình.

    Args:
        spec: Mapping phẳng, vd `{"k1": 1.5, "pool": "mean_top2", "verbose": False}`.
        *configs: Các lớp dataclass, mỗi khoá thuộc về lớp ĐẦU TIÊN có trường cùng tên.

    Returns:
        `(danh sách kwargs theo thứ tự configs, phần còn lại không thuộc lớp nào)`.
    """
    rest = dict(spec)
    parts: list[dict] = []
    for cls in configs:
        names = {f.name for f in fields(cls)}
        parts.append({k: rest.pop(k) for k in list(rest) if k in names})
    return parts, rest


def reject_unknown(cls_name: str, rest: dict, allowed: Sequence[str] = ()) -> None:
    """Báo lỗi nếu spec còn khoá không thuộc cấu hình nào — gõ sai tên khoá YAML phải dừng.

    Raises:
        TypeError: Còn khoá lạ ngoài `allowed`.
    """
    unknown = sorted(set(rest) - set(allowed))
    if unknown:
        raise TypeError(f"{cls_name}: khoá không nhận ra trong config: {unknown}")


# ─────────────────────────────────────────────────────────────────────────────
# Lớp cha
# ─────────────────────────────────────────────────────────────────────────────
class BaseRetriever(ABC):
    """Lớp cha mọi retriever tầng 1 (BM25, dense, hybrid). KPI: Recall@50.

    Attributes:
        name: Tên đăng ký trong registry (giá trị `type:` trong YAML).
        pooling: Cấu hình gộp chunk→doc.
        chunk_ids: `chunk_id` theo thứ tự đã index.
        chunk_doc_ids: `doc_id` của từng chunk, cùng thứ tự với `chunk_ids`.
    """

    name: str = "base"

    def __init__(self, pooling: PoolingConfig | None = None) -> None:
        self.pooling = pooling or PoolingConfig()
        self.chunk_ids: list[str] = []
        self.chunk_doc_ids: list[str] = []
        self._doc_ids_arr: np.ndarray | None = None

    @classmethod
    def from_spec(cls, spec: dict) -> "BaseRetriever":
        """Dựng retriever từ spec phẳng của YAML. Subclass có cấu hình riêng thì override."""
        return cls(**spec)

    # ── thuộc tính gộp, chỉ đọc ──────────────────────────────────────────────
    @property
    def pool(self) -> str:
        """Chiến lược gộp chunk→doc."""
        return self.pooling.pool

    @property
    def pool_tau(self) -> float:
        """Nhiệt độ của `logsumexp`."""
        return self.pooling.pool_tau

    @property
    def candidate_chunks(self) -> int:
        """Số chunk ứng viên giữ lại trước khi gộp."""
        return self.pooling.candidate_chunks

    # ── hai hàm subclass phải viết ──────────────────────────────────────────
    @abstractmethod
    def index(self, chunks: list[dict]) -> None:
        """Đánh chỉ mục kho chunk. Gọi một lần.

        Args:
            chunks: Đọc từ `chunks.jsonl` (INTERFACES.md §2).
        """

    @abstractmethod
    def _score_chunks(self, queries: list[str], n_candidates: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Chấm điểm chunk cho từng query.

        Args:
            queries: Câu hỏi.
            n_candidates: Số chunk tốt nhất tối đa cần trả cho mỗi query.

        Returns:
            Mỗi query một cặp `(chỉ số chunk, điểm)` đã sắp giảm dần. Chỉ số trỏ vào
            `self.chunk_ids`. Query không khớp gì → hai mảng RỖNG, không phải None.
        """

    # ── phần khung lo ────────────────────────────────────────────────────────
    def _register_chunks(self, chunks: list[dict]) -> None:
        """Ghi nhận kho chunk, ép `str`, tự sinh `chunk_id` nếu thiếu. Gọi đầu `index()`.

        Raises:
            ValueError: `chunk_id` trùng — INTERFACES.md §2 yêu cầu duy nhất toàn corpus.
        """
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
        """Số chunk đã index."""
        return len(self.chunk_ids)

    @property
    def n_docs(self) -> int:
        """Số văn bản khác nhau đã index."""
        return len(set(self.chunk_doc_ids))

    def candidates(
        self, queries: list[str], n_candidates: int | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Ứng viên mức chunk, chưa gộp — cho RRF mức chunk và lưới bench.

        Args:
            queries: Câu hỏi.
            n_candidates: Số chunk tối đa mỗi query; None = `candidate_chunks`.

        Returns:
            Như `_score_chunks`.

        Raises:
            RuntimeError: Chưa gọi `index()`.
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
        """Gộp ứng viên chunk của MỘT query lên doc, sort giảm dần, cắt `top_k`.

        Hoà điểm thì sort phụ theo `doc_id` để hai lần chạy ra cùng thứ tự.

        Args:
            idx: Chỉ số chunk.
            scores: Điểm chunk, cùng độ dài `idx`.
            top_k: Số doc giữ lại.
            pool: Ghi đè chiến lược gộp (lưới bench); None = của retriever.
            tau: Ghi đè `pool_tau`; None = của retriever.

        Returns:
            Danh sách `(doc_id, score)`.
        """
        if len(idx) == 0:
            return []
        assert self._doc_ids_arr is not None
        pooled = pool_scores(
            self._doc_ids_arr[idx], scores, pool or self.pool, self.pool_tau if tau is None else tau
        )
        ranked = sorted(pooled.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(d, float(s)) for d, s in ranked[:top_k]]

    def search(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        """Hợp đồng INTERFACES.md §3. Không override ở subclass.

        Args:
            queries: Câu hỏi.
            top_k: Số doc mỗi query.

        Returns:
            Mỗi query một danh sách `(doc_id, score)` đã gộp, dedupe, sort giảm dần, cắt top_k.
        """
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        n_cand = max(self.candidate_chunks, top_k)
        results = [
            self.pool_candidates(idx, sc, top_k) for idx, sc in self.candidates(list(queries), n_cand)
        ]
        check_contract(results, len(queries), top_k)
        return results

    def _anchor_chunks(self, idx: np.ndarray, scores: np.ndarray) -> dict[str, str]:
        """Chunk đại diện của từng doc: điểm cao nhất, hoà thì `chunk_id` NHỎ NHẤT.

        Quy ước phá hoà phải tường minh: 17/300 câu error_pool có ≥2 chunk cùng văn bản hoà
        điểm tuyệt đối — không phá hoà thì chunk được chọn phụ thuộc `candidate_chunks`.
        """
        anchor: dict[str, tuple[float, str]] = {}
        for i, s in zip(idx, scores):
            d = self.chunk_doc_ids[int(i)]
            key = (-float(s), self.chunk_ids[int(i)])
            if d not in anchor or key < anchor[d]:
                anchor[d] = key
        return {d: key[1] for d, key in anchor.items()}

    def search_with_anchor(self, queries: list[str], top_k: int) -> list[list[tuple[str, float, str]]]:
        """Như `search()` nhưng kèm chunk đại diện của mỗi doc (INTERFACES.md §3b).

        Reranker cần một đoạn văn bản cụ thể để chấm; `search()` gộp lên doc rồi bỏ mất
        chunk nào đã thắng.

        Args:
            queries: Câu hỏi.
            top_k: Số doc mỗi query.

        Returns:
            Mỗi query một danh sách `(doc_id, score, chunk_id)`.
        """
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        n_cand = max(self.candidate_chunks, top_k)
        out: list[list[tuple[str, float, str]]] = []
        for idx, sc in self.candidates(list(queries), n_cand):
            ranked = self.pool_candidates(idx, sc, top_k)
            anchor = self._anchor_chunks(idx, sc)
            out.append([(d, s, anchor[d]) for d, s in ranked])
        check_contract([[(d, s) for d, s, _ in r] for r in out], len(queries), top_k)
        return out

    def search_chunks(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        """Như `search()` nhưng trả `(chunk_id, score)`, chưa gộp lên doc."""
        return [
            [(self.chunk_ids[int(i)], float(s)) for i, s in zip(idx, sc)]
            for idx, sc in self.candidates(list(queries), top_k)
        ]

    def stats(self) -> dict:
        """Số liệu ghi vào log / `run_meta.json`. Subclass mở rộng thêm."""
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
    """Decorator đăng ký một lớp retriever dưới tên `name`.

    `python -m src.retrieval.bm25` nạp module hai lần (qua package và dưới tên `__main__`)
    nên cùng một lớp có thể đăng ký hai lần — đó không phải xung đột.

    Raises:
        KeyError: Tên đã thuộc về một lớp KHÁC.
    """

    def deco(cls: type[BaseRetriever]) -> type[BaseRetriever]:
        cu = _REGISTRY.get(name)
        if cu is not None and cu.__qualname__ != cls.__qualname__:
            raise KeyError(f"Retriever '{name}' đã đăng ký bởi {cu.__qualname__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_retrievers() -> list[str]:
    """Tên các retriever đã đăng ký."""
    return sorted(_REGISTRY)


def infer_retriever_kind(cfg: dict) -> str:
    """Suy ra loại retriever khi `retrieval` chỉ có đúng một khối.

    Raises:
        ValueError: Có 0 hoặc ≥2 khối — buộc khai `pipeline.retriever` cho rõ.
    """
    kinds = [k for k in cfg.get("retrieval", {}) if isinstance(cfg["retrieval"][k], dict)]
    if len(kinds) != 1:
        raise ValueError(
            f"`retrieval` có {len(kinds)} khối ({kinds}). Khai rõ `pipeline.retriever: <tên>` "
            f"để không ai phải đoán bản chạy dùng cái nào."
        )
    return kinds[0]


def retriever_spec(cfg: dict, kind: str | None = None, demo: bool = False) -> dict:
    """Config đã đọc → spec truyền thẳng vào `build_retriever()`.

    Làm ba việc mọi script đều cần: chọn khối retriever (`pipeline.retriever` hoặc suy ra);
    nối `paths.embeddings` / `paths.cache_dir` vào nguồn cần chúng, kể cả nguồn nằm trong
    `hybrid.sources`; và ở `--demo` thì gỡ cache/embedding (corpus giả mà nạp embedding thật
    là lỗi vân tay, hoặc tệ hơn, không lỗi).

    Args:
        cfg: Config đã đọc bằng `load_config`.
        kind: Ép loại retriever; None = theo config.
        demo: Chạy trên corpus giả.

    Returns:
        Spec có khoá `type`, đệ quy vào `sources`.
    """
    kind = kind or cfg.get("pipeline", {}).get("retriever") or infer_retriever_kind(cfg)
    return _resolve_spec(kind, cfg["retrieval"][kind], cfg.get("paths", {}) or {}, demo)


def _resolve_spec(kind: str, raw: dict, paths: dict, demo: bool) -> dict:
    """Đệ quy của `retriever_spec`: gắn `type` và đường dẫn cho một khối."""
    spec = dict(raw)
    spec["type"] = kind
    if "sources" in spec:
        # `weight` giữ nguyên trong spec con; `HybridRetriever` lấy nó ra.
        spec["sources"] = {
            name: _resolve_spec(sub.get("type", name), sub, paths, demo)
            for name, sub in spec["sources"].items()
        }
        return spec
    if demo:
        spec.pop("cache_dir", None)
        spec.pop("embeddings_path", None)
    elif kind == "dense" and paths.get("embeddings"):
        spec.setdefault("embeddings_path", paths["embeddings"])
    elif kind == "bm25" and paths.get("cache_dir"):
        spec.setdefault("cache_dir", paths["cache_dir"])
    return spec


def build_retriever(spec: dict) -> BaseRetriever:
    """Dựng retriever từ spec (`{"type": "bm25", "k1": 1.5, ...}`).

    Lớp chưa đăng ký thì nạp lười `src/retrieval/<type>.py` — dense kéo theo torch nên
    không import sẵn.

    Raises:
        ValueError: Thiếu khoá `type`.
        KeyError: `type` không ứng với retriever nào.
    """
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError("Thiếu khoá 'type' trong config retriever")
    if kind not in _REGISTRY:
        try:
            importlib.import_module(f"{__package__}.{kind}")
        except ImportError:
            pass
    if kind not in _REGISTRY:
        raise KeyError(f"Retriever '{kind}' chưa đăng ký. Có: {', '.join(available_retrievers())}")
    return _REGISTRY[kind].from_spec(spec)


__all__ = [
    "BaseRetriever",
    "PoolingConfig",
    "build_retriever",
    "check_contract",
    "pool_scores",
    "register_retriever",
    "reject_unknown",
    "retriever_spec",
    "split_spec",
    "top_n_desc",
]
