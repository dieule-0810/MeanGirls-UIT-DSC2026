"""Hợp nhất nhiều retriever tầng 1 bằng RRF, khớp `BaseRetriever` (INTERFACES.md §3). P3.

Theo §3, gộp chunk→doc **và** hợp nhất nhiều nguồn đều là việc NỘI BỘ của retriever: người gọi
chỉ thấy `search()`, và `scripts/run_pipeline.py` chỉ cần `pipeline.retriever: hybrid`.

HỢP NHẤT THỨ HẠNG, KHÔNG CỘNG ĐIỂM (giả thuyết H4). BM25 cho điểm dương không chặn trên,
cosine ∈ [−1, 1] — cộng thẳng là để thang lớn nhất nuốt các thang còn lại. RRF chỉ dùng thứ
hạng nên miễn nhiễm:

    score(x) = Σ_nguồn  w_nguồn / (k + rank_nguồn(x))

Hai mức hợp nhất, cùng đi qua một đường của khung (`_score_chunks` → `pool_candidates` →
`check_contract`) nên so sánh chúng là so hai chiến lược, không phải hai đoạn code:

* `fuse_level: chunk` — RRF trên thứ hạng CHUNK rồi mới gộp chunk→doc bằng `pool` của hybrid.
* `fuse_level: doc` — mỗi nguồn tự gộp chunk→doc bằng `pool` của chính nó, rồi RRF trên thứ
  hạng DOC. Đây là cấu hình nộp bài v0.8.

Typical usage example (YAML):

    retrieval:
      hybrid:
        fuse_level: doc
        rrf_k: 20
        sources:
          bm25:  {type: bm25,  weight: 0.4, tokenizer: syllable_bigram, ...}
          dense: {type: dense, weight: 0.6, repo: ..., revision: ...}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.retrieval.base import (
    BaseRetriever,
    PoolingConfig,
    build_retriever,
    register_retriever,
    reject_unknown,
    split_spec,
)

FUSE_LEVELS = ("chunk", "doc")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Chuẩn hoá tổng trọng số về 1.

    Không đổi thứ hạng (co giãn đều là biến đổi đơn điệu), chỉ để `0,6/0,4` trong YAML và điểm
    in ra log đọc được như nhau ở mọi cấu hình.

    Raises:
        ValueError: Tổng trọng số ≤ 0 — mọi câu sẽ rỗng, tức 0 điểm im lặng.
    """
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(
            f"Tổng trọng số = {total}. Ít nhất một nguồn phải có weight > 0, "
            f"nếu không hợp nhất trả về danh sách rỗng cho mọi câu — 0 điểm im lặng."
        )
    return {name: float(w) / total for name, w in weights.items()}


def rrf_from_ranks(
    ranked: dict[str, Sequence],
    weights: dict[str, float],
    rrf_k: int,
    absent_rank: int | None = None,
) -> dict:
    """RRF có trọng số trên nhiều danh sách đã xếp hạng.

    Args:
        ranked: `{nguồn: [phần tử tốt nhất trước, ...]}`.
        weights: Trọng số từng nguồn; nguồn trọng số 0 bị bỏ qua.
        rrf_k: Hằng số k của RRF.
        absent_rank: None = phần tử vắng mặt ở một nguồn thì nguồn đó không đóng góp gì (RRF cổ
            điển, Cormack 2009). Số nguyên = coi như phần tử đứng ở hạng đó — giữ để tái lập
            đúng `scripts/p4_fuse.py` (`top_k+1000`) và `scripts/p5_knn_fuse.py` (`10^6`).
            Lưu ý nó cộng một lượng DƯƠNG cho phần tử vắng mặt, nên chỉ lật được các ca hoà
            tuyệt đối — nhưng vẫn có thể đổi thứ tự ở vài câu.

    Returns:
        `{phần tử: điểm RRF}`.
    """
    out: dict = {}
    for name, items in ranked.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        for rank, key in enumerate(items, start=1):
            out[key] = out.get(key, 0.0) + w / (rrf_k + rank)

    if absent_rank is not None:
        for name, items in ranked.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
            present = set(items)
            bonus = w / (rrf_k + absent_rank)
            for key in out:
                if key not in present:
                    out[key] += bonus
    return out


@dataclass(frozen=True)
class FusionConfig:
    """Cách hợp nhất các nguồn.

    Attributes:
        fuse_level: `chunk` hoặc `doc`.
        rrf_k: Hằng số k của RRF, ≥ 1 (k=0 làm hạng 1 trội tuyệt đối).
        absent_rank: Xem `rrf_from_ranks`.
        doc_depth: Ở mức doc, số doc mỗi nguồn đưa vào RRF.
        source_depth: Số chunk ứng viên lấy từ mỗi nguồn; None = `candidate_chunks`.
    """

    fuse_level: str = "chunk"
    rrf_k: int = 60
    absent_rank: int | None = None
    doc_depth: int = 200
    source_depth: int | None = None

    def __post_init__(self) -> None:
        """Kiểm `fuse_level` và `rrf_k` ngay lúc dựng."""
        if self.fuse_level not in FUSE_LEVELS:
            raise ValueError(f"fuse_level '{self.fuse_level}' không có. Dùng: {', '.join(FUSE_LEVELS)}")
        if self.rrf_k < 1:
            raise ValueError("rrf_k phải >= 1 (k=0 làm hạng 1 trội tuyệt đối, RRF hết tác dụng)")


def _build_sources(sources: dict[str, dict]) -> tuple[dict[str, BaseRetriever], dict[str, float]]:
    """Dựng các retriever con và tách `weight` khỏi spec của chúng.

    Raises:
        ValueError: Ít hơn 2 nguồn, hoặc có trọng số âm.
    """
    if not isinstance(sources, dict) or len(sources) < 2:
        raise ValueError(
            f"hybrid cần ÍT NHẤT 2 nguồn, nhận {len(sources) if sources else 0}. "
            f"Hợp nhất một nguồn với chính nó chỉ đổi thang điểm chứ không đổi thứ hạng — "
            f"nếu chỉ muốn một retriever thì khai thẳng nó, đừng bọc thêm một lớp."
        )
    built: dict[str, BaseRetriever] = {}
    raw_weights: dict[str, float] = {}
    for name, spec in sources.items():
        spec = dict(spec)
        w = float(spec.pop("weight", 1.0))
        if w < 0:
            raise ValueError(f"nguồn '{name}': weight={w} âm. RRF không định nghĩa với trọng số âm.")
        raw_weights[name] = w
        built[name] = build_retriever(spec)
    return built, raw_weights


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """Hợp nhất ≥2 retriever con. Bản thân nó cũng là `BaseRetriever` nên lồng được.

    Attributes:
        fusion: Cấu hình hợp nhất.
        sources: `{tên: retriever con}` theo thứ tự khai trong YAML.
        weights: Trọng số đã chuẩn hoá về tổng 1.
    """

    def __init__(
        self,
        sources: dict[str, dict],
        fusion: FusionConfig | None = None,
        pooling: PoolingConfig | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        super().__init__(pooling)
        self.fusion = fusion or FusionConfig()
        # Ở mức doc, mỗi doc chỉ còn một chunk đại diện nên `pool` của hybrid là phép đồng nhất
        # với MỌI chiến lược. Cho khai giá trị khác là để một tham số đọc như đang làm gì đó.
        if self.fusion.fuse_level == "doc" and self.pool != "max":
            raise ValueError(
                f"fuse_level='doc' thì `pool` của hybrid phải là 'max' (nhận '{self.pool}'). "
                f"Ở mức doc, mỗi nguồn ĐÃ tự gộp chunk→doc bằng pool của riêng nó — khai pool "
                f"thứ hai ở tầng hybrid là một phép đồng nhất đội lốt tham số. Muốn đổi cách "
                f"gộp thì đổi `pool` TRONG từng nguồn."
            )
        self.verbose = verbose
        self.sources, raw_weights = _build_sources(sources)
        self.weights = normalize_weights(raw_weights)

    @classmethod
    def from_spec(cls, spec: dict) -> "HybridRetriever":
        """Dựng từ spec phẳng của YAML (`retrieval.hybrid`).

        Raises:
            TypeError: Spec có khoá không thuộc `FusionConfig`/`PoolingConfig`/`sources`/`verbose`.
        """
        (fuse_kw, pool_kw), rest = split_spec(spec, FusionConfig, PoolingConfig)
        reject_unknown(cls.__name__, rest, ("sources", "verbose"))
        return cls(
            rest.get("sources", {}),
            FusionConfig(**fuse_kw),
            PoolingConfig(**pool_kw),
            verbose=rest.get("verbose", True),
        )

    # ── thuộc tính rút gọn ───────────────────────────────────────────────────
    @property
    def fuse_level(self) -> str:
        """Mức hợp nhất (`chunk` | `doc`)."""
        return self.fusion.fuse_level

    @property
    def rrf_k(self) -> int:
        """Hằng số k của RRF."""
        return self.fusion.rrf_k

    @property
    def absent_rank(self) -> int | None:
        """Hạng gán cho phần tử vắng mặt ở một nguồn (None = RRF cổ điển)."""
        return self.fusion.absent_rank

    @property
    def doc_depth(self) -> int:
        """Số doc mỗi nguồn đưa vào RRF ở mức doc."""
        return self.fusion.doc_depth

    @property
    def source_depth(self) -> int | None:
        """Số chunk ứng viên lấy từ mỗi nguồn."""
        return self.fusion.source_depth

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        """Index chính mình rồi index từng nguồn TRÊN CÙNG danh sách chunk.

        RRF mức chunk cộng điểm theo CHỈ SỐ hàng: hai nguồn lệch một chunk thì chỉ số i của
        nguồn A trỏ vào chunk khác chỉ số i của nguồn B, và lỗi đó chỉ lộ ra qua recall tụt.

        Raises:
            ValueError: Một nguồn đăng ký danh sách chunk khác hybrid.
        """
        self._register_chunks(chunks)
        for name, r in self.sources.items():
            if self.verbose:
                print(f"  [hybrid] index nguồn '{name}' ({r.name})")
            r.index(chunks)
            if r.chunk_ids != self.chunk_ids:
                raise ValueError(
                    f"nguồn '{name}' đăng ký {len(r.chunk_ids)} chunk, hybrid có "
                    f"{len(self.chunk_ids)} — hai bên không nhìn cùng một kho chunk. "
                    f"Nhiều khả năng một nguồn lọc chunk trong `index()` của nó."
                )
        if self.verbose:
            ws = " · ".join(f"{n}={w:.3f}" for n, w in self.weights.items())
            print(
                f"  [hybrid] {len(self.sources)} nguồn · fuse_level={self.fuse_level} · "
                f"rrf_k={self.rrf_k} · {ws}"
            )

    # ── truy vấn ─────────────────────────────────────────────────────────────
    def source_candidates(
        self, queries: list[str], n_candidates: int | None = None
    ) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
        """Ứng viên mức chunk của TỪNG nguồn, chưa hợp nhất.

        Công khai cho `scripts/tune_rrf.py`: chấm điểm một lần rồi quét lưới (w, k) trong RAM,
        thay vì index lại toàn corpus ở mỗi ô lưới.

        Args:
            queries: Câu hỏi.
            n_candidates: Số chunk mỗi nguồn; None = `source_depth` hoặc `candidate_chunks`.
        """
        n = n_candidates or self.source_depth or self.candidate_chunks
        return {name: r.candidates(list(queries), n) for name, r in self.sources.items()}

    def fuse_query(
        self,
        per_source: dict[str, tuple[np.ndarray, np.ndarray]],
        weights: dict[str, float] | None = None,
        rrf_k: int | None = None,
        fuse_level: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hợp nhất ứng viên của MỘT câu hỏi.

        Args:
            per_source: `{nguồn: (chỉ số chunk, điểm)}`.
            weights: Ghi đè trọng số (quét lưới); None = của retriever.
            rrf_k: Ghi đè k; None = của retriever.
            fuse_level: Ghi đè mức hợp nhất; None = của retriever.

        Returns:
            `(chỉ số chunk, điểm)` như `_score_chunks` yêu cầu.
        """
        w = normalize_weights(weights) if weights else self.weights
        k = self.rrf_k if rrf_k is None else int(rrf_k)
        if (fuse_level or self.fuse_level) == "chunk":
            return self._fuse_chunk_level(per_source, w, k)
        return self._fuse_doc_level(per_source, w, k)

    def _fuse_chunk_level(self, per_source, weights, rrf_k) -> tuple[np.ndarray, np.ndarray]:
        """RRF trên thứ hạng chunk, vector hoá bằng numpy.

        Không phải tối ưu sớm: `tune_rrf.py` gọi hàm này một lần mỗi ô lưới × mỗi câu, và đây
        là ĐÚNG hàm production gọi — số đo lúc chỉnh tham số và lúc nộp không thể lệch nhau.
        Hoà điểm thì phá bằng chỉ số chunk (trong một văn bản trùng quy ước "chunk_id nhỏ
        nhất" của `search_with_anchor`).
        """
        parts_idx, parts_sc, present = [], [], []
        for name, (idx, _) in per_source.items():
            w = weights.get(name, 0.0)
            idx = np.asarray(idx, dtype=np.int64)
            if w == 0.0 or idx.size == 0:
                continue
            parts_idx.append(idx)
            parts_sc.append(w / (rrf_k + np.arange(1, idx.size + 1, dtype=np.float64)))
            present.append((w, idx))
        if not parts_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        uniq, inv = np.unique(np.concatenate(parts_idx), return_inverse=True)
        fused = np.bincount(inv, weights=np.concatenate(parts_sc), minlength=uniq.size)
        if self.absent_rank is not None:
            for w, idx in present:
                fused += np.where(np.isin(uniq, idx), 0.0, w / (rrf_k + self.absent_rank))
        order = np.lexsort((uniq, -fused))
        return uniq[order], fused[order]

    def _fuse_doc_level(self, per_source, weights, rrf_k) -> tuple[np.ndarray, np.ndarray]:
        """Mỗi nguồn tự gộp lên doc, RRF trên thứ hạng doc, rồi chiếu ngược về chunk.

        Phát ra đúng MỘT chunk đại diện cho mỗi doc, mang điểm RRF của doc đó — nên
        `pool_scores` của khung là phép đồng nhất và mức doc dùng lại nguyên
        `pool_candidates()` + `check_contract()`. Chunk đại diện chọn theo THỨ HẠNG trong nguồn
        (không theo điểm, vì điểm hai nguồn không so được).
        """
        ranked_docs: dict[str, list[str]] = {}
        anchor: dict[str, tuple[int, str, int]] = {}
        for name, (idx, sc) in per_source.items():
            if weights.get(name, 0.0) == 0.0:
                continue
            ranked_docs[name] = [d for d, _ in self.sources[name].pool_candidates(idx, sc, self.doc_depth)]
            for pos, i in enumerate(idx):
                i = int(i)
                d = self.chunk_doc_ids[i]
                key = (pos, self.chunk_ids[i], i)
                if d not in anchor or key < anchor[d]:
                    anchor[d] = key

        fused = rrf_from_ranks(ranked_docs, weights, rrf_k, self.absent_rank)
        if not fused:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        order = sorted(fused, key=lambda d: (-fused[d], d))
        return (
            np.asarray([anchor[d][2] for d in order], dtype=np.int64),
            np.asarray([fused[d] for d in order], dtype=np.float64),
        )

    def _score_chunks(self, queries: list[str], n_candidates: int) -> list[tuple[np.ndarray, np.ndarray]]:
        per_source = self.source_candidates(queries, self.source_depth or n_candidates)
        out = []
        for qi in range(len(queries)):
            idx, sc = self.fuse_query({name: per_source[name][qi] for name in self.sources})
            out.append((idx[:n_candidates], sc[:n_candidates]))
        return out

    def stats(self) -> dict:
        s = super().stats()
        s.update(
            fuse_level=self.fuse_level,
            rrf_k=self.rrf_k,
            absent_rank=self.absent_rank,
            doc_depth=self.doc_depth if self.fuse_level == "doc" else None,
            source_depth=self.source_depth,
            weights={n: round(w, 4) for n, w in self.weights.items()},
            sources={n: r.stats() for n, r in self.sources.items()},
        )
        return s


def sweep_weights(names: Iterable[str], w_first: float, base: dict[str, float]) -> dict[str, float]:
    """Lát cắt 1 chiều qua đơn hình trọng số.

    Nguồn ĐẦU nhận `w_first`, phần còn lại chia `1 − w_first` theo tỉ lệ trọng số gốc. Với 2
    nguồn đây chính là `w` / `1−w`; với ≥3 nguồn nó giữ nguyên tỉ lệ tương đối của các nguồn
    phụ (quét cả đơn hình trên 4.689 câu là overfit tập fit).

    Args:
        names: Tên nguồn theo thứ tự khai.
        w_first: Trọng số của nguồn đầu.
        base: Trọng số gốc (đã chuẩn hoá) để chia phần còn lại.

    Returns:
        `{nguồn: trọng số}`.
    """
    names = list(names)
    first, rest = names[0], names[1:]
    out = {first: float(w_first)}
    rest_total = sum(base.get(n, 0.0) for n in rest)
    for n in rest:
        share = (base.get(n, 0.0) / rest_total) if rest_total > 0 else 1.0 / max(len(rest), 1)
        out[n] = (1.0 - float(w_first)) * share
    return out
