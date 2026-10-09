"""
Hợp nhất nhiều retriever tầng 1 bằng RRF — CHỦ SỞ HỮU: P3. Khớp `BaseRetriever` (INTERFACES §3).

Theo §3, gộp chunk→doc **và** hợp nhất nhiều nguồn đều là việc NỘI BỘ của retriever: người gọi
chỉ thấy `search()`. Nhờ vậy `scripts/run_pipeline.py` không cần biết gì về fusion — nó chỉ đổi
`pipeline.retriever: hybrid` trong YAML.

HỢP NHẤT THỨ HẠNG, KHÔNG CỘNG ĐIỂM (giả thuyết H4, docs/pipeline_e2e_plan.md vòng 3). BM25 cho
điểm dương không chặn trên, cosine ∈ [−1,1], logit cross-encoder có dấu — cộng thẳng là để thang
điểm lớn nhất nuốt các thang còn lại. RRF chỉ dùng thứ hạng nên miễn nhiễm:

    score(x) = Σ_nguồn  w_nguồn / (k + rank_nguồn(x))

HAI MỨC HỢP NHẤT, đo được, không đoán (đây là phần prototype `scripts/p4_fuse.py` của P4 chưa
làm — nó chỉ hợp nhất ở mức doc):

  fuse_level: chunk   RRF trên thứ hạng CHUNK, rồi mới gộp chunk→doc bằng `pool` của hybrid.
                      Một văn bản có nhiều chunk được cả hai nguồn xếp cao sẽ cộng dồn bằng
                      chứng — đúng tinh thần `mean_topN`. Đổi lại, văn bản dài có nhiều cơ hội
                      lọt vào top chunk của cả hai nguồn hơn văn bản ngắn.
  fuse_level: doc     Mỗi nguồn tự gộp chunk→doc bằng `pool` CỦA CHÍNH NÓ (đúng §3: mỗi thang
                      điểm có cách gộp tối ưu riêng), rồi RRF trên thứ hạng DOC. Đây là cách
                      `p4_fuse.py` làm, tái lập được để so sánh.

Cả hai mức đều đi qua đúng một đường của khung (`_score_chunks` → `pool_candidates` →
`check_contract`), nên so sánh chúng là so hai chiến lược chứ không phải so hai đoạn code.

    retrieval:
      hybrid:
        fuse_level: chunk
        rrf_k: 60
        pool: mean_top2
        sources:
          bm25:  {type: bm25,  weight: 0.6, tokenizer: syllable_bigram, ...}
          dense: {type: dense, weight: 0.4, repo: ..., revision: ...}
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.retrieval.base import BaseRetriever, build_retriever, register_retriever

FUSE_LEVELS = ("chunk", "doc")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Chuẩn hoá tổng trọng số về 1. Không đổi thứ hạng (co giãn đều là phép biến đổi đơn điệu),
    chỉ để `w=0,6/0,4` trong YAML và điểm in ra log đọc được như nhau ở mọi cấu hình.
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
    """
    RRF có trọng số trên nhiều danh sách đã xếp hạng. `ranked[nguồn]` = phần tử tốt nhất trước.

    `absent_rank`:
      - `None` (mặc định) — phần tử VẮNG MẶT ở một nguồn thì nguồn đó KHÔNG đóng góp gì.
        Đây là RRF cổ điển (Cormack 2009).
      - số nguyên — coi như phần tử đứng ở hạng đó. `scripts/p4_fuse.py` dùng `top_k+1000`,
        `scripts/p5_knn_fuse.py` dùng `10^6`. Giữ tuỳ chọn này để tái lập ĐÚNG hai prototype đó.

        ⚠️ Lưu ý dấu: `absent_rank` cộng một lượng DƯƠNG cho phần tử vắng mặt, tức thưởng nhẹ
        cho việc vắng mặt. Với k=60 và absent_rank=10^6 thì lượng đó ≈ 1e-6·w, còn hạng 1 cho
        ≈ 0,016·w — nên nó chỉ lật được những trường hợp HOÀ TUYỆT ĐỐI. Không phải lỗi chết
        người, nhưng nó có nghĩa là `absent_rank=None` và `absent_rank=10^6` có thể ra thứ tự
        khác nhau ở vài câu. Khai tường minh còn hơn để hai bản "gần giống nhau".
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


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """
    Hợp nhất ≥2 retriever con. Bản thân nó cũng là một `BaseRetriever`, nên lồng được
    (hybrid của hybrid) và cắm thẳng vào `build_retriever` / `run_pipeline.py`.
    """

    def __init__(
        self,
        sources: dict[str, dict],
        *,
        fuse_level: str = "chunk",
        rrf_k: int = 60,
        absent_rank: int | None = None,
        doc_depth: int = 200,
        source_depth: int | None = None,
        pool: str = "max",
        pool_tau: float = 1.0,
        candidate_chunks: int = 2000,
        verbose: bool = True,
    ) -> None:
        super().__init__(pool=pool, pool_tau=pool_tau, candidate_chunks=candidate_chunks)
        if fuse_level not in FUSE_LEVELS:
            raise ValueError(f"fuse_level '{fuse_level}' không có. Dùng: {', '.join(FUSE_LEVELS)}")
        if not isinstance(sources, dict) or len(sources) < 2:
            raise ValueError(
                f"hybrid cần ÍT NHẤT 2 nguồn, nhận {len(sources) if sources else 0}. "
                f"Hợp nhất một nguồn với chính nó chỉ đổi thang điểm chứ không đổi thứ hạng — "
                f"nếu chỉ muốn một retriever thì khai thẳng nó, đừng bọc thêm một lớp."
            )
        if rrf_k < 1:
            raise ValueError("rrf_k phải >= 1 (k=0 làm hạng 1 trội tuyệt đối, RRF hết tác dụng)")

        # Ở mức doc, việc gộp chunk→doc đã xảy ra BÊN TRONG từng nguồn rồi; hybrid chỉ còn nhận
        # đúng một chunk đại diện cho mỗi doc, nên `pool` của hybrid là phép đồng nhất với MỌI
        # chiến lược. Config ghi `pool: mean_top2` ở đây sẽ không làm gì cả, mà lại đọc như
        # đang làm gì đó — đúng loại nhầm lẫn im lặng mà repo này chặn bằng lỗi.
        if fuse_level == "doc" and pool != "max":
            raise ValueError(
                f"fuse_level='doc' thì `pool` của hybrid phải là 'max' (nhận '{pool}'). "
                f"Ở mức doc, mỗi nguồn ĐÃ tự gộp chunk→doc bằng pool của riêng nó — khai pool "
                f"thứ hai ở tầng hybrid là một phép đồng nhất đội lốt tham số. Muốn đổi cách "
                f"gộp thì đổi `pool` TRONG từng nguồn."
            )

        self.fuse_level = fuse_level
        self.rrf_k = int(rrf_k)
        self.absent_rank = absent_rank
        self.doc_depth = int(doc_depth)
        self.source_depth = source_depth
        self.verbose = verbose

        self.sources: dict[str, BaseRetriever] = {}
        raw_weights: dict[str, float] = {}
        for name, spec in sources.items():
            spec = dict(spec)
            w = float(spec.pop("weight", 1.0))
            if w < 0:
                raise ValueError(f"nguồn '{name}': weight={w} âm. RRF không định nghĩa với trọng số âm.")
            raw_weights[name] = w
            self.sources[name] = build_retriever(spec)
        self.weights = normalize_weights(raw_weights)

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        """
        Index chính mình rồi index từng nguồn TRÊN CÙNG danh sách chunk.

        Kiểm `chunk_ids` khớp tuyệt đối là bắt buộc, không phải phòng xa: RRF ở mức chunk cộng
        điểm theo CHỈ SỐ hàng. Hai nguồn lệch một chunk thì chỉ số i của nguồn A trỏ vào chunk
        khác chỉ số i của nguồn B, và sai lệch đó không có biểu hiện nào ngoài recall tụt.
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
        """
        Ứng viên mức chunk của TỪNG nguồn, chưa hợp nhất.

        Công khai vì `scripts/tune_rrf.py` cần đúng thứ này: chấm điểm một lần rồi quét lưới
        (w, k) trong RAM. Nếu không có nó thì mỗi ô lưới phải index lại toàn corpus, và lưới
        11×3 sẽ tốn 33 lần encode 524.422 chunk cho một kết quả không đổi.
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
        """
        Hợp nhất ứng viên của MỘT câu hỏi → `(chỉ số chunk, điểm)` như `_score_chunks` yêu cầu.

        `weights`/`rrf_k`/`fuse_level` cho phép ghi đè để quét lưới mà không dựng lại retriever.
        """
        w = normalize_weights(weights) if weights else self.weights
        k = self.rrf_k if rrf_k is None else int(rrf_k)
        level = fuse_level or self.fuse_level
        if level == "chunk":
            return self._fuse_chunk_level(per_source, w, k)
        return self._fuse_doc_level(per_source, w, k)

    def _fuse_chunk_level(self, per_source, weights, rrf_k):
        """
        Vector hoá bằng numpy, KHÔNG phải tối ưu sớm: `scripts/tune_rrf.py` gọi hàm này một
        lần cho mỗi ô lưới × mỗi câu hỏi (lưới 11×3 trên 1.000 câu = 33.000 lần), mỗi lần trên
        tới 2×`candidate_chunks` chunk. Vòng lặp Python ở đây biến việc quét lưới thành việc
        qua đêm. Và quan trọng hơn: đây là ĐÚNG hàm mà production gọi, nên số đo lúc chỉnh
        tham số và số đo lúc nộp bài không thể lệch nhau vì hai đoạn code khác nhau.
        """
        parts_idx, parts_sc, present = [], [], []
        for name, (idx, _) in per_source.items():
            w = weights.get(name, 0.0)
            idx = np.asarray(idx, dtype=np.int64)
            if w == 0.0 or idx.size == 0:
                continue
            parts_idx.append(idx)
            # rank 1..n → w/(k+rank)
            parts_sc.append(w / (rrf_k + np.arange(1, idx.size + 1, dtype=np.float64)))
            present.append((w, idx))
        if not parts_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        uniq, inv = np.unique(np.concatenate(parts_idx), return_inverse=True)
        fused = np.bincount(inv, weights=np.concatenate(parts_sc), minlength=uniq.size)

        if self.absent_rank is not None:
            for w, idx in present:
                fused += np.where(np.isin(uniq, idx), 0.0, w / (rrf_k + self.absent_rank))

        # Phá hoà bằng CHỈ SỐ chunk (thứ tự trong chunks.jsonl). Trong cùng một văn bản, chỉ số
        # tăng theo `position` mà `chunk_id` cũng zero-pad theo `position` ⇒ trùng khít quy ước
        # "chunk_id nhỏ nhất" của `search_with_anchor()`. Giữa hai văn bản khác nhau thì thứ tự
        # hai bên khác nhau, nhưng ở đó ta chỉ cần một quy tắc XÁC ĐỊNH, không cần quy tắc nào
        # cụ thể — và so số nguyên thì `lexsort` làm được, so chuỗi thì không.
        order = np.lexsort((uniq, -fused))
        return uniq[order], fused[order]

    def _fuse_doc_level(self, per_source, weights, rrf_k):
        """
        Mỗi nguồn tự gộp lên doc rồi RRF trên thứ hạng doc, sau đó CHIẾU NGƯỢC về chunk:
        phát ra đúng MỘT chunk đại diện cho mỗi doc, mang điểm RRF của doc đó.

        Vì mỗi doc chỉ còn một chunk, `pool_scores` của khung là phép đồng nhất với mọi chiến
        lược (max/sum/mean_topN/logsumexp trên tập một phần tử đều trả về chính phần tử đó).
        Nhờ thế mức doc dùng lại nguyên `pool_candidates()` + `check_contract()` + phần tính
        chunk đại diện của `search_with_anchor()` — không có nhánh code riêng nào để lệch.
        """
        ranked_docs: dict[str, list[str]] = {}
        # (thứ hạng trong danh sách chunk của nguồn, chunk_id) → chunk đại diện của doc.
        # Dùng THỨ HẠNG chứ không dùng điểm vì điểm của hai nguồn không so được với nhau —
        # cùng lý do khiến ta chọn RRF thay vì cộng điểm.
        anchor: dict[str, tuple[int, str, int]] = {}
        for name, (idx, sc) in per_source.items():
            if weights.get(name, 0.0) == 0.0:
                continue
            ranked_docs[name] = [
                d for d, _ in self.sources[name].pool_candidates(idx, sc, self.doc_depth)
            ]
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

    def _score_chunks(
        self, queries: list[str], n_candidates: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        per_source = self.source_candidates(queries, self.source_depth or n_candidates)
        out = []
        for qi in range(len(queries)):
            one = {name: per_source[name][qi] for name in self.sources}
            idx, sc = self.fuse_query(one)
            out.append((idx[:n_candidates], sc[:n_candidates]))
        return out

    # ── log ──────────────────────────────────────────────────────────────────
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
    """
    Một lát cắt 1 chiều qua đơn hình trọng số: nguồn ĐẦU nhận `w_first`, phần còn lại chia
    `1 - w_first` theo tỉ lệ trọng số gốc của chúng.

    Với đúng 2 nguồn đây chính là `w` và `1-w` như prototype của P4. Với ≥3 nguồn (BM25 +
    reranker + kNN) nó là lát cắt giữ nguyên tỉ lệ tương đối của các nguồn phụ — quét cả
    đơn hình là bài toán khác, và quét nó trên tập fit 4.689 câu thì overfit chính tập fit.
    """
    names = list(names)
    first, rest = names[0], names[1:]
    out = {first: float(w_first)}
    rest_total = sum(base.get(n, 0.0) for n in rest)
    for n in rest:
        share = (base.get(n, 0.0) / rest_total) if rest_total > 0 else 1.0 / max(len(rest), 1)
        out[n] = (1.0 - float(w_first)) * share
    return out
