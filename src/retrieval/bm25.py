"""BM25 Okapi trên chunk, khớp `BaseRetriever` (INTERFACES.md §3).

Dùng ma trận thưa scipy thay vì `rank_bm25` (vòng lặp Python: giờ thay vì phút trên corpus
thật). Phần phụ thuộc văn bản của BM25 nhúng sẵn vào ma trận lúc index, nên truy vấn chỉ còn
một phép nhân ma trận thưa theo lô.

Tái lập v0.1 chính xác: `tokenizer: regex`, `tokenizer_opts: {fold_tone: false}`,
`pool: max`, `candidate_chunks: null` (configs/v0.1_bm25.yaml).

Typical usage example:

    python -m src.retrieval.bm25 --config configs/v0.3_bm25_best.yaml \\
        --questions data/dev.json --out outputs/tmp/preds.json --eval
"""
from __future__ import annotations

import argparse
import json
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from src.common.config import load_config, resolve_out_dir
from src.common.io import load_chunks, load_questions, write_predictions
from src.retrieval.base import (
    BaseRetriever,
    PoolingConfig,
    register_retriever,
    reject_unknown,
    split_spec,
    top_n_desc,
)
from src.retrieval.tokenizers import Tokenizer, get_tokenizer, tokenize_many

NO_CAP = 10**9  # `candidate_chunks: null` = không cắt ứng viên


@dataclass(frozen=True)
class BM25Config:
    """Siêu tham số BM25 và cách tách từ.

    Attributes:
        k1: Độ bão hoà tần suất term.
        b: Mức chuẩn hoá theo độ dài chunk.
        tokenizer: Tên tokenizer (`regex`, `whitespace`, `syllable_bigram`, `pyvi`,
            `underthesea`) hoặc một `Tokenizer` dựng sẵn.
        tokenizer_opts: Tuỳ chọn truyền vào `get_tokenizer` (vd `fold_tone`, `min_len`).
        min_df: Bỏ term xuất hiện ở ít hơn `min_df` chunk.
        n_jobs: Số tiến trình tách từ (chỉ đáng kể với pyvi/underthesea).
        batch_size: Số query mỗi phép nhân ma trận thưa.
        cache_dir: Thư mục cache token đã tách; None = không cache.
    """

    k1: float = 1.5
    b: float = 0.75
    tokenizer: str | Tokenizer = "regex"
    tokenizer_opts: dict = field(default_factory=dict)
    min_df: int = 1
    n_jobs: int = 1
    batch_size: int = 64
    cache_dir: str | Path | None = None


@register_retriever("bm25")
class BM25Retriever(BaseRetriever):
    """BM25 Okapi. Điểm chunk cho query q:

        Σ_{t∈q} tf_q(t)·idf(t)·tf_d(t)·(k1+1) / (tf_d(t) + k1·(1−b+b·|d|/avgdl))

    Attributes:
        config: Siêu tham số BM25.
        tokenizer: Tokenizer đã dựng từ `config`.
        vocab: `{term: chỉ số cột}` sau khi lọc `min_df`.
        matrix: Ma trận (n_chunk × n_term) đã nhúng phần chuẩn hoá BM25.
        idf: IDF theo cột.
        avgdl: Độ dài chunk trung bình (token).
    """

    def __init__(
        self,
        config: BM25Config | None = None,
        pooling: PoolingConfig | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        super().__init__(pooling)
        self.config = config or BM25Config()
        self.verbose = verbose
        self.tokenizer = get_tokenizer(self.config.tokenizer, **(self.config.tokenizer_opts or {}))
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.avgdl: float = 0.0
        self._index_seconds: float = 0.0

    @classmethod
    def from_spec(cls, spec: dict) -> "BM25Retriever":
        """Dựng từ spec phẳng của YAML (`retrieval.bm25`).

        `candidate_chunks: null` (hoặc 0) nghĩa là không cắt ứng viên.

        Raises:
            TypeError: Spec có khoá không thuộc `BM25Config`/`PoolingConfig`/`verbose`.
        """
        (bm25_kw, pool_kw), rest = split_spec(spec, BM25Config, PoolingConfig)
        reject_unknown(cls.__name__, rest, ("verbose",))
        if "candidate_chunks" in pool_kw:
            pool_kw["candidate_chunks"] = pool_kw["candidate_chunks"] or NO_CAP
        return cls(BM25Config(**bm25_kw), PoolingConfig(**pool_kw), verbose=rest.get("verbose", True))

    # ── thuộc tính rút gọn ───────────────────────────────────────────────────
    @property
    def k1(self) -> float:
        """Tham số `k1` của BM25."""
        return self.config.k1

    @property
    def b(self) -> float:
        """Tham số `b` của BM25."""
        return self.config.b

    @property
    def min_df(self) -> int:
        """Ngưỡng df tối thiểu của term."""
        return self.config.min_df

    @property
    def batch_size(self) -> int:
        """Số query mỗi phép nhân ma trận."""
        return self.config.batch_size

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        """Tách từ toàn kho, dựng từ vựng rồi dựng ma trận BM25.

        Args:
            chunks: Đọc từ `chunks.jsonl`.

        Raises:
            ValueError: Từ vựng rỗng sau khi lọc `min_df`.
        """
        t0 = time.perf_counter()
        self._register_chunks(chunks)
        tokenized = tokenize_many(
            [c["text"] for c in chunks],
            self.tokenizer,
            n_jobs=self.config.n_jobs,
            cache_dir=self.config.cache_dir,
            verbose=self.verbose,
        )
        self.vocab = self._build_vocab(tokenized)
        triplets, doc_len = self._count_terms(tokenized)
        del tokenized  # ~1 GB con trỏ token trên corpus thật — thả trước khi dựng ma trận
        self.matrix = self._bm25_matrix(triplets, doc_len)
        self._index_seconds = time.perf_counter() - t0
        if self.verbose:
            print(
                f"  [bm25] {self.matrix.shape[0]} chunk / {self.n_docs} văn bản, "
                f"từ vựng {self.matrix.shape[1]} term, avgdl {self.avgdl:.0f} token, "
                f"nnz {self.matrix.nnz} ({self._index_seconds:.1f}s)"
            )

    def _build_vocab(self, tokenized: list[list[str]]) -> dict[str, int]:
        """Đếm df rồi giữ term có df ≥ `min_df` (cắt lỗi OCR, số hiệu lạ).

        Đánh số theo thứ tự term ĐƯỢC GIỮ — đánh theo enumerate thì chỉ số nhảy cóc qua term
        bị loại và vượt kích thước ma trận.
        """
        df_counter: Counter[str] = Counter()
        for toks in tokenized:
            df_counter.update(set(toks))
        vocab: dict[str, int] = {}
        for term, df in df_counter.items():
            if df >= self.min_df:
                vocab[term] = len(vocab)
        if not vocab:
            raise ValueError(f"Từ vựng rỗng sau khi lọc min_df={self.min_df} trên {len(tokenized)} chunk.")
        return vocab

    def _count_terms(self, tokenized: list[list[str]]) -> tuple[tuple[array, array, array], np.ndarray]:
        """Đếm tf theo cặp (chunk, term) ở dạng COO và độ dài từng chunk.

        Dùng `array` có kiểu thay vì list Python: nnz thật ~96 triệu, list[int] ≈ 8 GB còn
        array('i'/'f') ≈ 1,1 GB.

        Returns:
            `((rows, cols, tf), doc_len)`.
        """
        rows, cols, vals = array("i"), array("i"), array("f")
        doc_len = np.zeros(len(tokenized), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(tf)
        return (rows, cols, vals), doc_len

    def _bm25_matrix(self, triplets: tuple[array, array, array], doc_len: np.ndarray) -> sparse.csr_matrix:
        """Dựng ma trận BM25 đã nhúng sẵn chuẩn hoá độ dài; đặt luôn `idf` và `avgdl`.

        df đếm thẳng trên COO (mỗi cặp chunk-term xuất hiện đúng một lần), tránh dựng thêm hai
        bản sao ma trận chỉ để cộng theo cột.
        """
        rows_np = np.frombuffer(triplets[0], dtype=np.int32)
        cols_np = np.frombuffer(triplets[1], dtype=np.int32)
        tf_np = np.frombuffer(triplets[2], dtype=np.float32)
        n_chunks, n_terms = len(doc_len), len(self.vocab)
        df = np.bincount(cols_np, minlength=n_terms)
        self.idf = np.log(1 + (n_chunks - df + 0.5) / (df + 0.5)).astype(np.float32)

        # |d| = 0 (chunk rỗng) vẫn chia được: avgdl > 0 vì vocab không rỗng.
        self.avgdl = float(doc_len.mean()) or 1.0
        norm = (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)).astype(np.float32)
        weighted = (tf_np * (self.k1 + 1)) / (tf_np + norm[rows_np])
        return sparse.csr_matrix(
            (weighted.astype(np.float32), (rows_np, cols_np)), shape=(n_chunks, n_terms)
        )

    # ── truy vấn ─────────────────────────────────────────────────────────────
    def _query_matrix(self, queries: list[str]) -> sparse.csr_matrix:
        """Ma trận (n_query × n_term), giá trị `tf_query(t)·idf(t)`. Term lạ bị bỏ qua."""
        assert self.idf is not None
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, q in enumerate(queries):
            ids = [self.vocab[t] for t in self.tokenizer(q) if t in self.vocab]
            for j, tf in Counter(ids).items():
                rows.append(i)
                cols.append(j)
                vals.append(tf * float(self.idf[j]))
        return sparse.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(queries), len(self.vocab)),
        ).tocsr()

    def _score_chunks(self, queries: list[str], n_candidates: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Nhân ma trận thưa theo lô `batch_size` query; chỉ chunk có điểm > 0 là ứng viên."""
        assert self.matrix is not None, "Gọi .index() trước"
        q_matrix = self._query_matrix(queries)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, q_matrix.shape[0], self.batch_size):
            block = q_matrix[start : start + self.batch_size]
            scores = (self.matrix @ block.T).toarray()  # (n_chunks, batch)
            for col in range(scores.shape[1]):
                s = scores[:, col]
                out.append(top_n_desc(s, n_candidates, np.flatnonzero(s > 0)))
        return out

    # ── chẩn đoán ────────────────────────────────────────────────────────────
    def explain_query(self, query: str, top_n: int = 10) -> dict:
        """Term nào của câu hỏi khớp/OOV — công cụ đọc lỗi.

        Args:
            query: Câu hỏi.
            top_n: Số term IDF cao nhất cần liệt kê.

        Returns:
            `{"tokenizer", "n_tokens", "oov", "top_terms"}`.
        """
        assert self.idf is not None, "Gọi .index() trước"
        toks = self.tokenizer(query)
        known = [(t, float(self.idf[self.vocab[t]])) for t in toks if t in self.vocab]
        return {
            "tokenizer": self.tokenizer.key,
            "n_tokens": len(toks),
            "oov": sorted({t for t in toks if t not in self.vocab}),
            "top_terms": sorted(known, key=lambda x: -x[1])[:top_n],
        }

    def stats(self) -> dict:
        s = super().stats()
        s.update(
            tokenizer=self.tokenizer.key,
            k1=self.k1,
            b=self.b,
            min_df=self.min_df,
            vocab=len(self.vocab),
            avgdl=round(self.avgdl, 1),
            nnz=int(self.matrix.nnz) if self.matrix is not None else 0,
            index_seconds=round(self._index_seconds, 1),
        )
        return s


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def retriever_from_config(cfg: dict) -> BM25Retriever:
    """Dựng BM25 từ config đã đọc; nối `paths.cache_dir` nếu có.

    Args:
        cfg: Config đọc bằng `load_config`, có khối `retrieval.bm25`.
    """
    spec = dict(cfg["retrieval"].get("bm25", {}))
    if "candidate_chunks" in cfg["retrieval"]:
        spec.setdefault("candidate_chunks", cfg["retrieval"]["candidate_chunks"])
    cache_dir = cfg.get("paths", {}).get("cache_dir")
    if cache_dir:
        spec.setdefault("cache_dir", cache_dir)
    return BM25Retriever.from_spec(spec)


def _print_eval(preds_full: dict[str, list[str]], questions: str, top_k_full: int) -> None:
    """In Recall@k và điểm BTC top-5 của CLI."""
    from src.evaluate import eval_official, load_truth, recall_at_k

    truth = load_truth(questions)
    for k in sorted({k for k in (5, 20, 50, 100) if k <= top_k_full} | {top_k_full}):
        print(f"   Recall@{k:<3}: {recall_at_k(preds_full, truth, k):.4f}")
    if top_k_full >= 5:
        s = eval_official({q: v[:5] for q, v in preds_full.items()}, truth)
        print(f"   Chấm như BTC (top-5): recall={s['recall']:.4f} precision={s['precision']:.4f}")


def main() -> int:
    """CLI: BM25 → `predictions.json` + `ranking_full.json`."""
    ap = argparse.ArgumentParser(description="BM25 trên chunk → top-k doc_id mỗi câu hỏi.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True, help="dev.json hoặc public-official.json")
    ap.add_argument("--out", required=True, help="predictions.json")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--eval", action="store_true", help="file câu hỏi có nhãn → in Recall@k")
    args = ap.parse_args()

    cfg = load_config(args.config)
    top_k_submit = args.top_k or cfg["retrieval"]["top_k_submit"]
    top_k_full = max(top_k_submit, cfg["retrieval"].get("ranking_full_k", cfg["retrieval"]["top_k_retrieve"]))

    chunks = load_chunks(cfg["paths"]["chunks"])
    r = retriever_from_config(cfg)
    r.index(chunks)
    qids, texts = load_questions(args.questions)
    t0 = time.perf_counter()
    ranked = r.search(texts, top_k_full)
    print(f"  {len(texts)} câu trong {time.perf_counter() - t0:.1f}s")

    out_dir = resolve_out_dir(cfg, fallback="bm25")
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "ranking_full.json"
    ranking_path.write_text(
        json.dumps({q: [[d, round(s, 4)] for d, s in res] for q, res in zip(qids, ranked)},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    preds_full = {q: [d for d, _ in res] for q, res in zip(qids, ranked)}
    out = write_predictions({q: v[:top_k_submit] for q, v in preds_full.items()}, args.out)
    print(f"\n✅ {out}\n✅ {ranking_path}  (top-{top_k_full})\n   {json.dumps(r.stats(), ensure_ascii=False)}")
    n_empty = sum(1 for v in preds_full.values() if not v)
    if n_empty:
        print(f"   ⚠️  {n_empty} câu không truy hồi được gì (0 term khớp từ vựng)")
    if args.eval:
        _print_eval(preds_full, args.questions, top_k_full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
