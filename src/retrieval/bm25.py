"""
BM25 Okapi trên chunk — CHỦ SỞ HỮU: P3. Khớp `BaseRetriever` (INTERFACES.md mục 3).

scipy sparse thay vì rank_bm25 (vòng lặp Python, giờ thay vì phút trên corpus thật).
v0.2 thêm tokenizer cắm rời, gộp chunk→doc do khung lo, chấm theo lô + argpartition —
đều là trục ablation, mặc định vẫn ra kết quả v0.1 (docs/p3_retrieval.md).

Tái lập v0.1 chính xác: `tokenizer: regex`, `tokenizer_opts: {fold_tone: false}`,
`pool: max`, `candidate_chunks: null`.
"""
from __future__ import annotations

import argparse
import json
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from src.common.io import load_chunks, load_questions, write_predictions
from src.retrieval.base import BaseRetriever, register_retriever
from src.retrieval.tokenizers import Tokenizer, get_tokenizer, tokenize_many


@register_retriever("bm25")
class BM25Retriever(BaseRetriever):
    """
    BM25 Okapi. Điểm chunk cho query q:
        Σ_{t∈q} tf_q(t)·idf(t)·tf_d(t)·(k1+1) / (tf_d(t) + k1·(1-b+b·|d|/avgdl))
    Phần phụ thuộc văn bản nhúng sẵn vào ma trận lúc index → truy vấn chỉ còn nhân ma trận thưa.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        pool: str = "max",
        pool_tau: float = 1.0,
        candidate_chunks: int | None = 2000,
        tokenizer: str | Tokenizer = "regex",
        tokenizer_opts: dict | None = None,
        min_df: int = 1,
        n_jobs: int = 1,
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        # candidate_chunks: null → không cắt ứng viên, đúng hành vi v0.1 (mốc kiểm chứng).
        super().__init__(
            pool=pool,
            pool_tau=pool_tau,
            candidate_chunks=candidate_chunks if candidate_chunks else 10**9,
        )
        self.k1, self.b = k1, b
        self.min_df = min_df
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.verbose = verbose
        self.tokenizer = get_tokenizer(tokenizer, **(tokenizer_opts or {}))

        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.avgdl: float = 0.0
        self._index_seconds: float = 0.0

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        t0 = time.perf_counter()
        self._register_chunks(chunks)
        tokenized = tokenize_many(
            [c["text"] for c in chunks],
            self.tokenizer,
            n_jobs=self.n_jobs,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )

        # df trước, vocab sau: min_df cắt term hiếm (lỗi OCR, số hiệu lạ), vocab nhẹ hơn.
        df_counter: Counter[str] = Counter()
        for toks in tokenized:
            df_counter.update(set(toks))
        # Đánh số theo thứ tự term ĐƯỢC GIỮ, không theo enumerate — nếu không chỉ số sẽ
        # nhảy cóc qua term bị loại và vượt kích thước ma trận.
        self.vocab = {}
        for term, df in df_counter.items():
            if df >= self.min_df:
                self.vocab[term] = len(self.vocab)
        if not self.vocab:
            raise ValueError(
                f"Từ vựng rỗng sau khi lọc min_df={self.min_df} trên {len(chunks)} chunk."
            )

        # array typed thay vì list Python: nnz thật ~90-100 triệu, list[int] ≈ 8 GB,
        # array('i'/'f') ≈ 1,1 GB.
        rows = array("i")
        cols = array("i")
        vals = array("f")
        doc_len = np.zeros(len(tokenized), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(tf)

        n_chunks, n_terms = len(tokenized), len(self.vocab)
        del tokenized  # ~1 GB con trỏ token trên corpus thật — thả trước khi dựng ma trận

        rows_np = np.frombuffer(rows, dtype=np.int32)
        cols_np = np.frombuffer(cols, dtype=np.int32)
        tf_np = np.frombuffer(vals, dtype=np.float32)

        # df đếm thẳng trên COO (mỗi cặp chunk-term chỉ xuất hiện một lần, từ Counter):
        # rẻ hơn coo→csr→sum→coo→csr, tránh dựng thêm hai bản sao ma trận (~1 GB mỗi bản).
        df = np.bincount(cols_np, minlength=n_terms)
        self.idf = np.log(1 + (n_chunks - df + 0.5) / (df + 0.5)).astype(np.float32)

        # |d| = 0 (chunk rỗng) vẫn phải chia được: avgdl luôn > 0 vì vocab không rỗng.
        self.avgdl = float(doc_len.mean()) or 1.0
        norm = (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)).astype(np.float32)
        # Nhúng sẵn phần chuẩn hoá BM25 vào ma trận → lúc query chỉ còn một phép nhân
        weighted = (tf_np * (self.k1 + 1)) / (tf_np + norm[rows_np])
        self.matrix = sparse.csr_matrix(
            (weighted.astype(np.float32), (rows_np, cols_np)), shape=(n_chunks, n_terms)
        )
        self._index_seconds = time.perf_counter() - t0
        if self.verbose:
            print(
                f"  [bm25] {n_chunks} chunk / {self.n_docs} văn bản, "
                f"từ vựng {n_terms} term, avgdl {self.avgdl:.0f} token, "
                f"nnz {self.matrix.nnz} ({self._index_seconds:.1f}s)"
            )

    # ── truy vấn ─────────────────────────────────────────────────────────────
    def _query_matrix(self, queries: list[str]) -> sparse.csr_matrix:
        """(n_query × n_term), giá trị = tf_query(t) · idf(t). Term lạ bị bỏ qua."""
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

    def _score_chunks(
        self, queries: list[str], n_candidates: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        assert self.matrix is not None, "Gọi .index() trước"
        Q = self._query_matrix(queries)
        out: list[tuple[np.ndarray, np.ndarray]] = []

        # Theo lô: 1 phép nhân thưa cho 64 query rẻ hơn 64 lần riêng lẻ, không dựng ma
        # trận đặc (n_chunk × n_query) cho cả 1.000 câu.
        for start in range(0, Q.shape[0], self.batch_size):
            block = Q[start : start + self.batch_size]
            scores = (self.matrix @ block.T).toarray()  # (n_chunks, batch)
            for col in range(scores.shape[1]):
                s = scores[:, col]
                hit = np.flatnonzero(s > 0)
                if hit.size == 0:
                    out.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
                    continue
                if hit.size > n_candidates:
                    # argpartition: chỉ cần biết top-n là ai, không cần sắp xếp phần còn lại
                    keep = np.argpartition(-s[hit], n_candidates - 1)[:n_candidates]
                    hit = hit[keep]
                order = np.argsort(-s[hit], kind="stable")
                idx = hit[order]
                out.append((idx.astype(np.int64), s[idx].astype(np.float32)))
        return out

    # ── chẩn đoán ────────────────────────────────────────────────────────────
    def explain_query(self, query: str, top_n: int = 10) -> dict:
        """Term nào của câu hỏi khớp/OOV — công cụ đọc lỗi cho P4 (plan.md mục 4)."""
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
    """Dựng retriever từ YAML, không hằng số hard-code (INTERFACES.md mục 6)."""
    spec = dict(cfg["retrieval"].get("bm25", {}))
    if "candidate_chunks" in cfg["retrieval"]:
        spec.setdefault("candidate_chunks", cfg["retrieval"]["candidate_chunks"])
    cache_dir = cfg.get("paths", {}).get("cache_dir")
    if cache_dir:
        spec.setdefault("cache_dir", cache_dir)
    return BM25Retriever(**spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="BM25 trên chunk → top-k doc_id mỗi câu hỏi.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True, help="holdout.json hoặc public-official.json")
    ap.add_argument("--out", required=True, help="predictions.json")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument(
        "--eval",
        action="store_true",
        help="File câu hỏi có nhãn (holdout) → in luôn Recall@k, KPI của P3",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    top_k_submit = args.top_k or cfg["retrieval"]["top_k_submit"]
    # ranking_full_k: độ sâu dump cho P4 (yêu cầu team: tách trần L1-RETRIEVE khỏi dư địa
    # L1-RERANK). Không có key riêng thì dùng top_k_retrieve — vẫn không hard-code gì mới.
    top_k_full = max(top_k_submit, cfg["retrieval"].get("ranking_full_k", cfg["retrieval"]["top_k_retrieve"]))

    chunks = load_chunks(cfg["paths"]["chunks"])
    print(f"Đánh chỉ mục {len(chunks)} chunk…")
    r = retriever_from_config(cfg)
    r.index(chunks)

    qids, texts = load_questions(args.questions)
    print(f"Truy vấn {len(texts)} câu hỏi (top_k={top_k_full}, pool={r.pool})…")
    t0 = time.perf_counter()
    ranked = r.search(texts, top_k_full)
    print(f"  {time.perf_counter() - t0:.1f}s")

    # Dump TRƯỚC khi cắt top-5, cùng một lần search — P4 cần bảng đầy đủ để tách
    # L1-RETRIEVE (trần cứng, việc P3) khỏi L1-RERANK (dư địa, việc P4).
    out_dir = Path(cfg["paths"].get("out_dir", f"outputs/{cfg.get('exp_id', 'bm25')}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "ranking_full.json"
    ranking_full = {q: [[d, round(s, 4)] for d, s in res] for q, res in zip(qids, ranked)}
    ranking_path.write_text(json.dumps(ranking_full, ensure_ascii=False, indent=1), encoding="utf-8")

    preds_full = {q: [d for d, _ in res] for q, res in zip(qids, ranked)}
    preds = {q: v[:top_k_submit] for q, v in preds_full.items()}
    out = write_predictions(preds, args.out)

    n_empty = sum(1 for v in preds_full.values() if not v)
    print(f"\n✅ {out}")
    print(f"✅ {ranking_path}  (top-{top_k_full}, cho P4)")
    print(f"   {json.dumps(r.stats(), ensure_ascii=False)}")
    if n_empty:
        print(f"   ⚠️  {n_empty} câu không truy hồi được gì (0 term khớp từ vựng)")

    if args.eval:
        from src.evaluate import eval_official, load_truth, recall_at_k

        truth = load_truth(args.questions)
        # Luôn có ít nhất một mốc: nếu top_k_full nhỏ hơn 5 thì in đúng Recall@top_k_full,
        # đừng để --eval chạy xong mà không in ra con số nào.
        ks = sorted({k for k in (5, 20, 50, 100) if k <= top_k_full} | {top_k_full})
        for k in ks:
            print(f"   Recall@{k:<3}: {recall_at_k(preds_full, truth, k):.4f}")
        if top_k_submit >= 5:
            top5 = {q: v[:5] for q, v in preds_full.items()}
            s = eval_official(top5, truth)
            print(f"   Chấm như BTC (top-5): recall={s['recall']:.4f} precision={s['precision']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
