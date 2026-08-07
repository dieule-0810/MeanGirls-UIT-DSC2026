"""
BM25 baseline — v0.1

CHỦ SỞ HỮU: P3.

Tự cài đặt bằng scipy sparse thay vì rank_bm25 vì:
  - rank_bm25 chạy vòng lặp Python: ~200k chunk × 1000 query = hàng giờ
  - bản sparse dưới đây: cùng khối lượng đó tính bằng phút, không thêm dependency nặng

Khớp giao diện BaseRetriever trong INTERFACES.md mục 3. P4 và P1 chỉ thấy .search(),
không thấy gì bên trong.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    v0.1 cố ý KHÔNG tách từ tiếng Việt (pyvi/underthesea) — giữ dependency tối thiểu
    và có một mốc tham chiếu sạch. P3 thêm word-segment ở v0.2 và đo mức cải thiện
    trên cùng held-out. Đó mới là ablation có ý nghĩa cho bài báo.
    """
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


class BM25Retriever:
    """BM25 Okapi trên chunk, gộp lên document bằng max-pooling."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, pool: str = "max"):
        self.k1, self.b, self.pool = k1, b, pool
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.chunk_doc_ids: list[str] = []

    def index(self, chunks: list[dict]) -> None:
        self.chunk_doc_ids = [c["doc_id"] for c in chunks]
        tokenized = [tokenize(c["text"]) for c in chunks]

        for toks in tokenized:
            for t in toks:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)

        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(tokenized), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                rows.append(i)
                cols.append(self.vocab[term])
                vals.append(tf)

        n_docs, n_terms = len(tokenized), len(self.vocab)
        tf_mat = sparse.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)), shape=(n_docs, n_terms)
        )

        df = np.asarray((tf_mat > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Nhúng sẵn phần chuẩn hoá BM25 vào ma trận → lúc query chỉ còn một phép nhân
        avgdl = doc_len.mean()
        norm = (self.k1 * (1 - self.b + self.b * doc_len / avgdl)).astype(np.float32)
        tf_mat = tf_mat.tocoo()
        weighted = (tf_mat.data * (self.k1 + 1)) / (tf_mat.data + norm[tf_mat.row])
        self.matrix = sparse.csr_matrix(
            (weighted.astype(np.float32), (tf_mat.row, tf_mat.col)), shape=(n_docs, n_terms)
        )

    def search(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        assert self.matrix is not None, "Gọi .index() trước"
        results = []
        for q in queries:
            idx = [self.vocab[t] for t in tokenize(q) if t in self.vocab]
            if not idx:
                results.append([])
                continue
            counts = Counter(idx)
            qvec = np.zeros(len(self.vocab), dtype=np.float32)
            for term_id, c in counts.items():
                qvec[term_id] = c * self.idf[term_id]
            scores = self.matrix @ qvec  # (n_chunks,)

            # Gộp chunk → doc. Trách nhiệm nội bộ của retriever (INTERFACES mục 3).
            best: dict[str, float] = {}
            for cid, s in zip(self.chunk_doc_ids, scores):
                if s <= 0:
                    continue
                if self.pool == "max":
                    if s > best.get(cid, -1e9):
                        best[cid] = float(s)
                else:  # "sum"
                    best[cid] = best.get(cid, 0.0) + float(s)

            ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
            results.append(ranked)
        return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True, help="holdout.json hoặc public-official.json")
    ap.add_argument("--out", required=True, help="predictions.json")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    top_k = args.top_k or cfg["retrieval"]["top_k_submit"]

    chunks = [
        json.loads(l)
        for l in Path(cfg["paths"]["chunks"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Đánh chỉ mục {len(chunks)} chunk…")

    r = BM25Retriever(**cfg["retrieval"]["bm25"])
    r.index(chunks)
    print(f"  từ vựng: {len(r.vocab)} term")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qids = list(questions)
    texts = [questions[q]["question"] for q in qids]
    print(f"Truy vấn {len(texts)} câu hỏi (top_k={top_k})…")

    ranked = r.search(texts, top_k)
    preds = {q: [d for d, _ in res] for q, res in zip(qids, ranked)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")

    n_empty = sum(1 for v in preds.values() if not v)
    print(f"\n✅ {out}")
    if n_empty:
        print(f"   ⚠️  {n_empty} câu không truy hồi được gì (0 term khớp từ vựng)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
