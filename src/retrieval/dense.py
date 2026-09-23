"""
Dense bi-encoder trên chunk — CHỦ SỞ HỮU: P3. Khớp `BaseRetriever` (INTERFACES.md mục 3).

MỘT đường code cho cả họ BGE-M3 (`AITeamVN/Vietnamese_Embedding_v2` / `_v1` / `BAAI/bge-m3`):
cùng XLM-R-large, cùng tokenizer, cùng pooling CLS. Đổi model = đổi `repo`/`revision` trong YAML,
không sửa dòng code nào. Model khác họ (Qwen3 cần instruction prefix, E5 cần `query:`/`passage:`)
khai bằng `query_prefix`/`passage_prefix`/`pooling` trong config — KHÔNG hard-code ở đây.

Vì sao không dùng vector DB: corpus chỉ 8.507 văn bản / 524.422 chunk. Ma trận 1024 chiều fp16
≈ 1,07 GB, nằm gọn trong RAM, và matmul numpy cho kết quả CHÍNH XÁC thay vì xấp xỉ như ANN
(plan.md mục 0.2).

torch/transformers nạp LƯỜI: registry và `--demo` của bench phải chạy được trên máy chưa cài
torch (requirements.txt v0.1 cố ý chỉ có numpy/scipy/PyYAML).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from src.retrieval.base import BaseRetriever, register_retriever

POOLINGS = ("cls", "mean")


def chunk_fingerprint(chunk_ids: Sequence[str]) -> str:
    """
    Vân tay của TẬP chunk đã encode. Ràng `embeddings.npy` vào đúng bộ chunk sinh ra nó —
    P2 đổi chunker mà ta vẫn dùng embedding cũ thì hàng i của ma trận không còn là chunk i,
    và sai lệch đó KHÔNG có biểu hiện nào ngoài việc recall tụt không rõ lý do.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(str(len(chunk_ids)).encode())
    for cid in chunk_ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def resolve_device(requested: str | None = None) -> str:
    """auto → cuda > mps > cpu. MPS chạy được để encode (plan.md 0.1), fine-tune thì không."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """
    Bi-encoder: điểm chunk = cosine(query, chunk). Gộp chunk→doc do khung lo (INTERFACES §3).

    Hai chế độ nạp:
      - `embeddings_path` trỏ tới file đã encode sẵn (`scripts/encode_corpus.py`) → `index()` chỉ
        đọc file, không cần GPU. Đây là đường chạy bình thường.
      - không có file → tự encode trong `index()`. Chỉ hợp lý khi chạy thử vài nghìn chunk.
    """

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        pool: str = "mean_top2",
        pool_tau: float = 1.0,
        candidate_chunks: int = 2000,
        pooling: str = "cls",
        normalize: bool = True,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_length: int = 512,
        batch_size: int = 32,
        query_batch_size: int = 64,
        device: str = "auto",
        fp16: bool = True,
        embeddings_path: str | Path | None = None,
        trust_remote_code: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(pool=pool, pool_tau=pool_tau, candidate_chunks=candidate_chunks)
        if not revision:
            raise ValueError(
                f"Model '{repo}' thiếu `revision`. Pin revision hash là quy tắc tái lập của team "
                f"(plan.md mục 7): tác giả cập nhật repo HF giữa chừng thì kết quả BTC tái lập "
                f"lệch đi mà không ai tìm ra nguyên nhân."
            )
        if pooling not in POOLINGS:
            raise ValueError(f"pooling '{pooling}' không có. Dùng: {', '.join(POOLINGS)}")
        self.repo, self.revision = repo, revision
        self.pooling = pooling
        self.normalize = normalize
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.max_length = max_length
        self.batch_size, self.query_batch_size = batch_size, query_batch_size
        self.device_request, self.fp16 = device, fp16
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.trust_remote_code = trust_remote_code
        self.verbose = verbose

        self.matrix: np.ndarray | None = None  # (n_chunks, dim), fp16 hoặc fp32
        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        self._index_seconds = 0.0

    # ── model, nạp lười ──────────────────────────────────────────────────────
    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "DenseRetriever cần `torch` + `transformers` (chưa nằm trong requirements.txt v0.1). "
                "Chỉ đọc embedding đã encode sẵn thì truyền `embeddings_path` và không gọi encode."
            ) from e

        self._device = resolve_device(self.device_request)
        if self.verbose:
            print(f"  [dense] nạp {self.repo}@{self.revision[:12]} trên {self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.repo, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            self.repo,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16 if (self.fp16 and self._device != "cpu") else torch.float32,
        )
        self._model = model.to(self._device).eval()

    def encode(self, texts: Sequence[str], *, is_query: bool, batch_size: int | None = None,
               show_every: int = 50) -> np.ndarray:
        """Encode một lô văn bản → (n, dim) float32 (đã chuẩn hoá nếu `normalize`)."""
        import torch

        self._ensure_model()
        prefix = self.query_prefix if is_query else self.passage_prefix
        bs = batch_size or (self.query_batch_size if is_query else self.batch_size)
        out: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = [prefix + t for t in texts[start : start + bs]]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if self.normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                out.append(vec.float().cpu().numpy())
                if self.verbose and not is_query and show_every and (start // bs) % show_every == 0:
                    done = min(start + bs, len(texts))
                    rate = done / max(1e-9, time.perf_counter() - t0)
                    print(f"    {done}/{len(texts)} ({rate:.0f}/s, còn ~{(len(texts)-done)/max(rate,1e-9)/60:.1f} phút)")
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        t0 = time.perf_counter()
        self._register_chunks(chunks)

        if self.embeddings_path and self.embeddings_path.exists():
            self.matrix = self._load_embeddings()
        else:
            if self.embeddings_path:
                print(
                    f"  ⚠️  chưa có {self.embeddings_path} — encode tại chỗ. "
                    f"Với corpus thật hãy chạy `python scripts/encode_corpus.py` trước."
                )
            self.matrix = self.encode([c["text"] for c in chunks], is_query=False)

        if self.matrix.shape[0] != self.n_chunks:
            raise ValueError(
                f"Embedding có {self.matrix.shape[0]} hàng nhưng nhận {self.n_chunks} chunk. "
                f"Hai thứ này phải khớp theo TỪNG HÀNG — encode lại."
            )
        self._index_seconds = time.perf_counter() - t0
        if self.verbose:
            print(
                f"  [dense] {self.n_chunks} chunk / {self.n_docs} văn bản, dim {self.matrix.shape[1]}, "
                f"{self.matrix.dtype}, {self.matrix.nbytes / 2**30:.2f} GB ({self._index_seconds:.1f}s)"
            )

    def _load_embeddings(self) -> np.ndarray:
        assert self.embeddings_path is not None
        mat = np.load(self.embeddings_path)
        # File trên Drive lưu fp16 cho nhẹ (1,07 GB thay vì 2,15 GB), nhưng numpy KHÔNG có nhân
        # ma trận fp16 thật trên CPU — nó ép kiểu qua lại từng phép, chậm hơn fp32 nhiều lần.
        # Nạp xong thì nâng lên fp32 một lần, tốn RAM gấp đôi và đổi lại tốc độ truy vấn.
        if mat.dtype == np.float16 and (self.device_request in ("auto", "cpu")):
            print(f"  [dense] fp16 → fp32 khi nạp ({mat.nbytes * 2 / 2**30:.2f} GB RAM) cho truy vấn CPU")
            mat = mat.astype(np.float32)
        meta_path = self.embeddings_path.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"  ⚠️  không có {meta_path.name} — không kiểm được embedding này encode từ đâu.")
            return mat
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        want = chunk_fingerprint(self.chunk_ids)
        if meta.get("chunk_fingerprint") != want:
            raise ValueError(
                f"{self.embeddings_path} encode từ MỘT BỘ CHUNK KHÁC "
                f"(vân tay {meta.get('chunk_fingerprint')} ≠ {want}). Hàng i của ma trận không còn "
                f"là chunk i — chạy lại `scripts/encode_corpus.py`."
            )
        if meta.get("repo") != self.repo or meta.get("revision") != self.revision:
            raise ValueError(
                f"{self.embeddings_path} encode bằng {meta.get('repo')}@{meta.get('revision')}, "
                f"config đang yêu cầu {self.repo}@{self.revision}. Trộn embedding của hai model "
                f"là so cosine giữa hai không gian khác nhau."
            )
        return mat

    # ── truy vấn ─────────────────────────────────────────────────────────────
    def _score_chunks(
        self, queries: list[str], n_candidates: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.matrix is None:
            raise RuntimeError("Gọi .index() trước")
        Q = self.encode(queries, is_query=True)
        mat = self.matrix
        out: list[tuple[np.ndarray, np.ndarray]] = []
        # Theo lô: (n_chunks × n_query) đặc cho 1.000 câu là 2 GB fp32. Lô 64 còn ~134 MB.
        for start in range(0, Q.shape[0], self.query_batch_size):
            block = Q[start : start + self.query_batch_size].astype(mat.dtype, copy=False)
            scores = (mat @ block.T).astype(np.float32)  # (n_chunks, batch)
            for col in range(scores.shape[1]):
                s = scores[:, col]
                n = min(n_candidates, s.size)
                if n < s.size:
                    idx = np.argpartition(-s, n - 1)[:n]
                else:
                    idx = np.arange(s.size)
                order = np.argsort(-s[idx], kind="stable")
                idx = idx[order]
                out.append((idx.astype(np.int64), s[idx]))
        return out

    def stats(self) -> dict:
        s = super().stats()
        s.update(
            repo=self.repo,
            revision=self.revision,
            pooling=self.pooling,
            normalize=self.normalize,
            max_length=self.max_length,
            dim=int(self.matrix.shape[1]) if self.matrix is not None else 0,
            device=self._device or self.device_request,
            index_seconds=round(self._index_seconds, 1),
        )
        return s
