"""Dense bi-encoder trên chunk, khớp `BaseRetriever` (INTERFACES.md §3).

Một đường code cho cả họ BGE-M3 (`AITeamVN/Vietnamese_Embedding_v2`, `BAAI/bge-m3`): cùng
XLM-R-large, cùng tokenizer, cùng pooling CLS. Đổi model = đổi `repo`/`revision` trong YAML.
Model khác họ (Qwen3 cần instruction prefix, E5 cần `query:`/`passage:`) khai bằng
`query_prefix`/`passage_prefix`/`pooling` trong config — không hard-code ở đây.

Không dùng vector DB: 432.142 chunk × 1024 chiều nằm gọn trong RAM, và matmul numpy cho kết
quả CHÍNH XÁC thay vì xấp xỉ như ANN (plan.md §0.2).

torch/transformers nạp LƯỜI: registry và `--demo` phải chạy được trên máy chưa cài torch.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from src.retrieval.base import (
    BaseRetriever,
    PoolingConfig,
    register_retriever,
    reject_unknown,
    split_spec,
    top_n_desc,
)

POOLINGS = ("cls", "mean")


def chunk_fingerprint(chunk_ids: Sequence[str]) -> str:
    """Vân tay của TẬP chunk đã encode.

    Ràng `embeddings.npy` vào đúng kho chunk sinh ra nó: đổi chunker mà vẫn dùng embedding cũ
    thì hàng i của ma trận không còn là chunk i, và sai lệch đó không có biểu hiện nào ngoài
    recall tụt không rõ lý do.

    Args:
        chunk_ids: `chunk_id` theo đúng thứ tự hàng của ma trận.

    Returns:
        Chuỗi hex 24 ký tự.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(str(len(chunk_ids)).encode())
    for cid in chunk_ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def resolve_device(requested: str | None = None) -> str:
    """Chọn thiết bị: `auto` → cuda > mps > cpu; giá trị khác giữ nguyên.

    MPS chạy được để encode (plan.md §0.1), fine-tune thì không.
    """
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class EncoderConfig:
    """Model và cách encode của bi-encoder.

    Attributes:
        repo: Repo HuggingFace.
        revision: Commit hash đã pin — BẮT BUỘC (tái lập, plan.md §7).
        pooling: `cls` (họ BGE-M3) hoặc `mean`.
        normalize: Chuẩn hoá L2 ⇒ dot product = cosine.
        query_prefix: Tiền tố cho câu hỏi (E5: `"query: "`, Qwen3: instruction).
        passage_prefix: Tiền tố cho chunk.
        max_length: Số token tối đa; phải khớp lần encode kho.
        batch_size: Lô khi encode chunk.
        query_batch_size: Lô khi encode câu hỏi và khi nhân ma trận lúc truy vấn.
        device: `auto`, `cuda`, `mps` hoặc `cpu`.
        fp16: Dùng fp16 khi có GPU.
        trust_remote_code: Cho phép code tuỳ biến của repo (vd gte-multilingual).
    """

    repo: str
    revision: str
    pooling: str = "cls"
    normalize: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = 512
    batch_size: int = 32
    query_batch_size: int = 64
    device: str = "auto"
    fp16: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        """Bắt buộc có `revision` và `pooling` hợp lệ."""
        if not self.revision:
            raise ValueError(
                f"Model '{self.repo}' thiếu `revision`. Pin revision hash là quy tắc tái lập của "
                f"team (plan.md mục 7): tác giả cập nhật repo HF giữa chừng thì kết quả BTC tái lập "
                f"lệch đi mà không ai tìm ra nguyên nhân."
            )
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling '{self.pooling}' không có. Dùng: {', '.join(POOLINGS)}")


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """Bi-encoder: điểm chunk = cosine(query, chunk). Gộp chunk→doc do khung lo.

    Hai chế độ nạp:

    * `embeddings_path` trỏ tới file đã encode sẵn (`scripts/encode_corpus.py`) → `index()`
      chỉ đọc file, không cần GPU. Đây là đường chạy bình thường.
    * Không có file → tự encode trong `index()`. Chỉ hợp lý khi chạy thử vài nghìn chunk.

    Attributes:
        encoder: Cấu hình model.
        embeddings_path: File embedding encode sẵn, hoặc None.
        matrix: Ma trận (n_chunks × dim) sau `index()`.
    """

    def __init__(
        self,
        encoder: EncoderConfig,
        pooling: PoolingConfig | None = None,
        *,
        embeddings_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(pooling or PoolingConfig(pool="mean_top2"))
        self.encoder = encoder
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.verbose = verbose
        self.matrix: np.ndarray | None = None
        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        self._index_seconds = 0.0

    @classmethod
    def from_spec(cls, spec: dict) -> "DenseRetriever":
        """Dựng từ spec phẳng của YAML (`retrieval.dense`).

        Raises:
            TypeError: Spec có khoá không thuộc `EncoderConfig`/`PoolingConfig`/
                `embeddings_path`/`verbose`.
        """
        (enc_kw, pool_kw), rest = split_spec(spec, EncoderConfig, PoolingConfig)
        reject_unknown(cls.__name__, rest, ("embeddings_path", "verbose"))
        return cls(
            EncoderConfig(**enc_kw),
            PoolingConfig(**{"pool": "mean_top2", **pool_kw}),
            embeddings_path=rest.get("embeddings_path"),
            verbose=rest.get("verbose", True),
        )

    def with_encoder(self, **changes) -> "DenseRetriever":
        """Bản sao chưa index với vài trường `EncoderConfig` được ghi đè (vd `batch_size`)."""
        return type(self)(
            replace(self.encoder, **changes),
            self.pooling,
            embeddings_path=self.embeddings_path,
            verbose=self.verbose,
        )

    # ── thuộc tính rút gọn (log, meta embedding) ─────────────────────────────
    @property
    def repo(self) -> str:
        """Repo HuggingFace của model."""
        return self.encoder.repo

    @property
    def revision(self) -> str:
        """Commit hash đã pin."""
        return self.encoder.revision

    @property
    def batch_size(self) -> int:
        """Lô khi encode chunk."""
        return self.encoder.batch_size

    # ── model, nạp lười ──────────────────────────────────────────────────────
    def _ensure_model(self) -> None:
        """Nạp tokenizer + model một lần, đúng `revision`, đúng dtype theo thiết bị.

        Raises:
            ImportError: Chưa cài torch/transformers.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "DenseRetriever cần `torch` + `transformers` (xem requirements.txt). "
                "Chỉ đọc embedding đã encode sẵn thì truyền `embeddings_path` và không gọi encode."
            ) from e

        enc = self.encoder
        self._device = resolve_device(enc.device)
        if self.verbose:
            print(f"  [dense] nạp {enc.repo}@{enc.revision[:12]} trên {self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            enc.repo, revision=enc.revision, trust_remote_code=enc.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            enc.repo,
            revision=enc.revision,
            trust_remote_code=enc.trust_remote_code,
            torch_dtype=torch.float16 if (enc.fp16 and self._device != "cpu") else torch.float32,
        )
        self._model = model.to(self._device).eval()

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        """Encode một lô đã gắn prefix → (n, dim) float32, đã pooling và chuẩn hoá."""
        import torch

        enc = self._tokenizer(
            batch, padding=True, truncation=True,
            max_length=self.encoder.max_length, return_tensors="pt",
        ).to(self._device)
        hidden = self._model(**enc).last_hidden_state
        if self.encoder.pooling == "cls":
            vec = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.encoder.normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        return vec.float().cpu().numpy()

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int | None = None,
        show_every: int = 50,
    ) -> np.ndarray:
        """Encode một dãy văn bản.

        Args:
            texts: Văn bản cần encode.
            is_query: True thì dùng `query_prefix`/`query_batch_size`, ngược lại dùng của chunk.
            batch_size: Ghi đè kích thước lô.
            show_every: In tiến độ sau mỗi chừng này lô (chỉ khi encode chunk); 0 = không in.

        Returns:
            Mảng (n, dim) float32.
        """
        import torch

        self._ensure_model()
        enc = self.encoder
        prefix = enc.query_prefix if is_query else enc.passage_prefix
        bs = batch_size or (enc.query_batch_size if is_query else enc.batch_size)
        out: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                out.append(self._embed_batch([prefix + t for t in texts[start : start + bs]]))
                if self.verbose and not is_query and show_every and (start // bs) % show_every == 0:
                    _print_progress(min(start + bs, len(texts)), len(texts), t0)
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    # ── index ────────────────────────────────────────────────────────────────
    def index(self, chunks: list[dict]) -> None:
        """Nạp embedding encode sẵn (hoặc encode tại chỗ) cho kho chunk.

        Raises:
            ValueError: Số hàng embedding khác số chunk.
        """
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
        """Đọc `embeddings.npy` và đối chiếu meta đi kèm.

        File trên Drive lưu fp16 cho nhẹ, nhưng numpy không có nhân ma trận fp16 thật trên
        CPU — nạp xong thì nâng lên fp32 một lần, đổi RAM lấy tốc độ truy vấn.
        """
        assert self.embeddings_path is not None
        mat = np.load(self.embeddings_path)
        if mat.dtype == np.float16 and self.encoder.device in ("auto", "cpu"):
            print(f"  [dense] fp16 → fp32 khi nạp ({mat.nbytes * 2 / 2**30:.2f} GB RAM) cho truy vấn CPU")
            mat = mat.astype(np.float32)
        meta_path = self.embeddings_path.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"  ⚠️  không có {meta_path.name} — không kiểm được embedding này encode từ đâu.")
            return mat
        self._check_meta(json.loads(meta_path.read_text(encoding="utf-8")))
        return mat

    def _check_meta(self, meta: dict) -> None:
        """Từ chối embedding encode từ kho chunk khác hoặc model khác.

        Raises:
            ValueError: Vân tay chunk hoặc `repo`/`revision` không khớp.
        """
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

    # ── truy vấn ─────────────────────────────────────────────────────────────
    def _score_chunks(self, queries: list[str], n_candidates: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Cosine với mọi chunk, theo lô `query_batch_size` (1.000 câu đặc là 2 GB fp32)."""
        if self.matrix is None:
            raise RuntimeError("Gọi .index() trước")
        q_emb = self.encode(queries, is_query=True)
        mat = self.matrix
        bs = self.encoder.query_batch_size
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, q_emb.shape[0], bs):
            block = q_emb[start : start + bs].astype(mat.dtype, copy=False)
            scores = (mat @ block.T).astype(np.float32)  # (n_chunks, batch)
            for col in range(scores.shape[1]):
                out.append(top_n_desc(scores[:, col], n_candidates))
        return out

    def stats(self) -> dict:
        s = super().stats()
        enc = self.encoder
        s.update(
            repo=enc.repo,
            revision=enc.revision,
            pooling=enc.pooling,
            normalize=enc.normalize,
            max_length=enc.max_length,
            dim=int(self.matrix.shape[1]) if self.matrix is not None else 0,
            device=self._device or enc.device,
            index_seconds=round(self._index_seconds, 1),
        )
        return s


def _print_progress(done: int, total: int, t0: float) -> None:
    """In tiến độ encode và thời gian còn lại ước tính."""
    rate = done / max(1e-9, time.perf_counter() - t0)
    print(f"    {done}/{total} ({rate:.0f}/s, còn ~{(total - done) / max(rate, 1e-9) / 60:.1f} phút)")
