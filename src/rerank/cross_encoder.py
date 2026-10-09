"""Cross-encoder reranker: chấm cặp (câu hỏi, đoạn văn bản) để xếp lại top-K. CHỦ SỞ HỮU: P4.

KẾT QUẢ ĐÃ ĐO: dùng reranker để THAY THẾ thứ hạng làm TỆ R@5 ở cả ba model zero-shot (bge-m3
−0,0252 trên dev, −0,0253 trên holdout) — chỉ dùng ở dạng hợp nhất RRF (`scripts/run_pipeline.py`
`pipeline.rerank.mode: rrf`). Đường nộp bài v0.8 KHÔNG có tầng này.

Bối cảnh lúc thiết kế (BM25 regex + logsumexp, dev n=1000, docs/stratified_baseline.md):
R@5 0,8067, trần R@50 0,9568 — reranker không bao giờ vượt trần đó. Dư địa lớn nhất ở tầng
freq>=11: BM25 tìm được luật khung nhưng xếp sai chỗ vì câu hỏi chỉ khác nhau ở token thực thể.

max_length=512 CHỦ Ý dù models.yaml ghi max_seq_len 8192: chunk tối đa 256 từ, và đo ở v0.5
(max_length 1024) chỉ tốn thêm 17% thời gian — truncation vốn không phổ biến.
"""
from __future__ import annotations

from dataclasses import dataclass


# Lấy từ configs/models.yaml. LUÔN truyền revision — không pin là BTC tái lập ra
# trọng số khác và kết quả trong bài báo thành vô nghĩa.
MODELS: dict[str, dict] = {
    "bge-m3": {
        "repo": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a",
        "params": 567_800_000,
        "note": "XLM-R-large, chuẩn tham chiếu",
    },
    "mminilm": {
        "repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "revision": "1427fd652930",
        "params": 117_600_000,
        "note": "ô ablation nhỏ-vs-lớn; 96M/117,6M nằm ở lớp embedding, transformer thật ~21,6M",
    },
    "viranker": {
        "repo": "namdp-ptit/ViRanker",
        "revision": "922bd0d698b1",
        "params": 567_800_000,
        "note": "ô ablation chuyên Việt vs đa ngữ; cùng XLM-R-large nên KHÔNG phải kiến trúc mới",
    },
    "vi-reranker": {
        "repo": "AITeamVN/Vietnamese_Reranker",
        "revision": "f53697624840",
        "params": 567_800_000,
        "note": (
            "fine-tune tiếng Việt TỪ CHÍNH bge-reranker-v2-m3 ⇒ so với 'bge-m3' là phép so "
            "cùng trọng số gốc, chỉ khác phần fine-tune. ViRanker cũng chuyên Việt nhưng train "
            "riêng nên lẫn hai biến; ô này tách được biến đó ra. Cùng họ XLM-R-large với "
            "Vietnamese_Embedding_v2 (configs/models.yaml) ⇒ một tokenizer cho cả embed lẫn rerank."
        ),
    },
}


@dataclass
class CrossEncoderReranker:
    """Chấm điểm liên quan cho từng cặp (query, passage).

    Không phải retriever: nó không tìm gì, chỉ xếp lại danh sách đã có. Nạp model ngay lúc
    dựng và kiểm hai điều kiện fail-loud: `num_labels == 1` và số tham số khớp models.yaml.

    Attributes:
        name: Khoá trong `MODELS` (`bge-m3`, `mminilm`, `viranker`, `vi-reranker`).
        device: `cuda`/`cpu`/...; None = cuda nếu có.
        max_length: Số token tối đa của cặp.
        batch_size: Số cặp mỗi lô.
        fp16: Dùng fp16 (tự tắt trên CPU).
    """

    name: str
    device: str | None = None
    max_length: int = 512
    batch_size: int = 16
    fp16: bool = True

    def __post_init__(self) -> None:
        """Chọn thiết bị, nạp model và chạy hai kiểm fail-loud."""
        import torch

        if self.name not in MODELS:
            raise KeyError(f"{self.name} không có trong MODELS: {sorted(MODELS)}")
        self.spec = MODELS[self.name]
        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and self.fp16:
            self.fp16 = False  # fp16 trên CPU chậm hơn fp32, không nhanh hơn
        self._load_model()
        self._check_num_labels()
        self._check_param_count()

    def _load_model(self) -> None:
        """Nạp tokenizer + model đúng `revision` đã pin."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = self.spec
        self.tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()

    def _check_num_labels(self) -> None:
        """`score()` lấy `logits[:, 0]`: chỉ đúng khi num_labels=1.

        Với num_labels=2 cột 0 là logit lớp KHÔNG liên quan ⇒ thứ hạng bị ĐẢO NGƯỢC, và biểu
        hiện duy nhất là recall tụt không rõ lý do.

        Raises:
            SystemExit: num_labels khác 1.
        """
        n_labels = int(getattr(self.model.config, "num_labels", 1))
        if n_labels != 1:
            raise SystemExit(
                f"❌ {self.spec['repo']} có num_labels={n_labels}, không phải 1. `score()` đang lấy "
                f"logits[:, 0] — với model này cột đó không phải điểm liên quan. Sửa `score()` "
                f"cho đúng quy ước của model rồi mới chạy."
            )

    def _check_param_count(self) -> None:
        """Ngân sách 4 tỷ tính trên số tham số ĐẾM ĐƯỢC, không phải số khai.

        Raises:
            SystemExit: Lệch quá 2% so với `params` trong models.yaml.
        """
        spec = self.spec
        n = sum(p.numel() for p in self.model.parameters())
        print(
            f"[rerank] {spec['repo']}@{spec['revision']}  "
            f"{n:,} tham số (models.yaml khai {spec['params']:,})  "
            f"device={self.device} fp16={self.fp16} max_len={self.max_length}"
        )
        if abs(n - spec["params"]) > 0.02 * spec["params"]:
            raise SystemExit(
                f"❌ Số tham số đếm được ({n:,}) lệch >2% so với models.yaml "
                f"({spec['params']:,}). Ngân sách 4 tỷ tính trên số ĐẾM ĐƯỢC, "
                "không phải số khai. Sửa models.yaml rồi chạy lại."
            )

    def score(self, pairs: list[tuple[str, str]], quiet: bool = False) -> list[float]:
        """Chấm điểm liên quan của từng cặp.

        Gom batch theo độ dài passage để giảm padding (rút ~30-40% thời gian), không đổi kết
        quả vì mỗi cặp được chấm độc lập.

        Args:
            pairs: Danh sách `(câu hỏi, đoạn văn bản)`.
            quiet: Không in tiến độ.

        Returns:
            Điểm theo ĐÚNG thứ tự `pairs`.
        """
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        out = [0.0] * len(pairs)
        done = 0
        for s in range(0, len(order), self.batch_size):
            sl = order[s : s + self.batch_size]
            for i, v in zip(sl, self._score_batch([pairs[i] for i in sl])):
                out[i] = v
            done += len(sl)
            if not quiet and (done % (self.batch_size * 50) < self.batch_size):
                print(f"  {done}/{len(pairs)}", flush=True)
        return out

    def _score_batch(self, batch: list[tuple[str, str]]) -> list[float]:
        """Một lô cặp → cột 0 của logits (num_labels=1 đã được kiểm lúc nạp)."""
        enc = self.tok(
            [q for q, _ in batch],
            [p for _, p in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        return [float(v) for v in logits[:, 0].float().tolist()]
