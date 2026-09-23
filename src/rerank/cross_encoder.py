"""Cross-encoder reranker — P4 sở hữu.

Chấm cặp (câu hỏi, đoạn văn bản) bằng cross-encoder rồi xếp lại top-K của BM25.

BỐI CẢNH SỐ LIỆU (dev n=1000, pool=logsumexp, xem docs/stratified_baseline.md):
    xuất phát R@5  = 0,8067
    TRẦN     R@50  = 0,9568   ← reranker không bao giờ vượt được
    dư địa         = 0,1501
Dư địa lớn nhất ở tầng freq>=11 (+0,2982): BM25 tìm được luật khung nhưng xếp sai chỗ,
vì câu hỏi pháp luật dùng chung khung diễn đạt dài và chỉ khác nhau ở token thực thể.
Đó chính là thứ cross-encoder đọc được mà BM25 thì không.

max_length=512 CHỦ Ý, dù models.yaml ghi max_seq_len 8192: chunk của P2 tối đa 180 từ
≈ 350-450 token tiếng Việt, cộng câu hỏi vẫn dưới 512. Cách chia chunk khiến cửa sổ
8192 thành vô dụng — không có lý do trả bộ nhớ attention bình phương cho nó.
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

    Không phải retriever: nó không tìm gì cả, chỉ xếp lại danh sách đã có.
    """

    name: str
    device: str | None = None
    max_length: int = 512
    batch_size: int = 16
    fp16: bool = True

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.name not in MODELS:
            raise KeyError(f"{self.name} không có trong MODELS: {sorted(MODELS)}")
        spec = MODELS[self.name]
        self.spec = spec

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and self.fp16:
            self.fp16 = False  # fp16 trên CPU chậm hơn fp32, không nhanh hơn

        self.tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec["repo"], revision=spec["revision"]
        )
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()

        # `score()` lấy thẳng logits[:, 0]. Với num_labels=1 đó là điểm liên quan; với
        # num_labels=2 đó là logit lớp KHÔNG liên quan ⇒ thứ hạng bị ĐẢO NGƯỢC, và biểu
        # hiện duy nhất là recall tụt không rõ lý do. Kiểm một lần lúc nạp, không đoán.
        n_labels = int(getattr(self.model.config, "num_labels", 1))
        if n_labels != 1:
            raise SystemExit(
                f"❌ {spec['repo']} có num_labels={n_labels}, không phải 1. `score()` đang lấy "
                f"logits[:, 0] — với model này cột đó không phải điểm liên quan. Sửa `score()` "
                f"cho đúng quy ước của model rồi mới chạy."
            )

        n = sum(p.numel() for p in self.model.parameters())
        self._torch = torch
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
        """Trả điểm theo ĐÚNG thứ tự `pairs` truyền vào.

        Gom batch theo độ dài để giảm padding: cặp ngắn không phải đệm cho bằng cặp
        dài nhất trong batch. Trên tập này rút ~30-40% thời gian, và không đổi kết
        quả vì mỗi cặp được chấm độc lập.
        """
        torch = self._torch
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        out = [0.0] * len(pairs)

        done = 0
        for s in range(0, len(order), self.batch_size):
            sl = order[s : s + self.batch_size]
            enc = self.tok(
                [pairs[i][0] for i in sl],
                [pairs[i][1] for i in sl],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            # num_labels=1 với cả ba model → điểm là cột 0
            vals = logits[:, 0].float().tolist()
            for i, v in zip(sl, vals):
                out[i] = float(v)
            done += len(sl)
            if not quiet and (done % (self.batch_size * 50) < self.batch_size):
                print(f"  {done}/{len(pairs)}", flush=True)
        return out
