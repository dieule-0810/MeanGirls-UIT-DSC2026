# Kiểm toán số tham số — ứng viên đăng ký với BTC

> Người thực hiện: P4 · Ngày: 13/08/2026 · Công cụ: `scripts/count_params.py`
> Nguồn: metadata `safetensors` trên HuggingFace (đếm thật, không lấy từ tên model).
> Ngân sách BTC: **tổng toàn pipeline < 4.000M**, tính cả lớp embedding.

## Cross-encoder / reranker

| Model | Tham số | Nền kiến trúc | dtype | revision | license |
|---|---:|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 567.8M | XLM-R large, 8192 ctx | F32 | `953dc6f6f85a` | apache-2.0 |
| `AITeamVN/Vietnamese_Reranker` | 567.8M | XLM-R large, 8192 ctx | F32 | `f53697624840` | apache-2.0 |
| `namdp-ptit/ViRanker` | 567.8M | XLM-R large, 8192 ctx | F32 | `922bd0d698b1` | apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | 595.8M | Qwen3 0.6B | BF16 | `e61197ed4502` | apache-2.0 |
| `jinaai/jina-reranker-v2-base-multilingual` | 278.4M | XLM-R base biến thể | BF16 | `9cfeff2df7d4` | **cc-by-nc-4.0** |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 117.6M | mMiniLMv2 L12-H384 | F32 | `1427fd652930` | apache-2.0 |

## Bi-encoder / embedding

| Model | Tham số | Nền kiến trúc | dtype | revision | license |
|---|---:|---|---|---|---|
| `BAAI/bge-m3` | **567.8M** ⚠️ | XLM-R large, 8192 ctx | F32 | `5617a9f61b02` | mit |
| `AITeamVN/Vietnamese_Embedding` | 567.8M | XLM-R large, 8192 ctx | F32 | `dea33aa1ab33` | apache-2.0 |
| `Qwen/Qwen3-Embedding-0.6B` | 595.8M | Qwen3 0.6B | BF16 | `97b0c614be4d` | apache-2.0 |
| `intfloat/multilingual-e5-large` | 559.9M | XLM-R large, 512 ctx | F32 | `3d7cfbdacd47` | mit |
| `Alibaba-NLP/gte-multilingual-base` | 305.4M | new-impl, 8192 ctx | F16 | `9bbca17d9273` | apache-2.0 |
| `intfloat/multilingual-e5-base` | 278.0M | XLM-R base | F32 | `d12875059715` | mit |
| `bkai-foundation-models/vietnamese-bi-encoder` | 135.0M | PhoBERT base | F32 | `84f9d9ada0d1` | apache-2.0 |

⚠️ **`BAAI/bge-m3` không có safetensors** (`NotASafetensorsRepoError`) — repo chỉ chứa
`pytorch_model.bin`. Con số 567.8M là **suy ra bằng kiến trúc**, chưa đếm trực tiếp.
Xem phần "Suy diễn BGE-M3" bên dưới và **chạy `scripts/verify_bge_m3.py` để xác nhận
trước khi khoá Model Card**.

---

## Suy diễn BGE-M3 = 567.75M

XLM-RoBERTa-large với cửa sổ ngữ cảnh mở rộng lên 8192
(`max_position_embeddings = 8194`, gồm 2 vị trí đệm của RoBERTa):

| Thành phần | Công thức | Tham số |
|---|---|---:|
| Word embeddings | 250.002 × 1024 | 256.002.048 |
| Position embeddings | 8.194 × 1024 | 8.390.656 |
| Token-type + LayerNorm | 1024 + 2×1024 | 3.072 |
| 24 lớp transformer | 24 × 12.596.224 | 302.309.376 |
| Pooler | 1024×1024 + 1024 | 1.049.600 |
| **Tổng backbone** | | **567.754.752** |

Kiểm chứng chéo bằng một model **đã đếm được**: `multilingual-e5-large` là cùng
XLM-R large nhưng ngữ cảnh 512 (`max_position_embeddings = 514`).

```
567.754.752 − 559.890.432 = 7.864.320 = (8194 − 514) × 1024   ✓ khớp chính xác
```

Chênh lệch duy nhất giữa hai model đúng bằng phần position embedding mở rộng — xác nhận
suy diễn đúng, và đồng thời **giải thích vì sao ba reranker cùng ra 567.8M**: tất cả đều
là XLM-R large ngữ cảnh 8192.

**Hai head phụ của BGE-M3** nằm ở file riêng, không nằm trong `pytorch_model.bin`:

| Head | Tham số | Khi nào tính vào ngân sách |
|---|---:|---|
| `sparse_linear.pt` | 1.025 | chỉ khi dùng chế độ sparse |
| `colbert_linear.pt` | 1.049.600 | chỉ khi dùng chế độ multi-vector |

→ Chỉ dùng dense: khai **567.8M**. Dùng cả ba chế độ: khai **568.8M**.
Ghi rõ chế độ nào trong Model Card, đừng khai chung chung.

---

## Tổ hợp pipeline

Tổ hợp **nặng nhất** trong toàn bộ danh sách:

```
Qwen3-Embedding-0.6B (595.8M) + Qwen3-Reranker-0.6B (595.8M) = 1.191,6M
```

**≈ 29,8% ngân sách 4.000M.** Không có tổ hợp nào trong danh sách vượt hạn mức, kể cả khi
Tuần 5 ensemble hai bi-encoder:

```
BGE-M3 + Vietnamese_Embedding + bge-reranker-v2-m3 = 1.703,4M  (42,6%)
```

→ **Ràng buộc 4B không siết team này.** Ràng buộc thật là 120 giờ T4/tuần và thời gian
inference ở private test (3 lượt/ngày). Nếu phải chọn giữa "model lớn hơn" và "chạy nhanh
hơn", ngân sách tham số không phải lý do để từ chối model lớn.

---

## Ba ghi chú cho Model Card

**1. Lớp embedding nuốt phần lớn ngân sách, không phải năng lực mô hình.**

| Model | Embedding | % tổng |
|---|---:|---:|
| `mmarco-mMiniLMv2-L12-H384` | 250.002 × 384 ≈ 96,0M | **81,6%** |
| `bge-m3` / XLM-R large | 250.002 × 1024 ≈ 256,0M | **45,1%** |
| `Qwen3-*-0.6B` | 151.936 × 1024 ≈ 155,6M | 26,1% |

Với mmarco-mMiniLMv2, phần transformer thật chỉ ~21,6M — 82% ngân sách nằm ở từ điển đa
ngữ 250k. Đây là quan sát dùng được cho phần thảo luận bài báo: trong bài toán đa ngữ,
hạn mức tham số phần lớn bị vocab tiêu thụ chứ không phải chiều sâu mô hình.

**2. Ba reranker XLM-R large là ba biến thể của cùng một kiến trúc.**
`Vietnamese_Reranker` gần như chắc chắn fine-tune trực tiếp từ `bge-reranker-v2-m3`.
Chúng sẽ sai ở những câu giống nhau → benchmark Tuần 2 trên ba model này là so ba biến thể,
không phải ba cách tiếp cận. Muốn có đa dạng thật cho ensemble và cho ô ablation, phải kéo
`mmarco-mMiniLMv2` (khác nền, khác kích thước) hoặc `jina-reranker-v2` vào.

**3. `jina-reranker-v2` là `cc-by-nc-4.0`.** Hợp lệ theo mục "Quy định về API và giấy phép"
của thể lệ (chấp nhận giấy phép phi thương mại / nghiên cứu). **Ghi rõ điều này trong đơn
đăng ký** để BTC không phải tự tra rồi loại vì nghi ngờ.

---

## Còn thiếu

- [ ] Chạy `scripts/verify_bge_m3.py`, thay con số suy diễn bằng con số đếm thật.
- [ ] `Alibaba-NLP/gte-multilingual-base` dùng `trust_remote_code` (kiến trúc `new-impl`).
      Kiểm tra có chạy được offline trong Docker không trước khi đưa vào pipeline chính.
- [ ] Xác nhận `Qwen3-*-0.6B` có `tie_word_embeddings: true` (nếu `false` thì lớp output
      nhân đôi phần embedding và con số 595.8M cần xem lại).
