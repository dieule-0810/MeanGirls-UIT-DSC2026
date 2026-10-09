# Model Card — DSC2026 Task 1 LegalIR

> Điều 9 yêu cầu **mỗi bài nộp** kèm Model Card và Data Statement. Cập nhật ở MỖI release.

## Phiên bản

| | |
|---|---|
| Release | v0.8 — `configs/v0.8_hybrid_rrf.yaml` |
| Commit | _(điền commit/tag của lượt nộp — xem `docs/releases/`)_ |
| Cập nhật | 09/10/2026 |

## Thành phần hệ thống và số tham số

⚠️ Ngân sách BTC: **tổng toàn pipeline < 4 tỷ tham số**, tính cả lớp embedding. LoRA/quantization
**không** làm giảm số tham số.

| Thành phần | Mô hình | Revision | Tham số (đếm thật) | BTC duyệt |
|---|---|---|---:|---|
| Tầng 1 — lexical | BM25 Okapi (tự cài, `src/retrieval/bm25.py`) | — | 0 | không cần |
| Tầng 1 — dense | `AITeamVN/Vietnamese_Embedding_v2` (XLM-R large, fine-tune từ BGE-M3) | `18b44161e041bf1d3a333ab5144b5b7b93f914d2` | 567.754.752 | ✅ (13/09/2026) |
| Hợp nhất | RRF có trọng số (`src/retrieval/hybrid.py`) | — | 0 | không cần |
| Số lượng doc | Ngưỡng khe hở điểm (`src/rerank/calibrate.py`) | — | 0 (1 ngưỡng) | không cần |
| **Tổng** | | | **567.754.752 (14,2% trần)** | |

Số tham số đếm bằng `sum(p.numel() for p in AutoModel.from_pretrained(...).parameters())` — gồm cả
lớp pooler mà pipeline không dùng (lấy vector CLS), nên là cận trên. 46,6% số tham số nằm ở lớp
embedding từ vựng.

Không dùng reranker trong bản này: ba cross-encoder zero-shot đã thử đều làm TỆ R@5 khi dùng để
thay thế thứ hạng (bge-reranker-v2-m3 −0,0252 dev / −0,0253 holdout).

## Mục đích và phạm vi

Truy hồi văn bản pháp luật tiếng Việt: nhận một câu hỏi, trả về 1–5 `doc_id` trong kho 8.507
văn bản của BTC. Chỉ dùng cho nghiên cứu/giáo dục trong khuôn khổ UIT DSC 2026.

## Phương pháp

Không huấn luyện trọng số nào. Ba thứ được CHỌN từ dữ liệu, đều chọn trên `train_split` (4.689 câu)
và báo cáo một lần trên `dev` (1.000 câu):

| Siêu tham số | Giá trị | Cách chọn |
|---|---|---|
| Tokenizer BM25 + cách gộp chunk→doc | `syllable_bigram` + `mean_top2` | lưới 5 × 5 (`scripts/bench_retrieval.py`) |
| RRF: mức, k, trọng số BM25/dense | doc, 20, 0,4/0,6 | lưới 66 ô (`scripts/tune_rrf.py`), độ lạc quan +0,0050 |
| Ngưỡng số lượng doc θ | 0,502793 | θ nhỏ nhất giữ recall trong ngân sách −0,003 (`scripts/fit_calibration.py`) |

## Đánh giá

Recall (chính) và Precision (phụ) theo đúng `scoring.py` của BTC (`src/evaluate.py`,
`tests/test_scoring.py`). Trên `dev` n=1.000:

| | R@5 | R@50 | BTC recall | BTC precision |
|---|---:|---:|---:|---:|
| BM25 một mình | 0,8547 | 0,9708 | — | — |
| Dense một mình | 0,9234 | 0,9813 | — | — |
| Hợp nhất RRF (luôn 5 doc) | 0,9384 | 0,9857 | 0,9384 | 0,1984 |
| **+ bộ quyết định số lượng doc** | | | **0,9354** | **0,2451** |

Bộ quyết định số lượng: ΔRecall −0,0030 (KTC95 cặp [−0,0065, −0,0005], 4/1.000 câu xấu đi),
precision ×1,24; 907/1.000 câu vẫn trả 5 doc.

## Hạn chế đã biết

- **Nhãn không đầy đủ**: cùng câu hỏi có gold khác nhau trong `train.json` (vd "Tham nhũng là gì?"
  → `211897` và `279667`). Recall thật cao hơn con số đo được.
- Chưa đo v0.8 trên `holdout`, nên chưa có ước lượng không thiên lệch cho bản này. Các bản trước
  cho thấy chênh dev → holdout khoảng −0,01 đến −0,02.
- Precision vẫn thấp (0,2451) vì phần lớn câu vẫn trả 5 doc — khe hở điểm RRF chỉ nhận một số ít
  giá trị rời rạc nên tín hiệu tự tin yếu.
- Kho chunk theo Điều không áp dụng được cho TCVN/QCVN (~8,7% văn bản) → cắt cửa sổ trượt.

## Biện pháp giảm thiểu rủi ro

- Chỉ truy hồi, không sinh văn bản → không có rủi ro bịa nội dung pháp lý.
- Kết quả là ID văn bản gốc, người dùng luôn truy được về nguồn.
- Không dùng API bên thứ ba; mọi trọng số tải về và chạy cục bộ, revision được pin.
