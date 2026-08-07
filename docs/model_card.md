# Model Card — DSC2026 Task 1 LegalIR

> Điều 9 yêu cầu **mỗi bài nộp** kèm Model Card và Data Statement.
> Cập nhật ở MỖI release, không để đến cuối. Đây là điều kiện nghiệm thu release (Track B).

## Phiên bản
| | |
|---|---|
| Release | v0.1 |
| Commit | _(điền)_ |
| Ngày | _(điền)_ |

## Thành phần hệ thống và số tham số

⚠️ Ngân sách BTC: **tổng toàn pipeline < 4 tỷ tham số**, tính cả lớp embedding.
LoRA/quantization **không** làm giảm số tham số.

| Thành phần | Mô hình | Tham số | Đã BTC duyệt |
|---|---|---|---|
| Tầng 1 — lexical | BM25 Okapi (tự cài đặt) | 0 (không có tham số học được) | — |
| Tầng 1 — dense | _(v0.2)_ | | |
| Tầng 2 — reranker | _(v0.4)_ | | |
| **Tổng** | | **0** | |

## Mục đích và phạm vi
Truy hồi văn bản pháp luật tiếng Việt: nhận một câu hỏi, trả về tối đa 5 `document_id`.
Chỉ dùng cho mục đích nghiên cứu/giáo dục trong khuôn khổ UIT DSC 2026.

## Phương pháp huấn luyện
v0.1: không huấn luyện.

## Đánh giá
Recall (chính) và Precision (phụ) theo đúng `scoring.py` của BTC, đo trên held-out 1.000 câu
tách từ `train.json` với seed 42. Xem `docs/scoring_behaviour.md`.

## Hạn chế đã biết
- **Nhãn không đầy đủ**: cùng một câu hỏi trong `train.json` có gold doc khác nhau
  (vd "Tham nhũng là gì?" → `211897` và `279667`). Recall thực tế cao hơn con số đo được.
- BM25 thuần không xử lý được diễn đạt khác từ vựng (vocabulary mismatch).
- Chunker cắt theo `Điều` không áp dụng được cho nhóm TCVN/QCVN → dùng sliding window.

## Biện pháp giảm thiểu rủi ro
- Không sinh văn bản, chỉ truy hồi → không có rủi ro bịa nội dung pháp lý.
- Kết quả trả về là ID văn bản gốc, người dùng luôn truy được về nguồn.
- Không dùng API bên thứ ba; toàn bộ chạy cục bộ, kiểm soát được.
