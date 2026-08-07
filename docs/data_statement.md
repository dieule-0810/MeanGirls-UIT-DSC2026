# Data Statement — DSC2026 Task 1 LegalIR

## Nguồn dữ liệu
**Chỉ dùng dữ liệu do BTC cung cấp.** Không thu thập ngoài, không gán nhãn thủ công,
không áp dụng data augmentation (Điều 4, Điều 9).

| Tệp | Nội dung | Cách dùng |
|---|---|---|
| `selected-contexts.zip` | 8.532 văn bản pháp luật (`context_*.json`) | Kho truy hồi |
| `train.json` | 7.000 câu hỏi + `document_id` đúng | 6.000 train / 1.000 held-out (seed 42) |
| `public-official.json` | 1.000 câu hỏi | Public test |

Văn bản gốc từ thuvienphapluat.vn (trường `link`) — văn bản quy phạm pháp luật công khai.

> Mô hình pretrained/LLM **không** bị coi là "dữ liệu từ nguồn khác" (Q&A của BTC):
> đội thi dùng mô hình ước lượng, không trực tiếp dùng ngữ liệu huấn luyện của chúng.

## Đặc điểm dữ liệu (thống kê thực tế)
- Độ dài câu hỏi: trung vị 19 tiếng, p95 31, max 52.
- Số đáp án/câu: 92,1% có **1** văn bản; 6,9% có 2; tối đa 5.
- 3.105/8.532 văn bản xuất hiện trong nhãn train (36%) — phần còn lại vẫn phải index.
- `doc_id` thưa, rải đều trên dải 69–306.081.

## Tiền xử lý
1. Chuẩn hoá Unicode NFC.
2. Gỡ artefact `\r\n` chèn giữa mệnh đề (vd `"tháng\r\n\n11 năm"` → `"tháng 11 năm"`).
3. Bỏ boilerplate quốc hiệu/tiêu ngữ và dòng phân cách.
4. Chunk theo `Điều N`, fallback sliding window 180 tiếng / overlap 45.

Đầu ra tất định, xác minh bằng SHA-256 ghi trong README.

## Quyền riêng tư
Dữ liệu là văn bản quy phạm pháp luật công khai, không chứa thông tin cá nhân nhạy cảm.
Không thực hiện thao tác nào có thể tái định danh cá nhân.
