# Data Statement — DSC2026 Task 1 LegalIR

## Nguồn dữ liệu

**Chỉ dùng dữ liệu do BTC cung cấp.** Không thu thập ngoài, không gán nhãn thủ công, không áp dụng
data augmentation (Điều 4, Điều 9).

| Tệp | Nội dung | Cách dùng |
|---|---|---|
| `selected-contexts.zip` | 8.532 văn bản pháp luật (`context_*.json`) | Kho truy hồi (8.507 sau khi loại) |
| `train.json` | 7.000 câu hỏi + `document_id` đúng | Chia 4 tập, xem dưới |
| `public-official.json` | 1.000 câu hỏi | Public test |
| `private-official.json` | 2.080 câu hỏi | Private test |

Văn bản gốc từ thuvienphapluat.vn (trường `link`) — văn bản quy phạm pháp luật công khai.

> Mô hình pretrained **không** bị coi là "dữ liệu từ nguồn khác" (Q&A của BTC): đội thi dùng mô
> hình ước lượng, không trực tiếp dùng ngữ liệu huấn luyện của chúng.

## Đặc điểm dữ liệu (thống kê thực tế, sinh bằng `scripts/eda.py`)

- Độ dài câu hỏi: trung vị 19 tiếng, p95 31, max 52.
- Số đáp án/câu: 92,1% có **1** văn bản; 6,9% có 2; tối đa 5.
- 3.105/8.532 văn bản xuất hiện trong nhãn train (36%) — phần còn lại vẫn phải index.
- `doc_id` thưa, rải trên dải 69–306.081.

## Loại trừ (quyết định ghi trong `docs/exclusion_decisions.json`, sinh bởi `scripts/eda.py`)

- **25 văn bản** bị loại khỏi kho: 20 văn bản rỗng nội dung và 5 bản trùng-dư (giữ một bản trong
  mỗi cụm trùng) → 8.507 văn bản.
- **11 câu hỏi "vùng chết"** (gold trỏ vào văn bản rỗng) bị loại khỏi `train.json` → 6.989 câu.

## Chia tập (`src/data/split_data.py`, seed 42, một lần shuffle)

| Tập | Số câu | Vai trò |
|---|---:|---|
| `holdout` | 1.000 | Đo lần cuối — chạm đúng một lần |
| `dev` | 1.000 | Đo lúc thử nghiệm |
| `error_pool` | 300 | Đọc tay, phân tích lỗi |
| `train_split` | 4.689 | Chọn siêu tham số; bộ nhớ kNN (v0.6) |

Câu hỏi chạm cùng một cụm văn bản trùng được gom vào cùng một tập (Union-Find) để không rò rỉ qua
bản trùng. `train_1000 ⊂ train_2500 ⊂ train_4689` là các mốc learning-curve lồng nhau.

## Tiền xử lý (`src/data/parse_corpus.py`, `src/data/chunker.py`)

1. Ép `doc_id` về `str` ngay tại điểm đọc.
2. Chuẩn hoá Unicode NFC; kéo phẳng mọi khoảng trắng/xuống dòng (`\r\n\n`) thành một dấu cách.
3. Chặt đuôi rác crawler ("quý khách vui lòng đăng nhập", ...) từ vị trí xuất hiện trở đi.
4. `name` thiếu → `""`; `link` về chữ thường. Không bịa nội dung cho trường thiếu.
5. Chunk theo tiêu đề `Điều N.`; Điều dài hơn 256 từ thì cắt cửa sổ trượt 256 / gối 64 bên trong
   Điều, mảnh sau gắn nhãn `[Điều N. - tiếp theo]`; văn bản không có Điều thì cắt cửa sổ trượt.
   Bỏ nhãn đi thì mọi chunk là chuỗi con nguyên văn của văn bản gốc.

Đầu ra tất định, xác minh bằng số dòng và SHA-256 ghi trong `docs/reproduce.md` mục 1.

## Quyền riêng tư

Dữ liệu là văn bản quy phạm pháp luật công khai, không chứa thông tin cá nhân nhạy cảm. Không thực
hiện thao tác nào có thể tái định danh cá nhân.
