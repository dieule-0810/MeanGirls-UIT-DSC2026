# EDA Notes — chạy trước khi chunker/parser bị chỉnh lần cuối

> Sinh bởi `scripts/eda.py`. Điền thủ công phần nhận xét sau mỗi mục.

## 1. Phân bố độ dài văn bản

```json
{
  "n_docs": 8532,
  "words_mean": 8535.5,
  "words_median": 4813.0,
  "words_p90": 18406,
  "words_p99": 59251,
  "words_max": 1242409,
  "chars_mean": 41454.7
}
```

**Nhận xét:** Độ dài văn bản lớn và dữ liệu lệch phải nặng. Độ dài trung bình là 8.535 từ, gấp đôi trung vị (chỉ 4.813 từ). Có outlier bất thường lên tới 1.242.409 từ (gấp ~21 lần mốc p99 là 59k từ). 
* Chia nhỏ văn bản (chunking) là bắt buộc vì không mô hình nhỏ nào nuốt trọn được ngữ cảnh quá lớn. 
* Tuy nhiên, cần kiểm tra lại file siêu dài này xem có phải lỗi parser gộp nhầm dữ liệu hay không.

**Phương án tạm thời:** 
* P2 Tra cứu thủ công top outlier theo words_max để xác định lỗi parser hay dữ liệu thật.
* Bổ sung phân bố độ dài tính theo từng "Điều" (per-Điều) để có cơ sở chuẩn xác nhất cho câu hỏi chunk size 256 token.

## 2. Cấu trúc Điều N vs fallback

```json
{
  "pct_with_dieu": 91.3,
  "pct_needs_fallback": 8.7,
  "avg_dieu_per_doc_when_present": 37.6
}
```

**Nhận xét:** 91.3% văn bản chứa cấu trúc "Điều N" rõ ràng (trung bình 37.6 điều/văn bản)
* Tỷ lệ 8.7% còn lại cần fallback sang sliding-window chủ yếu thuộc về các thông tư, quy chuẩn (TCVN/QCVN).

**Phương án tạm thời:** 
* Ưu tiên cắt theo Điều N
* Thiết kế parser hỗ trợ cơ chế fallback sliding-window cho 8.7% văn bản phi cấu trúc này.

## 3. Trường thiếu (name/link/text)

```json
{
  "pct_missing_name": 13.2,
  "pct_missing_link": 0.0,
  "pct_missing_text": 0.2,
  "missing_name_files": [
    "context_100050.json",
    "context_100363.json",
    "context_100419.json",
    "context_100642.json",
    "context_101298.json",
    "context_101341.json",
    "context_101445.json",
    "context_101508.json",
    "context_101591.json",
    "context_101733.json",
    "context_102059.json",
    "context_102462.json",
    "context_102964.json",
    "context_103038.json",
    "context_10304.json",
    "context_103049.json",
    "context_103155.json",
    "context_104343.json",
    "context_104973.json",
    "context_105081.json"
  ]
}
```

**Nhận xét:** Có 13.2% tài liệu khuyết trường name (~1.126 file) và 0.2% khuyết trường text (passage rỗng, ~17 file). Không có file nào thiếu link
* Nếu code parse gọi thẳng d["name"] hoặc d["passage"] mà không có cơ chế an toàn, hệ thống sẽ crash ngay.

**Phương án tạm thời:** 
* Dùng .get("name", "Không có tiêu đề") và .get("passage", "").
* Đối chiếu 17 file rỗng text với nhãn gold trong train.json để quyết định loại bỏ khỏi corpus sạch, tránh nhiễu mô hình truy hồi.

## 4. Phân bố số đáp án / câu hỏi

```json
{
  "n_questions": 7000,
  "pct_exactly_1_answer": 92.1,
  "distribution": {
    "1": 6447,
    "2": 485,
    "3": 53,
    "4": 14,
    "5": 1
  },
  "n_empty_question_text": 0
}
```

**Nhận xét:** Có tới 92.1% câu hỏi chỉ có đúng 1 đáp án đúng, phần còn lại (khoảng 8%) rải rác từ 2 đến 5 đáp án. Không có câu hỏi nào bị rỗng text.
* Nếu luôn nộp đủ 5 tài liệu cho mọi câu hỏi, Recall lên tối đa nhưng Precision đo thực tế bằng scoring.py của BTC là 0.300 (theo scoring_behaviour.md). 
--> Việc luôn nộp đủ 5 doc làm sụt giảm Precision nghiêm trọng.

**Phương án tạm thời:** 
* Tận dụng bộ quyết định số lượng tài liệu (Calibration) ở Tuần 5 của P4 để tối ưu hóa Precision.

## 5. Trùng / gần trùng văn bản

```json
{
  "n_exact_duplicate_groups": 5,
  "example_groups": [
    [
      10533,
      131890,
      149317,
      177151,
      181693,
      187338,
      191261,
      196918,
      208668,
      210808,
      232489,
      255762,
      263763,
      288457,
      34810,
      55497,
      56098,
      57978,
      67660,
      71014
    ],
    [
      121575,
      84226
    ],
    [
      158189,
      184972,
      206810
    ],
    [
      254937,
      280171
    ],
    [
      277743,
      35337
    ]
  ],
  "note": "Đây chỉ là trùng CHÍNH XÁC sau normalize whitespace. Gần trùng (near-duplicate thật, khác vài từ) cần MinHash/SimHash nếu số exact-dup thấp mà vẫn nghi ngờ — chưa làm ở bản khung này."
}
```

**Nhận xét:** Có 5 nhóm trùng lặp tuyệt đối, trong đó có một nhóm khổng lồ chứa tới 20 file trùng nội dung hoàn toàn.
* Mã chấm của BTC dùng giao tập hợp theo doc_id chính xác, tạo ra nguy cơ nhiễu nhãn (Label Noise): retriever trả về đúng nội dung nhưng khác ID sẽ bị tính sai điểm.

**Phương án tạm thời:** Đối chiếu các cụm trùng với tập train.json để đánh dấu riêng.
* P3 loại các thành viên trong cùng cụm trùng ra khỏi tập negative khi chạy hard-negative mining để tránh việc mô hình bị dạy sai.
* P4 dùng nhãn lỗi riêng "trùng nội dung, sai ID" khi phân tích lỗi.

## 6. Độ dài câu hỏi

```json
{
  "words_mean": 19.8,
  "words_min": 4,
  "words_max": 50,
  "n_suspiciously_short": 0,
  "examples_suspiciously_short": []
}
```

**Nhận xét:** Độ dài trung bình của câu hỏi là 19.8 từ (từ 4 đến 50 từ)
* Không phát hiện câu hỏi ngắn bất thường hay rỗng text. Dữ liệu đầu vào ở phần này rất sạch sẽ.

**Phương án tạm thời:** 
* Không cần can thiệp thêm, giữ nguyên quy trình hiện tại.

## 7. Độ phủ doc_id (train vs corpus)

```json
{
  "n_corpus_ids": 8532,
  "n_train_referenced_ids": 3105,
  "n_orphan_ids_in_train": 0,
  "orphan_examples": [],
  "n_corpus_ids_never_answer_in_train": 5427
}
```

**Nhận xét:** Corpus có tổng cộng 8.532 tài liệu nhưng tập train chỉ tham chiếu tới 3.105 tài liệu
* Nghĩa là có tới 5.427 tài liệu (63.6%) chưa từng xuất hiện làm đáp án trong train. Không có id mồ côi (id trong train nhưng không có trong corpus).
* Chưa có kiểm chứng xem các tài liệu này có xuất hiện trong tập public-official.json hay không.

**Phương án tạm thời:** 
* Vẫn bắt buộc phải index đầy đủ toàn bộ 8.532 tài liệu trong hệ thống retrieval, tuyệt đối không cắt xén theo tập train. 
* Tuy nhiên,giả thuyết "5.427 tài liệu này là kho chứa đáp án cho test" mới chỉ là phỏng đoán kiến trúc, chưa được kiểm chứng trực tiếp qua EDA. P2 chạy thêm đối chiếu doc_id giữa public-official.json và corpus (như với train.json) để kiểm chứng