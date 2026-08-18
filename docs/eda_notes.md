# EDA Notes - chạy trước khi chunker/parser bị chỉnh lần cuối

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
  "chars_mean": 41454.7,
  "mean_to_median_ratio": 1.77,
  "max_to_p99_ratio": 21.0
}
```

**Nhận xét:** Độ dài văn bản lớn và dữ liệu lệch phải nặng. Độ dài trung bình là 8.535 từ, gấp 1.77 lần trung vị. Có outlier bất thường lên tới 1.242.409 từ (gấp ~21 lần mốc p99 là 59k từ). 
* Chia nhỏ văn bản (chunking) là bắt buộc vì không mô hình nhỏ nào nuốt trọn được ngữ cảnh quá lớn.
* Mục 11: outlier lớn nhất (id 68843) là dữ liệu thật (QCVN bảng biểu dài), không phải lỗi gộp văn bản. Riêng outlier thứ 2 (id 4644) phát hiện nhiễm nội dung rác từ giao diện website.

**Phương án tạm thời:**
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
* Mục 11: "có Điều" không đảm bảo cắt theo Điều ra chunk kích thước hợp lý; văn bản QCVN/TCVN bảng biểu lớn cần sliding-window lồng bên trong Điều quá dài, không chỉ áp dụng cho 8.7% fallback.

**Phương án tạm thời:** 
* Ưu tiên cắt theo Điều N
* Thiết kế parser hỗ trợ cơ chế fallback sliding-window cho 8.7% văn bản phi cấu trúc này.
* Dù có Điều (91.3%), nếu một Điều đơn lẻ vượt ngưỡng độ dài, vẫn áp sliding-window lồng bên trong đoạn đó, không chỉ dành riêng cho nhóm 8.7% fallback.

## 3. Trường thiếu (id/name/link/text)

```json
{
  "ID_missing_summary": {},
  "NAME_missing_summary": {
    "pct_missing": 13.2,
    "key_missing_count": 1125,
    "val_empty_count": 0,
    "eg": [
      "context_100050.json",
      "context_100363.json",
      "context_100419.json",
      "context_100642.json",
      "context_101298.json",
      "context_101341.json",
      "context_101445.json",
      "context_101508.json",
      "context_101591.json",
      "context_101733.json"
    ]
  },
  "LINK_missing_summary": {},
  "PASSAGE_missing_summary": {
    "pct_missing": 0.2,
    "key_missing_count": 0,
    "val_empty_count": 20,
    "eg": [
      "context_10533.json",
      "context_131890.json",
      "context_149317.json",
      "context_177151.json",
      "context_181693.json",
      "context_187338.json",
      "context_191261.json",
      "context_196918.json",
      "context_208668.json",
      "context_210808.json"
    ]
  },
  "EXTRA_keys_analysis": {}
}
```

**Nhận xét:** Có 13.2% tài liệu khuyết trường name (1.125 file) và 0.2% khuyết trường text (passage rỗng, 20 file).
* Nếu downstream code gọi trực tiếp doc["name"] hoặc doc["passage"] mà không có phòng ngự, hệ thống sẽ bị crash (sập) ngay lập tức.

**Phương án tạm thời:** 
* Bắt buộc dùng phương thức an toàn .get("name") và .get("passage", "") khi viết parser parse_corpus.py.

## 4. Phân bố số đáp án / câu hỏi

```json
{
  "n_questions": 7000,
  "pct_exactly_1_answer": 92.1,
  "pct_multi_answer": 7.9,
  "distribution": {
    "1": 6447,
    "2": 485,
    "3": 53,
    "4": 14,
    "5": 1
  },
  "n_empty_question_text": 0,
  "n_zero_answer": 0,
  "mean_gold_per_q": 1.091,
  "precision_ceiling_if_always_5": 0.218
}
```

**Nhận xét:** Có tới 92.1% câu hỏi chỉ có đúng 1 đáp án đúng, phần còn lại (khoảng 8%) rải rác từ 2 đến 5 đáp án. Không có câu hỏi nào bị rỗng text.
* Nếu luôn nộp đủ 5 tài liệu cho mọi câu hỏi, Recall đạt tối đa nhưng Precision bị chặn trên ở 0.218 (theo phân bố ngay phía trên). Đây là trần, chỉ đạt khi mọi gold lọt top-5; thực tế Recall@5 < 1 nên Precision còn thấp hơn. Con số 0.300 trong scoring_behaviour.md là hiện vật của fixture 2 câu dùng để dò hành vi scoring, không phải Precision của tập test - không dùng lẫn hai ngữ cảnh.
--> Việc luôn nộp đủ 5 doc làm sụt giảm Precision nghiêm trọng.

**Phương án tạm thời:** 
* Tận dụng bộ quyết định số lượng tài liệu (Calibration) ở Tuần 5 của P4 để tối ưu hóa Precision.

## 5. Trùng / gần trùng văn bản

```json
{
  "PASSAGE_duplicate_summary": {
    "n_groups": 4,
    "total_duplicate_files": 9,
    "eg": [
      {
        "ids": [
          "121575",
          "84226"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspxhttps://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspx",
          "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspx"
        ],
        "members_ever_gold": [
          "84226"
        ]
      },
      {
        "ids": [
          "158189",
          "184972",
          "206810"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-91-2016-NÐ-CP-quan-ly-hoa-chat-che-pham-diet-con-trung-diet-khuan-dung-gia-dung-y-te-315454.aspx",
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Decree-91-2016-ND-CP-management-insecticidal-germicidal-chemicals-preparations-household-medical-use-318036.aspx",
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-91-2016-N%C3%90-CP-quan-ly-hoa-chat-che-pham-diet-con-trung-diet-khuan-dung-gia-dung-y-te-315454.aspx"
        ],
        "members_ever_gold": [
          "206810"
        ]
      },
      {
        "ids": [
          "254937",
          "280171"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-1242-QD-BYT-2022-Tai-lieu-Phuc-hoi-chuc-nang-benh-co-lien-quan-sau-mac-COVID19-513657.aspx",
          "https://thuvienphapluat.vn/van-ban/the-thao-y-te/Quyet-dinh-1242-QD-BYT-2022-Tai-lieu-Phuc-hoi-chuc-nang-benh-lien-quan-sau-mac-COVID19-513657.aspx"
        ],
        "members_ever_gold": [
          "254937"
        ]
      },
      {
        "ids": [
          "277743",
          "35337"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/lao-dong-tien-luong/Nghi-dinh-38-2022-ND-CP-muc-luong-toi-thieu-nguoi-lao-dong-lam-viec-theo-hop-dong-515984.aspx",
          "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Nghi-dinh-muc-luong-toi-thieu-doi-voi-lao-dong-lam-viec-theo-hop-dong-lao-dong-515984.aspx"
        ],
        "members_ever_gold": []
      }
    ]
  },
  "LINK_duplicate_summary": {},
  "Join_1": {
    "total_multi_answer_questions": 553,
    "count": 0,
    "eg": []
  },
  "Join_2": {
    "total_questions_at_risk": 4,
    "pct_of_total_train": 0.06,
    "distribution_by_answer_count": {
      "1": 2,
      "2": 2,
      "3": 0,
      "4": 0,
      "5": 0
    },
    "eg": [
      {
        "qid": "130058",
        "answer_count": 1,
        "question": "Nguyên nhân khó thở sau mắc COVID-19 là gì? Kiểm soát khó th...",
        "answers_in_train": [
          "254937"
        ],
        "duplicate_missing_from_gold": [
          "280171"
        ]
      },
      {
        "qid": "22884",
        "answer_count": 2,
        "question": "Yêu cầu đối với cơ sở vật chất phòng thử nghiệm chế phẩm diệ...",
        "answers_in_train": [
          "270233",
          "206810"
        ],
        "duplicate_missing_from_gold": [
          "158189",
          "184972"
        ]
      },
      {
        "qid": "159914",
        "answer_count": 2,
        "question": "Đề xuất người chưa thành niên được cung cấp dịch vụ thân thi...",
        "answers_in_train": [
          "81598",
          "84226"
        ],
        "duplicate_missing_from_gold": [
          "121575"
        ]
      },
      {
        "qid": "127798",
        "answer_count": 1,
        "question": "Bài tập cơ tay, cơ chân được hướng dẫn như thế nào?...",
        "answers_in_train": [
          "254937"
        ],
        "duplicate_missing_from_gold": [
          "280171"
        ]
      }
    ]
  },
  "dup_passage_groups": [
    [
      "121575",
      "84226"
    ],
    [
      "158189",
      "184972",
      "206810"
    ],
    [
      "254937",
      "280171"
    ],
    [
      "277743",
      "35337"
    ]
  ]
}
```

**Nhận xét:** Kho dữ liệu tồn tại 4 nhóm trùng lặp nội dung thực tế (9 file).
* Join #1 (đa đáp án): Trùng khớp 0 câu.
* Join #2 (Rủi ro Label Noise): Chỉ có đúng 4 câu hỏi thực sự nằm trong vùng rủi ro (2 câu 1-answer, 2 câu 2-answer). Mô hình trả về file trùng nội dung nhưng khác ID sẽ bị tính 0 điểm im lặng.
* Có hai URL dính liền nhau, không có dấu phân cách, trong khi 84226 thì link sạch bình thường. Lỗi nằm ngay trong dữ liệu gốc context_121575.json (có thể lỗi crawler nối chuỗi 2 lần khi lưu). 
* 2/3 nhóm còn lại (254937/280171, 277743/35337) chỉ khác nhau ở cách viết hoa URL (The-thao-Y-te vs the-thao-y-te) (nhiều khả năng do server không phân biệt hoa/thường nên bị crawl trùng 2 lần thành 2 "văn bản" khác id). 
* Riêng nhóm 158189/184972/206810: 1 link tiếng Anh (Decree-91-2016...), nội dung passage vẫn y hệt bản tiếng Việt, trang EN chưa dịch, crawler lấy nhầm nội dung gốc.

**Phương án tạm thời:**
* P2: Giữ nguyên đủ 8.532 dòng
* P3: Loại bỏ các thành viên trong cùng cụm trùng ra khỏi tập negative của chunk gold khi chạy hard-negative mining để tránh dạy sai mô hình.
* P4: Sử dụng nhãn lỗi riêng "trùng nội dung, sai ID" khi chạy khâu phân tích lỗi (Error Analysis) hàng tuần. Không can thiệp vào logic sinh file nộp (make_submission.py).
* P2: Chuẩn hoá URL về lowercase trước khi coi là nguồn riêng; cụm song ngữ vẫn coi là duplicate ở tầng retrieval.
* 20 file rỗng passage không được tính là nhóm trùng (đã xử lý riêng ở mục 3/8); phép so trùng chỉ chạy trên passage khác rỗng.

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
* Không cần can thiệp thêm, giữ nguyên quy trình hiện tại

## 7. Độ phủ doc_id (train vs corpus)

```json
{
  "n_corpus_ids": 8532,
  "n_train_referenced_ids": 3105,
  "n_orphan_ids_in_train": 0,
  "orphan_examples": [],
  "n_corpus_ids_never_answer_in_train": 5427,
  "pct_corpus_never_answer_in_train": 63.6
}
```

**Nhận xét:** Corpus có tổng cộng 8.532 tài liệu nhưng tập train chỉ tham chiếu tới 3.105 tài liệu
* Nghĩa là có tới 5.427 tài liệu (63.6%) chưa từng xuất hiện làm đáp án trong train. Không có id mồ côi (id trong train nhưng không có trong corpus).
* Chưa có kiểm chứng xem các tài liệu này có xuất hiện trong tập public-official.json hay không.

**Phương án tạm thời:** 
* Vẫn bắt buộc phải index đầy đủ toàn bộ 8.532 tài liệu trong hệ thống retrieval, tuyệt đối không cắt xén theo tập train. 
* Giả thuyết '5.427 tài liệu chưa dùng là nguồn đáp án cho test' không thể kiểm chứng trực tiếp bằng doc_id (xem mục 10 - public-official.json ẩn nhãn hoàn toàn). Theo dõi gián tiếp qua chênh lệch Recall held-out vs leaderboard (ngưỡng <3%)

## 8. Mối quan hệ lỗi (3.1)

```json
{
  "total_missing_name_keys": 1125,
  "total_empty_passages": 20,
  "both_errors_overlap_count": 20,
  "overlap_percentage_of_empty_passage": 100.0,
  "eg_overlap_files": [
    "context_10533.json",
    "context_131890.json",
    "context_149317.json",
    "context_177151.json",
    "context_181693.json",
    "context_187338.json",
    "context_191261.json",
    "context_196918.json",
    "context_208668.json",
    "context_210808.json"
  ]
}
```

**Nhận xét:** Số file khuyết name là 1.125, số file rỗng passage là 20. 
* Phép giao cho thấy đúng 20 file bị cả hai lỗi. Toàn bộ file rỗng text thực chất nằm gọn hoàn toàn bên trong nhóm khuyết name. 
* Đây là các "file siêu lỗi" do hệ thống cào dữ liệu bị hỏng cả cấu trúc lẫn nội dung.

**Phương án tạm thời:**
* P2: bù đắp cấu trúc (null/"") để đảm bảo đúng 8.532 dòng, xử lý "khuyết" ở tầng index (P3 skip khi index()), không đụng vào nội dung.

## 9. Kiểm chứng Gold ID trỏ vào file lỗi

```json
{
  "gold_missing_name_summary": {
    "count_questions": 543,
    "pct_of_train_questions": 7.76,
    "n_unique_docs_affected": 294,
    "pct_of_error_set_that_matters": 26.1,
    "top_offenders": [
      [
        "23402",
        65
      ],
      [
        "132545",
        15
      ],
      [
        "299574",
        14
      ],
      [
        "285041",
        14
      ],
      [
        "161768",
        14
      ]
    ],
    "top_offender_pct_of_cases": 12.0,
    "total_docs_with_missing_name": 1125,
    "cases": [
      {
        "qid": "61600",
        "gold_id": "77960",
        "question": "Tần suất thử nghiệm định kỳ các thông số chất lượng nước sạch sử dụng cho mục đí..."
      },
      {
        "qid": "153714",
        "gold_id": "166505",
        "question": "Mức lương của giảng viên cao cấp (tiến sỹ) hiện nay là bao nhiêu?"
      },
      {
        "qid": "32282",
        "gold_id": "79898",
        "question": "Việc chẩn đoán bệnh giun xoắn ở lợn bằng phương pháp ép cơ thì cần thực hiện ra ..."
      },
      {
        "qid": "32928",
        "gold_id": "218610",
        "question": "Muốn tìm hiểu thông tin về biện pháp bảo đảm thì thực hiện như thế nào?"
      },
      {
        "qid": "86284",
        "gold_id": "132545",
        "question": "Thủ tục đổi lại bằng lái xe B2 như thế nào? "
      },
      {
        "qid": "120914",
        "gold_id": "151666",
        "question": "Thực nghiệm chữa cháy bằng bình chữa cháy như thế nào?"
      },
      {
        "qid": "124514",
        "gold_id": "237238",
        "question": "Cách lấy mẫu sản phẩm xử lý môi trường nuôi trồng thủy sản ra sao?"
      },
      {
        "qid": "15338",
        "gold_id": "96507",
        "question": "Chu kỳ đánh giá tổ chức kiểm định chất lượng giáo dục trong nước là bao lâu?"
      },
      {
        "qid": "137840",
        "gold_id": "295664",
        "question": "Ai được xem là người phụ thuộc để tính giảm trừ gia cảnh?"
      },
      {
        "qid": "91878",
        "gold_id": "23402",
        "question": "Điều kiện hưởng chế độ ốm đau theo quy định pháp luật"
      },
      {
        "qid": "81636",
        "gold_id": "61357",
        "question": "Cách ghi thành phần dinh dưỡng, giá trị dinh dưỡng "
      },
      {
        "qid": "90028",
        "gold_id": "242143",
        "question": "Xem xét ban hành Nghị quyết mở rộng diện áp dụng miễn thị thực đơn phương theo y..."
      },
      {
        "qid": "104760",
        "gold_id": "155461",
        "question": "Bệnh gan tụy do Parvovirus ở tôm thường được chẩn đoán bằng những loại thuốc thử..."
      },
      {
        "qid": "54564",
        "gold_id": "176459",
        "question": "Thủ tục mua chứng từ khấu trừ thuế TNCN được thực hiện thế nào?"
      },
      {
        "qid": "155930",
        "gold_id": "23402",
        "question": "Mức trợ cấp đối với lao động nam đóng bảo hiểm xã hội khi vợ sinh con là bao nhi..."
      },
      {
        "qid": "3380",
        "gold_id": "290149",
        "question": "Những trường hợp nào phải tổ chức Hội đồng an toàn, vệ sinh lao động cơ sở?"
      },
      {
        "qid": "82348",
        "gold_id": "290149",
        "question": "Hồ sơ vụ tai nạn lao động gồm những giấy tờ gì?"
      },
      {
        "qid": "155810",
        "gold_id": "63721",
        "question": "Công tác chuẩn bị trong thi công công trình xây dựng phải căn cứ vào đâu?"
      },
      {
        "qid": "163306",
        "gold_id": "299574",
        "question": "Thời hiệu kỷ luật tổ chức đảng khi phải áp dụng hình thức cảnh cáo là bao lâu?"
      },
      {
        "qid": "36144",
        "gold_id": "136441",
        "question": "Thẻ an toàn nhóm 3 do ai cấp? Doanh nghiệp có thể tự cấp Thẻ an toàn nhóm 3 cho ..."
      }
    ]
  },
  "gold_empty_passage_VÙNG_CHẾT_summary": {
    "count_questions": 11,
    "pct_of_train_questions": 0.16,
    "n_unique_docs_affected": 6,
    "pct_of_error_set_that_matters": 30.0,
    "top_offenders": [
      [
        "288457",
        3
      ],
      [
        "263763",
        2
      ],
      [
        "131890",
        2
      ],
      [
        "149317",
        2
      ],
      [
        "55497",
        1
      ]
    ],
    "top_offender_pct_of_cases": 27.3,
    "total_docs_with_empty_passage": 20,
    "cases": [
      {
        "qid": "116906",
        "gold_id": "55497",
        "question": "Sản phẩm thuốc lá có phải công bố hợp quy hay không?"
      },
      {
        "qid": "28410",
        "gold_id": "263763",
        "question": "Thực hiện nổ mìn công nghiệp thăm dò địa chấn trên sông, biển như thế nào?"
      },
      {
        "qid": "42298",
        "gold_id": "288457",
        "question": "Khái niệm bảo hiểm y tế theo Luật Bảo hiểm y tế 2008"
      },
      {
        "qid": "163826",
        "gold_id": "131890",
        "question": "Cần dùng những thiết bị và dụng cụ nào để chẩn đoán bệnh viêm phế quản truyền nh..."
      },
      {
        "qid": "93066",
        "gold_id": "263763",
        "question": "Hủy vật liệu cháy nổ công nghiệp bằng cách làm mất tác dụng được quy định như th..."
      },
      {
        "qid": "49334",
        "gold_id": "149317",
        "question": "Điều kiện để doanh nghiệp được lập mã truy vết tài sản là gì? Có thể sử dụng mã ..."
      },
      {
        "qid": "39790",
        "gold_id": "10533",
        "question": "Đơn vị lắp đặt ngoài việc có đủ điều kiện trang thiết bị kỹ thuật phục vụ cho cô..."
      },
      {
        "qid": "5728",
        "gold_id": "288457",
        "question": "Phương thức chi trả khám bệnh, chữa bệnh bảo hiểm y tế theo Dự thảo Luật Bảo hiể..."
      },
      {
        "qid": "13426",
        "gold_id": "131890",
        "question": "Máu gà dùng cho việc chẩn đoán bệnh viêm phế quản truyền nhiễm cần được bảo quản..."
      },
      {
        "qid": "17708",
        "gold_id": "149317",
        "question": "Thực hiện gán mã truy vết vật phẩm cho sản phẩm như thế nào?"
      },
      {
        "qid": "60066",
        "gold_id": "288457",
        "question": "Nguyên tắc về bảo hiểm y tế tại Dự thảo Luật Bảo hiểm y tế ngày 15/02/2022"
      }
    ]
  }
}
```

**Nhận xét:** Phát hiện 543 câu hỏi trỏ đáp án vào file khuyết name và đặc biệt có 11 câu hỏi (0.16%) trỏ đáp án đúng vào các file rỗng passage (như file context_10533.json). 
* 11 câu này thuộc "Vùng Chết Retrieval" (Unsolvable Questions) vì không chứa bất kỳ chữ nào để mô hình so khớp tìm kiếm.
* Name: Chỉ 294/1125 (26.1%) file khuyết name từng thực sự được hỏi tới - phần lớn (73.9%) là lỗi vô hại về hiển thị. Đáng chú ý: một mình gold_id 23402 xuất hiện 65 lần trong 543 case - chiếm ~12% tổng số lần khuyết-name-là-gold, mức tập trung bất thường
* Passage: chỉ 6/20 (30.0%) file từng là gold, top offender 288457 gây ảnh hưởng 3 câu

**Phương án tạm thời:**
* P2: Sử dụng hàm trích xuất động extract_vung_chet_qids để tự động lọc bỏ hoàn toàn 11 câu hỏi vùng chết này trước khi chia tập train_split.json và holdout.json nhằm tránh gây nhiễu khi train và đảm bảo công bằng cho tập held-out.
* Ghi lại danh sách 11 qid bị loại vào docs/excluded_questions.md hoặc cột riêng trong experiments.csv, vì việc này làm n_questions đổi từ 7000 -> 6989 xuyên suốt mọi báo cáo sau này - để người dry-run sau thấy số liệu lệch và hiểu
* Loại khỏi holdout giả định tập test của BTC không chứa câu trỏ vào doc rỗng/hỏng tương tự; nếu test có mà holdout đã bỏ, held-out sẽ lạc quan hơn leaderboard một cách hệ thống - theo dõi như một nguồn của gap, nhưng đóng góp bị chặn ở 0.16% nên không thể một mình giải thích chênh lệch vượt ngưỡng - vẫn cần soi các nguyên nhân lớn hơn.

## 10. Đối chiếu Train vs Public

```json
{
  "n_train_qids": 7000,
  "n_public_qids": 1000,
  "n_overlap_qids": 0,
  "overlap_qids_sample": [],
  "interpretation": "0 overlap KHÔNG bác bỏ giả thuyết 5.427-tài-liệu - chỉ nghĩa là train/public dùng không gian qid tách biệt, không thể kiểm chứng bằng cách này.",
  "overlap_pointing_to_broken_doc": {
    "count": 0,
    "cases": []
  }
}
```

**Nhận xét:** Phép đối chiếu cho thấy giữa hai tập dữ liệu có đúng 0 mã câu hỏi trùng lặp (n_overlap_qids = 0), xác nhận không gian câu hỏi của hai tập hoàn toàn tách biệt. 
* Do tập dữ liệu Public chính thức không đi kèm nhãn đáp án ("answer": null), team hoàn toàn không thể thực hiện đối chiếu chéo cấp độ ID tài liệu (doc_id) để kiểm chứng trực tiếp giả thuyết 5.427 tài liệu chưa dùng qua dữ liệu tĩnh.
* Việc kiểm chứng trực tiếp giả thuyết 5.427 tài liệu là bất khả thi bằng phương pháp tĩnh do BTC ẩn nhãn.

**Phương án hành động:**
* Quan sát chênh lệch điểm số Recall thực tế: đối chiếu trực tiếp giữa điểm số trên tập kiểm thử nội bộ (Held-out Recall) và điểm số chạy thực tế trên Leaderboard công khai (Public Leaderboard Recall). 
* Nếu độ lệch này được giữ ở mức nhỏ hơn 3%, giả thuyết hệ thống hoạt động ổn định trên toàn bộ không gian 8.532 tài liệu sẽ được xác minh.

## 11. Chi tiết outlier độ dài (Tra cứu thủ công)

```json
{
  "top_n": 5,
  "outliers": [
    {
      "id": "68843",
      "source_file": "context_68843.json",
      "n_words": 1242409,
      "has_dieu": true,
      "n_dieu": 37,
      "words_per_dieu_gap": 34511.4,
      "link": "https://thuvienphapluat.vn/tcvn/Dien-dien-tu/QCVN-118-2018-BTTTT-tuong-thich-dien-tu-thiet-bi-da-phuong-tien-Yeu-cau-phat-xa-919645.aspx",
      "text_head": "\n\n\n\n1. QUY\r\n\nĐỊNH CHUNG.. 51.1. Phạm vi điều chỉnh.. 51.2. Đối tượng áp dụng. 51.3. Tài liệu viện dẫn.. 51.4. Giải thích từ ngữ.. 61.5. Chữ viết tắt 101.6. Phân loại thiết bị 122. QUY ĐỊNH KỸ THUẬT. 1",
      "text_tail": "Electrical and Electronic Equipment in the Range of\r\n\n9 kHz to 40 GHz.\n\n\n\n...\n\n...\n\n...\n\n\n\nCISPR 32:2015/COR1:2016: “Electromagnetic\r\n\ncompatibility of multimedia equipment - Emission requirements”.\n\n"
    },
    {
      "id": "4644",
      "source_file": "context_4644.json",
      "n_words": 571358,
      "has_dieu": true,
      "n_dieu": 203,
      "words_per_dieu_gap": 2828.5,
      "link": "https://thuvienphapluat.vn/tcvn/Giao-thong/QCVN-22-2018-BGTVT-Che-tao-kiem-tra-phuong-tien-thiet-bi-xep-do-918111.aspx",
      "text_head": "\n\nĐộ sâu hào, hố (m)\n\nKhoảng cách cho\r\n\n  phép nhỏ nhất đối với các loại đất (m)\n\nCát sỏi\n\nÁ cát\n\nÁ sét\n\nSét\n\nHoàng thổ\n\n1\n\n1,5\n\n1,25\n\n1\n\n1\n\n1\n\n2\n\n3\n\n2,4\n\n2\n\n1,5\n\n2\n\n3\n\n4\n\n3,6\n\n3,25\n\n1,75\n\n2,5\n\n4\n\n5\n\n",
      "text_tail": "iều thiết bị khác nhau, Quý Khách có thể vào đây để xem chi tiết lịch sử đăng nhập\n\nCó thể tài khoản của bạn đã bị rò rỉ mật khẩu và mất bảo mật, xin vui lòng đổi mật khẩu tại đây để tiếp tục sử dụng "
    },
    {
      "id": "42223",
      "source_file": "context_42223.json",
      "n_words": 237110,
      "has_dieu": true,
      "n_dieu": 143,
      "words_per_dieu_gap": 1669.8,
      "link": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-200-2014-TT-BTC-huong-dan-Che-do-ke-toan-Doanh-nghiep-263599.aspx",
      "text_head": "BỘ TÀI CHÍNH\r\n\n  --------\n\nCỘNG\r\n\n  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\r\n\n  Độc lập - Tự do - Hạnh phúc \r\n\n  ---------------\n\nSố: 200/2014/TT-BTC\n\nHà Nội, ngày 22 tháng 12 năm 2014\n\n \n\nTHÔNG TƯ\n\nHƯỚNG DẪN C",
      "text_tail": "hỉ đạo TW về phòng, chống tham nhũng;\r\n\n  - Website Chính phủ; Website Bộ Tài chính;\r\n\n  - Lưu: VT, Vụ CĐKT.\n\nKT.\r\n\n  BỘ TRƯỞNG\r\n\n  THỨ TRƯỞNG\n\n\r\n\n  Trần Xuân Hà\n\n \n\nFILE ĐƯỢC ĐÍNH KÈM THEO VĂN BẢN\n\n "
    },
    {
      "id": "164898",
      "source_file": "context_164898.json",
      "n_words": 189589,
      "has_dieu": true,
      "n_dieu": 5,
      "words_per_dieu_gap": 47397.2,
      "link": "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-1832-QD-BYT-2022-tai-lieu-huong-dan-chan-doan-dieu-tri-benh-ly-huyet-hoc-520596.aspx",
      "text_head": "BỘ Y TẾ\r\n\n  -------\n\nCỘNG HÒA\r\n\n  XÃ HỘI CHỦ NGHĨA VIỆT NAM\r\n\n  Độc lập - Tự do - Hạnh phúc \r\n\n  ---------------\n\nSố:\r\n\n  1832/QĐ-BYT\n\nHà Nội,\r\n\n  ngày 01 tháng 7 năm 2022\n\n\n\nQUYẾT ĐỊNH\n\nVỀ VIỆC BAN H",
      "text_tail": "Gene ABO basic của hãng Inno-Train, Đức.\n\n8. Hướng\r\n\ndẫn sử dụng bộ kít RBC-FluoGene VERYfy của hãng Inno-Train, Đức.\n\n9. Hướng\r\n\ndẫn sử dụng bộ kít RBC-FluoGene D screen của hãng Inno-Train, Đức.\n\n\n\n"
    },
    {
      "id": "12964",
      "source_file": "context_12964.json",
      "n_words": 164588,
      "has_dieu": true,
      "n_dieu": 14,
      "words_per_dieu_gap": 12660.6,
      "link": "https://thuvienphapluat.vn/tcvn/Giao-thong/QCVN-42-2015-BGTVT-trang-bi-an-toan-tau-bien-916747.aspx",
      "text_head": "\n\nTT\n\nTên\r\n\n  thiết bị\n\nKiểm\r\n\n  tra tàu\n\nHàng\r\n\n  năm lần 1\n\nHàng\r\n\n  năm lần 2\n\nHàng\r\n\n  năm lần 3\n\nHàng\r\n\n  năm lần 4\n\nĐịnh\r\n\n  kỳ\n\n(1)\n\n(2)\n\n(3)\n\n(4)\n\n(5)\n\n(6)\n\n(7)\n\n1\n\nThiết bị cứu sinh\n\n\n\n\n\n\n\n\n\n",
      "text_tail": " âm\n\n3 dB\n\nVùng biển A3 là vùng không\r\n\nthuộc vùng biển A1 hoặc A2 trong phạm vi góc nâng của vệ tinh INMARSAT 5 độ hoặc\r\n\nlớn hơn.\n\nVùng biển A4 là vùng biển\r\n\nkhông thuộc vùng biển A1, A2 hoặc A3.\n\n"
    }
  ],
  "note": "text_head/text_tail để soi nhanh dấu hiệu lỗi parser (nội dung lặp, nhiều văn bản dính liền) mà không cần mở file gốc. has_dieu/n_dieu để biết outlier rơi vào nhóm nào ở mục 2 (ảnh hưởng ước lượng sliding-window cho 8.7% văn bản fallback)."
}
```

**Nhận xét:** Top 5 outlier đều has_dieu: true - tức đều rơi vào nhóm 91.3% "có cấu trúc Điều" (mục 2), không phải nhóm 8.7% cần fallback như nghi ngờ ban đầu ở mục 1. Tuy nhiên mức độ tin cậy của từng file khác nhau rõ rệt sau khi soi text_head/text_tail. 
(Lưu ý) đây là top 5 văn bản dài nhất toàn corpus theo n_words. Không loại trừ khả năng còn văn bản khác ngoài top 5 này cũng bị nhiễm rác tương tự id 4644
* id 68843 (1.242.409 từ, lớn nhất): nhiều khả năng là dữ liệu thật. text_head có dấu hiệu lỗi định dạng nhỏ ở mục lục, nhưng không có bằng chứng gộp nhầm 2 văn bản.
* id 4644 (571.358 từ): phát hiện nghiêm trọng - text_tail chứa nguyên văn thông báo bảo mật của chính website nguồn, không phải nội dung QCVN 22-2018. Đây là ô nhiễm dữ liệu do crawler lấy nhầm phần giao diện/popup tài khoản lẫn vào nội dung văn bản.
* id 42223, 164898, 12964: text_head/text_tail đều khớp hợp lý với link tương ứng (chữ ký "BỘ TRƯỞNG", danh sách bộ kit xét nghiệm, bảng vùng biển hàng hải) - chưa thấy dấu hiệu bất thường, nhưng mới chỉ soi 200 ký tự đầu/cuối, chưa xác nhận toàn văn.
* Ngoài vấn đề đúng/sai dữ liệu, ~34511 từ cho mỗi khoảng giữa 2 Điều ở file 68843 cho thấy: có cấu trúc Điều không đảm bảo cắt theo Điều sẽ ra chunk kích thước hợp lý - với văn bản QCVN/TCVN nhiều bảng biểu, một Điều đơn lẻ vẫn có thể dài tới hàng chục nghìn từ.

**Phương án tạm thời:**
* P2: Kiểm tra chay context_4644.json trước khi đưa vào corpus_clean.jsonl; grep thử các cụm từ đặc trưng ("đăng nhập", "rò rỉ mật khẩu", "Quý Khách") trên toàn corpus để biết đây là lỗi cá biệt hay lỗi crawler lặp lại ở nhiều file khác.
* P2: Lưu ý lỗi định dạng mục lục nhỏ của id 68843; giữ nguyên trong corpus.
* P2: Rà chay toàn văn 3 file còn lại (42223, 164898, 12964) - xác nhận cả 3 file đều sạch sẽ(không phải lỗi parser gộp file hay lỗi crawler). Độ dài lớn hoàn toàn do đặc thù văn bản gốc

## 12. Kiểm tra ô nhiễm dữ liệu do lỗi Crawler

```json
{
  "total_infected_files": 239,
  "percentage_infected": 2.8,
  "eg_first_10": [
    "context_100139.json",
    "context_101375.json",
    "context_103064.json",
    "context_104500.json",
    "context_105190.json",
    "context_106740.json",
    "context_107059.json",
    "context_107707.json",
    "context_110894.json",
    "context_113090.json"
  ]
}
```

**Nhận xét:** Có 239 ứng viên nghi vấn (~2.8% corpus) dính từ khóa rộng.
* Kiểm tra trực tiếp: gần như toàn bộ là False Positive - dùng "đăng nhập" đúng ngữ cảnh hành chính hợp pháp (cổng dịch vụ công, TABMIS, định danh điện tử...). KHÔNG coi các văn bản này là nhiễu BM25.
* Số file thực sự ô nhiễm: chỉ 1 file - ID 4644, chứa chuỗi rác "bị rò rỉ mật khẩu và mất bảo mật...". Chỉ file này mới có rủi ro mất dữ liệu đuôi văn bản và gây term pollution thật cho BM25.

**Phương án tạm thời:**
* P2: Bổ sung Regex lọc chuỗi vào clean_text (parse_corpus.py). Xác nhận sạch qua verify tự động + scan rộng độc lập.
* Cập nhật: Đã chạy scan rộng không giới hạn độ dài lần cuối trước v0.1 (333 ứng viên, 0/333 khớp pattern rác - xem docs/final_pollution_scan_v0.1.txt)
* P4: Riêng ID 4644 - passage đã bị cắt cụt phần đuôi (do clean_text xóa từ vị trí "rò rỉ mật khẩu" trở đi). Nếu case lỗi BM25 rơi vào ID này, không phải bug retrieval mà là hệ quả chủ động của bước làm sạch.
* Giới hạn đã biết: "Sạch 100%" ở corpus_clean.jsonl chỉ nghĩa là sạch-theo-5-pattern-đã-biết (xem CRAWLER_JUNK_PATTERNS trong parse_corpus.py). Nếu BM25 error analysis gặp case lạ dính rác web (đăng nhập/mật khẩu/quý khách) không thuộc 5 cụm này, đó là bằng chứng cần bổ sung pattern mới, không phải bug ở tầng retrieval.
