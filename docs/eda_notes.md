# EDA Notes - chạy trước khi chunker/parser bị chỉnh lần cuối

> Sinh bởi `scripts/eda.py`. Điền thủ công phần nhận xét sau mỗi mục.

Cập nhật ngày 20/8/2026: BTC đã xác nhận trên tập train đang có một số context có passage rỗng hoặc trùng, trên tập public test và private test đáp án không có hiện tượng context trùng hay rỗng, phương án xử lý sau cập nhật sẽ được đưa ra tùy mỗi mục

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

## 2. Cấu trúc Điều N vs fallback

```json
{
  "pct_with_dieu": 91.3,
  "pct_needs_fallback": 8.7,
  "avg_dieu_per_doc_when_present": 37.6
}
```

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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
  }
}
```

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

## 9. Kiểm chứng Gold ID trỏ vào file lỗi (rỗng + trùng)

```json
{
  "gold_missing_name_summary": {
    "count_questions": 543,
    "n_unique_docs_affected": 294,
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
  "gold_empty_passage_VUNG_CHET_summary": {
    "count_questions": 11,
    "n_unique_docs_affected": 6,
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
  },
  "gold_duplicate_passage_summary": {
    "n_groups": 4,
    "n_groups_needs_manual_review": 0,
    "n_ids_recommended_remove": 5,
    "groups": [
      {
        "group_ids": [
          "121575",
          "84226"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspxhttps://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspx",
          "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-dan-so-443680.aspx"
        ],
        "gold_members": {
          "84226": [
            "159914"
          ]
        },
        "recommended_keep": "84226",
        "recommended_remove": [
          "121575"
        ],
        "action": "Giữ ID đang là gold (84226).",
        "needs_manual_review": false
      },
      {
        "group_ids": [
          "158189",
          "184972",
          "206810"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-91-2016-NÐ-CP-quan-ly-hoa-chat-che-pham-diet-con-trung-diet-khuan-dung-gia-dung-y-te-315454.aspx",
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Decree-91-2016-ND-CP-management-insecticidal-germicidal-chemicals-preparations-household-medical-use-318036.aspx",
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-91-2016-N%C3%90-CP-quan-ly-hoa-chat-che-pham-diet-con-trung-diet-khuan-dung-gia-dung-y-te-315454.aspx"
        ],
        "gold_members": {
          "206810": [
            "22884"
          ]
        },
        "recommended_keep": "206810",
        "recommended_remove": [
          "158189",
          "184972"
        ],
        "action": "Giữ ID đang là gold (206810).",
        "needs_manual_review": false
      },
      {
        "group_ids": [
          "254937",
          "280171"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-1242-QD-BYT-2022-Tai-lieu-Phuc-hoi-chuc-nang-benh-co-lien-quan-sau-mac-COVID19-513657.aspx",
          "https://thuvienphapluat.vn/van-ban/the-thao-y-te/Quyet-dinh-1242-QD-BYT-2022-Tai-lieu-Phuc-hoi-chuc-nang-benh-lien-quan-sau-mac-COVID19-513657.aspx"
        ],
        "gold_members": {
          "254937": [
            "130058",
            "127798"
          ]
        },
        "recommended_keep": "254937",
        "recommended_remove": [
          "280171"
        ],
        "action": "Giữ ID đang là gold (254937).",
        "needs_manual_review": false
      },
      {
        "group_ids": [
          "277743",
          "35337"
        ],
        "links": [
          "https://thuvienphapluat.vn/van-ban/lao-dong-tien-luong/Nghi-dinh-38-2022-ND-CP-muc-luong-toi-thieu-nguoi-lao-dong-lam-viec-theo-hop-dong-515984.aspx",
          "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Nghi-dinh-muc-luong-toi-thieu-doi-voi-lao-dong-lam-viec-theo-hop-dong-lao-dong-515984.aspx"
        ],
        "gold_members": {},
        "recommended_keep": "35337",
        "recommended_remove": [
          "277743"
        ],
        "action": "Không thành viên nào là gold - tie-break, giữ ID nhỏ nhất.",
        "needs_manual_review": false
      }
    ]
  },
  "ket_luan": {
    "docs_recommended_exclude_from_corpus": [
      "10533",
      "121575",
      "131890",
      "149317",
      "158189",
      "177151",
      "181693",
      "184972",
      "187338",
      "191261",
      "196918",
      "208668",
      "210808",
      "232489",
      "255762",
      "263763",
      "277743",
      "280171",
      "288457",
      "34810",
      "55497",
      "56098",
      "57978",
      "67660",
      "71014"
    ],
    "n_docs_recommended_exclude_from_corpus": 25,
    "n_corpus_after_exclusion": 8507,
    "qids_exclude_unsolvable_vung_chet": [
      "116906",
      "13426",
      "163826",
      "17708",
      "28410",
      "39790",
      "42298",
      "49334",
      "5728",
      "60066",
      "93066"
    ],
    "n_qids_exclude_unsolvable": 11,
    "qids_need_gold_remap_not_exclude": [],
    "n_qids_need_gold_remap": 0,
    "qids_needs_manual_review": [],
    "n_qids_needs_manual_review": 0,
    "ready_to_apply": true
  }
}
```

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_

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

**Nhận xét:** _(điền)_

**Phương án tạm thời:** _(điền)_
