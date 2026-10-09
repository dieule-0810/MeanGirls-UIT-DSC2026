# Bench tầng 1 — tokenizer × gộp chunk→doc

> Sinh bởi `scripts/bench_retrieval.py`. Phần **Nhận xét** điền tay — đó mới là thứ
> đi vào bài báo (plan.md mục 0.6: mỗi phương pháp phải nói rõ yếu ở đâu).

```json
{
  "exp_id": "v0.2_bm25_tok",
  "config": "configs/v0.2_bm25_tokenizer.yaml",
  "commit": "04d2167-dirty",
  "questions": "data/dev.json",
  "demo": false,
  "n_chunks": 524422,
  "n_questions": 1000,
  "top_k": 100,
  "bm25": {
    "k1": 1.5,
    "b": 0.75,
    "pool": "logsumexp",
    "pool_tau": 1.0,
    "tokenizer_opts": {
      "fold_tone": true,
      "min_len": 1
    },
    "min_df": 1,
    "candidate_chunks": 2000,
    "n_jobs": 8,
    "batch_size": 64
  },
  "ngay": "2026-09-11"
}
```

| tokenizer | pool | recall@5 | recall@20 | recall@50 | recall@100 | btc_recall@5 | btc_precision@5 | vocab | index_s | score_s |
|---|---|---|---|---|---|---|---|---|---|---|
| syllable_bigram | mean_top2 | 0.8555 | 0.9427 | 0.97 | 0.9778 | 0.8555 | 0.1792 | 1212503 | 34.7 | 23.6 |
| syllable_bigram | logsumexp | 0.8413 | 0.9422 | 0.969 | 0.9768 | 0.8413 | 0.176 | 1212503 | 34.7 | 23.6 |
| syllable_bigram | max | 0.8423 | 0.9412 | 0.9685 | 0.9768 | 0.8423 | 0.1762 | 1212503 | 34.7 | 23.6 |
| syllable_bigram | mean_top3 | 0.8557 | 0.9447 | 0.9685 | 0.9778 | 0.8557 | 0.1792 | 1212503 | 34.7 | 23.6 |
| underthesea | mean_top2 | 0.8219 | 0.9357 | 0.9647 | 0.9783 | 0.8219 | 0.172 | 262774 | 11.3 | 14.3 |
| pyvi | logsumexp | 0.8014 | 0.9318 | 0.9643 | 0.9783 | 0.8014 | 0.1676 | 148283 | 11.1 | 14.3 |
| underthesea | logsumexp | 0.8079 | 0.9384 | 0.9642 | 0.9798 | 0.8079 | 0.1692 | 262774 | 11.3 | 14.3 |
| underthesea | mean_top3 | 0.8211 | 0.9312 | 0.9635 | 0.9785 | 0.8211 | 0.1716 | 262774 | 11.3 | 14.3 |
| pyvi | mean_top2 | 0.8228 | 0.9362 | 0.963 | 0.9788 | 0.8228 | 0.1722 | 148283 | 11.1 | 14.3 |
| syllable_bigram | sum | 0.6483 | 0.9023 | 0.9622 | 0.9753 | 0.6483 | 0.1362 | 1212503 | 34.7 | 23.6 |
| pyvi | mean_top3 | 0.8166 | 0.9334 | 0.962 | 0.9778 | 0.8166 | 0.1706 | 148283 | 11.1 | 14.3 |
| pyvi | max | 0.7949 | 0.9303 | 0.9603 | 0.9773 | 0.7949 | 0.1664 | 148283 | 11.1 | 14.3 |
| underthesea | max | 0.8008 | 0.9334 | 0.9592 | 0.9778 | 0.8008 | 0.1678 | 262774 | 11.3 | 14.3 |
| regex | logsumexp | 0.8068 | 0.9273 | 0.9568 | 0.9717 | 0.8068 | 0.1686 | 93644 | 13.6 | 19.6 |
| whitespace | mean_top2 | 0.8144 | 0.9261 | 0.9565 | 0.9692 | 0.8144 | 0.1698 | 245195 | 13.5 | 19.9 |
| whitespace | logsumexp | 0.8083 | 0.9248 | 0.9563 | 0.9727 | 0.8083 | 0.169 | 245195 | 13.5 | 19.9 |
| regex | mean_top2 | 0.8126 | 0.9283 | 0.955 | 0.9687 | 0.8126 | 0.1692 | 93644 | 13.6 | 19.6 |
| regex | mean_top3 | 0.8126 | 0.9258 | 0.955 | 0.9697 | 0.8126 | 0.1692 | 93644 | 13.6 | 19.6 |
| whitespace | mean_top3 | 0.8103 | 0.9223 | 0.9545 | 0.9697 | 0.8103 | 0.1686 | 245195 | 13.5 | 19.9 |
| pyvi | sum | 0.6163 | 0.8774 | 0.9532 | 0.9768 | 0.6163 | 0.1296 | 148283 | 11.1 | 14.3 |
| whitespace | max | 0.7928 | 0.9203 | 0.9518 | 0.9703 | 0.7928 | 0.1656 | 245195 | 13.5 | 19.9 |
| underthesea | sum | 0.6133 | 0.8796 | 0.9513 | 0.9758 | 0.6133 | 0.1288 | 262774 | 11.3 | 14.3 |
| regex | max | 0.7863 | 0.9213 | 0.9503 | 0.9708 | 0.7863 | 0.1642 | 93644 | 13.6 | 19.6 |
| whitespace | sum | 0.587 | 0.8496 | 0.9468 | 0.9677 | 0.587 | 0.1228 | 245195 | 13.5 | 19.9 |
| regex | sum | 0.5835 | 0.8511 | 0.9447 | 0.9662 | 0.5835 | 0.1222 | 93644 | 13.6 | 19.6 |

**Sắp xếp theo recall@50 — KPI của P3.**

Cấu hình tốt nhất: `tokenizer=syllable_bigram`, `pool=mean_top2` → recall@50 = 0.9700, Precision@5 = 0.1792.

## Nhận xét (điền tay)

**Đối chứng trước khi đọc bảng.** Ô `regex` × `max` ra **R@5 = 0,7863 · R@50 = 0,9503**, trùng
tuyệt đối dòng `v0.1_pool_max` của P4. Vì `dev.json` ở máy này vừa được sinh lại bằng
`src.data.split_data`, con số trùng khít đó chứng minh tập dev tái tạo **giống hệt** bản P4 đã
đo — mọi so sánh dưới đây so trực tiếp được với số liệu của P4, không cần quy đổi.

**H1 — word-segment hơn âm tiết thuần? ĐÚNG, nhưng nhỏ.** Ở `mean_top2`, R@50: `underthesea`
0,9647 · `pyvi` 0,9630 · `regex` 0,9550 (+0,0097 / +0,0080). Đúng chiều giả thuyết: term ghép
hiếm hơn ⇒ IDF cao hơn ⇒ ít khớp nhiễu.

**H1b — bigram âm tiết lấy lại được bao nhiêu phần? BỊ BÁC BỎ, theo chiều NGƯỢC lại.** Giả
thuyết đặt ra là bigram *xấp xỉ* được word-segment. Thực tế nó **vượt cả hai segmenter thật**:

| | R@5 | R@50 | vocab | index |
|---|---:|---:|---:|---:|
| `syllable_bigram` + `mean_top2` | **0,8555** | **0,9700** | 1.212.503 | 34,7s |
| `underthesea` + `mean_top2` | 0,8219 | 0,9647 | 262.774 | 11,3s (+304s tách từ) |
| `pyvi` + `mean_top2` | 0,8228 | 0,9630 | 148.283 | 11,1s (+46s tách từ) |
| `regex` + `mean_top2` (v0.1 + fold_tone) | 0,8126 | 0,9550 | 93.644 | 13,6s |

Cơ chế đề xuất (cần đọc lỗi để xác nhận, chưa phải kết luận): segmenter **cam kết vào một cách
tách duy nhất**, tách sai là hỏng cả term — đúng mã lỗi `R-SEG` trong `error_taxonomy.md`. Bigram
âm tiết không cam kết: nó sinh **cả** âm tiết rời **lẫn** cặp liền kề, để IDF của BM25 tự quyết
định cụm nào mang thông tin. Đổi lại là vocab gấp 13 lần và nnz gấp 2,2 lần — nhưng index vẫn
34,7s và truy vấn 23,6 ms/câu, tức **rẻ hơn `underthesea` một bậc** (304s chỉ riêng tách từ).

**H2 — chiến lược gộp.** Xác nhận lại kết quả của P4 trên tokenizer mới: `mean_top2` ≈ `mean_top3`
> `logsumexp` ≈ `max` ≫ `sum`; `sum` sập ở mọi tokenizer (R@5 0,58–0,65) vì thưởng độ dài văn bản.
Nhưng điều đáng chú ý là **biên độ**: chênh lệch giữa các pooling tốt nhất/kém nhì chỉ ~0,005 R@50,
trong khi đổi tokenizer được **+0,0197**. Nhóm đã dồn công vào pooling; biến có đòn bẩy lớn hơn
nằm ở tầng tách từ.

**Câu `n_empty`:** 0/1.000 ở cả 5 tokenizer — không câu nào mất trắng vì không khớp term.

**Điểm yếu còn lại của phương pháp này.** BM25 + bigram vẫn là **đếm trùng token**: nó không đọc
được thực thể phân biệt (`R-ENTITY`), và mức tăng ở đây đến từ việc *giữ được cụm từ*, không phải
từ việc *hiểu nghĩa*. Trần hiện tại: R@50 = 0,9700 ⇒ 3,0% số câu không một reranker nào cứu được,
và đó là phần việc còn lại của P3 ở tầng dense.

**Việc tiếp theo để nâng Recall@50:**

1. **Quét `candidate_chunks`** cho tổ hợp thắng. `docs/candidate_cap_check.md` cho thấy cap=2000
   **không** tương đương `null` với `mean_topN` (mất tới 0,0167 R@50 ở `pyvi`+`mean_top3`). Toàn bộ
   bảng trên đo ở cap=2000, nên 0,9700 có thể vẫn chưa phải trần của chính cấu hình này.
2. **Dense zero-shot + RRF** — `syllable_bigram` là nền BM25 mạnh hơn hẳn để hợp nhất, và nó
   không kéo theo dependency nào.
3. Báo P4: ứng viên rerank nên sinh lại từ `syllable_bigram`, vì trần R@50 tăng từ 0,9568 lên
   0,9700 (dư địa reranker đổi từ 0,1501 thành 0,1145 — reranker có ít đất hơn nhưng xuất phát cao hơn).
