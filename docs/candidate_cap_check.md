# Kiểm tương đương `candidate_chunks: 2000` vs `null`

> Sinh bởi `scripts/check_candidate_cap.py`. Chủ sở hữu: P3.
> Cấu hình `configs/v0.2_bm25_tokenizer.yaml` · `data/error_pool.json` n=300 · corpus 524422 chunk · top-50.

```json
{
  "config": "configs/v0.2_bm25_tokenizer.yaml",
  "questions": "data/error_pool.json",
  "cap": 2000,
  "top_k": 50,
  "n_chunks": 524422,
  "ngay": "2026-09-11",
  "ket_luan": "KHÔNG tương đương ở 19/25 ô"
}
```

| tokenizer | pool | lệch danh sách | lệch TẬP | R@50 null | R@50 cap | ms/câu null | ms/câu cap | ứng viên (trung vị) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| regex | max | 0/300 | 0/300 | 0.9356 | 0.9356 | 60.9 | 19.7 | 467691 |
| regex | mean_top2 | 295/300 | 295/300 | 0.9322 | 0.9322 | 60.9 | 19.7 | 467691 |
| regex | mean_top3 | 300/300 | 300/300 | 0.9356 | 0.9289 | 60.9 | 19.7 | 467691 |
| regex | logsumexp | 84/300 | 11/300 | 0.9356 | 0.9356 | 60.9 | 19.7 | 467691 |
| regex | sum | 300/300 | 300/300 | 0.4461 | 0.9256 | 60.9 | 19.7 | 467691 |
| whitespace | max | 0/300 | 0/300 | 0.9356 | 0.9356 | 60.4 | 19.2 | 467665 |
| whitespace | mean_top2 | 295/300 | 295/300 | 0.9322 | 0.9322 | 60.4 | 19.2 | 467665 |
| whitespace | mean_top3 | 300/300 | 300/300 | 0.9322 | 0.9289 | 60.4 | 19.2 | 467665 |
| whitespace | logsumexp | 76/300 | 12/300 | 0.9356 | 0.9356 | 60.4 | 19.2 | 467665 |
| whitespace | sum | 300/300 | 300/300 | 0.4417 | 0.9256 | 60.4 | 19.2 | 467665 |
| syllable_bigram | max | 0/300 | 0/300 | 0.9422 | 0.9422 | 65.4 | 23.7 | 467691 |
| syllable_bigram | mean_top2 | 290/300 | 288/300 | 0.9389 | 0.9422 | 65.4 | 23.7 | 467691 |
| syllable_bigram | mean_top3 | 299/300 | 299/300 | 0.9456 | 0.9422 | 65.4 | 23.7 | 467691 |
| syllable_bigram | logsumexp | 9/300 | 0/300 | 0.9422 | 0.9422 | 65.4 | 23.7 | 467691 |
| syllable_bigram | sum | 300/300 | 300/300 | 0.5639 | 0.9489 | 65.4 | 23.7 | 467691 |
| pyvi | max | 0/300 | 0/300 | 0.9389 | 0.9389 | 50.6 | 14.2 | 411747 |
| pyvi | mean_top2 | 291/300 | 291/300 | 0.9506 | 0.9422 | 50.6 | 14.2 | 411747 |
| pyvi | mean_top3 | 299/300 | 299/300 | 0.9606 | 0.9439 | 50.6 | 14.2 | 411747 |
| pyvi | logsumexp | 90/300 | 19/300 | 0.9422 | 0.9422 | 50.6 | 14.2 | 411747 |
| pyvi | sum | 300/300 | 300/300 | 0.5439 | 0.935 | 50.6 | 14.2 | 411747 |
| underthesea | max | 0/300 | 0/300 | 0.9356 | 0.9356 | 48.6 | 13.9 | 409182 |
| underthesea | mean_top2 | 289/300 | 289/300 | 0.9456 | 0.9389 | 48.6 | 13.9 | 409182 |
| underthesea | mean_top3 | 299/300 | 299/300 | 0.9506 | 0.9406 | 48.6 | 13.9 | 409182 |
| underthesea | logsumexp | 86/300 | 20/300 | 0.9422 | 0.9422 | 48.6 | 13.9 | 409182 |
| underthesea | sum | 300/300 | 300/300 | 0.5489 | 0.9433 | 48.6 | 13.9 | 409182 |

**Đọc bảng:** *lệch TẬP* = 0 nghĩa là hai chế độ trả về đúng cùng một tập doc ở độ sâu 50 ⇒ mọi con số Recall đều không đổi (mã chấm chỉ dùng phép giao tập hợp). *lệch danh sách* > 0 mà *lệch TẬP* = 0 chỉ là khác thứ tự trong cùng tập.

**Kết luận:** KHÔNG tương đương ở 19/25 ô.

## Nhận xét (điền tay)

- Ô nào lệch, và lệch ở pooling cộng dồn (`mean_topN`/`logsumexp`) hay ở `max`? 
- Nếu có ô lệch: cap phải nâng lên bao nhiêu, hay tokenizer đó buộc phải chạy `null`? 
