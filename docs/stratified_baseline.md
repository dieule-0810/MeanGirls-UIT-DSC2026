# Đường cơ sở BM25 phân tầng theo `train_gold_freq`

> Chủ sở hữu: **P4**. Ngày: 07/09/2026. Đây là **đường cơ sở phải có trước mọi fine-tune**;
> không có nó thì không kết luận được chênh lệch của mô hình học là ghi nhớ hay khái quát hoá.
>
> Tái lập toàn bộ tài liệu này:
> ```bash
> python -m scripts.p4_build_ranking --config configs/v0.1_bm25.yaml \
>     --questions data/dev.json --out outputs/p4_bm25_ranking/dev_top50.json --top-k 50
> python -m scripts.p4_stratified_recall \
>     --ranking outputs/p4_bm25_ranking/dev_top50.json --questions data/dev.json
> ```

## 1. Thiết lập

- **Hệ thống:** BM25 Okapi trên chunk, `configs/v0.1_bm25.yaml`
  (`k1=1.5`, `b=0.75`, `tokenizer=regex`, `fold_tone=false`, `pool=max`, `candidate_chunks=null`).
- **Corpus:** 524.422 chunk / 8.507 văn bản. Từ vựng 93.700 term, nnz 43.803.763.
- **Tập đo:** `dev.json` n=1.000 (chính) và `error_pool.json` n=300 (lặp lại độc lập).
- **Tầng:** tần suất doc xuất hiện làm gold trong `train_split.json` (**n=4.689**, bản chia v6).
  Câu nhiều gold lấy `min` — bảo thủ: câu là "khó" nếu có **bất kỳ** gold nào chưa từng thấy.
- **Vì sao phân tầng:** `dev`/`holdout` tách từ `train`, nên gold của chúng theo cấu tạo nằm
  trong 2.435 doc quen thuộc, trong khi **71,4% corpus chưa từng là gold**. Public/private
  test không có ràng buộc đó. Tầng `freq=0` là đại diện gần nhất cho phần corpus đó.

## 2. Kết quả — `dev`, n=1.000

| K | `freq=0` (n=304) | `freq=1-2` (n=263) | `freq=3-10` (n=262) | `freq>=11` (n=171) | toàn cục | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0,8073 | 0,8511 | 0,7958 | **0,6345** | 0,7863 | **0,2166** |
| 20 | 0,9142 | 0,9487 | 0,9179 | 0,8908 | 0,9203 | 0,0578 |
| 50 | 0,9402 | 0,9696 | 0,9542 | 0,9327 | 0,9503 | 0,0368 |

**Dư địa reranker** (`R@50 − R@5`, tức phần một reranker hoàn hảo có thể giành):

| Tầng | dư địa |
|---|---:|
| `freq=0` | +0,1329 |
| `freq=1-2` | +0,1185 |
| `freq=3-10` | +0,1584 |
| `freq>=11` | **+0,2982** |
| toàn cục | +0,1640 |

Ba con số chốt của toàn hệ thống:

```
R@5  = 0,7863   điểm xuất phát
R@50 = 0,9503   TRẦN CỨNG của mọi reranker — không bao giờ vượt được
dư địa = 0,1640
```

**Ranh giới trách nhiệm:** 4,97% còn lại (1 − 0,9503) là việc của **P3** (không có ứng viên
thì reranker chịu). 16,40% ở giữa là việc của **P4**.

## 3. Kết luận

### 3.1 Hiệu ứng tầng nằm TRỌN ở khâu xếp hạng, không ở khâu tìm kiếm

Spread co từ **0,2166 (K=5)** xuống **0,0368 (K=50)**. Ở độ sâu 50, bốn tầng bao phủ gần như
đồng đều (0,933–0,970). Ở độ sâu 5 — đúng độ sâu ta nộp bài — chúng chênh 21,7 điểm.

BM25 **tìm được** văn bản đúng cho câu luật khung với xác suất ngang các tầng khác; nó chỉ
**xếp sai chỗ**. Cơ chế đã dự đoán trước ở `error_taxonomy.md` (`R-ENTITY`): câu hỏi pháp
luật VN dùng chung khung diễn đạt dài, phần phân biệt chỉ là một token thực thể; BM25 đếm
trùng token nên kéo hàng loạt văn bản cùng khuôn lên trên.

⇒ **Đây là bài toán reranker giải được, không phải bài toán retriever.** Và tầng `freq>=11`
có dư địa gấp ~1,8 lần trung bình.

### 3.2 Giả thiết "BM25 không học nên bốn tầng phải bằng nhau" — đúng một nửa

Đúng ở K=50 (spread 0,0368 < 5%). **Sai ở K=5** (spread 0,2166).

⇒ `train_gold_freq` **không phải thước đo thuần tuý về ghi nhớ**. Ở độ sâu nộp bài nó còn đo
độ khó xếp hạng nội tại của câu hỏi. Vì thế mọi so sánh về sau phải viết:

```
lợi thế thật của fine-tune  =  spread(mô hình fine-tuned, K=5)  −  0,2166
```

Bỏ vế trừ là quy nhầm toàn bộ chênh lệch cho ghi nhớ. Đây là chỗ phản biện tạp chí sẽ bắt.

### 3.3 Thứ tự giữa các tầng KHÔNG đơn điệu — chưa giải thích được

`freq=1-2` (0,8511) > `freq=0` (0,8073) > `freq=3-10` (0,7958) ≫ `freq>=11` (0,6345).
Nếu `freq` thuần tuý đo độ hiếm thì thứ tự phải đơn điệu. Nó không đơn điệu ⇒ có biến ẩn thứ
ba, nhiều khả năng là độ đặc thù của thực thể trong câu hỏi và độ dài văn bản gold.

**Chưa kiểm được thì không giải thích trong bài báo.** Báo cáo con số, ghi nhận là câu hỏi mở.
Giả thiết đáng kiểm nếu có thời gian: cắt chéo `freq` × `độ dài doc gold`.

## 4. Lặp lại độc lập trên `error_pool` (n=300)

| | error_pool (n=300) | dev (n=1.000) | Phán quyết |
|---|---:|---:|---|
| `freq>=11` tại K=5 | 0,6415 | 0,6345 | ✅ lặp lại |
| spread tại K=5 | 0,2210 | 0,2166 | ✅ lặp lại |
| `freq>=11` tại K=50 | 0,9811 (cao nhất) | 0,9327 (thấp nhất) | ❌ **không lặp lại** |
| spread tại K=50 | 0,0775 | 0,0368 | ❌ **không lặp lại** |

⚠️ **Ghi lại để không tái phạm:** từ n=300 nhóm đã suýt kết luận *"BM25 luôn tìm thấy luật
khung, chỉ xếp sai chỗ"* — dựa trên con số 0,9811 ở tầng chỉ có **n=53**. Trên n=171 con số
đó lật thành thấp nhất. Chênh lệch 0,048 với SE ≈ 0,027 là **chưa tới 2 SE**, tức nhiễu.

**Kết luận nào cũng phải kiểm trên n≥1.000 trước khi viết ra.** Kết luận 3.1 sống sót được
là vì nó dựa trên spread tại K=5 — đại lượng lặp lại gần như hoàn hảo trên cả hai tập.

## 5. Ghi chú về `holdout`

`outputs/p4_bm25_ranking/holdout_top50.json` đã sinh và **đã đọc một lần**: R@5 = 0,7797,
R@20 = 0,9072, R@50 = 0,9450 — nhất quán với `dev` (0,7863 / 0,9203 / 0,9503).

Việc đọc này **chấp nhận được vì BM25 không học gì từ dữ liệu**, nên không có rủi ro
overfit. **Không lặp lại điều này với bất kỳ hệ thống có học nào.** Từ đây trở đi mọi lựa
chọn siêu tham số đo trên `dev`; `holdout` mở lại đúng một lần lúc chốt cuối.

## 6. Hạn chế

- Một cấu hình BM25 duy nhất. Chưa biết đường cơ sở này ổn định thế nào khi đổi
  tokenizer hoặc cách gộp chunk→doc (chờ P3 chạy `bench_retrieval.py`).
- `dev` và `error_pool` đều đã lọc 11 câu "Vùng Chết"; public/private test **không** được
  lọc ⇒ cả hai tập **lạc quan hơn test theo cấu tạo**.
- Tầng gán bằng `min`. Bản `max` cho phân bố khác (282/238/288/192 trên holdout) nhưng chưa đo.
