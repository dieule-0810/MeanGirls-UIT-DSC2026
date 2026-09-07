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

Ba con số chốt **của cấu hình `pool=max`**:

```
R@5  = 0,7863   điểm xuất phát
R@50 = 0,9503   trần của reranker
dư địa = 0,1640
```

> ⚠️ **`max` KHÔNG còn là cấu hình được chọn.** Mục 7 cho thấy nó thua có ý nghĩa thống
> kê. Các mốc hiện hành: **R@5 = 0,8116** (`mean_top2`, bản BM25 thuần) và
> **R@50 = 0,9568** (`logsumexp`, trần reranker) ⇒ **dư địa reranker thật = 0,1501**.
> Bảng ở mục 2 giữ nguyên vì nó là ĐƯỜNG CƠ SỞ dùng để trừ, không phải cấu hình nộp bài.

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

- Cách gộp chunk→doc đã quét (mục 7). **Tokenizer thì chưa** — cả đường cơ sở này lẫn
  mọi kết luận của nó đều đứng trên `regex`, `fold_tone=false`. Chờ P3 chạy
  `bench_retrieval.py` để biết chúng có sống sót khi đổi tokenizer không.
- `dev` và `error_pool` đều đã lọc 11 câu "Vùng Chết"; public/private test **không** được
  lọc ⇒ cả hai tập **lạc quan hơn test theo cấu tạo**.
- Tầng gán bằng `min`. Bản `max` cho phân bố khác (282/238/288/192 trên holdout) nhưng chưa đo.

## 7. Cập nhật 07/09: quét cách gộp chunk→doc

Đường cơ sở ở mục 2 dùng `pool=max`. Quét đủ sáu cách gộp trên cùng `dev` n=1.000
(`configs/v0.1_bm25_cap2000.yaml`) cho thấy **`max` là lựa chọn tệ**:

| pool | R@5 | R@50 | spread@5 | `freq=0` @5 | `freq>=11` @5 |
|---|---:|---:|---:|---:|---:|
| `max` (= N=1) | 0,7863 | 0,9503 | 0,2166 | 0,8073 | 0,6345 |
| `mean_top2` | **0,8116** | 0,9540 | **0,1694** | 0,8067 | 0,6988 |
| `mean_top3` | 0,8106 | 0,9540 | 0,1751 | 0,7952 | 0,6988 |
| `mean_top4` | 0,7961 | 0,9530 | — | — | — |
| `mean_top5` | 0,7869 | 0,9490 | — | — | — |
| `logsumexp` | 0,8067 | **0,9568** | 0,2024 | **0,8172** | 0,6696 |
| `sum` | sập | sập | — | — | — |

**Kiểm định cặp** (bootstrap 10.000 lần + McNemar, `scripts/p4_paired_test.py`):

| So sánh | K | hiệu | KTC 95% | McNemar | kết luận |
|---|---:|---:|---|---|---|
| `mean_top3` − `max` | 5 | +0,0243 | [+0,0073, +0,0415] | 56-33, p=0,0197 | **khác biệt thật** |
| `mean_top3` − `logsumexp` | 5 | +0,0038 | [−0,0107, +0,0187] | 33-32, p≈1 | hoà |
| `mean_top2` − `mean_top3` | 5 | +0,0010 | [−0,0100, +0,0120] | 18-17, p≈1 | hoà |
| `logsumexp` − `mean_top3` | 50 | +0,0028 | [−0,0007, +0,0065] | 7-1, p=0,0771 | hoà |

### 7.1 Đường cong N — độ liên quan tập trung ở 2–3 chunk

`max` chính là `mean_top1`, nên sáu số trên hợp thành một đường cong theo N:

```
N=1  0,7863      N=2  0,8116      N=3  0,8106      N=4  0,7961      N=5  0,7869
```

Chữ U ngược, đỉnh tại N=2–3, hai đầu tụt về nhau. Cơ chế:

- **N nhỏ:** một chunk may mắn quyết định cả văn bản. Luật khung có một đoạn khớp phần
  khung diễn đạt chung là leo lên top — đúng cơ chế `R-ENTITY` trong `error_taxonomy.md`.
- **N lớn:** pha loãng bằng những đoạn không liên quan. Văn bản mà độ liên quan tập trung
  ở vài điều khoản bị thiệt nhất — mà đó là dạng điển hình của hỏi đáp pháp luật.
- **N → ∞** chính là `sum` chuẩn hoá độ dài, và `sum` thì sập (0,0683 so với 0,2583 của
  `max` ở thăm dò 150k chunk). Hai đầu đường cong khớp nhau.

Cơ chế "không đệm 0 để khỏi phạt văn bản ngắn" **không chi phối kết quả này**: trung vị
36 chunk/văn bản, p90 = 133, chỉ **1,0%** văn bản có ≤3 chunk.

⇒ **Độ liên quan trong hỏi đáp pháp luật tiếng Việt tập trung trong ~2–3 chunk
(~360–540 từ) của văn bản** — thường là một hai điều khoản.

### 7.2 Cú tụt ở `freq=0` là của riêng N=3

`mean_top3` là cách gộp duy nhất kéo tụt tầng `freq=0` (0,7952 so với 0,8073 của `max`).
Đáng lo vì `freq=0` là tầng gần private test nhất. Nhưng `mean_top2` lấy **trọn** mức tăng
ở `freq>=11` (0,6988, y hệt N=3) mà `freq=0` gần như không đổi (0,8067, −0,0006).

⇒ Giả thiết *"`mean_topN` hại văn bản chưa từng thấy"* **bị bác bỏ**. Đó là đặc tính của
riêng N=3, không phải của cả họ.

### 7.3 Quyết định, và mức độ chắc chắn của nó

```
pool = logsumexp   cho pipeline CÓ rerank
pool = mean_top2   cho bản BM25 thuần
pool = max         KHÔNG dùng nữa
```

Lý do tách hai chế độ: reranker xáo lại top-50, nên R@5 của BM25 gần như vô nghĩa với
pipeline có rerank — chỉ **R@50, tức trần của reranker**, là ràng buộc. Không rerank thì
ngược lại.

⚠️ **Đây là quyết định trên bằng chứng yếu.** `mean_top2`, `mean_top3`, `logsumexp` hoà
nhau về thống kê ở mọi cặp đã kiểm. Phân thắng bại bằng **nguyên tắc thiết kế** —
`logsumexp` không có tham số tự do nào để chọn sai; `mean_top2` không gây thoái lui ở
`freq=0`. Không được viết trong bài báo như thể số liệu đã phân thắng bại.
