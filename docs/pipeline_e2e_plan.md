# Kế hoạch 5 vòng — pipeline end-to-end (trừ chunking)

> Chủ sở hữu bản kế hoạch: **P3**. Ngày lập: 12/09/2026. Nhánh: `p3/pipeline-e2e`.
> Phạm vi: từ `chunks.jsonl` (P2 giao) đến `submission.zip`. **Không đụng chunker.**
>
> ⚠️ **Chế độ làm việc của vòng này: CHỈ VIẾT SCRIPT, KHÔNG CHẠY TRÊN DỮ LIỆU.**
> Mọi script phải chạy được ở chế độ `--demo` / `--dry-run` với corpus giả (khuôn mẫu:
> `tests/test_retrieval.py`, `scripts/bench_retrieval.py --demo`) để kiểm đúng đường ống
> trước khi tiêu một giờ CPU/GPU nào. Người chạy thật là bạn, không phải script tự chạy.

---

## 0. Lịch thật, đối chiếu trước khi đọc tiếp

`plan.md` mục 5 chốt: **code freeze 15/09**, private test **19/09 → 23/09** (3 lượt nộp/ngày),
hạn đăng ký model thực tế **~10/09 (đã qua)**. Hôm nay là **12/09**.

Vậy "5 vòng" dưới đây **không phải 5 tuần lịch**. Đọc nó là **5 vòng có cổng nghiệm thu**,
xếp theo thứ tự giá trị giảm dần: nếu hết thời gian ở vòng 3 thì dừng ở đó và nộp bản tốt nhất
đã chứng minh, chứ không bỏ dở vòng 4 rồi nộp thứ chưa đo. Nếu BTC đã dời hạn thì giãn ra theo
lịch mới — thứ tự vòng không đổi.

**Quy tắc cứng cho mọi vòng** (vi phạm là mất điểm im lặng, xem `INTERFACES.md` mục 0):
`doc_id`/`qid` luôn `str` · dedupe giữ thứ tự rồi mới cắt 5 · không câu nào > 5 doc hay rỗng ·
mọi siêu tham số nằm trong YAML · đo trên `data/dev.json`, **holdout chạm đúng một lần** ·
không gọi API bên thứ ba · pin `revision` hash HuggingFace · tổng tham số < 4B.

---

## 1. Trạng thái xuất phát (đã có, đã đo trên dev n=1000)

| Thành phần | Trạng thái | Số |
|---|---|---|
| `BaseRetriever` + registry + `pool_scores` | xong | — |
| BM25 + 5 tokenizer tiếng Việt | xong | `syllable_bigram`+`mean_top2`: **R@5 0,8555 · R@50 0,9700** |
| Kiểm tương đương `candidate_chunks` | xong | cap=2000 **không** trung tính với `mean_topN` |
| Dense retrieval | **chưa có** | — |
| Hợp nhất RRF | mới có prototype của P4 trên `dev_sub300` | w=0,6 k=60, ΔR@5 +0,0300 |
| DL train-from-scratch | **chưa có** | ô trống trong ma trận bài báo |
| Cross-encoder rerank | có `src/rerank/cross_encoder.py` | cả 3 model đều **hại** R@5 khi thay thế thứ hạng |
| Bộ quyết định số lượng doc | **chưa có** | — |
| Pipeline E2E một lệnh | `scripts/run_v0.1.py` **hỏng** | gọi `split_holdout` không tồn tại |

Trần hiện tại: **R@50 = 0,9700** ⇒ 3,0% số câu không reranker nào cứu được. Dư địa cho tầng
xếp hạng: `0,9700 − 0,8555 = 0,1145`.

---

## 2. Năm vòng

### Vòng 1 — Đóng lại phần cổ điển, dựng khung E2E chạy được

**Mục tiêu:** có một lệnh chạy từ `chunks.jsonl` → `submission.zip`, và chốt cấu hình BM25 tốt nhất.

| Script cần viết | Việc |
|---|---|
| `scripts/run_pipeline.py` | Runner E2E **mới**, config-driven: retrieve → (fuse) → (rerank) → (calibrate) → `predictions.json` → `submission.zip`. Mỗi tầng bật/tắt bằng YAML để đo đóng góp riêng (`plan.md` mục 2 yêu cầu cho ablation). **Không sửa `scripts/run_v0.1.py`** — file của P1, đang hỏng, để P1 xử lý. |
| `scripts/sweep_candidate_cap.py` | Quét `candidate_chunks` ∈ {2000, 20000, null} cho tổ hợp thắng. Bảng hiện tại đo ở cap=2000 mà cap đã chứng minh **không** trung tính ⇒ 0,9700 có thể chưa phải trần. |
| `configs/v0.3_bm25_best.yaml` | Chốt `syllable_bigram` + `mean_top2` (+ cap sau khi quét). Đây là bản BM25 để nộp nếu mọi thứ sau đó đổ vỡ. |

**Cổng:** `run_pipeline.py --config v0.3_bm25_best.yaml --demo` sinh được `submission.zip` hợp lệ
trên corpus giả; `make_submission.py` của P1 không kêu.

**Nhóm so sánh:** `co_dien`.

---

### Vòng 2 — Dense zero-shot (tận dụng LLM ở tầng biểu diễn)

**Giả thuyết H3:** biểu diễn dày (dense) bắt được `R-TERM` — câu hỏi dùng từ thường dân, văn bản
dùng thuật ngữ pháp lý — thứ BM25 không thể khớp vì không trùng token. Kỳ vọng mức tăng rơi vào
đúng nhóm lỗi đó, **không** rơi vào `R-NUM` (số hiệu văn bản), nơi BM25 vốn mạnh.

| Script cần viết | Việc |
|---|---|
| `src/retrieval/dense.py` | Subclass `BaseRetriever`: `index()` nạp/encode, `_score_chunks()` = matmul + `argpartition`. Nạp model bằng `repo` + **`revision`** đọc từ `configs/models.yaml`. Tự dò thiết bị MPS/CUDA/CPU, fp16 khi có GPU, chuẩn hoá L2 để cosine = dot. |
| `scripts/encode_corpus.py` | Encode 524.422 chunk → `data/embeddings.npy` (tên đã khoá, `INTERFACES.md` mục 7) + `data/embeddings_meta.json` ghi `repo`/`revision`/`n_chunks`/checksum để không bao giờ dùng nhầm embedding của model khác. Có `--resume`, ghi theo lô, `--limit` để thử 1.000 chunk trước. |
| `scripts/bench_dense.py` *(hoặc mở rộng `bench_retrieval.py` để nhận `build_retriever`)* | Lưới zero-shot 4–5 model × pooling, cùng khuôn báo cáo với bench BM25. |

**Ngân sách tham số:** chỉ chọn trong `configs/models.yaml` đã pin revision. Nặng nhất
(Qwen3-Embed 0,6B + Qwen3-Rerank 0,6B) = 1.191,6M = 29,8% của 4B ⇒ không siết.

**Cổng đi tiếp (`plan.md` Tuần 2):** hybrid zero-shot phải **vượt rõ** BM25. Không vượt thì dừng
lại debug chunking/gộp, **không** đi tiếp sang fine-tune.

**Nhóm so sánh:** `llm_leverage_embed`.

---

### Vòng 3 — Hợp nhất RRF (việc của P3, không phải của P4)

**Giả thuyết H4:** hợp nhất **thứ hạng** giữ được bề rộng của BM25 và lấy phần đỉnh của dense;
cộng **điểm** thì không được, vì hai thang điểm không so được (BM25 dương không chặn trên,
cosine ∈ [−1,1], logit reranker có dấu).

| Script cần viết | Việc |
|---|---|
| `src/retrieval/hybrid.py` | RRF ở **mức chunk** rồi mới gộp lên doc — `candidates()` / `search_chunks()` đã có sẵn cho đúng việc này. Prototype của P4 (`scripts/p4_fuse.py`) hợp nhất ở mức doc; bản của P3 phải so hai mức. |
| `scripts/tune_rrf.py` | Quét `w` và `k` trên **`train_split`**, chốt giá trị, rồi mới đo **một lần** trên `dev`. Prototype của P4 chọn w trên chính tập đo ⇒ lạc quan; đây là chỗ sửa điểm yếu đó. |

**Cổng:** `w` được chốt trên tập khác tập đo, ghi rõ trong `experiments.csv`.

**Nhóm so sánh:** `chien_luoc_du_lieu`.

---

### Vòng 4 — Hai ô còn thiếu của ma trận bài báo

Không có hai ô này thì phần trả lời câu hỏi nghiên cứu của BTC **hổng từ gốc** (`plan.md` 0.6).

**4a. DL train-from-scratch** — BTC xác nhận không cần đăng ký (`QA.md` mục 11).

| Script cần viết | Việc |
|---|---|
| `src/retrieval/scratch_encoder.py` | Bi-encoder nhỏ, **khởi tạo ngẫu nhiên**, không nạp checkpoint nào. Vocab tự dựng từ tokenizer tiếng Việt đã có (dùng lại `src/retrieval/tokenizers.py` — đây là lợi thế sẵn có). |
| `scripts/train_scratch.py` | Huấn luyện trên `train_split` (4.689 câu) với in-batch negatives. Chạy được cả 3 mốc learning-curve `train_1000/2500/4689` — đường cong đó **chính là** bằng chứng "10k điểm dữ liệu đủ hay không đủ để học biểu diễn từ số 0". |

**4b. Fine-tune bi-encoder (MultipleNegativesRankingLoss)**

| Script cần viết | Việc |
|---|---|
| `scripts/mine_hard_negatives.py` | Mining từ retriever tốt nhất, **bỏ top-2** (nhãn nhiễu: doc hạng 1 không phải gold vẫn có thể đúng — `plan.md` Tuần 3). Xuất `data/train_pairs.jsonl`. |
| `scripts/train_biencoder.py` | Fine-tune, chạy được trên Kaggle T4×2 / Colab Pro; checkpoint ra ngoài session (Kaggle Datasets), seed cố định, log mỗi epoch. |

⚠️ `train_split` đã co từ 5.689 xuống **4.689** khi `dev` được carve ra. Checkpoint hoặc hard
negative nào mine từ bản 5.689 **phải chạy lại**, nếu không `dev` không còn "chưa từng thấy".

**Nhóm so sánh:** `dl_from_scratch` và `llm_leverage_embed`.

---

### Vòng 5 — Xếp hạng lại + bộ quyết định số lượng doc

**Giả thuyết H5 (đã có bằng chứng ngược, phải thiết kế cẩn thận):** P4 đo được cả 3 reranker
zero-shot đều **hại** R@5 khi dùng để **thay thế** thứ hạng (bge-m3 −0,0183, ViRanker −0,0906),
dù probe gold-vs-bừa đạt 95%. Diễn giải: giỏi phân biệt theo **cặp** không kéo theo giỏi theo
**danh sách** — bài toán thật là gold vs 4 văn bản khó nhất. ⇒ reranker chỉ được dùng ở dạng
**hợp nhất**, hoặc phải fine-tune trên chính phân bố hard negative của mình.

| Script cần viết | Việc |
|---|---|
| `scripts/train_cross_encoder.py` | Fine-tune cross-encoder trên (câu hỏi, chunk) lấy từ top-50 của retriever đã fine-tune — khác hẳn phân bố mà model pretrained từng thấy. |
| `src/rerank/calibrate.py` | Bộ quyết định số lượng doc: dựa trên margin điểm hạng 1 vs hạng 2, trả về 1..5 doc. |
| `scripts/fit_calibration.py` | Fit ngưỡng trên `train_split`/`dev`, ràng buộc: **Recall không giảm quá 0,3%**, Precision tăng 3–4 lần. Đây là đòn bẩy Precision lớn nhất còn lại, và Precision là tie-break của giải. |

**Nhóm so sánh:** `llm_leverage_rerank` và `chien_luoc_du_lieu`.

---

## 3. Thứ tự viết script (đề xuất)

1. `scripts/run_pipeline.py` — có khung rồi thì mọi thứ sau chỉ là cắm thêm tầng.
2. `src/retrieval/dense.py` + `scripts/encode_corpus.py` — đường găng dài nhất (encode 524k chunk).
3. `src/retrieval/hybrid.py` + `scripts/tune_rrf.py`.
4. `src/rerank/calibrate.py` + `scripts/fit_calibration.py` — rẻ, không cần GPU, ăn thẳng vào Precision.
5. `scripts/mine_hard_negatives.py` → `scripts/train_biencoder.py`.
6. `scripts/train_scratch.py` — ô bài báo, chi phí thấp, đừng để rơi xuống cuối rồi mất.
7. `scripts/sweep_candidate_cap.py`, `scripts/train_cross_encoder.py` — làm khi còn thời gian.

## 4. Khuôn bắt buộc cho mọi script mới

- `--config configs/*.yaml`, **không hằng số hard-code**; `--demo`/`--dry-run` chạy được không cần `data/`.
- Retriever mới: chỉ viết `index()` + `_score_chunks()`, để khung lo gộp/dedupe/sort/cắt và `check_contract()`.
- In ra `stats()` và ghi **một dòng** `experiments.csv` (13 cột, 3 cột cuối phải cụ thể).
- Job dài: `python -u`, log ra file, in tiến độ theo lô.
- Lưới nhiều cấu hình: `assert` đối tượng đang mang đúng cấu hình được yêu cầu.
- Fail loud: thà dừng với thông báo rõ còn hơn nộp một file sai im lặng.
