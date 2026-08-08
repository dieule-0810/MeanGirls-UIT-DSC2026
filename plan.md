# DSC2026 — Task 1 LegalIR: Kế hoạch 6 tuần cho team 4 người

> Bản v3 — cập nhật 07/08/2026, sau khi phân tích `scoring.py` + `metadata.yaml` của BTC.
> Public test: 06/08 → 18/09/2026. Private test: 19/09 → 23/09/2026.
> Trạng thái: **đã đăng ký đội xong**, đang trong public test.
> Đăng ký mô hình: mở đến hết public test (18/09), nhưng **hạn thực tế ~10/09** vì BTC cần 5 ngày làm việc để duyệt.

---

## 0. Ba quyết định nền tảng

### 0.1 Kaggle là engine tính toán chính, không phải máy của team

| Thiết bị | Vai trò | Không dùng để |
|---|---|---|
| MacBook M4 Pro (MPS, ARM64) | Encode corpus, BM25, eval, phân tích lỗi, dev hằng ngày | Fine-tune (MPS thiếu kernel, chậm 5–10× so với T4); build Docker image cuối |
| Máy GPU 6GB | Dev, chạy BM25, test code trước khi đẩy lên Kaggle | Model > 300M tham số ở batch thật |

**Giải pháp**: 4 tài khoản Kaggle → **30 giờ GPU T4×2/tuần/tài khoản = 120 giờ/tuần**. Colab free dự phòng. Toàn bộ fine-tune chạy ở đây.

> Không vi phạm quy định cấm API: BTC cấm dùng API *của mô hình/dịch vụ bên thứ ba trong pipeline*. Kaggle/Colab chỉ là hạ tầng tính toán — trọng số do team tự tải, tự chạy, tự kiểm soát. Dùng Claude Pro hỗ trợ viết code cũng hợp lệ; **tuyệt đối không** gọi bất kỳ API nào bên trong hệ thống dự thi.

### 0.2 Corpus chỉ 8.532 văn bản — đừng over-engineer

- **Không cần vector DB.** `numpy` matmul hoặc `faiss-cpu` IndexFlatIP là đủ, và cho kết quả chính xác tuyệt đối thay vì xấp xỉ như ANN.
- Ước tính ~150k–250k chunk (256 token, overlap 64). Ma trận embedding 1024 chiều fp16 ≈ 0.5 GB → nằm gọn trong RAM máy M4.
- Rerank top-100 cho cả 1.000 câu public test chạy trong vài chục phút trên T4.
- Ngân sách 4B tham số rất thoải mái: embedding 568M + reranker 568M ≈ 1.14B, dư > 2.8B.

### 0.3 "Vibecode" là rủi ro lớn nhất của team này

Top 10 **bắt buộc** nộp mã nguồn tái lập + viết bài báo khoa học có phản biện (top 1–2–3 không viết thì không được công nhận kết quả).

**Quy tắc bắt buộc từ ngày 1:**
- Mỗi người viết **1 trang notes/tuần** bằng tiếng Việt: tuần này làm gì, tại sao chọn cách đó, kết quả ra sao. Đây là nguyên liệu thô của bài báo.
- Mỗi thí nghiệm ghi 1 dòng vào `experiments.csv`: `exp_id, ngày, người chạy, config, Recall@5, Precision, ghi chú`.
- Dùng Claude để *giải thích* code trước khi copy, không chỉ để *sinh* code. Không merge PR nào mà tác giả không giải thích được cho thành viên khác trong 3 phút.

---

### 0.4 Mã chấm của BTC có hai cái bẫy im lặng

Đã chạy trực tiếp `eval_retrieval` của BTC với input biên. Kết quả (chi tiết: `docs/scoring_behaviour.md`):

| Nộp gì | Recall | Precision | |
|---|---|---|---|
| `["100"]` đúng | 1.000 | 1.000 | |
| Đủ 5 doc/câu | 1.000 | 0.300 | Recall **không** tăng khi nộp thừa |
| **`[100]` kiểu int** | **0.000** | **0.000** | 🔴 Không lỗi, không cảnh báo |
| **`["100","100","100"]`** | 1.000 | **0.667** | 🔴 Trùng lặp không được khử |
| Thiếu 1 câu hỏi | 💥 | 💥 | `Exception` → submission FAILED |
| Sai qid | 💥 | 💥 | `TypeError` → submission FAILED |

**Bẫy 1 — `doc_id` kiểu int cho 0 điểm im lặng.** `context_*.json` để `id` là **int**, nhãn để **str**.
Mã chấm dùng `set(a) & set(b)` nên `100 != "100"`. Leaderboard hiện 0.0 và team đi tìm bug trong mô hình
suốt nhiều ngày. → Ép `str()` **tại điểm đọc corpus**, assert ở `make_submission.py`.

**Bẫy 2 — trùng lặp tính vào cả giới hạn 5 lẫn mẫu số Precision.** Mẫu số là `len(list)`, không phải
`len(set)`. Trùng lặp phát sinh tự nhiên khi gộp nhiều retriever (RRF, ensemble) — đúng thứ team làm từ Tuần 2.
→ Dedupe giữ thứ tự rồi mới cắt còn 5.

**Lỗi cấu trúc thì CRASH, không phải 0 điểm.** Thiếu câu / sai qid / thiếu key `answer` đều làm container
chấm điểm văng exception → CodaLab báo *Failed* và nhiều khả năng vẫn tiêu một lượt nộp. Ở private test
chỉ có 3 lượt/ngày → mất 1/3 ngân sách ngày hôm đó vì một lỗi format.

**Mã chấm KHÔNG dùng thứ tự.** Toàn bộ là phép giao tập hợp — không MRR, không NDCG. Doc đúng nằm ở
vị trí 1 hay 5 đều như nhau. → Đừng tốn công tối ưu thứ tự *bên trong* top-5. Toàn bộ giá trị nằm ở
**doc đúng có lọt vào tập 5 hay không** và ở **kích thước tập trả về**.

---

## 1. Hai luồng công việc song song

Đây là thay đổi quan trọng nhất so với bản v1. Việc đóng gói **không phải là giai đoạn cuối** mà là thuộc tính của mọi cột mốc.

### Track A — Mô hình (P2, P3, P4)
Đi từ đơn giản đến phức tạp, tối ưu Recall@5.

### Track B — Tái lập & đóng gói (P1, chạy song song từ Tuần 1)
Mỗi 1–2 tuần chốt một **release chạy được**:

| Release | Thời điểm | Nội dung | Chi phí đóng gói |
|---|---|---|---|
| v0.1 | cuối Tuần 1 | BM25 thuần. Dockerfile ~15 dòng, không PyTorch, không trọng số | ~2 giờ |
| v0.2 | cuối Tuần 2 | + dense retrieval. Thêm `torch`, `sentence-transformers` | ~1 giờ |
| v0.3 | cuối Tuần 3 | + bi-encoder đã fine-tune. Thêm cơ chế nạp checkpoint | ~1 giờ |
| v0.4 | cuối Tuần 4 | + cross-encoder reranker | ~1 giờ |
| v0.5 | cuối Tuần 5 | + bộ calibration số lượng doc | ~1 giờ |
| **v1.0** | **15/09** | Bản chốt, đóng băng | ~2 giờ |

**Điều kiện nghiệm thu mỗi release** (không đạt thì không được gắn tag):
1. Chạy end-to-end bằng **một lệnh** trên máy sạch → sinh ra `submission.zip` hợp lệ.
2. Model Card + Data Statement đã cập nhật (Điều 9 yêu cầu kèm theo mỗi bài nộp).
3. README ghi rõ từng bước tái lập.
4. Đã gắn tag Git, ghi lại điểm held-out + điểm leaderboard tương ứng.

Lợi ích: **không bao giờ tồn tại thời điểm phải "làm Docker từ đầu"**. Chi phí trải mỏng thành 6 lần × 1–2 giờ thay vì một lần 20 giờ vào tuần bận nhất.

### Ba điều cần biết về Docker trong cuộc thi này

1. **Docker không cần để *chạy* private test.** Theo Điều 6, Docker image + mã nguồn là thứ top 10 gửi *sau* để BTC tái lập. Trong 5 ngày private test, team tự chạy inference rồi nộp file dự đoán.
2. **Nhưng ràng buộc thật nghiêm hơn**: thứ sinh ra kết quả private test phải tái lập được. Quy tắc cứng — **mọi submission private test phải sinh ra từ một Git tag**, không từ notebook đang mở.
3. **Q&A đã nới lỏng hình thức**: BTC chấp nhận GitHub repo hoặc zip mã nguồn/trọng số, không bắt buộc Docker, miễn README chi tiết. Tải trọng số từ Internet lúc chạy cũng hợp lệ. Vẫn nên làm Docker vì nó *chứng minh* tính tái lập thay vì chỉ tuyên bố.

### ⚠️ Bẫy ARM64 — xác nhận ngay ở v0.1

Máy M4 Pro là **ARM64**. Docker image build mặc định trên đó là `linux/arm64`, trong khi BTC gần như chắc chắn chạy x86_64 + CUDA. Image sẽ không khởi động được ở phía họ, và team **không phát hiện ra** vì trên máy mình nó chạy hoàn hảo.

Bắt buộc: `docker buildx build --platform linux/amd64`, hoặc build trên Kaggle/máy x86. Kiểm tra ở v0.1 khi image còn nhỏ, không để đến v1.0.

Bổ sung: **pin revision hash của model HuggingFace** trong config. Repo trên HF có thể được tác giả cập nhật, làm kết quả tái lập lệch đi mà không ai biết tại sao.

---

## 2. Kiến trúc mục tiêu

```
Câu hỏi
  │
  ├─► BM25 (bm25s / rank_bm25)      ──► top 100
  │
  ├─► Dense bi-encoder (fine-tuned) ──► top 100
  │
  ▼
Hợp nhất bằng RRF (Reciprocal Rank Fusion) ──► top 50 chunk
  │
  ▼  gộp chunk → document (max-pooling điểm)
  │
Cross-encoder reranker (fine-tuned) ──► xếp hạng lại top 20 doc
  │
  ▼
Bộ quyết định số lượng (calibration) ──► trả về 1..5 doc_id
```

Ba tầng bật/tắt độc lập để đo đóng góp riêng — cần cho phần ablation study của bài báo.

---

## 3. Chiến lược điểm số: lợi thế cạnh tranh chính

Từ phân tích train: **92,1% câu hỏi chỉ có đúng 1 văn bản đáp án**, tối đa là 5 → trần 5 doc của BTC **không hề ràng buộc**.

Vì Recall là metric chính và 92% câu chỉ nhận giá trị 0 hoặc 1, tổng Recall trên 1.000 câu có **độ hạt rất thô → hòa điểm giữa các đội top rất dễ xảy ra**. Lúc đó Precision quyết định thứ hạng.

| Chiến lược | Recall | Precision (câu 1 đáp án) |
|---|---|---|
| Luôn nộp 5 doc | tối đa | ~0.20 |
| Nộp 1 doc khi tự tin, 5 doc khi phân vân | ~tối đa | 0.5 – 0.8 |

Đã **kiểm chứng trên mã chấm thật**: nộp 5 doc cho câu có 1 đáp án đúng cho Recall 1.000 và Precision 0.300;
nộp đúng 1 doc cho cả hai bằng 1.000. Recall không hề tăng khi nộp thừa.

**Kết luận**: xây *bộ quyết định số lượng* dựa trên margin điểm giữa doc hạng 1 và hạng 2 sau rerank.
Calibrate trên held-out để Recall không giảm quá 0.3% nhưng Precision tăng 3–4 lần. Việc của P4 ở Tuần 5,
nhưng eval harness phải hỗ trợ nó từ Tuần 1 — `src/evaluate.py` đã có sẵn `recall_at_k()` và `diagnose()`.

---

## 4. Phân vai 4 người

Vai trò là **trách nhiệm sở hữu**, không phải hàng rào.

### P1 — Lead / Infra / Track B Owner
- Repo Git (MIT ngay từ đầu), cấu trúc thư mục, `requirements.txt`, seed cố định.
- `evaluate.py` cài **đúng công thức BTC**, gồm cả luật > 5 doc → 0 điểm.
- `make_submission.py` sinh `submission.zip` chứa duy nhất `submission.json`, có assert chặn lỗi.
- **Toàn bộ Track B**: Dockerfile, README, Model Card, Data Statement, tag release.
- Quản lý CodaLab, đăng ký mô hình, `experiments.csv`.
- Là **người duy nhất** nộp bài lên leaderboard.

### P2 — Data Engineer
- Parser chịu lỗi (`context_69.json` **thiếu trường `name`** → dùng `.get()`).
- Normalize `\r\n\n` chèn giữa câu; tách boilerplate quốc hiệu/tiêu ngữ.
- Chunker lai: ưu tiên cắt theo `Điều N`, fallback sliding-window cho TCVN/QCVN không có `Điều`.
- Tập train pairs: positive từ nhãn + **hard negative mining** từ BM25.
- Chia held-out 1.000 câu từ train, cố định seed, không ai được đụng.

### P3 — First-stage Retrieval (KPI: **Recall@50**)
- BM25 với tokenizer tiếng Việt (thử cả `underthesea`/`pyvi` word-segment và whitespace thuần).
- Benchmark rồi fine-tune bi-encoder (MultipleNegativesRankingLoss).
- Hợp nhất RRF, tune trọng số lai.
- Chiến lược gộp chunk → doc (max / mean-top-3 / logsumexp).

### P4 — Rerank, Calibration & Error Analysis (KPI: **Recall@5, Precision**)
- Benchmark rồi fine-tune cross-encoder.
- Bộ quyết định số lượng doc — đóng góp lớn nhất về Precision.
- Phân loại 50 câu sai mỗi tuần thành nhóm nguyên nhân, báo cho P2/P3.
- Theo dõi **nhiễu nhãn** (đã phát hiện: cùng câu hỏi, gold doc khác nhau).

**Tuần 0–1 mọi người làm chung** để ai cũng hiểu toàn cảnh, sau đó mới tách vai.

---

## 5. Timeline

### Tuần 0 — Khởi động kỹ thuật (07/08 → 10/08)

Đăng ký đội đã xong → dồn toàn bộ thời gian cho kỹ thuật.

| Việc | Người |
|---|---|
| 4 tài khoản Kaggle, verify phone để mở GPU | Cả team |
| Xác nhận CodaLab hoạt động, Team Name đúng quy định | P1 |
| Tải `selected-contexts.zip`, xác nhận đủ 8.532 file, thống kê độ dài passage thật | P2 |
| **Gửi danh sách mô hình xin BTC duyệt** (mục 6) | P1 |
| Repo Git + `INTERFACES.md` + `.gitignore` + `experiments.csv` (đã có sẵn, xem bộ khung v0.1) | P1 |
| Đọc `docs/scoring_behaviour.md` — cả 4 người, không ai được bỏ qua | Cả team |

**🎯 Cột mốc bắt buộc**: nộp thành công một **submission giả** (5 doc_id ngẫu nhiên mỗi câu) lên CodaLab và **nhìn thấy điểm hiện trên leaderboard**.

> Rủi ro giết chết nhiều đội không phải mô hình yếu mà là sai format zip. Xác nhận đường ống thông suốt khi còn 6 tuần, không phải khi còn 6 giờ.

### Tuần 1 — Baseline & hạ tầng đo lường (11/08 → 17/08)

| Người | Việc |
|---|---|
| P1 | `evaluate.py`, `make_submission.py`. **Dockerfile v0.1 + build thử `--platform linux/amd64`** |
| P2 | Parser + normalize 8.532 doc → `corpus_clean.jsonl`. Chunker v1. Tách held-out |
| P3 | BM25 baseline toàn corpus, đo Recall@5 / @20 / @50 |
| P4 | Đọc 50 câu BM25 sai, báo cáo phân loại lỗi đầu tiên |

**Deliverable: release v0.1** + điểm BM25 trên held-out **và** trên leaderboard. Hai con số phải chênh < 3%; chênh nhiều nghĩa là held-out chia sai.

*Mốc tham chiếu (không phải cam kết)*: BM25 thuần thường cho Recall@5 vùng 0.6–0.75 nhờ độ trùng từ vựng cao giữa câu hỏi và văn bản luật.

### Tuần 2 — Dense retrieval + Hybrid (18/08 → 24/08)

| Người | Việc |
|---|---|
| P2 | Chunker v2 dựa trên lỗi tuần 1. Bắt đầu hard negative mining |
| P3 | Benchmark **zero-shot** 4–5 embedding model, chọn 2 tốt nhất. Hợp nhất RRF, tune trọng số |
| P4 | Benchmark **zero-shot** 2–3 reranker trên top-50 của BM25 |
| P1 | Script encode chạy được cả MPS (M4) lẫn CUDA (Kaggle). **Release v0.2** |

**Điều kiện đi tiếp**: hybrid zero-shot phải vượt BM25 rõ rệt. Nếu không → có bug ở chunking hoặc ở cách gộp chunk→doc. Dừng lại debug, đừng đi tiếp.

### Tuần 3 — Fine-tune bi-encoder (25/08 → 31/08)

Tuần ngốn GPU nhất → phân bổ quota 4 tài khoản Kaggle từ đầu tuần.

| Người | Việc |
|---|---|
| P2 | Hoàn thiện tập train: 6.000 câu × (1 positive + 8 hard negative) |
| P3 | Fine-tune bi-encoder, thử 2–3 cấu hình LR/epoch |
| P4 | Giữ reranker zero-shot cố định để đo đóng góp thuần của fine-tune |
| P1 | Backup checkpoint lên Kaggle Datasets (không mất khi hết session). **Release v0.3** |

⚠️ **Cẩn thận hard negative**: nhãn có nhiễu — một doc BM25 xếp hạng 1 mà không phải gold **có thể là đáp án đúng chưa được gán**. Đề xuất: bỏ qua top-2, chỉ mining negative từ hạng 3 trở xuống.

### Tuần 4 — Fine-tune reranker (01/09 → 07/09)

| Người | Việc |
|---|---|
| P4 | Fine-tune cross-encoder trên (câu hỏi, chunk) từ top-50 của retriever đã fine-tune |
| P3 | Tối ưu Recall@50 lần cuối — đây là **trần cứng** của toàn hệ thống, reranker không cứu được doc không lọt top 50 |
| P2 | Rà nhóm văn bản gây lỗi nhiều nhất (TCVN/QCVN dài, không có `Điều`) |
| P1 | README tái lập + Data Statement + Model Card bản đầy đủ. **Release v0.4** |

### Tuần 5 — Calibration & Ensemble (08/09 → 14/09)

🔴 **~10/09: hạn thực tế cuối cùng để đăng ký mô hình mới.** Cổng đăng ký mở đến 18/09 nhưng BTC cần 5 ngày làm việc để duyệt — đăng ký sau 10/09 thì phê duyệt về sau khi private test đã bắt đầu.

| Người | Việc |
|---|---|
| P4 | **Bộ quyết định số lượng doc** — việc giá trị nhất tuần này |
| P3 | Ensemble 2 bi-encoder nếu ngân sách tham số cho phép |
| P2 | Tổng hợp bảng ablation cho bài báo |
| P1 | **Release v0.5.** Dry-run tái lập lần 1 trên máy sạch |

### Tuần 6 — Đóng băng (15/09 → 18/09)

- **15/09: code freeze.** Sau mốc này chỉ sửa bug, không thêm ý tưởng.
- **Release v1.0.** Dry-run tái lập lần 2: xóa sạch môi trường, làm theo đúng README, phải ra đúng con số. Không ra đúng → README sai, sửa README chứ không sửa trí nhớ.
- Nộp submission tốt nhất lên public test.
- Script private test sẵn sàng: chỉ cần đổi đường dẫn file input.
- Viết draft bài báo từ notes hằng tuần.

### 19/09 → 23/09 — Private test

- **Chỉ 3 lượt submit/ngày.** Kế hoạch: ngày 1 nộp cấu hình an toàn nhất đã chứng minh trên public; các ngày sau mới thử biến thể.
- P1 là người duy nhất bấm nút nộp.
- **Mọi submission phải sinh ra từ một Git tag.** Không sửa code trừ khi crash.
- Ghi lại: tag nào → điểm nào. Đây là bằng chứng tái lập gửi BTC sau này.

### 24/09 → 24/10 — Viết bài báo (nếu vào top 10)

---

## 6. Danh sách mô hình cần đăng ký (đăng ký sớm, đăng ký rộng)

Cổng mở đến 18/09 nhưng **hạn thực tế là ~10/09**. Đăng ký toàn bộ ứng viên ngay Tuần 0, kể cả model có thể không dùng — đăng ký thừa không mất gì, đăng ký thiếu thì mất cả giải.

**Bi-encoder / embedding:**
- `BAAI/bge-m3` (~568M, context 8192 — rất hợp văn bản luật dài)
- `AITeamVN/Vietnamese_Embedding` (fine-tune từ BGE-M3 cho tiếng Việt)
- `Qwen/Qwen3-Embedding-0.6B`
- `intfloat/multilingual-e5-base` và `-large`
- `Alibaba-NLP/gte-multilingual-base`
- `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base, nhẹ, tốt để prototype)

**Cross-encoder / reranker:**
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-Reranker-0.6B`
- `AITeamVN/Vietnamese_Reranker`
- `namdp-ptit/ViRanker`

**Lưu ý bắt buộc:**
1. **Tự kiểm tra số tham số thực tế trên HuggingFace trước khi đăng ký** — danh sách trên dựa theo hiểu biết đến giữa 2025, con số có thể đã thay đổi.
2. Ngân sách 4B tính **tổng toàn pipeline**, gồm cả lớp embedding. BGE-M3 + bge-reranker-v2-m3 ≈ 1.14B → an toàn.
3. **LoRA/quantization KHÔNG hợp lệ hoá model > 4B.** Qwen3-Reranker-4B: 4B **không phải "dưới 4 tỷ"** → loại.
4. Kiểm tra danh sách BTC đã duyệt trước, tránh đăng ký trùng.
5. BM25 không có tham số học được nhưng vẫn nên khai báo trong Model Card.

---

## 7. Bảng rủi ro

| Rủi ro | Mức | Phòng ngừa |
|---|---|---|
| Docker build ARM64 trên M4, BTC không chạy được | **Cao** | `--platform linux/amd64` từ v0.1; hoặc build trên Kaggle |
| Đóng gói bị đẩy đến phút chót rồi hỏng | **Cao** | Track B song song, 6 release có điều kiện nghiệm thu |
| Kết quả private test không tái lập được | **Cao** | Mọi submission sinh từ Git tag; 2 lần dry-run trên máy sạch |
| **`doc_id` kiểu int → 0 điểm im lặng** | **Cao** | Ép `str()` tại điểm đọc corpus; assert trong `make_submission.py`; test tự động |
| **Trùng lặp doc_id khi ensemble** | **Cao** | `dedupe_keep_order()` trước khi cắt 5 |
| **Thiếu/sai qid → submission FAILED, mất lượt nộp** | **Cao** | Assert khớp CHÍNH XÁC tập qid, không chỉ số lượng |
| Sai format `submission.zip` | Cao | Submission giả ngay Tuần 0 |
| Một câu lỡ có > 5 doc → cả câu 0 điểm | Cao | Assert cứng trong `make_submission.py`, fail loud |
| Model chưa được duyệt kịp | Cao | Đăng ký rộng Tuần 0; chốt tuyệt đối 10/09 |
| Giới hạn ứng viên vào 3.105 doc xuất hiện trong train | Cao | 5.427 doc còn lại chính là nơi chứa đáp án public/private. Luôn index đủ 8.532 |
| Model HF bị tác giả cập nhật giữa chừng | Trung bình | Pin revision hash trong config |
| Hết quota GPU Kaggle giữa tuần train | Trung bình | 4 tài khoản, checkpoint thường xuyên |
| Overfit vào nhãn nhiễu | Trung bình | Bỏ top-2 khi mining hard negative; theo dõi held-out |
| Không viết được paper vì không hiểu code | **Cao với team này** | Notes 1 trang/người/tuần; quy tắc giải thích 3 phút |
| Held-out lệch so với leaderboard | Trung bình | Đối chiếu 2 con số ngay Tuần 1 |

---

## 8. Nhịp làm việc

- **Thứ 2, 15 phút**: chốt mục tiêu tuần, phân bổ quota GPU.
- **Thứ 6, 30 phút**: demo kết quả, cập nhật `experiments.csv`, nộp notes cá nhân, **nghiệm thu release nếu đến hạn**.
- Mọi con số quan trọng ghi vào repo, không để trôi trong chat.
- Không push thẳng lên `main`; PR + 1 review, kể cả review sơ sài — mục đích là để người thứ hai biết code đó tồn tại.

---

## 9. Checklist 72 giờ tới

- [ ] 4 tài khoản Kaggle đã verify, thấy được GPU quota (cả team)
- [ ] Xác nhận CodaLab hoạt động, Team Name đúng quy định (P1)
- [ ] Giải nén `selected-contexts.zip`, xác nhận đủ 8.532 file, thống kê độ dài passage thật (P2)
- [ ] Gửi danh sách mô hình xin duyệt cho BTC (P1)
- [ ] Repo Git khởi tạo, MIT license, `experiments.csv` trống (P1)
- [ ] **Submission giả nộp thành công, thấy điểm trên leaderboard** (P1)
