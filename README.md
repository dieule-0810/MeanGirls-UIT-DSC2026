# DSC2026 — Task 1: Legal Information Retrieval

> ⚠️ **README này là deliverable chính gửi BTC.** Nó được *kiểm thử*, không phải được *viết*.
> Quy tắc: P1 viết, nhưng **P2 hoặc P3 chạy dry-run** và chỉ được làm đúng những gì ghi ở đây.
> Vướng chỗ nào thì **không hỏi P1** — ghi lại. Mỗi câu phải hỏi là một lỗ hổng đã tìm ra.
> Máy sạch để dry-run: một notebook Kaggle mới (x86_64, không state cũ, gần môi trường BTC hơn máy M4).

---

## 1. Tóm tắt phương pháp

<!-- P1 cập nhật ở mỗi release -->

**v0.1 (hiện tại)** — BM25 baseline. Chunk văn bản theo `Điều`, fallback sliding window; truy hồi bằng
BM25 Okapi trên chunk; gộp lên document bằng max-pooling; trả về top-5.

| Release | Phương pháp | Recall held-out | Precision held-out | Recall LB | Commit |
|---|---|---|---|---|---|
| v0.1 | BM25 | _(điền)_ | _(điền)_ | _(điền)_ | _(điền)_ |

---

## 2. Yêu cầu hệ thống

| Hạng mục | Giá trị |
|---|---|
| OS | Ubuntu 22.04 / macOS 14+ (x86_64 hoặc ARM64) |
| Python | **3.11.x** (bản đã kiểm chứng: 3.11.9) — `verify_env.py` sẽ fail nếu khác |
| CUDA | v0.1 **không cần GPU**. Từ v0.2: CUDA 12.1 |
| RAM | ≥ 16 GB |
| Đĩa trống | ≥ 10 GB |

**Thời gian chạy ước tính (v0.1, CPU 8 nhân):**

| Bước | Thời gian |
|---|---|
| Parse 8.532 văn bản | ~1 phút |
| Chunking | ~2 phút |
| Đánh chỉ mục BM25 | ~3 phút |
| Truy vấn 1.000 câu | ~2 phút |
| **Tổng** | **~8 phút** |

---

## 3. Cài đặt

```bash
git clone <repo-url> dsc2026 && cd dsc2026
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.verify_env          # phải in "✅ Môi trường OK"
```

---

## 4. Chuẩn bị dữ liệu

Giải nén dữ liệu BTC vào `data/` theo đúng cây sau:

```
data/
├── selected-contexts/        ← giải nén selected-contexts.zip vào đây
│   ├── context_21.json
│   └── … (8.532 file)
├── train.json
└── public-official.json      (hoặc private-official.json)
```

```bash
mkdir -p data/selected-contexts
unzip selected-contexts.zip -d data/selected-contexts
ls data/selected-contexts | wc -l      # phải in đúng 8532
```

> Dữ liệu **không** được commit lên Git (`.gitignore` đã chặn) — đây là dữ liệu của BTC,
> không phát tán ra ngoài phạm vi cuộc thi.

---

## 5. Tải trọng số

v0.1 **không dùng mô hình pretrained nào** — BM25 không có tham số học được.

Từ v0.2: `python -m src.download_weights` (revision hash được pin trong `configs/`, không chỉ tên repo,
vì repo HuggingFace có thể bị tác giả cập nhật làm kết quả lệch mà không ai biết tại sao).

---

## 6. Smoke test (2 phút) — **chạy cái này TRƯỚC**

```bash
bash scripts/smoke_test.sh
```

Kiểm tra môi trường và xác nhận `src/evaluate.py` khớp chính xác `scoring.py` của BTC (14 test).
Phải in `✅ PASS`. Nếu fail thì **dừng lại** — chạy tiếp 1.000 câu chỉ tốn thời gian vô ích.

---

## 7. Chạy inference → `submission.zip`

```bash
bash scripts/run_v0.1.sh
```

Một lệnh, sinh ra `outputs/v0.1_bm25/submission.zip` sẵn sàng nộp CodaLab.

Chạy trên private test: đổi `paths.questions` trong `configs/v0.1_bm25.yaml` sang
`data/private-official.json`, chạy lại đúng lệnh trên.

---

## 8. Con số kỳ vọng (để đối chiếu ngay)

Đối chiếu ba con số này. Lệch thì dừng và xem mục 10.

| Kiểm tra | Giá trị kỳ vọng |
|---|---|
| Số văn bản trong `corpus_clean.jsonl` | `8532` |
| SHA-256 của `corpus_clean.jsonl` | `_(điền sau khi chạy trên corpus đầy đủ)_` |
| Số chunk trong `chunks.jsonl` | `_(điền)_` |
| Recall trên `holdout.json` (seed 42) | `_(điền)_` |

Checksum khớp nghĩa là bước tiền xử lý đã đúng, và mọi sai lệch còn lại nằm ở phần mô hình.
Tách được hai nguồn lỗi này tiết kiệm rất nhiều thời gian cho cả hai bên.

---

## 9. Tái lập huấn luyện (không bắt buộc để verify kết quả)

v0.1 không có bước huấn luyện. Từ v0.3 xem `docs/training.md`.

---

## 10. Troubleshooting

> Mục này chỉ được điền bằng **lỗi thật gặp trong dry-run**, không phải lỗi tưởng tượng.
> Sau mỗi lần dry-run, người chạy bổ sung vào đây.

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `verify_env` báo sai Python | Máy dùng 3.10/3.12 | Tạo venv bằng đúng `python3.11` |
| `ls data/selected-contexts \| wc -l` ≠ 8532 | Giải nén tạo thêm một cấp thư mục | `mv data/selected-contexts/*/*.json data/selected-contexts/` |
| Recall ≈ 0.000 nhưng không có lỗi | 🔴 `doc_id` bị ép thành int ở đâu đó | Xem `docs/scoring_behaviour.md`. `make_submission.py` lẽ ra đã chặn — báo P1 |
| _(bổ sung sau dry-run 1)_ | | |
| _(bổ sung sau dry-run 2)_ | | |

---

## 11. Cấu trúc repo

```
configs/          mỗi thí nghiệm = 1 YAML, được commit
src/
  data/           parse_corpus, chunker, split_holdout          [P2]
  retrieval/      bm25, dense, hybrid                            [P3]
  rerank/         cross-encoder, calibration                     [P4]
  evaluate.py     bản sao chính xác scoring.py của BTC      [P1, KHOÁ]
  make_submission.py  chốt chặn cuối trước CodaLab          [P1, KHOÁ]
  verify_env.py                                             [P1, KHOÁ]
scripts/          entrypoint chạy được bằng 1 lệnh
tests/            kiểm chứng evaluate.py khớp mã BTC
docker/           Dockerfile — đặc tả môi trường
docs/             scoring_behaviour.md, model_card.md, data_statement.md, notes tuần
INTERFACES.md     hợp đồng dữ liệu giữa 4 người — phần KHOÁ của repo
experiments.csv   mọi thí nghiệm, kèm commit SHA
```

Đọc `INTERFACES.md` trước khi viết dòng code đầu tiên.

## Giấy phép

MIT — xem `LICENSE`. Repo để **private** trong thời gian thi (Điều 11 cấm trao đổi mã nguồn giữa các đội);
chuyển public trong vòng 30 ngày sau chung kết (13/11/2026) theo Điều 10.
