# Truy hồi văn bản pháp luật tiếng Việt

Hệ thống truy hồi văn bản pháp luật tiếng Việt: nhận một câu hỏi, trả về **1–5 `doc_id`** liên quan
nhất trong kho 8.507 văn bản. Kết hợp BM25 và dense retrieval bằng Reciprocal Rank Fusion, sau đó
tự quyết định số văn bản trả về để cân bằng Recall và Precision.

Dữ liệu: UIT Data Science Challenge 2026 — Task 1 (Legal Information Retrieval).

## Phương pháp

```
câu hỏi ─┬─► BM25 (syllable_bigram) ──────────────► top-200 doc ─┐
         │                                                        ├─► RRF mức doc ─► cắt theo khe hở điểm ─► 1..5 doc_id
         └─► Dense (Vietnamese_Embedding_v2) ─────► top-200 doc ─┘
```

1. **Chunk** theo tiêu đề `Điều N.`; Điều dài cắt cửa sổ trượt 256 từ / gối 64; văn bản không có
   Điều cắt cửa sổ trượt toàn văn → 432.142 chunk.
2. **BM25** trên chunk với token âm tiết + bigram âm tiết (không cần thư viện tách từ); điểm doc =
   trung bình 2 chunk tốt nhất.
3. **Dense** zero-shot [`AITeamVN/Vietnamese_Embedding_v2`](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
   (568M tham số, pin revision), cosine trên chunk, cùng cách gộp.
4. **RRF** có trọng số trên thứ hạng doc (k=20, BM25/dense = 0,4/0,6).
5. **Quyết định số lượng doc**: cắt ở khe hở điểm chuẩn hoá đầu tiên vượt θ=0,502793.

Không fine-tune, không dùng dữ liệu ngoài. Siêu tham số chọn trên `train_split`, báo cáo trên `dev`.

## Kết quả

| | dev R@5 | dev R@50 | Recall | Precision |
|---|---:|---:|---:|---:|
| BM25 | 0,8547 | 0,9708 | — | — |
| Dense | 0,9234 | 0,9813 | — | — |
| BM25 + Dense (RRF) | 0,9384 | 0,9857 | 0,9384 | 0,1984 |
| **+ quyết định số lượng** | | | **0,9354** | **0,2451** |

Trên `holdout` (đo một lần): R@5 0,9160 · Recall 0,9115 · Precision 0,2379.
Recall/Precision tính theo `src/evaluate.py` (trung bình theo câu, tối đa 5 doc). Toàn bộ thí
nghiệm: [`experiments.csv`](experiments.csv).

### Siêu tham số

Không có trọng số nào được huấn luyện. Các giá trị dưới đây được chọn trên `train_split`
(4.689 câu) và báo cáo một lần trên `dev` (1.000 câu):

| Siêu tham số | Giá trị | Cách chọn |
|---|---|---|
| Tokenizer BM25, cách gộp chunk→doc | `syllable_bigram`, `mean_top2` | lưới 5 × 5 (`scripts/bench_retrieval.py`) |
| RRF: mức hợp nhất, k, trọng số BM25/dense | doc, 20, 0,4/0,6 | lưới 66 ô (`scripts/tune_rrf.py`) |
| Ngưỡng số lượng doc θ | 0,502793 | θ nhỏ nhất giữ mức giảm recall ≤ 0,003 (`scripts/fit_calibration.py`) |

### Ngân sách tham số

| Thành phần | Mô hình | Tham số |
|---|---|---:|
| Lexical | BM25 Okapi (`src/retrieval/bm25.py`) | 0 |
| Dense | `AITeamVN/Vietnamese_Embedding_v2` @ `18b44161` | 567.754.752 |
| Hợp nhất | RRF có trọng số (`src/retrieval/hybrid.py`) | 0 |
| Số lượng doc | ngưỡng khe hở điểm (`src/rerank/calibrate.py`) | 0 |
| **Tổng** | | **567.754.752** |

Tổng bằng 14,2% giới hạn 4 tỷ tham số của cuộc thi (tính cả lớp embedding). Số tham số của các
model ứng viên khác: [`docs/param_audit.md`](docs/param_audit.md).

### Lịch sử phiên bản

| Phiên bản | Phương pháp | dev R@5 | dev Recall / Precision | holdout R@5 | Leaderboard Recall / Precision | Kho chunk |
|---|---|---:|---|---:|---|---:|
| v0.1 | BM25 regex, gộp max | 0,7863 | 0,7863 / 0,1642 | — | — | 524.422 |
| v0.3 | BM25 `syllable_bigram`, `mean_top2` | 0,8555 | 0,8555 / 0,1792 | 0,8449 | public 0,8566 / — | 524.422 |
| v0.4 | + reranker `bge-reranker-v2-m3`, RRF | 0,8733 | — | 0,8559 | public 0,8566 / 0,1838 | 524.422 |
| v0.6 | + kNN câu hỏi train, RRF 3 nguồn | 0,9028 | — | 0,8855 | private 0,8818 / 0,1884 | 524.422 |
| **v0.8** | BM25 + dense, RRF + quyết định số lượng | **0,9384** | **0,9354 / 0,2451** | **0,9160** | — | 432.142 |

Kho chunk 524.422 dùng chiến lược `loose` (nhận cả trích dẫn chéo làm ranh giới), 432.142 dùng
`strict` (chỉ tiêu đề `Điều N.`). Cách tái lập từng bản: [`docs/reproduce.md`](docs/reproduce.md).

## Cài đặt

Yêu cầu: Python **3.11**, RAM ≥ 16 GB, ~10 GB đĩa. GPU chỉ cần để encode kho chunk (một lần);
truy vấn chạy được trên CPU / Apple MPS.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, torch, transformers, ...
pip install pytest
python scripts/smoke_test.py             # kiểm môi trường + test, không cần dữ liệu
```

Trọng số model tự tải từ HuggingFace ở revision pin trong `configs/v0.4_dense.yaml`.

## Dữ liệu

Dữ liệu không đi kèm repo. Đặt vào `data/`:

```
data/
├── selected-contexts/      # 8.532 file context_*.json
├── train.json
├── public-official.json
└── private-official.json
```

## Sử dụng

Chạy toàn bộ từ dữ liệu thô (tiền xử lý → chunk → chia tập → encode → dự đoán):

```bash
python scripts/run_e2e.py --questions data/private-official.json
```

Bước encode được bỏ qua nếu đã có `data/embeddings.npy`; kết quả ở `outputs/v0.8_hybrid_rrf/`.

Đánh giá trên tập dev, hoặc chạy thử không cần dữ liệu và torch:

```bash
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_rrf.yaml --questions data/dev.json --eval
python -u scripts/run_pipeline.py --config configs/v0.8_hybrid_demo.yaml --demo --eval
```

Thời gian tham khảo: encode 432.142 chunk ~2–2,5 giờ trên GPU T4; dự đoán 1.000 câu ~1,5 phút trên
MacBook M4. Từng bước riêng lẻ, giá trị hash kỳ vọng và cách chọn lại siêu tham số:
[`docs/reproduce.md`](docs/reproduce.md).

## Cấu trúc

```
configs/      cấu hình thí nghiệm (YAML, kế thừa bằng `extends`)
src/
  common/     đọc config, I/O file trung gian, tiện ích chung
  data/       parse corpus, chunker, chia tập
  retrieval/  BM25, dense, hybrid (RRF), tokenizer
  rerank/     cross-encoder, quyết định số lượng doc
  evaluate.py           tính Recall/Precision
  make_submission.py    xuất và kiểm file dự đoán
scripts/      run_e2e, run_pipeline, encode_corpus, tune_rrf, fit_calibration, ...
tests/        test cho metric, retriever, hybrid, calibrate, config, chunker
docker/       Dockerfile
docs/         tài liệu kỹ thuật (xem dưới)
```

## Tài liệu

- [`docs/reproduce.md`](docs/reproduce.md) — tái lập từng bước, hash kỳ vọng, chọn siêu tham số
- [`docs/model_card.md`](docs/model_card.md) — thành phần, số tham số, đánh giá, hạn chế
- [`docs/data_statement.md`](docs/data_statement.md) — nguồn dữ liệu, tiền xử lý, cách chia tập
- [`docs/scoring_behaviour.md`](docs/scoring_behaviour.md) — hành vi của hàm chấm ở các ca biên
- [`INTERFACES.md`](INTERFACES.md) — định dạng dữ liệu giữa các module

## Giấy phép

MIT — xem [`LICENSE`](LICENSE).
