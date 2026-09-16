# reproduce.md — tái lập kết quả nộp bài

> **Trạng thái: BẢN NHÁP, P4 dựng 16/09/2026.** Mục 1 và 3 CẦN P2/P3 xác nhận —
> xem các khối ⚠️ TODO. Mục 2, 4, 5, 6 đã được P4 chạy và kiểm.
>
> BTC yêu cầu tái lập được thực nghiệm. Tài liệu này là thứ họ sẽ chạy theo.
> Nguyên tắc: **mọi lệnh ở đây phải là lệnh ĐÃ CHẠY THẬT**, không phải lệnh lẽ ra
> nên chạy. Chỗ nào chưa xác nhận thì ghi rõ là chưa xác nhận.

---

## 0. Môi trường

```
Python 3.11  (src/verify_env.py chặn cứng, xem REQUIRED_PYTHON)
pip install -r requirements.txt        # numpy 1.26.4 · scipy 1.13.1 · PyYAML 6.0.1
pip install -r requirements-p3.txt     # CHỈ cần nếu dùng tokenizer pyvi/underthesea.
                                       # Đường nộp bài dùng syllable_bigram → KHÔNG cần.
```

Bước rerank cần thêm `torch` + `transformers`. GPU đã dùng: RTX 3050 6 GB, fp16.
Chạy được trên CPU nhưng chậm hơn khoảng hai bậc độ lớn.

```bash
python -m src.verify_env
python -m pytest tests/ -q          # kỳ vọng: 98 passed, 2 skipped
```

Trọng số tải từ HuggingFace, **pin revision theo `configs/models.yaml`**. Model duy nhất
trong đường nộp bài: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a`, 567.755.777 tham số.
Tổng tham số toàn hệ thống = 567.755.777 (BM25 không có tham số học được) — dưới trần 4 tỷ
của BTC. Kiểm toán: `docs/param_audit.md`, đếm lại bằng `scripts/count_params.py`.

---

## 1. Dữ liệu ⚠️ CẦN P2 XÁC NHẬN

Đặt dữ liệu BTC vào `data/`: `selected-contexts/`, `train.json`, `public-official.json`.
Không file nào trong số này được commit (`.gitignore` dòng 3).

```bash
python -m src.data.parse_corpus --corpus-dir data/selected-contexts --out data/corpus_clean.jsonl
python -m src.data.chunker      --input data/corpus_clean.jsonl --out data/chunks.jsonl
python -m src.data.split_data   --corpus data/corpus_clean.jsonl --train data/train.json --out-dir data
```

Kỳ vọng: `corpus_clean.jsonl` **8.507** dòng · `chunks.jsonl` **524.422** dòng ·
`split_data` sinh `holdout.json` 1.000 · `dev.json` 1.000 · `error_pool.json` 300 ·
`train_split.json` 4.689 (7.000 − 11 câu Vùng Chết).

> ⚠️ **TODO-P2 (1) — tham số chunking.** Lệnh trên dùng CLI default `--chunk-size 256
> --overlap 64`. Mọi file trong `configs/` ghi `max_words: 180 / overlap_words: 45`, và
> `chunker.py` **không nhận `--config`** nên các giá trị đó chưa bao giờ có tác dụng.
> Đo trên 50.000 chunk đầu của `data/chunks.jsonl`: mean 163,7 từ · median 187 · **max 261**
> — khớp 256, không khớp 180. Kết luận tạm: chunk hiện có dựng bằng default.
> P2 xác nhận rồi thì sửa cả ba config cho khớp thực tế, hoặc cho `chunker.py` đọc config.
>
> ⚠️ **TODO-P2 (2) — `run_v0.1.py` không chạy được.** Nó gọi `src.data.split_holdout`
> (module thật tên `split_data`) và gọi `src.data.chunker --config <cfg>` (chunker không
> có tham số đó). Pipeline một-lệnh gãy ở bước 2 và bước 3. Cho tới khi vá xong,
> **mục 1 này là quy trình chuẩn**, không phải `run_v0.1.py`.
>
> ⚠️ **TODO-P2 (3)** — `INTERFACES §1` ghi `corpus_clean` 8.532 dòng, thực tế 8.507.

---

## 2. Xác minh chỉ mục BM25 (tuỳ chọn, ~3 phút)

Bước này không sinh ra kết quả nào, nó chỉ trả lời "chunks.jsonl của tôi có giống của
team không". Nếu bốn con số dưới đây khớp thì mọi thứ sau đó sẽ khớp.

```bash
python - <<'PY'
import json, sys
sys.path.insert(0, '.')
from src.retrieval.tokenizers import get_tokenizer
tk = get_tokenizer('syllable_bigram', fold_tone=True)
v, n, ntok = set(), 0, 0
for line in open('data/chunks.jsonl', encoding='utf-8'):
    if not line.strip():
        continue
    t = tk(json.loads(line)['text']); v.update(t); ntok += len(t); n += 1
print(f"n_chunks={n}  vocab={len(v)}  avgdl={ntok/n:.1f}")
PY
```

Kỳ vọng: `n_chunks=524422  vocab=1212503  avgdl=309.9` — khớp `outputs/v0.3_bm25_best/run_meta.json`.

Đây cũng là bằng chứng khôi phục `fold_tone: true` cho `configs/v0.3_bm25_best.yaml`:
chạy lại với `fold_tone=False` cho `vocab=1215933`, không khớp `run_meta`.

---

## 3. Đường nộp bài ⚠️ CHỜ KẾT QUẢ RERANK TRÊN POOL MỚI

Cấu hình: `configs/v0.3_bm25_best.yaml` (BM25 `syllable_bigram` + `mean_top2` +
`candidate_chunks=2000`) hợp nhất RRF `w=0,6`, `rrf_k=60`, với `bge-reranker-v2-m3`
top-20, 1 chunk/doc, `--prepend-name`.

### 3.1 BM25 → top-50 (CPU)

Chi phí đo thật (16/09/2026, máy P4): đọc chunk 8,4 s · tách từ 82,7 s · dựng chỉ mục
242,8 s · truy vấn **72 ms/câu** → khoảng **6,5 phút** cho 1.000 câu kể cả index.
RAM đỉnh ~7 GB.

> ⚠️ `run_meta.json` của lần chạy v0.3 ghi `index_seconds: 34.3`, **không tái lập được**
> (đo lại ra 242,8 s). Nhưng bốn con số mô tả *nội dung* chỉ mục thì khớp tuyệt đối:
> vocab 1.212.503 · nnz 96.134.846 · avgdl 309,9 · 524.422 chunk / 8.507 doc. Kết luận:
> chỉ mục giống hệt, trường thời gian trong `run_meta` đo thứ khác (nhiều khả năng
> không tính bước tách từ, hoặc chạy với cache ấm). **Đừng dùng `index_seconds` làm tiêu
> chí kiểm tra tái lập** — dùng vocab/nnz/avgdl.

```bash
python -m scripts.p4_build_ranking --config configs/v0.3_bm25_best.yaml \
    --questions data/public-official.json \
    --out outputs/v0.4_submit/public_bm25_top50.json --top-k 50
```

Đối chứng trên tập có nhãn (phải ra **R@5 = 0,8555** trên dev n=1000):

```bash
python -m scripts.p4_build_ranking --config configs/v0.3_bm25_best.yaml \
    --questions data/dev.json \
    --out outputs/v0.3_bm25_best/ranking_full.json --top-k 50
```

### 3.2 Rerank top-20 (GPU, ~80 phút cho 20.000 cặp trên RTX 3050 6 GB)

`--model` nhận **alias** trong `MODELS` của `src/rerank/cross_encoder.py`
(`bge-m3` · `mminilm` · `viranker`), **không** nhận repo id HuggingFace. Alias mới là chỗ
pin `revision`; truyền repo id thẳng sẽ bỏ qua lớp pin đó, nên script chặn cứng.
`bge-m3` → `BAAI/bge-reranker-v2-m3@953dc6f6f85a`.

```bash
python -m scripts.p4_rerank --model bge-m3 \
    --ranking outputs/v0.4_submit/public_bm25_top50.json \
    --questions data/public-official.json \
    --rerank-top 20 --chunks 1 --prepend-name \
    --batch-size 8 --max-length 512 \
    --out outputs/p4_rerank/public_bge_top20_syl.json
```

> ⚠️ **TODO-P4 — `--max-length 512` có thể đang cắt cụt đầu vào.** Lý do chọn 512 ghi
> trong `P4_TASKS §3.2` là "chunk 180 từ". Chunk thật dài tới 261 từ (mục 1). Với
> XLM-R tiếng Việt ~1,8 subword/từ thì đuôi phân bố vượt 512. Chưa đo tỉ lệ bị cắt.
> Probe rẻ: chạy lại `dev_sub300` với `--max-length 1024` và so với
> `outputs/p4_rerank/devsub_bge_top20_name.json`.

### 3.3 Hợp nhất RRF, `w` CỐ ĐỊNH

```bash
python -m scripts.p4_fuse --bm25 outputs/v0.4_submit/public_bm25_top50.json \
    --rerank outputs/p4_rerank/public_bge_top20_syl.json \
    --questions data/public-official.json --top-k 20 --w 0.6 --rrf-k 60 \
    --out outputs/v0.4_submit/public_rrf_w06.json
```

**`--w` là bắt buộc.** Không có nó, script quét `w` 0→1 rồi ghi ra thứ hạng của `w` tốt
nhất — trên tập thi đó là tune trên tập thi. Bản vá hiện tại dừng hẳn nếu tập câu hỏi
không có nhãn mà thiếu `--w`.

`w = 0,6` chốt từ trước, không chọn trên tập này: `w*` rơi vào 0,6–0,7 ở cả 5 lần chạy
độc lập qua 3 reranker và 2 độ sâu (`P4_TASKS §0.2(5)`).

### 3.4 Dựng submission.zip

```bash
python -m scripts.p4_to_preds --ranking outputs/v0.4_submit/public_rrf_w06.json \
    --questions data/public-official.json --out outputs/v0.4_submit/public_preds.json

python -m src.make_submission --preds outputs/v0.4_submit/public_preds.json \
    --questions data/public-official.json --corpus data/corpus_clean.jsonl \
    --out outputs/v0.4_submit/submission.zip
```

`make_submission` khử trùng lặp **rồi mới** cắt 5 (mẫu số Precision là `len(list)` sau
khi cắt), ép mọi `doc_id` về `str` (int vs str = 0 điểm im lặng), và dừng nếu tập qid
không khớp chính xác (thiếu/thừa qid → BTC raise → **Failed**, mất một lượt nộp).
Hành vi mã chấm đã xác minh: `docs/scoring_behaviour.md`, test `tests/test_scoring.py` 24/24.

---

## 4. Số cần khớp

| Hệ thống | Tập | R@5 |
|---|---|---:|
| BM25 `regex` + `max` | dev n=1000 | 0,7863 |
| BM25 `regex` + `mean_top2` | dev n=1000 | 0,8116 |
| BM25 `syllable_bigram` + `mean_top2` | dev n=1000 | **0,8555** |
| + RRF `w=0,6` với rerank dựng trên pool `regex` | dev n=1000 | 0,8736 |
| + RRF `w=0,6` với rerank dựng trên pool `syllable_bigram` | dev n=1000 | ⚠️ **chưa đo** |

> ⚠️ Dòng 0,8736 **không mô tả hệ thống ở mục 3**. File rerank dùng để tạo ra nó
> (`outputs/p4_rerank/dev_bge_top20_name.json`) dựng trên pool `regex`, trong khi mục 3
> rerank trên pool `syllable_bigram`. Độ chồng lấn top-20 giữa hai pool: **44,0%** theo
> cặp (doc, chunk), 62,7% theo doc — tức 56% số cặp đi vào hợp nhất là cặp reranker chưa
> từng chấm. Dòng cuối bảng là con số phải điền trước khi báo cáo bất cứ thứ gì.

Mọi Δ trong bảng đều đã qua `scripts/p4_paired_test.py` (McNemar + bootstrap cặp).
Không dùng sai số biên: SE của *hiệu* chỉ phụ thuộc số câu bất đồng.

---

## 5. Hai kiểm tra còn nợ, đều đe doạ số ở mục 4

**(a) `candidate_chunks=2000` so với `null` trên `syllable_bigram`.**
Tương đương mới chỉ được chứng minh trên tokenizer `regex` (300/300 câu `error_pool`
có top-50 giống hệt). `syllable_bigram` có nnz 96,1M so với 43,8M nên phân bố điểm chunk
khác hẳn. Nếu 2000 không còn tương đương thì +0,0439 bị pha tạp bởi hiệu ứng cắt ứng viên.
Chạy trên `error_pool` 300 câu, hai config chỉ khác dòng `candidate_chunks`. **Cảnh báo
RAM: index sylbigram không cắt ứng viên tốn khoảng 6–7 GB, không phải 3,1 GB.**

**(b) `fold_tone` bị trộn vào khoản +0,0439.**
Nền 0,8116 chạy trên `configs/v0.1_bm25_cap2000.yaml` với `fold_tone: false`; lần chạy
0,8555 dùng `fold_tone: true` (chứng minh ở mục 2). Vậy +0,0439 gộp **hai** thay đổi.
Ablation tách đôi: `syllable_bigram` + `fold_tone: false` + `mean_top2` trên dev-1000.
Một lần index, vài phút. Bài báo không quy công cho tokenizer được trước khi có số này.

---

## 6. Ghi chú về `outputs/`

`.gitignore` dòng 25 (`outputs/*`) chặn toàn bộ thư mục kết quả, nên mọi file ranking chỉ
tồn tại trên ổ đĩa từng người. Tài liệu này tồn tại để bù cho điều đó: chạy theo mục 1→3
là dựng lại được tất cả.

Hai thư mục trong `outputs/` dễ gây hiểu nhầm:
- `outputs/v0.2_bm25_tok/` là **chạy demo** (`"demo": true`, 429 chunk, 60 câu). Không
  dùng được cho kết luận nào.
- `outputs/v0.3_bm25_best/` là chạy thật, đầy đủ 524.422 chunk, dev-1000. Đây là nguồn
  của 0,8555.

`.gitignore` dòng 44 có hai mẫu dính liền do thiếu newline, và mẫu đó còn trỏ sai thư mục
(`outputs/label_audit/error_audit_r*.csv`, file thật ở `outputs/error_audit/`).
