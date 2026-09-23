# Runbook — chạy toàn bộ pipeline

> Nhánh `p3/pipeline-e2e`. Cập nhật 13/09/2026.
> Máy local = MacBook M4 (ARM64, MPS). Kaggle = nơi duy nhất thật sự cần, để encode corpus.
> **Không bao giờ chạy bất cứ thứ gì trên `data/holdout.json`** — mọi script đã chặn cứng,
> muốn qua phải thêm `--allow-holdout`, và chỉ khi cả nhóm đã chốt đó là lần đo cuối.

---

## Giai đoạn 0 — kiểm môi trường (local, 2 phút)

```bash
cd ~/MeanGirls-UIT-DSC2026
source .venv/bin/activate
python scripts/smoke_test.py          # phải in "✅ PASS", 64 test
```

Hỏng ở đây thì dừng, đừng chạy tiếp — mọi con số sau đó vô nghĩa.

---

## Giai đoạn 1 — BM25 chốt + lưới an toàn (local, ~2 phút/lượt)

Đây là **bản nộp dự phòng**: có nó rồi thì mọi thứ sau đều là cải thiện, không phải điều kiện sống còn.

```bash
# 1a. Đo trên dev
python -u scripts/run_pipeline.py --config configs/v0.3_bm25_best.yaml \
    --questions data/dev.json --eval

# 1b. Sinh submission cho public test
python -u scripts/run_pipeline.py --config configs/v0.3_bm25_best.yaml \
    --questions data/public-official.json --submission
```

**Con số phải khớp ở 1a** (nếu lệch ⇒ `chunks.jsonl` hoặc `dev.json` đã khác, dừng lại tìm nguyên nhân):

```
Recall@5   : 0.8555      Recall@20  : 0.9427      Recall@50  : 0.9700
Chấm như BTC: recall=0.8555 precision=0.1792
```

Ra: `outputs/v0.3_bm25_best/submission.zip` — nộp được ngay.

---

## Giai đoạn 2 — encode trên Kaggle (`docs/kaggle_encode.ipynb`)

Bốn bước ở **mỗi** tài khoản, không có bước nào khác:

1. **Thả notebook** — New Notebook → File → Import Notebook → `docs/kaggle_encode.ipynb`.
2. **Add Data** — dataset Private chứa `data/chunks.jsonl`. Sửa `DATASET` ở cell 1 cho khớp
   tên thư mục thật trong `/kaggle/input/`.
3. **Add-ons → Secrets** — thêm `RCLONE_CONF_B64` (base64 của `rclone.conf`). Secret là
   per-account nên phải làm ở cả 4 tài khoản.
4. **Settings** — Accelerator **GPU T4 ×2**, Internet **ON**. Sửa `SHARDS_THIS_ACCOUNT`
   (tk1 `[0,1]` · tk2 `[2,3]` · tk3 `[4,5]` · tk4 `[6,7]`) → **Save Version → Save & Run All
   (Commit)** → tắt máy.

Notebook chạy trong container riêng của Kaggle nên không phụ thuộc trình duyệt. Mỗi mảnh xong là
đẩy ngay lên Drive; hết 12h thì bấm *Save & Run All* lần nữa — nó đọc trạng thái từ Drive, bỏ qua
mảnh đã xong, `--resume` mảnh dở. Khi đủ 8 mảnh, tài khoản chạy sau cùng tự ghép và đẩy
`embeddings.npy` (float16, ~1,07 GB) lên `Drive:DSC2026/embeddings/final/`.

Ước tính ~2,5 giờ GPU cho toàn corpus ⇒ ~20 phút/mảnh. Con số thật hiện ở bước thử 2.000 chunk
(cell 5 in tốc độ chunk/s) — nhân lên để biết trước khi cam kết cả 4 tài khoản.

---

## Giai đoạn 3 — dense + so với BM25 (local)

```bash
pip install torch transformers        # CHƯA có trong .venv; báo P1 để thêm vào requirements + README
python -u scripts/run_pipeline.py --config configs/v0.4_dense.yaml \
    --questions data/dev.json --eval
```

Chỉ mã hoá 1.000 câu hỏi (CPU vài phút), phần corpus đọc thẳng từ `embeddings.npy`.
Nếu vân tay chunk hoặc `repo/revision` không khớp meta, script **dừng ngay** thay vì cho ra số sai.

**Cổng đi tiếp (plan.md Tuần 2):** dense phải cho thấy nó bù được chỗ BM25 yếu.
So với mốc `R@5 = 0,8555 · R@50 = 0,9700`. Dense một mình thấp hơn BM25 là **bình thường** —
giá trị của nó nằm ở chỗ hợp nhất. Nhưng nếu R@50 của dense dưới ~0,90 thì dừng lại kiểm
`max_length`, `pooling`, prefix trước khi đi tiếp.

---

## Giai đoạn 4 — bộ quyết định số lượng doc (local, ~2 phút, không cần GPU)

Tầng duy nhất ăn thẳng vào Precision — tức vào tie-break của bảng xếp hạng. Nguyên tắc:
**fit trên `train_split`, báo cáo trên `dev`**, và một lệnh làm cả hai để không ai chọn
tham số trên tập đo.

```bash
# 4a. Ranking của tập fit (chưa có sẵn — dev có rồi, train_split thì chưa)
python -u scripts/run_pipeline.py --config configs/v0.3_bm25_best.yaml \
    --questions data/train_split.json --out-dir outputs/v0.7_calib/train_split --eval

# 4b. Fit θ trong ngân sách recall, rồi báo cáo lại trên dev với θ đã chốt
python scripts/fit_calibration.py \
    --ranking outputs/v0.7_calib/train_split/ranking_full.json \
    --questions data/train_split.json \
    --verify-ranking outputs/v0.3_bm25_best/ranking_full.json \
    --verify-questions data/dev.json \
    --max-recall-drop 0.003 --out outputs/v0.7_calib/calibration.json

# 4c. Dán khối YAML script in ra vào config, rồi chạy lại pipeline
python -u scripts/run_pipeline.py --config configs/v0.7_bm25_calib.yaml \
    --questions data/dev.json --eval
```

`--max-recall-drop` là **ngân sách khai trước**, không phải số đọc ra sau khi nhìn kết quả.
θ đơn điệu: tăng θ ⇒ recall tăng, precision giảm — script in vài dòng quanh θ\* để kiểm bằng mắt.

⚠️ **θ gắn với đúng loại ranking đã fit.** Tín hiệu bất biến với co giãn và tịnh tiến điểm,
nhưng *hình dạng* khe hở của BM25, của RRF và của cross-encoder thì khác nhau. Đổi tầng
trước đó ⇒ fit lại, đừng bê θ cũ sang.

## Giai đoạn 5 — những thứ CHƯA có code

| Cần | Để làm gì |
|---|---|
| `src/retrieval/hybrid.py` | hợp nhất RRF BM25 + dense ở mức chunk (`type: hybrid` trong YAML) |
| `scripts/tune_rrf.py` | chốt `w`/`k` trên `train_split` rồi mới đo dev — không chọn tham số trên tập đo |

---

## Phụ lục — các phép đo phụ, chạy khi nào muốn (local, đều vài phút)

```bash
# Đồ thị trích dẫn: dựng rồi đo TRẦN TRÊN. Dưới 1% thì bỏ hướng này.
python scripts/build_citation_graph.py build --config configs/v0.3_bm25_best.yaml
python scripts/build_citation_graph.py probe \
    --ranking outputs/v0.3_bm25_best/ranking_full.json \
    --questions data/dev.json --edge-kinds amend

# Lọc theo lĩnh vực (đã đo: hard filter phá 28% câu đang đúng, oracle +0,0588)
python scripts/domain_filter_probe.py \
    --ranking outputs/v0.3_bm25_best/ranking_full.json --questions data/dev.json

# Quét candidate_chunks cho cấu hình chốt (cap KHÔNG trung tính với mean_topN)
python -u scripts/check_candidate_cap.py --config configs/v0.3_bm25_best.yaml \
    --tokenizers syllable_bigram --pools mean_top2
```

---

## Thứ tự nếu thời gian eo hẹp

1. Giai đoạn 0 → 1 (có `submission.zip` trong tay: **30 phút**).
2. Giai đoạn 2 chạy nền trên Kaggle trong lúc làm việc khác.
3. `calibrate.py` — rẻ nhất, không cần GPU, ăn thẳng vào Precision (tie-break).
4. Giai đoạn 3 + hybrid.
5. Phụ lục: chỉ khi còn thời gian.
