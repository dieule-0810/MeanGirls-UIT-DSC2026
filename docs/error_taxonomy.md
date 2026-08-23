# Bảng phân loại lỗi — LegalIR

> Chủ sở hữu: P4 · Chốt ngày 13/08/2026, **trước** khi đọc câu sai đầu tiên.
>
> Vì sao chốt trước: nếu vừa đọc vừa nghĩ ra nhãn, nhãn sẽ trôi theo những gì mình
> gặp đầu tiên, và số liệu tuần này không so được với tuần sau. Muốn đổi taxonomy
> thì đổi có phiên bản (`v2`), ghi ngày, và **không sửa lại nhãn cũ**.

## Nguyên tắc

1. Mỗi câu sai nhận **đúng một** nhãn tầng 1 và **một hoặc nhiều** nhãn tầng 2.
2. Tầng 1 do máy gán (suy ra được từ thứ hạng). Tầng 2 do người đọc gán.
3. Mỗi nhãn tầng 2 phải có **một người chịu trách nhiệm**. Nhãn không ai nhận là
   nhãn vô dụng — bỏ khỏi bảng.
4. Chỉ đọc trên `data/error_pool.json`. **Không bao giờ** đọc `data/holdout.json`.

---

## Tầng 1 — Hỏng ở đâu (máy gán tự động)

| Mã | Định nghĩa | Ý nghĩa |
|---|---|---|
| `L1-RETRIEVE` | Gold doc **không có** trong top-K của retriever | Trần cứng. Reranker không cứu được. → P3 |
| `L1-RERANK` | Gold doc **có** trong top-K nhưng bị đẩy khỏi top-5 | Reranker chấm sai. → P4 |
| `L1-CALIBRATE` | Gold doc nằm trong top-5 nhưng bị cắt bởi bộ quyết định số lượng | Chỉ xuất hiện từ Tuần 5. → P4 |
| `L1-LABEL` | Doc hệ thống trả về **cũng trả lời được** câu hỏi, nhưng không phải gold | Không phải lỗi hệ thống. → không sửa được, chỉ đếm |

> Tuần 1 chưa có reranker: mọi lỗi hoặc là `L1-RETRIEVE` (không trong top-50),
> hoặc là `L1-RERANK` (trong top-50, ngoài top-5) — với "reranker" ở đây là chính
> thứ tự BM25. Vẫn ghi đúng mã để so được với Tuần 2.

---

## Tầng 2 — Vì sao (người đọc gán)

### Nhóm dữ liệu — P2 chịu trách nhiệm

| Mã | Dấu hiệu nhận biết | Hành động đề xuất |
|---|---|---|
| `D-CHUNK` | Đoạn chứa câu trả lời bị cắt đôi giữa chừng; hoặc chunk trúng chỉ toàn quốc hiệu/tiêu ngữ/nơi nhận | Sửa chunker: ưu tiên ranh giới `Điều N` |
| `D-NORM` | Còn `\r\n\n` chèn giữa câu, mất dấu tiếng Việt, ký tự lạ trong chunk gold | Sửa normalize |
| `D-BOILER` | Doc gold thắng/thua vì phần đầu-cuối hành chính chứ không vì nội dung | Tách boilerplate mạnh tay hơn |
| `D-EMPTY` | Doc gold có `passage` rỗng hoặc gần rỗng (17 file đã biết) | Quyết định loại khỏi corpus hay giữ |
| `D-LONG` | Doc gold rất dài, tín hiệu bị loãng trên nhiều chunk | Cân nhắc chiến lược gộp khác |

### Nhóm truy hồi — P3 chịu trách nhiệm

| Mã | Dấu hiệu nhận biết | Hành động đề xuất |
|---|---|---|
| `R-TERM` | Câu hỏi dùng từ thường dân, văn bản dùng thuật ngữ pháp lý (vd "tiền hỗ trợ" vs "khoản trợ cấp") | Dense retrieval / mở rộng truy vấn |
| `R-SEG` | Tách từ sai làm hỏng khớp (vd "quy định" bị tách thành "quy" + "định") | Đổi tokenizer, so `underthesea` vs whitespace |
| `R-NUM` | Câu hỏi nhắc số hiệu cụ thể (`Nghị định 15/2022/NĐ-CP`, `Điều 7`) mà hệ thống không ưu tiên khớp chính xác | BM25 nên mạnh ở đây — nếu vẫn sai là lỗi index |
| `R-SHORT` | Câu hỏi quá ngắn hoặc quá chung, ít tín hiệu để phân biệt | Ghi nhận, khó sửa |
| `R-DUP` | Trả về một văn bản **gần trùng** với gold (bản sửa đổi, bản hợp nhất) | Cần chiến lược khử gần trùng |

### Nhóm rerank / calibration — P4 chịu trách nhiệm

| Mã | Dấu hiệu nhận biết | Hành động đề xuất |
|---|---|---|
| `K-CHUNKPICK` | Chunk được chọn để chấm không phải chunk chứa câu trả lời | Đổi cách chọn chunk đại diện |
| `K-MERGE` | Doc gold có một chunk điểm rất cao nhưng bị chiến lược gộp dìm xuống | So max / mean-top-3 / logsumexp |
| `K-MULTI` | Câu hỏi cần nhiều văn bản, hệ thống chỉ tìm được một | Ảnh hưởng trực tiếp Recall — 7,9% số câu |
| `K-CONF` | Điểm của doc đúng và doc sai gần bằng nhau, không phân biệt được | Đầu vào cho thiết kế bộ calibration Tuần 5 |

### Nhóm nhãn — không ai sửa được, chỉ đếm

| Mã | Dấu hiệu nhận biết | Vì sao vẫn phải đếm |
|---|---|---|
| `N-ALTOK` | Doc hệ thống trả về **thật sự trả lời được** câu hỏi nhưng không nằm trong gold | Đây là **trần trên thật** của điểm số. Nếu 15% lỗi là loại này thì Recall 0,85 đã là kịch trần, và mọi nỗ lực thêm là lãng phí |
| `N-WRONG` | Gold doc **không** trả lời được câu hỏi — nhãn sai | Đã phát hiện dấu hiệu: cùng câu hỏi, gold khác nhau (`plan.md` mục P4) |
| `N-AMBIG` | Câu hỏi mơ hồ tới mức không xác định được doc nào đúng | Ghi nhận, dùng để giải thích phần Recall không bao giờ đạt được |

---

## Quy ước ghi nhãn

Ghi vào `outputs/error_analysis/<exp_id>.csv`, một dòng một câu hỏi:

```csv
qid,l1,l2,gold_rank,n_gold,confidence,note
147194,L1-RETRIEVE,R-TERM|D-CHUNK,none,1,high,"hỏi 'tiền hỗ trợ' - văn bản ghi 'khoản trợ cấp'"
```

- `l2` nhiều nhãn phân tách bằng `|`, **tối đa 3**. Nhiều hơn 3 nghĩa là chưa
  hiểu câu đó — ghi `confidence=low` và ghi rõ vào `note`.
- `confidence`: `high` / `low`. Khi tổng hợp, báo cáo riêng phần `low` — nếu
  quá 20% là `low` thì taxonomy chưa đủ tốt, phải sửa trước khi đọc tiếp.
- `note` là bắt buộc với mọi nhãn thuộc nhóm `N-*`, vì đó là những khẳng định
  mạnh nhất bạn đưa ra (nói nhãn của BTC sai) và phải có bằng chứng.

## Cảnh báo về lấy mẫu

Nếu lấy mẫu phân tầng (vd 25 câu `L1-RETRIEVE` + 25 câu `L1-RERANK`), thì mọi
phần trăm báo cáo là **phần trăm trong tầng đó**, không phải phần trăm toàn cục.
Khi viết bài báo phải quy đổi lại theo tỉ lệ thật của hai tầng. Ghi rõ cỡ mẫu và
cách lấy mẫu ngay đầu mỗi báo cáo tuần.
