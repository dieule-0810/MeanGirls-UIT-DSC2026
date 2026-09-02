# Bảng phân loại lỗi — LegalIR

> Chủ sở hữu: P4 · v1 chốt 13/08/2026, **trước** khi đọc câu sai đầu tiên.
> **v2 chốt 02/09/2026**, sau vòng đọc tay 1 (30 câu).
>
> Vì sao chốt trước: nếu vừa đọc vừa nghĩ ra nhãn, nhãn sẽ trôi theo những gì mình
> gặp đầu tiên, và số liệu tuần này không so được với tuần sau. Muốn đổi taxonomy
> thì đổi có phiên bản, ghi ngày, và **không sửa lại nhãn cũ**.
>
> **Thay đổi v1 → v2** (chi tiết ở mục *Nhật ký phiên bản* cuối file):
> thêm `N-TRUNC`, `N-CONSOLID`; sửa định nghĩa `N-VERSION`; thêm `R-ENTITY`;
> sửa nguyên tắc 4 (nguyên tắc bằng lời không đủ, phải có `assert`).
> Vòng đọc 1 đã dùng bộ v2 này — không có nhãn v1 nào cần sửa lại.

## Nguyên tắc

1. Mỗi câu sai nhận **đúng một** nhãn tầng 1 và **một hoặc nhiều** nhãn tầng 2.
2. Tầng 1 do máy gán (suy ra được từ thứ hạng). Tầng 2 do người đọc gán.
3. Mỗi nhãn tầng 2 phải có **một người chịu trách nhiệm**. Nhãn không ai nhận là
   nhãn vô dụng — bỏ khỏi bảng.
4. Chỉ đọc trên `data/error_pool.json`. **Không bao giờ** đọc `data/holdout.json`.
   🔴 **v2:** nguyên tắc này đã bị vi phạm một lần — `cmd_sample()` lấy mẫu từ
   `train.json` vốn *chứa cả* holdout, nên vòng đọc 1 dính 2 câu (`62912`,
   `119006`). Nguyên tắc bằng lời không đủ; nay `audit_labels.py` và
   `build_audit_packet.py` đều có `assert` chống rò rỉ.
5. Nhãn nhóm `N-*` là khẳng định về **bài toán và dữ liệu**, không phải về hệ
   thống. Khi báo cáo, **tách `N-WRONG` (nhãn BTC sai) khỏi các nhãn còn lại
   (bản chất kho văn bản)**. Gộp chung là tố sai BTC.

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
| `R-ENTITY` **(v2)** | Khớp mạnh vào **khung điều luật** dùng chung nhưng sai **thực thể/lĩnh vực** (vd Luật PPP Đ.43 ↔ Luật Dầu khí Đ.23; BLTTDS Đ.157 ↔ Luật TTHC Đ.358) | Loại hard negative giá trị nhất trong corpus pháp luật VN. Đưa vào mining; đây là chỗ cross-encoder phải ăn điểm |

### Nhóm rerank / calibration — P4 chịu trách nhiệm

| Mã | Dấu hiệu nhận biết | Hành động đề xuất |
|---|---|---|
| `K-CHUNKPICK` | Chunk được chọn để chấm không phải chunk chứa câu trả lời | Đổi cách chọn chunk đại diện |
| `K-MERGE` | Doc gold có một chunk điểm rất cao nhưng bị chiến lược gộp dìm xuống | So max / mean-top-3 / logsumexp |
| `K-MULTI` | Câu hỏi cần nhiều văn bản, hệ thống chỉ tìm được một | Ảnh hưởng trực tiếp Recall — 7,9% số câu |
| `K-CONF` | Điểm của doc đúng và doc sai gần bằng nhau, không phân biệt được | Đầu vào cho thiết kế bộ calibration Tuần 5 |

### Nhóm nhãn — không ai sửa được, chỉ đếm

> Dùng chung bộ mã này cho **cả hai** ngữ cảnh: phân tích lỗi hàng tuần (nhãn tầng 2)
> và audit nhãn đọc tay (cột `verdict` của `manual_audit_r*.csv`). Một bộ từ vựng,
> một nguồn sự thật — v1 để hai chỗ lệch nhau và đã gây lẫn lộn một lần.

| Mã | Dấu hiệu nhận biết | Vì sao vẫn phải đếm |
|---|---|---|
| `N-ALTOK` | Có văn bản khác **thật sự trả lời được** câu hỏi nhưng không nằm trong gold | Đây là **trần trên thật** của điểm số. Mô hình không có cách nào biết BTC chọn bản nào |
| `N-CONSOLID` **(v2)** | Gold **trùng nội dung** với một **văn bản hợp nhất** (VBHN) cũng có trong kho — cùng Điều, cùng con số | Trùng lặp trong kho, **không** phải xung đột phiên bản. Xử lý được bằng gộp cụm như dedup. Corpus có 27 VBHN (0,32%) |
| `N-VERSION` **(sửa ở v2)** | Nội dung gold **đã bị sửa đổi** bởi văn bản khác, và **bản sửa đúng hơn gold** | Khác `N-CONSOLID`: **không gộp cụm được**, phải chọn đúng đời. Từ neo thời gian trong câu hỏi ("hiện nay", "mới nhất") là dấu hiệu chính |
| `N-TRUNC` **(v2)** | Gold đúng văn bản, nhưng nội dung trả lời nằm ở **file đính kèm / phụ lục** không có trong corpus | Không mô hình nào giải được, nhưng **không phải nhãn sai**. Phát hiện tự động bằng `audit_labels.py --scan-truncated` |
| `N-WRONG` | Gold **không** trả lời được câu hỏi — nhãn sai thật sự | Khẳng định mạnh nhất. `note` bắt buộc, và phải có người thứ hai kiểm lại |
| `N-AMBIG` | Câu hỏi mơ hồ tới mức không xác định được doc nào **nên** đúng | Thiếu **phạm vi** (ngành/đối tượng). Phân biệt với `N-VERSION` = thiếu **mốc thời gian** |

**Cây quyết định khi phân vân giữa `N-CONSOLID` / `N-VERSION` / `N-ALTOK`:**

```
Văn bản thay thế có phải VBHN của chính gold không?
├─ có  → nội dung có giống hệt gold không?
│        ├─ giống  → N-CONSOLID
│        └─ khác   → N-VERSION
└─ không → nó SỬA ĐỔI gold, và bản sửa đúng hơn?
           ├─ đúng  → N-VERSION
           └─ không → hai văn bản độc lập cùng trả lời được → N-ALTOK
```

**Bốn ca mẫu từ vòng đọc 1** (tra khi phân vân):

| qid | Nhãn | Vì sao |
|---|---|---|
| 68836 | `N-ALTOK` | 4 sàn (HOSE/HNX/UPCoM/phái sinh) cùng quy tắc ưu tiên giá–thời gian; câu hỏi không nêu sàn |
| 116370 | `N-CONSOLID` | `288036` = VBHN 03/VBHN-BGDĐT 2017; **cùng Điều 6, cùng 19 tiết** |
| 72846 | `N-VERSION` | `232314` = NĐ 64/2022 **sửa** khoản 3 Điều 20; câu hỏi có "hiện nay" |
| 119006 | `N-TRUNC` | doc `169533` chỉ 1.347 ký tự, Quy chế nằm ở `FILE ĐÍNH KÈM` |

---

## Quy ước ghi nhãn

**Phân tích lỗi hàng tuần** — ghi vào `outputs/error_analysis/<exp_id>.csv`,
một dòng một câu hỏi:

```csv
qid,l1,l2,gold_rank,n_gold,confidence,note
147194,L1-RETRIEVE,R-TERM|D-CHUNK,none,1,high,"hỏi 'tiền hỗ trợ' - văn bản ghi 'khoản trợ cấp'"
```

- `l2` nhiều nhãn phân tách bằng `|`, **tối đa 3**. Nhiều hơn 3 nghĩa là chưa
  hiểu câu đó — ghi `confidence=low` và ghi rõ vào `note`.
- `confidence`: `high` / `medium` / `low` (**v2** thêm `medium` — dùng khi ranh
  giới giữa hai nhãn mảnh nhưng vẫn quyết được, vd `OK` vs `N-ALTOK`). Khi tổng
  hợp, báo cáo riêng phần `low` — nếu quá 20% là `low` thì taxonomy chưa đủ tốt,
  phải sửa trước khi đọc tiếp. Quá 3 phút chưa quyết được → ghi `low`, đi tiếp.
- `note` là bắt buộc với mọi nhãn thuộc nhóm `N-*`, vì đó là những khẳng định
  mạnh nhất bạn đưa ra (nói nhãn của BTC sai) và phải có bằng chứng.

**Audit nhãn đọc tay** — ghi vào `outputs/label_audit/manual_audit.csv` (file đang
làm việc), đổi tên thành `manual_audit_r{N}.csv` khi chốt xong một vòng:

```csv
qid,verdict,confidence,note,question,gold_ids,gold_names,gold_links
119006,N-TRUNC,high,"doc 169533 chỉ 1.347 ký tự, Quy chế ở FILE ĐÍNH KÈM",...
```

- `verdict` lấy từ **cùng bảng nhóm nhãn ở trên**, cộng thêm `OK`.
- `audit_labels.py` fail-loud nếu gặp verdict lạ — gõ sai là dừng, không đếm nhầm.
- `already_read()` chỉ quét `manual_audit_r*.csv`, nên file đang làm dở không ảnh
  hưởng pool của vòng sau.

## Cảnh báo về lấy mẫu

Nếu lấy mẫu phân tầng (vd 25 câu `L1-RETRIEVE` + 25 câu `L1-RERANK`), thì mọi
phần trăm báo cáo là **phần trăm trong tầng đó**, không phải phần trăm toàn cục.
Khi viết bài báo phải quy đổi lại theo tỉ lệ thật của hai tầng. Ghi rõ cỡ mẫu và
cách lấy mẫu ngay đầu mỗi báo cáo tuần.

---

## Nhật ký phiên bản

### v2 — 02/09/2026, sau vòng đọc tay 1 (30 câu)

| Thay đổi | Lý do |
|---|---|
| Thêm `N-TRUNC` | qid 119006: gold đúng nhưng nội dung ở file đính kèm. Không khớp bất kỳ nhãn nào của v1 — `N-WRONG` thì đổ oan cho BTC, `OK` thì sai |
| Thêm `N-CONSOLID` | Hai ca (116370, 5576) từng bị gán nhầm `N-VERSION` dù ghi chú tự nó nói "cùng Điều, cùng con số" — tức mâu thuẫn với định nghĩa `N-VERSION` của v1 |
| Sửa định nghĩa `N-VERSION` | Từ "một đời luật khác" thành "nội dung **đã bị sửa**; bản sửa **đúng hơn** gold". Hệ quả retrieval khác hẳn `N-CONSOLID` |
| Thêm `R-ENTITY` | 3 cặp tìm được ở vòng đọc 1: cùng khuôn mẫu điều luật, khác lĩnh vực. Luật VN tái sử dụng khuôn mẫu rất nhiều |
| Thêm `medium` vào `confidence` | v1 chỉ có `high`/`low`; 4/30 ca ở ranh giới mảnh nhưng vẫn quyết được — ép về `low` sẽ thổi phồng tỉ lệ `low` |
| Gộp một bộ từ vựng cho cả error analysis và label audit | v1 để hai chỗ dùng hai bộ khác nhau, đã gây lẫn lộn một lần |
| Nguyên tắc 4 + 5 | Rò rỉ holdout (§2.6a `P4_TASKS.md`); và yêu cầu tách `N-WRONG` khỏi phần còn lại khi báo cáo |

**Không có nhãn v1 nào bị sửa lại** — vòng đọc 1 diễn ra sau khi v2 được chốt.

### v1 — 13/08/2026
Bản đầu, chốt trước khi đọc câu sai đầu tiên.
