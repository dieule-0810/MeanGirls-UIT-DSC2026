# Hướng dẫn đọc 50 câu lỗi truy vấn — vòng 1

> **Người làm:** P4 · **Ngày tạo:** 09/09/2026 · **Thời lượng:** ~2,5 giờ (3 phút/câu + nghỉ)
>
> File này **tự chứa**: mở session mới, đọc file này là đủ để làm, không cần ngữ cảnh khác.
>
> **Đầu vào:** `outputs/error_audit/error_packet.html` (đọc) + `error_audit_r1.csv` (điền)
> **Đầu ra:** CSV đã điền → `docs/error_analysis_notes.md` → gửi P2 và P3

---

## 1. Việc này là gì, và KHÔNG phải là gì

Bạn đang trả lời **"vì sao hệ thống trượt câu này?"** — mã `R-*`.

Bạn **không** trả lời "gold BTC gán có đúng không?" — đó là việc khác, mã `N-*`, đã làm
xong ở `outputs/label_audit/manual_audit_r1.csv` (30 câu, 7 không-OK). Hai việc dùng hai
bộ mã, cho hai người nhận khác nhau. Đừng trộn.

Đây là task Tuần 1 trong `plan.md`, bị chặn 4 tuần vì chưa có ranking BM25 để đọc.

## 2. Bối cảnh số liệu — đọc trước khi bắt đầu

Hệ thống được đọc: **BM25 + `max`-pool**, `configs/v0.1_bm25.yaml`, trên `dev` n=1.000.

```
R@5  = 0,7863      R@50 = 0,9503
769 câu ĐÚNG (đủ gold trong top-5)
193 câu MISS5   — gold có trong top-50 nhưng ngoài top-5  → lỗi XẾP HẠNG → của P4
 38 câu MISS50  — không gold nào trong top-50             → lỗi BAO PHỦ  → của P3
```

**Hai loại thất bại thuộc hai người khác nhau.** Trộn chung rồi báo cáo một tỉ lệ là làm
cả P3 lẫn P4 không biết phần nào của mình. Cột `mode` trong CSV đã tách sẵn.

### 🔴 Hệ số quy đổi — chỗ dễ sai nhất

Mẫu 50 câu **lấy vượt tỉ lệ** nhóm MISS50 (20/38 câu) vì cả tập chỉ có 38 câu; lấy đúng
tỉ lệ thì chỉ được 8 câu, không đủ nói gì với P3.

| Nhóm | Trong mẫu | Tỉ lệ thật trên dev | Hệ số quy đổi |
|---|---:|---:|---:|
| MISS5 | 30 (60%) | 19,3% | **×0,322** |
| MISS50 | 20 (40%) | 3,8% | **×0,095** |

**Mọi % tính từ CSV là % TRONG NHÓM.** Viết "40% lỗi là do bao phủ" là sai gấp 10 lần.
Cách viết đúng: *"Trong 20 câu MISS50 đọc được, 12 câu là `R-TERM` (60%). Nhóm MISS50
chiếm 3,8% toàn tập, nên `R-TERM`-do-bao-phủ ≈ 2,3% tổng số câu."*

## 3. Bộ mã — dùng đúng 8 giá trị này, không tự thêm

Điền vào cột `code`. Định nghĩa đầy đủ ở `docs/error_taxonomy.md`.

| Mã | Nghĩa | Dấu hiệu nhận biết |
|---|---|---|
| `R-TERM` | Câu hỏi dùng từ thường dân, văn bản dùng thuật ngữ pháp lý | "tiền hỗ trợ" vs "khoản trợ cấp"; gold gần như không trùng từ nào với câu hỏi |
| `R-SEG` | Tách từ sai làm hỏng khớp | "quy định" bị tách thành "quy" + "định"; từ ghép bị cắt |
| `R-NUM` | Câu hỏi nhắc số hiệu cụ thể mà hệ thống không ưu tiên khớp chính xác | câu hỏi có `Nghị định 15/2022/NĐ-CP`, `Điều 7`… mà top-5 không chứa văn bản đó |
| `R-SHORT` | Câu hỏi quá ngắn hoặc quá chung, ít tín hiệu phân biệt | dưới ~10 từ, không có thực thể riêng nào |
| `R-DUP` | Trả về văn bản **gần trùng** gold (bản sửa đổi, bản hợp nhất) | top-5 có văn bản cùng tên/cùng chủ đề, khác năm hoặc khác số hiệu |
| `R-ENTITY` | Khớp đúng **khung điều luật** nhưng sai **thực thể/lĩnh vực** | top-5 toàn văn bản cùng khuôn diễn đạt, khác ngành (Luật PPP ↔ Luật Dầu khí) |
| `N-*` | Nghi **nhãn sai**, không phải lỗi truy vấn | doc top-1 trông đúng hơn cả gold, hoặc gold trông không liên quan |
| `khac` | Không khớp mã nào | **bắt buộc** ghi `note` mô tả |

**Nếu phân vân giữa hai mã:** chọn cái mô tả **nguyên nhân gần nhất**, ghi cái kia vào
`note`. Đừng tạo mã mới giữa chừng — bộ mã phải cố định trong suốt một vòng đọc, nếu
không thì các câu đọc trước và sau không so được với nhau.

## 4. Bốn quy tắc

1. **3 phút/câu.** Quá thì ghi `confidence=low`, điền mã tốt nhất đoán được, đi tiếp.
2. **Không bỏ câu khó.** Bỏ câu là tự lọc mẫu, và mẫu bị lọc thì mọi tỉ lệ đều vô nghĩa.
   Không nghĩ ra gì thì ghi `khac` + `confidence=low`.
3. **Trước khi ghi `R-NUM` hoặc `N-*`, mở link nguồn Ctrl-F lại.** Hai mã này là cáo buộc
   mạnh; HTML chỉ hiển thị một đoạn, không phải cả văn bản.
4. **`note` bắt buộc với mọi mã.** Một câu ngắn là đủ. Không có `note` thì 3 tuần nữa
   không ai (kể cả bạn) hiểu nổi vì sao ghi mã đó.

`confidence`: `high` / `medium` / `low`.

## 5. Cách đọc mỗi câu

Mở `outputs/error_audit/error_packet.html`. Mỗi khối gồm:

- **Nhãn màu** `MISS5` (cam) hoặc `MISS50` (đỏ) — biết ngay lỗi thuộc về ai.
- **Khối xanh GOLD**: văn bản đúng, thứ hạng nó bị xếp, tên, link, và đoạn văn bản.
  MISS50 thì không có đoạn (vì không lọt top-50) — đó là bình thường.
- **Top-5 hệ thống trả về**: tên + đoạn của từng doc.

Trình tự đọc hiệu quả:

1. Đọc câu hỏi. Câu hỏi hỏi về **thực thể/lĩnh vực nào**?
2. Đọc tên các doc trong top-5. Chúng **giống nhau ở điểm gì**? Cùng khuôn điều luật
   khác ngành → `R-ENTITY`. Cùng chủ đề khác năm → `R-DUP`.
3. Đọc đoạn của gold. Nó **dùng từ khác** câu hỏi không? → `R-TERM`.
4. Câu hỏi có **số hiệu** không, top-5 có văn bản đó không? → `R-NUM`.
5. Không mã nào khớp → `khac` + `note`.

## 6. Hai dự đoán cần kiểm — ĐỌC TRƯỚC KHI BẮT ĐẦU

Viết ra trước thì mới có giá trị khoa học. Nếu đọc xong mới nghĩ ra giả thuyết thì đó là
kể chuyện, không phải kiểm nghiệm — và phản biện tạp chí sẽ bắt.

> **H-E1.** `R-ENTITY` chiếm đa số nhóm MISS5, và tập trung ở tầng `freq>=11`.
>
> Căn cứ: `docs/stratified_baseline.md` mục 7 cho thấy tầng `freq>=11` có R@5 thấp nhất
> (0,6345) nhưng R@50 ngang các tầng khác — toàn bộ hiệu ứng nằm ở khâu xếp hạng. Cơ chế
> giả định là luật khung dùng chung khuôn diễn đạt. Đây là **lần đầu cơ chế đó được kiểm
> bằng mắt người**; trước giờ mới chỉ suy ra từ số.
>
> **Nếu H-E1 sai**, `stratified_baseline.md` mục 7.1 phải viết lại — cơ chế giải thích
> đường cong `mean_topN` đang đứng trên giả định này.

> **H-E2.** Nhóm MISS50 nghiêng về `R-TERM` và `R-SHORT`.
>
> Căn cứ: không lọt nổi top-50 nghĩa là gần như không trùng từ vựng. Hoặc câu hỏi dùng từ
> hoàn toàn khác văn bản (`R-TERM`), hoặc quá chung nên không có tín hiệu (`R-SHORT`).
>
> **Nếu H-E2 đúng**, đó là lý lẽ định lượng cho P3 dùng dense retrieval: BM25 không sửa
> được lỗi từ vựng dù có tune bao nhiêu. **Nếu MISS50 lại nhiều `R-NUM`** thì ngược lại —
> BM25 lẽ ra phải mạnh ở khớp số hiệu, nên đó là **lỗi index**, sửa được và rẻ hơn nhiều.

## 7. Sau khi đọc xong

### 7.1 Thống kê

```bash
py - <<'PY'
import csv
from collections import Counter, defaultdict
rows = list(csv.DictReader(open("outputs/error_audit/error_audit_r1.csv",
                                encoding="utf-8-sig")))
by_mode = defaultdict(Counter)
for r in rows:
    by_mode[r["mode"]][r["code"] or "(chưa điền)"] += 1
FACTOR = {"MISS5": 0.322, "MISS50": 0.095}   # hệ số quy đổi, mục 2
for mode, c in by_mode.items():
    n = sum(c.values())
    print(f"\n=== {mode} (n={n}, hệ số ×{FACTOR[mode]}) ===")
    for code, k in c.most_common():
        print(f"  {code:<10} {k:>3}  {k/n:>6.1%} trong nhóm  "
              f"→ {k/n*FACTOR[mode]*100*(60 if mode=='MISS5' else 40)/100:>5.2f}% toàn tập")
print("\n=== R-ENTITY theo tầng (kiểm H-E1) ===")
t = Counter(r["tier"] for r in rows if r["code"] == "R-ENTITY" and r["mode"] == "MISS5")
print(" ", dict(t) or "không có ca nào")
print("\nconfidence:", Counter(r["confidence"] for r in rows))
PY
```

### 7.2 Viết `docs/error_analysis_notes.md`

Phải có: bảng phân bố mã **tách theo `mode`**, hệ số quy đổi ghi rõ, phán quyết H-E1 và
H-E2 (đúng/sai/không kết luận được), và **danh sách việc cụ thể cho P2 và P3**.

Kết quả âm tính vẫn là kết quả. "Dự đoán `R-ENTITY` chiếm đa số, thực tế chỉ 20%" là một
phát hiện, không phải một thất bại.

### 7.3 Gửi cho ai

| Người | Nhận gì |
|---|---|
| **P3** | Phân bố mã của nhóm **MISS50**. Đây là 4,97% trần R@50 mà P3 đang phải nâng ở Tuần 4. `R-TERM` nhiều → cần dense; `R-NUM` nhiều → lỗi index, rẻ hơn nhiều |
| **P2** | Thống kê **họ văn bản** hay gây lỗi (TCVN/QCVN dài, văn bản không có `Điều`). Đúng task Tuần 4 của P2 |
| **P1** | Đây chính là "tổng hợp lỗi" P1 đang hỏi |

### 7.4 Đưa CSV vào repo

`.gitignore` chặn `outputs/*` trừ `outputs/label_audit/manual_audit_r*.csv`. Nên:

```bash
mkdir -p outputs/label_audit
cp outputs/error_audit/error_audit_r1.csv outputs/label_audit/error_audit_r1.csv
```

rồi thêm dòng ngoại lệ vào `.gitignore`:

```
!outputs/label_audit/error_audit_r*.csv
```

Không làm bước này thì 2,5 giờ đọc tay của bạn không vào được repo.

### 7.5 Bảng ablation gần như miễn phí

Sau khi xong vòng này, chạy lại `p4_error_sample.py` trên ranking **hợp nhất RRF** rồi so
hai bảng phân bố mã. Mã nào biến mất = hợp nhất đã chữa được; mã nào còn nguyên = phần
việc còn lại. Đó là bảng ablation thuyết phục nhất có thể đưa vào bài báo, và tốn thêm
đúng một lần chạy script.

---

## 8. Nếu gặp trục trặc

**HTML quá nặng, trình duyệt lag** → giảm mẫu: `--n-miss5 20 --n-miss50 15`, nhớ tính lại
hệ số quy đổi từ dòng script in ra.

**Nhiều câu trông như nhãn sai (`N-*`)** → đó là tín hiệu, không phải phiền toái. Vòng
đọc nhãn đã cho 7/30 không-OK; nếu vòng này cũng nhiều `N-*` thì hai vòng độc lập cùng
chỉ ra nhiễu nhãn, và đó là căn cứ mạnh để xin BTC làm rõ.

**Không chắc giữa `R-ENTITY` và `R-DUP`** → `R-DUP` là **cùng một văn bản, khác phiên
bản** (sửa đổi/hợp nhất). `R-ENTITY` là **văn bản khác hẳn, cùng khuôn diễn đạt**. Khác
ngành → `R-ENTITY`. Khác năm cùng chủ đề → `R-DUP`.
