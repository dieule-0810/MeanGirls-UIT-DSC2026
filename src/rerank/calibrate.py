"""
Bộ quyết định số lượng văn bản trả về (1..5) — CHỦ SỞ HỮU: P4. Tầng cuối của pipeline.

VÌ SAO TẦNG NÀY TỒN TẠI. Mã chấm BTC lấy recall = |gold ∩ pred| / |gold| và
precision = |gold ∩ pred| / len(pred) rồi trung bình trên mọi câu. Nộp thừa KHÔNG làm
tăng recall (docs/scoring_behaviour.md) nhưng chia nhỏ precision theo len(pred). 92,1%
câu chỉ có ĐÚNG MỘT văn bản đáp án ⇒ trần 5 doc của BTC không hề ràng buộc, và "luôn
nộp 5" là bỏ không 4/5 precision để mua một chút bảo hiểm recall.

Recall là metric chính, precision chỉ phá hoà ⇒ tầng này KHÔNG được tự do đánh đổi.
Nó giải một bài toán CÓ RÀNG BUỘC:

    max precision   sao cho   recall ≥ recall("luôn nộp max_count") − ngân sách

Ngân sách do người chạy khai trong YAML; `scripts/fit_calibration.py` đi tìm ngưỡng.

TÍN HIỆU: KHOẢNG CÁCH ĐIỂM ĐÃ CHUẨN HOÁ, KHÔNG PHẢI ĐIỂM THÔ.
Điểm thô không so được giữa các tầng: BM25 là số dương không chặn trên (top-1 dao động
90–300 tuỳ câu), cosine nằm trong [−1, 1], RRF là tổng 1/(k+rank) cỡ 0,016, logit
cross-encoder thì có dấu. Một ngưỡng tuyệt đối kiểu "s1 > 150" fit trên BM25 sẽ vô
nghĩa ở mọi tầng khác — và tệ hơn, nó vẫn CHẠY, chỉ là cắt sai, im lặng.

Nên tín hiệu là khoảng cách tương đối, bất biến với cả co giãn lẫn tịnh tiến:

    m_k = (s_k − s_{k+1}) / (s_1 − s_ref)        ref = hạng `scale_rank`

Mẫu số là bề rộng phổ điểm của CHÍNH câu đó ⇒ "hơn nhau nhiều" được đo bằng thang đo
riêng của câu hỏi, không phải thang đo của corpus.

LUẬT CẮT: n = k nhỏ nhất sao cho m_k ≥ ngưỡng_k; không có k nào thì trả `max_count`.
Đọc: đi từ trên xuống, gặp chỗ hụt điểm dứt khoát đầu tiên thì cắt, vì hạng 1..k đã
tách thành một cụm riêng. Không có chỗ hụt nào ⇒ phổ điểm phẳng ⇒ không đủ cơ sở để tự
tin ⇒ trả đủ `max_count`, đúng hành vi cũ.

Ngưỡng là một VECTƠ (mỗi hạng một giá trị) nhưng `fit_calibration.py` chỉ fit vectơ
HẰNG. Lý do là tính đơn điệu: với vectơ hằng, tăng ngưỡng làm MỌI câu trả về nhiều doc
hơn ⇒ tập dự đoán mới là tập CHA của tập cũ ⇒ recall chỉ có thể tăng, precision chỉ có
thể giảm. Nhờ đó "tìm ngưỡng tốt nhất trong ngân sách recall" là một phép quét một
chiều có biên rõ ràng, thay vì một cuộc dò dẫm 4 chiều dễ khớp quá mức trên tập fit.
Giao diện vẫn nhận vectơ vì nó không tốn gì: có bằng chứng cho ngưỡng riêng theo hạng
thì chỉ cần đổi YAML, không đụng code.
"""
from __future__ import annotations

import math
from typing import Sequence

# Trần cứng của BTC. Trả > 5 doc ⇒ câu đó 0 cả recall lẫn precision (INTERFACES.md mục 0).
HARD_LIMIT = 5


def _scores_of(row: Sequence) -> list[float]:
    """
    Lấy dãy điểm từ một dòng ranking, chấp nhận cả `(doc_id, score)` lẫn
    `(doc_id, score, chunk_id)` — `search()` trả dạng 2, `search_with_anchor()` và
    `ranking_full.json` trả dạng 3.
    """
    out = []
    for item in row:
        if isinstance(item, (int, float, str)):
            raise TypeError(
                f"Mỗi phần tử phải là (doc_id, score[, chunk_id]), nhận {item!r}. "
                f"Có phải đang truyền nhầm Predictions (chỉ có doc_id) thay vì ranking?"
            )
        out.append(float(item[1]))
    return out


def margins(scores: Sequence[float], max_count: int, scale_rank: int) -> list[float]:
    """
    `m_k` cho k = 1..max_count-1. Phổ điểm phẳng (mẫu số ≤ 0) → toàn 0, tức không có
    chỗ hụt nào, tức luật cắt sẽ trả `max_count`. Đó là phía an toàn cho recall.
    """
    if scale_rank < 2:
        raise ValueError("scale_rank phải ≥ 2 — mẫu số là s_1 − s_(scale_rank)")
    n = len(scores)
    if n < 2:
        return []
    ref = scores[min(scale_rank, n) - 1]
    scale = scores[0] - ref
    if not math.isfinite(scale) or scale <= 0:
        return [0.0] * (min(max_count, n) - 1)
    return [
        (scores[k - 1] - scores[k]) / scale
        for k in range(1, min(max_count, n))
    ]


def count_from_margins(
    m: Sequence[float],
    thresholds: Sequence[float],
    max_count: int = HARD_LIMIT,
    min_count: int = 1,
) -> int:
    """
    LUẬT CẮT — nguồn sự thật duy nhất. `decide_one` và phép quét θ của
    `scripts/fit_calibration.py` đều đi qua đây, để không bao giờ có chuyện fit một luật
    rồi chạy một luật khác.

    `m > 0` là điều kiện RIÊNG, không gộp vào ngưỡng: khe hở bằng 0 nghĩa là hai văn bản
    cùng điểm, cắt vào giữa chúng là chọn bừa một trong hai — đúng thứ INTERFACES.md
    mục 3b bắt phải xử lý tường minh. Không có điều kiện này thì θ=0 (đầu mút của phép
    quét) cắt phổ điểm PHẲNG còn 1 doc, tức là tự tin nhất ở đúng chỗ không có thông tin
    nhất. `thresholds` ở đây đã được chuẩn hoá thành list đúng độ dài.
    """
    for k, mk in enumerate(m, start=1):
        if mk > 0.0 and mk >= thresholds[k - 1]:
            return max(k, min_count)
    return max_count


def decide_one(
    scores: Sequence[float],
    thresholds: Sequence[float],
    max_count: int = HARD_LIMIT,
    min_count: int = 1,
    scale_rank: int = 10,
) -> int:
    """Số doc cho MỘT câu, đi từ điểm thô."""
    return count_from_margins(
        margins(scores, max_count, scale_rank), thresholds, max_count, min_count
    )


def normalize_params(
    thresholds: Sequence[float] | float, max_count: int, min_count: int
) -> list[float]:
    """Kiểm tham số đến từ YAML và trả vectơ ngưỡng đúng độ dài. Fail loud, không đoán."""
    if not 1 <= max_count <= HARD_LIMIT:
        raise ValueError(
            f"max_count={max_count} ngoài khoảng 1..{HARD_LIMIT}. Trả quá {HARD_LIMIT} doc "
            f"⇒ câu đó 0 cả recall lẫn precision, mã chấm KHÔNG báo lỗi (INTERFACES.md mục 0)."
        )
    if not 1 <= min_count <= max_count:
        raise ValueError(f"min_count={min_count} phải nằm trong 1..max_count={max_count}")
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * (max_count - 1)
    out = [float(t) for t in thresholds]
    if len(out) != max_count - 1:
        raise ValueError(
            f"thresholds có {len(out)} phần tử, cần đúng max_count-1 = {max_count - 1} "
            f"(một ngưỡng cho mỗi chỗ cắt k=1..{max_count - 1})."
        )
    return out


def decide_counts(
    ranked: Sequence[Sequence],
    *,
    thresholds: Sequence[float] | float,
    max_count: int = HARD_LIMIT,
    min_count: int = 1,
    scale_rank: int = 10,
) -> list[int]:
    """
    Số doc cho từng câu, cùng thứ tự với `ranked`. Đây là hàm `scripts/run_pipeline.py`
    gọi; mọi tham số đến từ khối `pipeline.calibrate` trong YAML.

    `thresholds` nhận một số (vectơ hằng — dạng đã fit) hoặc một list dài `max_count-1`.
    """
    thresholds = normalize_params(thresholds, max_count, min_count)
    counts = [
        decide_one(_scores_of(row), thresholds, max_count, min_count, scale_rank)
        for row in ranked
    ]

    # Lưới an toàn: rẻ, và bốn cách mất điểm im lặng của BTC đều đi qua đúng chỗ này.
    for i, (c, row) in enumerate(zip(counts, ranked)):
        if not 1 <= c <= HARD_LIMIT:
            raise AssertionError(f"câu {i}: calibrate trả {c} doc, ngoài 1..{HARD_LIMIT}")
        if c > len(row) and len(row) > 0:
            counts[i] = len(row)
    return counts


def count_histogram(counts: Sequence[int]) -> dict[int, int]:
    """Phân bố số doc đã trả — in ra để người chạy thấy ngay luật cắt có đang làm gì không."""
    return {n: counts.count(n) for n in sorted(set(counts))}
