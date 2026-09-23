"""
Kiểm chứng bộ quyết định số lượng doc (`src/rerank/calibrate.py`).

CHỦ SỞ HỮU: P4.

Chạy được KHÔNG cần `data/` — ranking giả dựng ngay trong file. Trọng tâm không phải
"hàm chạy đúng không" mà là bốn tính chất mà nếu hỏng thì hỏng ÂM THẦM trên leaderboard:

  1. không bao giờ trả 0 doc hay > 5 doc (cả hai đều cho 0 điểm, mã chấm không báo gì);
  2. bất biến với phép co giãn VÀ phép tịnh tiến điểm — BM25, cosine, RRF và logit
     cross-encoder có bốn thang điểm khác nhau, ngưỡng fit trên tầng này phải còn nghĩa
     ở tầng kia;
  3. đơn điệu theo θ — cơ sở để `scripts/fit_calibration.py` quét một chiều;
  4. phổ điểm phẳng ⇒ trả đủ `max_count` (phía an toàn cho recall, metric chính).

    python -m pytest tests/test_calibrate.py -q
"""
from __future__ import annotations

import pytest

from src.rerank.calibrate import (
    HARD_LIMIT,
    count_histogram,
    decide_counts,
    decide_one,
    margins,
)


def row(scores, prefix="d"):
    """Một dòng ranking giả: (doc_id, score, chunk_id) — dạng `search_with_anchor` trả về."""
    return [(f"{prefix}{i}", float(s), f"{prefix}{i}::0000") for i, s in enumerate(scores)]


# Top-1 tách hẳn khỏi phần còn lại: phải cắt còn 1 doc.
DUT_KHOAT = [100.0, 20.0, 19.0, 18.5, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0]
# Phổ điểm phẳng: không có chỗ hụt nào, không đủ cơ sở để tự tin.
PHANG = [50.0, 49.9, 49.8, 49.7, 49.6, 49.5, 49.4, 49.3, 49.2, 49.1]
# Hai văn bản đầu là một cụm, rồi hụt: phải cắt còn 2.
CUM_HAI = [80.0, 79.0, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0]


def test_cat_dung_cho_ba_dang_pho_diem():
    counts = decide_counts([row(DUT_KHOAT), row(PHANG), row(CUM_HAI)], thresholds=0.4)
    assert counts == [1, HARD_LIMIT, 2]


def test_khong_bao_gio_ra_ngoai_1_5():
    """Hai cách mất trắng một câu: trả rỗng, và trả quá 5 (INTERFACES.md mục 0)."""
    rows = [row(DUT_KHOAT), row(PHANG), row(CUM_HAI)]
    for theta in (-1.0, 0.0, 0.25, 1.0, 10.0, 1e9):
        for n in decide_counts(rows, thresholds=theta):
            assert 1 <= n <= HARD_LIMIT


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (1000.0, 0.0), (0.001, 0.0),
                                         (1.0, -500.0), (7.5, 3.25)])
def test_bat_bien_voi_co_gian_va_tinh_tien(scale, shift):
    """
    Cùng một hình dạng phổ điểm phải cho cùng một quyết định dù thang điểm khác nhau.
    Đây là lý do tín hiệu là (s_k - s_k+1) / (s_1 - s_ref) chứ không phải điểm thô:
    RRF cỡ 0,016 còn BM25 cỡ 170, một ngưỡng tuyệt đối sẽ cắt sai mà không báo gì.
    """
    goc = [row(DUT_KHOAT), row(PHANG), row(CUM_HAI)]
    doi_thang = [row([s * scale + shift for s in r]) for r in (DUT_KHOAT, PHANG, CUM_HAI)]
    assert decide_counts(goc, thresholds=0.4) == decide_counts(doi_thang, thresholds=0.4)


def test_don_dieu_theo_theta():
    """
    Tăng θ ⇒ mọi câu trả về nhiều doc hơn (tập dự đoán mới là tập CHA của tập cũ).
    `fit_calibration.py` dựa hẳn vào tính chất này để quét một chiều và khẳng định
    θ nhỏ nhất thoả ràng buộc recall chính là θ cho precision cao nhất.
    """
    rows = [row(DUT_KHOAT), row(PHANG), row(CUM_HAI), row([9.0, 8.0, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5])]
    truoc = None
    for theta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0):
        cur = decide_counts(rows, thresholds=theta)
        if truoc is not None:
            assert all(b >= a for a, b in zip(truoc, cur)), f"θ={theta} làm số doc GIẢM"
        truoc = cur


def test_pho_diem_bang_nhau_tuyet_doi_tra_du_max_count():
    """Mẫu số = 0. Không được chia cho 0, và không được tự tin khi không có thông tin."""
    assert decide_counts([row([5.0] * 10)], thresholds=0.0) == [HARD_LIMIT]
    assert margins([5.0] * 10, HARD_LIMIT, 10) == [0.0, 0.0, 0.0, 0.0]


def test_ranking_ngan_hon_max_count():
    """Retriever chỉ trả 2 doc thì không thể nộp 5 — đếm phải bám theo độ dài thật."""
    assert decide_counts([row([10.0, 1.0]), row([10.0])], thresholds=99.0) == [2, 1]


def test_nhan_ca_dang_2_phan_tu_lan_3_phan_tu():
    """`search()` trả (doc_id, score); `search_with_anchor()` trả thêm chunk_id."""
    hai = [[(d, s) for d, s, _ in row(DUT_KHOAT)]]
    ba = [row(DUT_KHOAT)]
    assert decide_counts(hai, thresholds=0.4) == decide_counts(ba, thresholds=0.4)


def test_nham_predictions_thay_vi_ranking_thi_bao_loi():
    """Truyền {qid: [doc_id]} vào đây là lỗi dễ mắc — phải fail loud chứ không đoán bừa."""
    with pytest.raises(TypeError, match="Predictions"):
        decide_counts([["740", "812", "915"]], thresholds=0.4)


def test_vecto_nguong_sai_do_dai_thi_bao_loi():
    with pytest.raises(ValueError, match="max_count-1"):
        decide_counts([row(DUT_KHOAT)], thresholds=[0.4, 0.4])


def test_max_count_vuot_tran_btc_thi_bao_loi():
    """Trần 5 của BTC là cứng: vượt là 0 điểm im lặng, không phải cảnh báo."""
    with pytest.raises(ValueError, match="0 cả recall lẫn precision"):
        decide_counts([row(DUT_KHOAT)], thresholds=0.4, max_count=6)


def test_min_count_ep_san_duoi():
    """Đội muốn bảo hiểm recall có thể ép sàn 2 doc mà không phải sửa code."""
    assert decide_counts([row(DUT_KHOAT)], thresholds=0.4, min_count=2) == [2]


def test_nguong_rieng_tung_hang():
    """Giao diện nhận vectơ; fitter chỉ fit vectơ hằng, nhưng YAML vẫn khai riêng được."""
    # m_1 của CUM_HAI nhỏ (hai doc đầu sát nhau), m_2 lớn → chặn k=1, cho phép k=2.
    assert decide_one([s for s in CUM_HAI], [9.9, 0.1, 0.1, 0.1]) == 2


def test_count_histogram():
    assert count_histogram([1, 1, 5, 2, 5, 5]) == {1: 2, 2: 1, 5: 3}
