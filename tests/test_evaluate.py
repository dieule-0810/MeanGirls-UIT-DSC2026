"""
Kiểm chứng src/evaluate.py khớp CHÍNH XÁC scoring.py của BTC.

Các con số kỳ vọng dưới đây lấy từ việc chạy trực tiếp hàm eval_retrieval của BTC.
Nếu một test ở đây fail → evaluate.py đã lệch khỏi mã chấm thật → mọi con số
trên held-out đều vô nghĩa. Chạy: python -m pytest tests/ -q
"""
import pytest

from src.evaluate import eval_official, recall_at_k
from src.make_submission import build_submission, dedupe_keep_order

TRUTH = {"q1": ["100"], "q2": ["200", "201"]}


def test_khop_hoan_toan():
    s = eval_official({"q1": ["100"], "q2": ["200", "201"]}, TRUTH)
    assert s == {"recall": 1.0, "precision": 1.0}


def test_nop_du_5_recall_khong_doi_precision_sut():
    """Bằng chứng cho chiến lược calibration: nộp thừa KHÔNG tăng Recall."""
    s = eval_official(
        {"q1": ["100", "a", "b", "c", "d"], "q2": ["200", "201", "x", "y", "z"]}, TRUTH
    )
    assert s["recall"] == 1.0
    assert s["precision"] == pytest.approx(0.3)


def test_vuot_5_doc_bi_zero_ca_hai():
    s = eval_official({"q1": ["100", "a", "b", "c", "d", "e"], "q2": ["200", "201"]}, TRUTH)
    assert s["recall"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(0.5)


def test_list_rong_bi_zero():
    s = eval_official({"q1": [], "q2": ["200", "201"]}, TRUTH)
    assert s["recall"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(0.5)


def test_trung_lap_khong_duoc_khu():
    """🔴 Mẫu số Precision là len(LIST). Trùng lặp bị phạt."""
    s = eval_official({"q1": ["100", "100", "100"], "q2": ["200", "201"]}, TRUTH)
    assert s["recall"] == 1.0
    assert s["precision"] == pytest.approx((1 / 3 + 1.0) / 2)


def test_doc_id_kieu_int_cho_0_diem_im_lang():
    """🔴 Chế độ hỏng nguy hiểm nhất: không lỗi, không cảnh báo, chỉ 0 điểm."""
    s = eval_official({"q1": [100], "q2": [200, 201]}, TRUTH)
    assert s == {"recall": 0.0, "precision": 0.0}


def test_thieu_cau_hoi_raise():
    with pytest.raises(ValueError, match="not match"):
        eval_official({"q1": ["100"]}, TRUTH)


def test_sai_qid_du_so_luong_raise():
    with pytest.raises(ValueError, match="không khớp"):
        eval_official({"q1": ["100"], "q9": ["200"]}, TRUTH)


def test_thu_tu_khong_anh_huong():
    """Không có MRR/NDCG — chỉ phép giao tập hợp."""
    a = eval_official({"q1": ["100", "x"], "q2": ["200", "201"]}, TRUTH)
    b = eval_official({"q1": ["x", "100"], "q2": ["201", "200"]}, TRUTH)
    assert a == b


def test_recall_at_k_khong_gioi_han_5():
    ranked = {"q1": ["a", "b", "c", "d", "e", "f", "100"]}
    assert recall_at_k(ranked, {"q1": ["100"]}, k=5) == 0.0
    assert recall_at_k(ranked, {"q1": ["100"]}, k=10) == 1.0


# ── make_submission: chốt chặn phòng thủ ────────────────────────────────────
def test_dedupe_giu_thu_tu():
    assert dedupe_keep_order(["b", "a", "b", 3, "3"]) == ["b", "a", "3"]


def test_build_submission_ep_str_va_cat_5():
    sub = build_submission({"q1": [1, 2, 2, 3, 4, 5, 6]}, {"q1"})
    assert sub["q1"]["answer"] == ["1", "2", "3", "4", "5"]
    assert all(isinstance(d, str) for d in sub["q1"]["answer"])


def test_build_submission_chan_thieu_cau_hoi():
    with pytest.raises(SystemExit, match="THIẾU"):
        build_submission({"q1": ["100"]}, {"q1", "q2"})


def test_build_submission_chan_qid_thua():
    with pytest.raises(SystemExit, match="THỪA"):
        build_submission({"q1": ["100"], "q9": ["1"]}, {"q1"})
