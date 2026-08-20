"""
tests/test_scoring.py — Đặc tả hành vi mã chấm của BTC.

MỤC ĐÍCH: không phải test code của team, mà là ĐO hành vi mã chấm của BTC và
đóng băng nó thành đặc tả. Mọi giả định trong `make_submission.py` phải truy
được về một test ở đây.

Chủ sở hữu: P4 (KPI Recall@5 + Precision).

CÁCH DÙNG
    pip install pytest
    # đặt scoring.py của BTC vào vendor/btc_scoring/scoring.py
    pytest tests/test_scoring.py -v
    pytest tests/test_scoring.py -v -s      # in bảng kết quả để dán vào docs/

Mọi fixture được ghi ra tests/fixtures/*.json khi chạy → tái lập được, và là
bằng chứng cho bảng 0.4 trong plan.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Nạp mã chấm của BTC
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
BTC_SCORING = REPO / "vendor" / "btc_scoring" / "scoring.py"
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

sys.path.insert(0, str(BTC_SCORING.parent))
try:
    from scoring import eval_retrieval  # noqa: E402
except ImportError as e:  # pragma: no cover
    pytest.skip(f"Chưa có mã chấm BTC tại {BTC_SCORING}: {e}", allow_module_level=True)


# --------------------------------------------------------------------------
# Truy xuất nguồn gốc — bắt buộc, mục 2 P4_TASKS
# --------------------------------------------------------------------------
def test_provenance_recorded():
    """Mã chấm phải có nguồn gốc ghi lại được.

    Đảm bảo hash sha256 của file vendor/btc_scoring/scoring.py trùng khớp với
    hash được ghi nhận trong bundle/tài liệu docs/scoring_provenance.md.
    """
    doc = REPO / "docs" / "scoring_provenance.md"
    assert doc.exists(), (
        "Thiếu docs/scoring_provenance.md. Tạo file ghi: nguồn bundle, sha256, "
        "và xác nhận hash khớp Scoring-Program-Task-LegalIR.zip của BTC."
    )
    actual = hashlib.sha256(BTC_SCORING.read_bytes()).hexdigest()
    assert actual in doc.read_text(encoding="utf-8"), (
        f"sha256 của scoring.py = {actual} nhưng không khớp với hash ghi nhận trong "
        f"docs/scoring_provenance.md. Mã chấm chưa khớp với bundle."
    )


def test_scoring_enforces_five_doc_limit_at_all():
    """Chốt chặn: bản mã chấm đang giữ CÓ luật >5 doc chưa?

    Nếu test này fail, team đang giữ bản trước khi BTC vá → mọi test dưới đây
    vô nghĩa. Tải lại từ CodaLab trước khi làm gì tiếp.
    """
    gold = {"q1": ["100"]}
    over = {"q1": {"answer": ["100", "2", "3", "4", "5", "6"]}}
    r, p = _score(over, gold, save_as="probe_six_docs")
    assert (r, p) == (0.0, 0.0), (
        "Nộp 6 doc mà vẫn được điểm → đang giữ mã chấm CŨ, chưa vá lỗ hổng. "
        "Dừng lại, tải bản mới từ CodaLab."
    )


# --------------------------------------------------------------------------
# Adapter: chuẩn hoá chữ ký hàm + lưu fixture
# --------------------------------------------------------------------------
def _score(submission: dict, gold: dict, save_as: str):
    """Chấm một submission, đồng thời ghi fixture ra đĩa.

    save_as: tên fixture. KHÔNG được bỏ qua — fixture là sản phẩm chính của
    file này, quan trọng ngang kết quả assert.
    """
    (FIXTURES / f"{save_as}.json").write_text(
        json.dumps({"submission": submission, "gold": gold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = eval_retrieval(submission, gold)
    if isinstance(result, dict):
        return float(result["recall"]), float(result["precision"])
    return float(result[0]), float(result[1])


def _approx(x):
    return pytest.approx(x, abs=1e-6)


# --------------------------------------------------------------------------
# NHÓM 1 — Tái lập bảng 0.4 của plan
# --------------------------------------------------------------------------
GOLD_1 = {"q1": ["100"]}


def test_row1_exact_single_answer():
    """Nộp đúng 1 doc đúng → cả hai = 1.0."""
    r, p = _score({"q1": {"answer": ["100"]}}, GOLD_1, save_as="row1_exact_single")
    assert (r, p) == (_approx(1.0), _approx(1.0))


def test_row2_five_docs_one_gold():
    """Nộp đủ 5 doc, 1 gold trên 1 câu hỏi."""
    r, p = _score(
        {"q1": {"answer": ["100", "2", "3", "4", "5"]}}, GOLD_1, save_as="row2_five_docs"
    )
    assert r == _approx(1.0), "Recall phải là 1.0 — nộp thừa không làm mất recall"
    assert p == _approx(0.2), (
        "0.300 trong plan là trung bình fixture 2 câu — không mâu thuẫn."
    )


def test_row3_int_doc_id_scores_zero_silently():
    """🔴 BẪY 1 — doc_id kiểu int cho 0 điểm, không lỗi, không cảnh báo.

    Đây là rủi ro 'Cao' trong bảng rủi ro. Test này là bằng chứng để
    make_submission.py fail-loud thay vì fail-silent.
    """
    r, p = _score({"q1": {"answer": [100]}}, GOLD_1, save_as="row3_int_doc_id")
    assert (r, p) == (_approx(0.0), _approx(0.0)), (
        "int doc_id KHÔNG còn cho 0 điểm nữa — mã chấm đã đổi. "
        "Cập nhật docs/scoring_behaviour.md và nới assert trong make_submission.py."
    )


def test_row4_duplicates_counted_in_precision_denominator():
    """🔴 BẪY 2 — trùng lặp không được khử.

    Kiểm tra hành vi tính toán mẫu số của Precision khi có ID trùng lặp.
    """
    r, p = _score(
        {"q1": {"answer": ["100", "100", "100"]}}, GOLD_1, save_as="row4_duplicates"
    )
    assert r == _approx(1.0)
    assert p == _approx(1 / 3)


def test_two_gold_with_duplicate():
    """Kiểm tra hành vi tính điểm khi nộp ID trùng lặp trên câu hỏi có 2 gold.

    Nộp ['100', '100', '200'] với gold {'100', '200'}:
    Kỳ vọng Recall = 2/2 = 1.0, Precision = 2/3 = 0.667.
    """
    gold = {"q1": ["100", "200"]}
    r, p = _score(
        {"q1": {"answer": ["100", "100", "200"]}}, gold, save_as="two_gold_with_duplicate"
    )
    assert (r, p) == (_approx(1.0), _approx(2 / 3))


def test_row5_missing_question_raises():
    """Thiếu câu hỏi → CRASH, không phải 0 điểm."""
    with pytest.raises(Exception):
        _score({}, {"q1": ["100"], "q2": ["200"]}, save_as="row5_missing_question")


def test_row6_unknown_qid_raises():
    """qid sai → CRASH."""
    with pytest.raises(Exception):
        _score({"KHONG_TON_TAI": {"answer": ["100"]}}, GOLD_1, save_as="row6_bad_qid")


# --------------------------------------------------------------------------
# NHÓM TÁI LẬP FIXTURE THẬT CỦA P1
# --------------------------------------------------------------------------
GOLD_P1 = {"q1": ["100"], "q2": ["200", "201"]}


def test_reproduce_p1_five_docs():
    sub = {
        "q1": {"answer": ["100", "2", "3", "4", "5"]},
        "q2": {"answer": ["200", "201", "3", "4", "5"]},
    }
    assert _score(sub, GOLD_P1, save_as="p1_five_docs") == (_approx(1.0), _approx(0.3))


def test_reproduce_p1_duplicates():
    sub = {
        "q1": {"answer": ["100", "100", "100"]},
        "q2": {"answer": ["200", "201"]},
    }
    assert _score(sub, GOLD_P1, save_as="p1_dupes") == (_approx(1.0), _approx(2 / 3))


# --------------------------------------------------------------------------
# NHÓM 2 — Ba case plan còn thiếu (mục 2 của P4_TASKS)
# --------------------------------------------------------------------------
def test_over_limit_scope_whole_submission_or_one_question():
    """Kiểm tra xem khi 1 câu vi phạm >5 doc thì bị hủy cả submission hay chỉ hủy câu đó.

    Fixture: 2 câu, câu 1 nộp đúng, câu 2 nộp 6 doc.
    """
    gold = {"q1": ["100"], "q2": ["200"]}
    sub = {
        "q1": {"answer": ["100"]},
        "q2": {"answer": ["200", "1", "2", "3", "4", "5"]},
    }
    r, p = _score(sub, gold, save_as="over_limit_scope")
    print(f"\n[KẾT QUẢ] 1 câu vi phạm trong 2 câu → Recall={r}, Precision={p}")

    if (r, p) == (_approx(0.0), _approx(0.0)):
        verdict = "TOÀN SUBMISSION = 0"
    elif r == _approx(0.5):
        verdict = "CHỈ CÂU VI PHẠM = 0 (câu q1 vẫn được tính)"
    else:
        verdict = f"HÀNH VI THỨ BA, ngoài dự đoán: recall={r}"
    print(f"[KẾT LUẬN] {verdict}")

    Path(REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / "over_limit_verdict.txt").write_text(
        f"{verdict}\nrecall={r} precision={p}\n", encoding="utf-8"
    )
    assert r in (_approx(0.0), _approx(0.5)), f"Hành vi lạ: {r}, {p} — điều tra tay"


def test_exactly_five_is_allowed():
    """Biên: đúng 5 doc PHẢI hợp lệ."""
    gold = {"q1": ["1", "2", "3", "4", "5"]}
    r, p = _score(
        {"q1": {"answer": ["1", "2", "3", "4", "5"]}}, gold, save_as="exactly_five_ok"
    )
    assert (r, p) == (_approx(1.0), _approx(1.0)), "5 doc bị coi là vi phạm → trần là 4!"


def test_two_gold_partial_hit():
    """Câu 2 gold, nộp 5 doc chỉ trúng 1 → recall 0.5, precision 0.2."""
    gold = {"q1": ["100", "200"]}
    r, p = _score(
        {"q1": {"answer": ["100", "8", "9", "10", "11"]}}, gold, save_as="two_gold_partial"
    )
    assert r == _approx(0.5), f"Recall={r}, kỳ vọng 1/2 — mẫu số phải là |gold|"
    assert p == _approx(0.2)


def test_empty_answer_list():
    """Danh sách rỗng → Precision = 0, KHÔNG ZeroDivisionError."""
    r, p = _score({"q1": {"answer": []}}, GOLD_1, save_as="empty_answer")
    assert (r, p) == (_approx(0.0), _approx(0.0))


# --------------------------------------------------------------------------
# NHÓM 3 — Bằng chứng cho chiến lược calibration (mục 3 plan)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_precision_by_submission_size_single_gold(k):
    """Bảng đánh đổi Precision theo số doc nộp, khi gold đúng nằm trong tập."""
    answer = ["100"] + [str(900 + i) for i in range(k - 1)]
    r, p = _score({"q1": {"answer": answer}}, GOLD_1, save_as=f"precision_k{k}")
    print(f"\nk={k}: Recall={r:.3f}  Precision={p:.3f}")
    assert r == _approx(1.0), "Recall phải bất biến theo k khi gold có trong tập"
    assert p == _approx(1.0 / k)


def test_recall_does_not_reward_padding():
    """Nộp thừa KHÔNG tăng Recall — nền tảng của toàn bộ chiến lược calibration."""
    r1, _ = _score({"q1": {"answer": ["100"]}}, GOLD_1, save_as="pad_k1")
    r5, _ = _score(
        {"q1": {"answer": ["100", "a", "b", "c", "d"]}}, GOLD_1, save_as="pad_k5"
    )
    assert r1 == r5 == _approx(1.0)


# --------------------------------------------------------------------------
# NHÓM 4 — Hợp đồng cho make_submission.py (P4 đặc tả → P1 thực thi)
# --------------------------------------------------------------------------
def dedupe_keep_order(ids):
    """Khử trùng lặp GIỮ THỨ TỰ, rồi mới cắt còn 5."""
    seen, out = set(), []
    for i in ids:
        s = str(i)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def test_dedupe_then_clip_not_clip_then_dedupe():
    """RRF/ensemble sinh trùng lặp tự nhiên — đúng thứ team làm từ Tuần 2."""
    raw = ["a", "a", "b", "b", "c", "c", "d", "e", "f"]
    assert dedupe_keep_order(raw)[:5] == ["a", "b", "c", "d", "e"]
    assert dedupe_keep_order(raw[:5])[:5] == ["a", "b", "c"]


def test_dedupe_coerces_int_to_str():
    """Ép str() ngay tại điểm khử trùng lặp, chặn BẪY 1 tại nguồn."""
    assert dedupe_keep_order([100, "100", 200]) == ["100", "200"]


def test_submission_contract():
    """Hợp đồng P4 giao cho P1. Mỗi assert truy về đúng một test ở trên."""

    def validate(submission: dict, expected_qids: set):
        assert set(submission) == expected_qids, (
            f"Lệch qid → CRASH, mất lượt nộp. "
            f"Thiếu: {expected_qids - set(submission)} | Thừa: {set(submission) - expected_qids}"
        )
        for qid, obj in submission.items():
            assert "answer" in obj, f"{qid}: thiếu key 'answer' → CRASH"
            ans = obj["answer"]
            assert isinstance(ans, list), f"{qid}: 'answer' phải là list"
            assert all(isinstance(d, str) for d in ans), f"{qid}: doc_id phải str (BẪY 1)"
            assert len(ans) == len(set(ans)), f"{qid}: có trùng lặp (BẪY 2)"
            assert 1 <= len(ans) <= 5, f"{qid}: {len(ans)} doc, phải trong [1,5]"

    validate({"q1": {"answer": ["100"]}}, {"q1"})
    for bad in (
        {"q1": {"answer": [100]}},
        {"q1": {"answer": ["1", "1"]}},
        {"q1": {"answer": ["1", "2", "3", "4", "5", "6"]}},
        {"q1": {"answer": []}},
        {"q1": {"doc_ids": ["1"]}},
        {"q2": {"answer": ["1"]}},
    ):
        with pytest.raises(AssertionError):
            validate(bad, {"q1"})