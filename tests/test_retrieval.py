"""
Kiểm chứng khung tầng 1: hợp đồng `BaseRetriever` + tokenizer tiếng Việt.

CHỦ SỞ HỮU: P3.

Chạy được KHÔNG cần `data/` — corpus giả dựng ngay trong file. Đó là chủ ý: dữ liệu BTC không
được commit, nên tầng retrieval phải tự kiểm chứng được ở máy sạch, và mọi bất biến ở đây
(`doc_id` là str, không trùng, sort giảm dần, không quá top_k) là những thứ hỏng ÂM THẦM
trên leaderboard chứ không văng exception.

    python -m pytest tests/test_retrieval.py -q
"""
from __future__ import annotations

import importlib.util

import pytest

from src.retrieval.base import (
    build_retriever,
    check_contract,
    parse_pool,
    pool_scores,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.tokenizers import (
    Tokenizer,
    fold_tone_placement,
    get_tokenizer,
    tokenize_many,
)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus giả: 3 văn bản, mỗi văn bản vài chunk
# ─────────────────────────────────────────────────────────────────────────────
CHUNKS = [
    {
        "chunk_id": "740::0000",
        "doc_id": "740",
        "position": 0,
        "text": "Điều 1. Cơ quan thuế có thẩm quyền huỷ bỏ hoá đơn điện tử đã lập sai.",
    },
    {
        "chunk_id": "740::0001",
        "doc_id": "740",
        "position": 1,
        "text": "Điều 2. Việc huỷ bỏ hoá đơn điện tử phải lập biên bản theo Thông tư số 78/2021/TT-BTC.",
    },
    {
        "chunk_id": "812::0000",
        "doc_id": "812",
        "position": 0,
        "text": "Điều 1. Cơ sở dữ liệu quốc gia về dân cư do Bộ Công an quản lý.",
    },
    {
        "chunk_id": "812::0001",
        "doc_id": "812",
        "position": 1,
        "text": "Điều 2. Quan hệ lao động giữa người sử dụng lao động và người lao động.",
    },
    {
        "chunk_id": "999::0000",
        "doc_id": "999",
        "position": 0,
        "text": "Điều 1. Uỷ ban nhân dân cấp huyện quyết định thu hồi đất theo quy định.",
    },
]


def build_bm25(**opts) -> BM25Retriever:
    r = BM25Retriever(verbose=False, **opts)
    r.index([dict(c) for c in CHUNKS])
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Chuẩn hoá dấu thanh
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hoà bình", "hòa bình"),
        ("khoẻ mạnh", "khỏe mạnh"),
        ("thuỷ lợi", "thủy lợi"),
        ("uỷ ban nhân dân", "ủy ban nhân dân"),
        ("hoá đơn", "hóa đơn"),
        ("hoà", "hòa"),
    ],
)
def test_gop_vi_tri_dau_thanh(raw, expected):
    """'hoà' và 'hòa' cùng một từ, khác chuỗi byte → với BM25 là hai term. Phải quy về một."""
    assert fold_tone_placement(raw) == expected


@pytest.mark.parametrize("giu_nguyen", ["quý khách", "quỷ kế", "của cải", "cửa hàng", "giá"])
def test_khong_dong_vao_tu_da_dung_chuan(giu_nguyen):
    """qu- và các tổ hợp không tranh chấp phải giữ nguyên, nếu không là tự tạo term rác."""
    assert fold_tone_placement(giu_nguyen) == giu_nguyen


def test_cau_hoi_va_van_ban_dat_dau_khac_nhau_van_khop():
    tok = get_tokenizer("regex", fold_tone=True)
    assert tok("Uỷ ban nhân dân") == tok("Ủy ban nhân dân")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def test_regex_giu_nguyen_hanh_vi_v01():
    """Mốc tham chiếu v0.1: NFC + lower + \\w+, không gộp dấu. Đừng đổi con số này."""
    tok = get_tokenizer("regex", fold_tone=False)
    assert tok("Điều 5. Cơ quan thuế") == ["điều", "5", "cơ", "quan", "thuế"]


def test_whitespace_giu_nguyen_so_hieu_van_ban():
    """Số hiệu văn bản là term định danh mạnh — regex làm nó vỡ vụn, whitespace thì không."""
    text = "Thông tư số 78/2021/TT-BTC quy định."
    assert "78/2021/tt-btc" in get_tokenizer("whitespace")(text)
    assert "78/2021/tt-btc" not in get_tokenizer("regex")(text)


def test_whitespace_got_dau_cau_hai_dau_token():
    assert get_tokenizer("whitespace")("(quy định),") == ["quy", "định"]


def test_syllable_bigram_sinh_tu_ghep():
    toks = get_tokenizer("syllable_bigram")("cơ quan thuế")
    assert toks == ["cơ", "quan", "thuế", "cơ_quan", "quan_thuế"]


def test_bigram_khong_bac_qua_dau_cau():
    """'…quy định. Điều 5…' không được sinh 'định_điều' — đó là term ma, không có nghĩa."""
    toks = get_tokenizer("syllable_bigram")("quy định. Điều 5")
    assert "định_điều" not in toks
    assert "quy_định" in toks


def test_bigram_only_bo_am_tiet_don():
    toks = get_tokenizer("syllable_bigram", keep_syllables=False)("cơ quan thuế")
    assert toks == ["cơ_quan", "quan_thuế"]


def test_min_len_loc_token_ngan():
    assert get_tokenizer("regex", min_len=2)("Điều 5 về cơ quan") == ["điều", "về", "cơ", "quan"]


def test_ten_tokenizer_sai_bao_loi_ngay():
    with pytest.raises(ValueError, match="không tồn tại"):
        get_tokenizer("vncorenlp")


def test_tokenizer_pickle_duoc():
    """Bắt buộc để tách từ đa tiến trình (n_jobs > 1) chạy được."""
    import pickle

    tok = get_tokenizer("syllable_bigram", fold_tone=False)
    assert pickle.loads(pickle.dumps(tok)) == tok


def test_tokenize_many_dung_cache(tmp_path):
    tok = get_tokenizer("regex")
    texts = ["Cơ quan thuế", "Uỷ ban nhân dân"]
    a = tokenize_many(texts, tok, cache_dir=tmp_path, verbose=False)
    files = list(tmp_path.glob("tokens_*.pkl"))
    assert len(files) == 1
    b = tokenize_many(texts, tok, cache_dir=tmp_path, verbose=False)
    assert a == b


def test_cache_khong_dung_lai_khi_corpus_doi(tmp_path):
    """Key cache băm theo nội dung: P2 đổi chunker là cache tự hết hiệu lực."""
    tok = get_tokenizer("regex")
    tokenize_many(["a b"], tok, cache_dir=tmp_path, verbose=False)
    tokenize_many(["a c"], tok, cache_dir=tmp_path, verbose=False)
    assert len(list(tmp_path.glob("tokens_*.pkl"))) == 2


@pytest.mark.skipif(importlib.util.find_spec("pyvi") is None, reason="chưa cài pyvi")
def test_pyvi_gop_tu_ghep():
    assert "cơ_quan" in get_tokenizer("pyvi")("Cơ quan thuế có thẩm quyền")


@pytest.mark.skipif(
    importlib.util.find_spec("underthesea") is None, reason="chưa cài underthesea"
)
def test_underthesea_gop_tu_ghep():
    assert "cơ_quan" in get_tokenizer("underthesea")("Cơ quan thuế có thẩm quyền")


# ─────────────────────────────────────────────────────────────────────────────
# Gộp chunk → doc
# ─────────────────────────────────────────────────────────────────────────────
DOCS = ["a", "a", "a", "b"]
SCORES = [3.0, 1.0, 2.0, 5.0]


def test_pool_max():
    assert pool_scores(DOCS, SCORES, "max") == {"a": 3.0, "b": 5.0}


def test_pool_sum_thien_vi_van_ban_dai():
    """Bằng chứng số cho lý do KHÔNG mặc định dùng sum: 'a' yếu hơn nhưng nhiều chunk hơn."""
    assert pool_scores(DOCS, SCORES, "sum") == {"a": 6.0, "b": 5.0}


def test_pool_mean_topN():
    assert pool_scores(DOCS, SCORES, "mean_top2") == {"a": 2.5, "b": 5.0}


def test_mean_topN_khong_dem_0_cho_van_ban_it_chunk():
    """Đệm 0 sẽ phạt oan văn bản chỉ có 1 chunk khớp — lấy trung bình trên số chunk thật."""
    assert pool_scores(["b"], [5.0], "mean_top3") == {"b": 5.0}


def test_logsumexp_nam_giua_max_va_sum():
    a = pool_scores(DOCS, SCORES, "logsumexp", tau=1.0)["a"]
    assert 3.0 < a < 6.0


def test_logsumexp_tau_nho_tien_ve_max():
    a = pool_scores(DOCS, SCORES, "logsumexp", tau=0.01)["a"]
    assert a == pytest.approx(3.0, abs=1e-6)


def test_parse_pool_sai_ten_bao_loi():
    with pytest.raises(ValueError, match="không có"):
        parse_pool("mean")
    with pytest.raises(ValueError, match="tau > 0"):
        pool_scores(DOCS, SCORES, "logsumexp", tau=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hợp đồng INTERFACES.md mục 3
# ─────────────────────────────────────────────────────────────────────────────
def test_check_contract_bat_doc_id_int():
    with pytest.raises(AssertionError, match="doc_id"):
        check_contract([[(740, 1.0)]], 1, 5)


def test_check_contract_bat_trung_lap():
    with pytest.raises(AssertionError, match="trùng lặp"):
        check_contract([[("a", 2.0), ("a", 1.0)]], 1, 5)


def test_check_contract_bat_sai_thu_tu():
    with pytest.raises(AssertionError, match="giảm dần"):
        check_contract([[("a", 1.0), ("b", 2.0)]], 1, 5)


def test_check_contract_bat_qua_top_k():
    with pytest.raises(AssertionError, match="top_k"):
        check_contract([[("a", 2.0), ("b", 1.0)]], 1, 1)


def test_search_dung_hop_dong():
    r = build_bm25()
    queries = ["huỷ bỏ hoá đơn điện tử", "quan hệ lao động", "xyz không có trong corpus"]
    res = r.search(queries, top_k=2)
    assert len(res) == len(queries)
    for one in res:
        assert len(one) <= 2
        ids = [d for d, _ in one]
        assert ids == list(dict.fromkeys(ids))          # không trùng doc_id
        assert all(isinstance(d, str) for d in ids)      # str, không phải int
        assert all(isinstance(s, float) for _, s in one)
        scores = [s for _, s in one]
        assert scores == sorted(scores, reverse=True)


def test_search_tim_dung_van_ban():
    r = build_bm25()
    (top,) = r.search(["cơ quan nào có thẩm quyền huỷ bỏ hoá đơn điện tử"], top_k=3)
    assert top[0][0] == "740"


def test_query_khong_khop_term_nao_tra_rong():
    """Phải là list rỗng, KHÔNG phải None — hạ nguồn (make_submission) đếm len()."""
    r = build_bm25()
    assert r.search(["zzzz qqqq"], top_k=5) == [[]]


def test_goi_search_truoc_index_bao_loi():
    with pytest.raises(RuntimeError, match="index"):
        BM25Retriever(verbose=False).search(["a"], top_k=5)


def test_search_chunks_tra_chunk_id():
    r = build_bm25()
    (res,) = r.search_chunks(["hoá đơn điện tử"], top_k=2)
    assert all("::" in cid for cid, _ in res)
    assert res[0][0] in {c["chunk_id"] for c in CHUNKS}


def test_candidates_va_search_khop_nhau():
    """Lưới bench gộp lại từ `candidates()`; kết quả phải trùng `search()` từng chữ."""
    r = build_bm25(pool="mean_top3")
    q = ["huỷ bỏ hoá đơn điện tử"]
    (idx, sc) = r.candidates(q, r.candidate_chunks)[0]
    assert r.pool_candidates(idx, sc, 3) == r.search(q, 3)[0]


def test_doi_chien_luoc_gop_doi_ket_qua():
    r = build_bm25()
    q = ["hoá đơn điện tử huỷ bỏ biên bản"]
    (idx, sc) = r.candidates(q, r.candidate_chunks)[0]
    assert (
        r.pool_candidates(idx, sc, 3, pool="max")
        != r.pool_candidates(idx, sc, 3, pool="sum")
    )


def test_top_k_khong_hop_le():
    with pytest.raises(ValueError, match="top_k"):
        build_bm25().search(["a"], top_k=0)


def test_pool_sai_ten_fail_ngay_luc_dung_retriever():
    """Fail trước khi index — không ai muốn biết mình gõ sai sau 3 phút đánh chỉ mục."""
    with pytest.raises(ValueError, match="Chiến lược gộp"):
        BM25Retriever(pool="maximum", verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# BM25: chỉ mục và tham số
# ─────────────────────────────────────────────────────────────────────────────
def test_word_segment_giup_phan_biet_co_quan_voi_co_so_quan_he():
    """
    Lý do tồn tại của cả trục ablation này: 'cơ quan' tách rời khớp nhầm văn bản chỉ có
    'cơ sở' và 'quan hệ'. Bigram/word-segment gộp lại thành một term hiếm.
    """
    r = build_bm25(tokenizer="syllable_bigram")
    assert "cơ_quan" in r.vocab
    (top,) = r.search(["cơ quan"], top_k=3)
    assert top[0][0] == "740"


def test_min_df_cat_bot_tu_vung():
    ít = build_bm25(min_df=1)
    nhiều = build_bm25(min_df=3)
    assert len(nhiều.vocab) < len(ít.vocab)


def test_candidate_chunks_null_giu_toan_bo():
    """`candidate_chunks: null` trong YAML = hành vi v0.1, dùng để kiểm chứng phần cắt."""
    r = build_bm25(candidate_chunks=None)
    assert r.candidate_chunks > len(CHUNKS)


def test_explain_query_chi_ra_term_ngoai_tu_vung():
    r = build_bm25()
    out = r.explain_query("hoá đơn điện tử và một từ lạ zzzz")
    assert "zzzz" in out["oov"]
    assert out["top_terms"]


def test_stats_du_thong_tin_ghi_log():
    s = build_bm25(tokenizer="whitespace").stats()
    assert s["retriever"] == "bm25"
    assert s["n_docs"] == 3 and s["n_chunks"] == len(CHUNKS)
    assert s["tokenizer"].startswith("whitespace") and s["vocab"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Đăng ký chunk & registry
# ─────────────────────────────────────────────────────────────────────────────
def test_ep_str_cho_doc_id_kieu_int():
    """INTERFACES.md mục 0 — int lọt xuống submission là 0 điểm im lặng."""
    r = BM25Retriever(verbose=False)
    r.index([{"doc_id": 740, "chunk_id": "740::0000", "text": "hoá đơn điện tử"}])
    assert r.chunk_doc_ids == ["740"]
    assert r.search(["hoá đơn"], top_k=1)[0][0][0] == "740"


def test_thieu_chunk_id_thi_tu_sinh():
    r = BM25Retriever(verbose=False)
    r.index([{"doc_id": "740", "position": 3, "text": "hoá đơn điện tử"}])
    assert r.chunk_ids == ["740::0003"]


def test_chunk_id_trung_bao_loi():
    r = BM25Retriever(verbose=False)
    with pytest.raises(ValueError, match="trùng"):
        r.index([dict(CHUNKS[0]), dict(CHUNKS[0])])


def test_build_retriever_tu_config():
    r = build_retriever({"type": "bm25", "k1": 1.2, "pool": "mean_top3", "verbose": False})
    assert isinstance(r, BM25Retriever) and r.k1 == 1.2 and r.pool == "mean_top3"


def test_build_retriever_ten_la():
    with pytest.raises(KeyError, match="chưa đăng ký"):
        build_retriever({"type": "dense"})


def test_build_retriever_thieu_type():
    with pytest.raises(ValueError, match="type"):
        build_retriever({"k1": 1.2})


def test_tokenizer_key_di_vao_log():
    """Key phải phân biệt được cấu hình, nếu không experiments.csv ghi hai dòng giống hệt."""
    assert Tokenizer("pyvi").key == "pyvi"
    assert Tokenizer("regex", fold_tone=False).key == "regex-notonefold"


# ─────────────────────────────────────────────────────────────────────────────
# Đọc file của P2: schema thật lệch INTERFACES.md (xem ghi chú đầu src/common/io.py)
# ─────────────────────────────────────────────────────────────────────────────
def _viet_jsonl(path, records):
    import json

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    return path


def test_doc_schema_thuc_te_cua_p2_van_nap_duoc(tmp_path):
    """`src/data/chunker.py` ghi 'passage' + 'id', hợp đồng ghi 'text' + 'doc_id'."""
    from src.common.io import load_chunks

    p = _viet_jsonl(
        tmp_path / "chunks.jsonl",
        [{"chunk_id": "740::0", "doc_id": 740, "passage": "hoá đơn điện tử", "name": "TT78"}],
    )
    (c,) = load_chunks(p)
    assert c["text"] == "hoá đơn điện tử"
    assert c["doc_id"] == "740" and isinstance(c["doc_id"], str)


def test_position_suy_ra_tu_chunk_id_khi_p2_khong_ghi(tmp_path):
    """P2 không ghi 'position'; suy ngược từ đuôi chunk_id để P4 tra lại được thứ tự."""
    from src.common.io import load_chunks

    p = _viet_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"chunk_id": "740::0", "doc_id": "740", "passage": "a"},
            {"chunk_id": "740::12", "doc_id": "740", "passage": "b"},
        ],
    )
    assert [c["position"] for c in load_chunks(p)] == [0, 12]


def test_bo_qua_chunk_rong(tmp_path):
    """EDA mục 3: 20 văn bản passage rỗng. Index chúng chỉ phồng ma trận, không khớp được gì."""
    from src.common.io import load_chunks

    p = _viet_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"chunk_id": "1::0", "doc_id": "1", "passage": "có nội dung"},
            {"chunk_id": "2::0", "doc_id": "2", "passage": "   "},
        ],
    )
    assert len(load_chunks(p)) == 1
    assert len(load_chunks(p, skip_empty=False)) == 2


def test_corpus_ids_doc_duoc_ca_hai_ten_truong(tmp_path):
    from src.common.io import load_corpus_ids

    p = _viet_jsonl(tmp_path / "c.jsonl", [{"id": 740, "passage": "x"}, {"id": "812", "passage": "y"}])
    assert load_corpus_ids(p) == {"740", "812"}


def test_thieu_ca_hai_ten_truong_thi_bao_loi_ro(tmp_path):
    from src.common.io import load_chunks

    p = _viet_jsonl(tmp_path / "chunks.jsonl", [{"chunk_id": "1::0", "noi_dung": "x"}])
    with pytest.raises(ValueError, match="INTERFACES"):
        load_chunks(p)
