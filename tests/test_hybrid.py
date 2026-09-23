"""
Kiểm chứng `src/retrieval/hybrid.py` — hợp nhất RRF nhiều nguồn. CHỦ SỞ HỮU: P3.

Chạy được KHÔNG cần `data/` và KHÔNG cần torch: hai nguồn trong corpus giả đều là BM25 với
tokenizer khác nhau. Đó không phải cách dùng thật (thật là BM25+dense) nhưng nó kiểm đúng thứ
cần kiểm — cơ chế hợp nhất — mà không kéo theo 2 GB trọng số vào CI.

    python -m pytest tests/test_hybrid.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.base import build_retriever, check_contract, retriever_spec
from src.retrieval.hybrid import (
    HybridRetriever,
    normalize_weights,
    rrf_from_ranks,
    sweep_weights,
)

CHUNKS = [
    {"chunk_id": "740::0000", "doc_id": "740", "position": 0,
     "text": "Điều 1. Cơ quan thuế có thẩm quyền huỷ bỏ hoá đơn điện tử đã lập sai."},
    {"chunk_id": "740::0001", "doc_id": "740", "position": 1,
     "text": "Điều 2. Việc huỷ bỏ hoá đơn điện tử phải lập biên bản theo Thông tư số 78/2021/TT-BTC."},
    {"chunk_id": "812::0000", "doc_id": "812", "position": 0,
     "text": "Điều 1. Cơ sở dữ liệu quốc gia về dân cư do Bộ Công an quản lý."},
    {"chunk_id": "812::0001", "doc_id": "812", "position": 1,
     "text": "Điều 2. Quan hệ lao động giữa người sử dụng lao động và người lao động."},
    {"chunk_id": "999::0000", "doc_id": "999", "position": 0,
     "text": "Điều 1. Uỷ ban nhân dân cấp huyện quyết định thu hồi đất theo quy định."},
    {"chunk_id": "999::0001", "doc_id": "999", "position": 1,
     "text": "Điều 2. Hồ sơ thu hồi đất gồm quyết định thu hồi và biên bản bàn giao."},
]

QUERIES = [
    "Huỷ bỏ hoá đơn điện tử lập sai thì cơ quan nào có thẩm quyền?",
    "Cơ sở dữ liệu quốc gia về dân cư do ai quản lý?",
    "Hồ sơ thu hồi đất gồm những gì?",
]


def bm25_source(tokenizer: str, weight: float, pool: str = "max") -> dict:
    return {
        "type": "bm25", "weight": weight, "tokenizer": tokenizer,
        "tokenizer_opts": {"fold_tone": True, "min_len": 1},
        "pool": pool, "candidate_chunks": 50, "verbose": False,
    }


def build_hybrid(**opts) -> HybridRetriever:
    opts.setdefault("sources", {
        "a": bm25_source("syllable_bigram", 0.6),
        "b": bm25_source("regex", 0.4),
    })
    opts.setdefault("verbose", False)
    opts.setdefault("candidate_chunks", 50)
    r = HybridRetriever(**opts)
    r.index([dict(c) for c in CHUNKS])
    return r


# ─────────────────────────────────────────────────────────────────────────────
# RRF thuần — hàm tham chiếu
# ─────────────────────────────────────────────────────────────────────────────
def test_rrf_cong_dung_cong_thuc():
    fused = rrf_from_ranks({"a": ["x", "y"], "b": ["y"]}, {"a": 0.5, "b": 0.5}, rrf_k=60)
    assert fused["x"] == pytest.approx(0.5 / 61)
    assert fused["y"] == pytest.approx(0.5 / 62 + 0.5 / 61)


def test_rrf_bo_qua_nguon_trong_so_0():
    """Trọng số 0 phải là KHÔNG ĐÓNG GÓP, không phải 'đóng góp ít' — để w=1 tái lập nguồn đơn."""
    fused = rrf_from_ranks({"a": ["x"], "b": ["z"]}, {"a": 1.0, "b": 0.0}, rrf_k=60)
    assert set(fused) == {"x"}


def test_absent_rank_thay_doi_ket_qua_theo_dung_dau_da_ghi():
    """`absent_rank` cộng lượng DƯƠNG cho phần tử VẮNG MẶT — ghi rõ trong docstring, kiểm ở đây."""
    om = rrf_from_ranks({"a": ["x"], "b": ["y"]}, {"a": 0.5, "b": 0.5}, 60, absent_rank=None)
    big = rrf_from_ranks({"a": ["x"], "b": ["y"]}, {"a": 0.5, "b": 0.5}, 60, absent_rank=10**6)
    assert om["x"] == pytest.approx(om["y"])
    assert big["x"] > om["x"]


def test_normalize_weights_tong_0_bao_loi():
    with pytest.raises(ValueError, match="Tổng trọng số"):
        normalize_weights({"a": 0.0, "b": 0.0})


def test_sweep_weights_hai_nguon_la_w_va_1_tru_w():
    assert sweep_weights(["a", "b"], 0.7, {"a": 0.6, "b": 0.4}) == pytest.approx({"a": 0.7, "b": 0.3})


def test_sweep_weights_ba_nguon_giu_ti_le_cac_nguon_phu():
    w = sweep_weights(["a", "b", "c"], 0.6, {"a": 0.5, "b": 0.3, "c": 0.1})
    assert w["a"] == pytest.approx(0.6)
    assert w["b"] + w["c"] == pytest.approx(0.4)
    assert w["b"] / w["c"] == pytest.approx(3.0)   # giữ nguyên 0,3 : 0,1


# ─────────────────────────────────────────────────────────────────────────────
# Hợp đồng INTERFACES §3
# ─────────────────────────────────────────────────────────────────────────────
def test_search_dung_hop_dong():
    r = build_hybrid()
    res = r.search(QUERIES, top_k=3)
    check_contract(res, len(QUERIES), 3)   # tự nó cũng chạy trong search(), gọi lại cho rõ ý


def test_search_tim_dung_van_ban():
    r = build_hybrid()
    for q, want in zip(QUERIES, ["740", "812", "999"]):
        assert r.search([q], top_k=1)[0][0][0] == want


def test_search_with_anchor_tra_chunk_thuoc_dung_doc():
    r = build_hybrid()
    for res in r.search_with_anchor(QUERIES, top_k=3):
        for doc_id, _, chunk_id in res:
            assert chunk_id.split("::")[0] == doc_id


@pytest.mark.parametrize("level,pool", [("chunk", "max"), ("chunk", "mean_top2"), ("doc", "max")])
def test_hai_lan_chay_giong_het_nhau(level, pool):
    """Phá hoà phải XÁC ĐỊNH — không thì mỗi lần chạy ra một bài nộp khác."""
    a = build_hybrid(fuse_level=level, pool=pool).search(QUERIES, top_k=3)
    b = build_hybrid(fuse_level=level, pool=pool).search(QUERIES, top_k=3)
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# w=1 phải tái lập CHÍNH XÁC nguồn đơn — nếu không thì mọi con số "hợp nhất hơn
# bao nhiêu" đều so với một đường cơ sở sai.
# ─────────────────────────────────────────────────────────────────────────────
def test_muc_doc_w1_tai_lap_nguon_don():
    r = build_hybrid(
        fuse_level="doc", pool="max",
        sources={"a": bm25_source("syllable_bigram", 1.0, pool="mean_top2"),
                 "b": bm25_source("regex", 0.0, pool="mean_top2")},
    )
    solo = r.sources["a"]
    for q in QUERIES:
        got = [d for d, _ in r.search([q], top_k=3)[0]]
        want = [d for d, _ in solo.search([q], top_k=3)[0]]
        assert got == want


def test_muc_chunk_w1_tai_lap_tap_van_ban_cua_nguon_don():
    """
    Ở mức chunk, điểm RRF là hàm ĐƠN ĐIỆU của thứ hạng chứ không phải chính điểm gốc, nên với
    `pool=max` tập văn bản trả về trùng nhau và văn bản đứng đầu trùng nhau; còn thứ tự phía
    sau có thể lệch ở những chỗ điểm gốc HOÀ NHAU (RRF phá hoà bằng thứ hạng, gốc phá bằng
    doc_id). So bằng tập là so đúng thứ cần so — metric của cuộc thi không dùng thứ tự.
    """
    r = build_hybrid(
        fuse_level="chunk", pool="max",
        sources={"a": bm25_source("syllable_bigram", 1.0),
                 "b": bm25_source("regex", 0.0)},
    )
    solo = r.sources["a"]
    for q in QUERIES:
        got = [d for d, _ in r.search([q], top_k=3)[0]]
        want = [d for d, _ in solo.search([q], top_k=3)[0]]
        assert set(got) == set(want)
        assert got[0] == want[0]


def test_duong_vector_hoa_khop_ham_tham_chieu():
    """`_fuse_chunk_level` vector hoá bằng numpy — phải ra ĐÚNG như `rrf_from_ranks`."""
    r = build_hybrid(fuse_level="chunk")
    per_source = r.source_candidates(QUERIES[:1])
    one = {n: per_source[n][0] for n in r.sources}
    idx, sc = r.fuse_query(one)
    want = rrf_from_ranks(
        {n: [int(i) for i in one[n][0]] for n in one}, r.weights, r.rrf_k, None
    )
    assert len(idx) == len(want)
    for i, s in zip(idx, sc):
        assert s == pytest.approx(want[int(i)])
    assert list(sc) == sorted(sc, reverse=True)


def test_hop_nhat_thay_doi_khi_doi_trong_so():
    """
    Nếu w không đổi được kết quả thì quét w là vô nghĩa — kiểm cơ chế có thật sự cắm vào.

    Ứng viên ở đây dựng TAY chứ không lấy từ BM25: trên corpus giả 6 chunk, `syllable_bigram`
    và `whitespace` cho ĐÚNG CÙNG thứ hạng (chỉ khác thang điểm), nên không trọng số nào đổi
    được gì và phép kiểm sẽ đạt một cách rỗng. Hai nguồn BẤT ĐỒNG mới là tình huống mà hợp
    nhất sinh ra để xử lý, và đó là thứ cần kiểm.
    """
    r = build_hybrid(fuse_level="chunk")
    a_idx = np.array([0, 2], dtype=np.int64)          # nguồn 'a': chunk 0 trước
    b_idx = np.array([2, 0], dtype=np.int64)          # nguồn 'b': chunk 2 trước
    one = {"a": (a_idx, np.array([9.0, 1.0])), "b": (b_idx, np.array([9.0, 1.0]))}

    assert list(r.fuse_query(one, {"a": 1.0, "b": 0.0})[0]) == [0, 2]
    assert list(r.fuse_query(one, {"a": 0.0, "b": 1.0})[0]) == [2, 0]
    # Hoà 50/50: hai chunk cùng điểm ⇒ phá hoà bằng chỉ số, xác định và không phụ thuộc
    # thứ tự duyệt dict các nguồn.
    idx, sc = r.fuse_query(one, {"a": 0.5, "b": 0.5})
    assert list(idx) == [0, 2] and sc[0] == pytest.approx(sc[1])


def test_muc_doc_cung_phan_ung_voi_trong_so():
    """Cùng phép kiểm ở mức doc — hai mức là hai nhánh code, đừng để một nhánh trôi."""
    r = build_hybrid(
        fuse_level="doc", pool="max",
        sources={"a": bm25_source("syllable_bigram", 0.5), "b": bm25_source("regex", 0.5)},
    )
    # chunk 0 → doc 740, chunk 2 → doc 812 (xem CHUNKS)
    one = {"a": (np.array([0, 2], dtype=np.int64), np.array([9.0, 1.0])),
           "b": (np.array([2, 0], dtype=np.int64), np.array([9.0, 1.0]))}
    top_a = r.chunk_doc_ids[int(r.fuse_query(one, {"a": 1.0, "b": 0.0})[0][0])]
    top_b = r.chunk_doc_ids[int(r.fuse_query(one, {"a": 0.0, "b": 1.0})[0][0])]
    assert (top_a, top_b) == ("740", "812")


def test_rrf_k_thay_doi_ket_qua():
    r = build_hybrid(fuse_level="chunk")
    one = {n: v[0] for n, v in r.source_candidates(QUERIES[:1]).items()}
    assert r.fuse_query(one, rrf_k=1)[1][0] != pytest.approx(r.fuse_query(one, rrf_k=1000)[1][0])


# ─────────────────────────────────────────────────────────────────────────────
# Fail loud
# ─────────────────────────────────────────────────────────────────────────────
def test_mot_nguon_bao_loi():
    with pytest.raises(ValueError, match="ÍT NHẤT 2 nguồn"):
        HybridRetriever(sources={"a": bm25_source("regex", 1.0)}, verbose=False)


def test_muc_doc_voi_pool_khac_max_bao_loi():
    """Pool ở tầng hybrid khi fuse_level=doc là phép đồng nhất — im lặng chấp nhận là bẫy."""
    with pytest.raises(ValueError, match="fuse_level='doc'"):
        HybridRetriever(
            sources={"a": bm25_source("regex", 0.5), "b": bm25_source("whitespace", 0.5)},
            fuse_level="doc", pool="mean_top2", verbose=False,
        )


def test_fuse_level_sai_ten_bao_loi_ngay_luc_dung():
    with pytest.raises(ValueError, match="fuse_level"):
        HybridRetriever(
            sources={"a": bm25_source("regex", 0.5), "b": bm25_source("whitespace", 0.5)},
            fuse_level="document", verbose=False,
        )


def test_trong_so_am_bao_loi():
    with pytest.raises(ValueError, match="âm"):
        HybridRetriever(
            sources={"a": bm25_source("regex", -0.5), "b": bm25_source("whitespace", 1.0)},
            verbose=False,
        )


def test_nguon_nhin_khac_kho_chunk_bao_loi():
    """
    Nguồn lọc bớt chunk trong `index()` của nó ⇒ chỉ số hàng lệch ⇒ RRF mức chunk cộng nhầm
    chunk. Không có biểu hiện nào ngoài recall tụt, nên phải chặn bằng lỗi.
    """
    r = HybridRetriever(
        sources={"a": bm25_source("regex", 0.5), "b": bm25_source("whitespace", 0.5)},
        verbose=False,
    )
    lech = r.sources["b"]
    _index_that = lech.index
    lech.index = lambda chunks: _index_that(chunks[:-1])
    with pytest.raises(ValueError, match="không nhìn cùng một kho chunk"):
        r.index([dict(c) for c in CHUNKS])


# ─────────────────────────────────────────────────────────────────────────────
# Cắm vào registry và vào config
# ─────────────────────────────────────────────────────────────────────────────
def test_build_retriever_dung_duoc_tu_config():
    r = build_retriever({
        "type": "hybrid", "verbose": False, "candidate_chunks": 50,
        "sources": {"a": bm25_source("regex", 0.5), "b": bm25_source("whitespace", 0.5)},
    })
    assert isinstance(r, HybridRetriever)
    assert r.weights == pytest.approx({"a": 0.5, "b": 0.5})


def test_retriever_spec_noi_duong_dan_vao_nguon_con():
    cfg = {
        "pipeline": {"retriever": "hybrid"},
        "paths": {"cache_dir": "data/cache", "embeddings": "data/embeddings.npy"},
        "retrieval": {"hybrid": {"sources": {
            "bm25": {"type": "bm25", "weight": 0.6},
            "dense": {"type": "dense", "weight": 0.4},
        }}},
    }
    spec = retriever_spec(cfg)
    assert spec["sources"]["bm25"]["cache_dir"] == "data/cache"
    assert spec["sources"]["dense"]["embeddings_path"] == "data/embeddings.npy"
    assert spec["sources"]["bm25"]["weight"] == 0.6

    demo = retriever_spec(cfg, demo=True)
    assert "cache_dir" not in demo["sources"]["bm25"]
    assert "embeddings_path" not in demo["sources"]["dense"]


def test_stats_ghi_du_de_tai_lap():
    r = build_hybrid(fuse_level="doc", pool="max", rrf_k=17)
    s = r.stats()
    assert s["retriever"] == "hybrid"
    assert s["fuse_level"] == "doc" and s["rrf_k"] == 17
    assert set(s["sources"]) == {"a", "b"}
    assert s["weights"]["a"] == pytest.approx(0.6)
