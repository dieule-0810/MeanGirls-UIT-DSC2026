"""Kiểm hai chiến lược của `src/data/chunker.py` (loose = kho 524.422, strict = kho 432.142)."""
from __future__ import annotations

import pytest

from src.data.chunker import ChunkerConfig, chunk_document, sliding_window, split_by_dieu, STRATEGIES

DOC = (
    "QUYẾT ĐỊNH về việc ban hành quy chế. "
    "Điều 1. Phạm vi điều chỉnh theo Điều 5 Luật Đất đai. "
    "Điều 2. Đối tượng áp dụng gồm cơ quan nhà nước."
)


def test_cua_so_truot_goi_dau_va_dung_o_cuoi():
    words = " ".join(f"w{i}" for i in range(10))
    assert sliding_window(words, chunk_size=4, overlap=1) == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert sliding_window("ngắn thôi", 4, 1) == ["ngắn thôi"]
    assert sliding_window("", 4, 1) == []


def test_loose_cat_ca_trich_dan_cheo_strict_thi_khong():
    """`Điều 5 Luật ...` giữa câu là trích dẫn, không phải ranh giới — chỉ strict nhận ra."""
    loose = split_by_dieu(DOC, STRATEGIES["loose"].dieu_pattern)
    strict = split_by_dieu(DOC, STRATEGIES["strict"].dieu_pattern)
    assert len(loose) == 4 and len(strict) == 3
    assert strict[1].startswith("Điều 1.") and "Điều 5 Luật" in strict[1]


def test_dieu_dai_gan_nhan_tiep_theo_va_strict_tru_ngan_sach():
    body = " ".join(f"t{i}" for i in range(30))
    text = f"Điều 7. {body}"
    for name in ("loose", "strict"):
        chunks = chunk_document(text, ChunkerConfig(chunk_size=10, overlap=2, strategy=name))
        assert chunks[0].startswith("Điều 7")
        assert all(c.startswith("[Điều 7") for c in chunks[1:]), name
        if name == "strict":
            assert all(len(c.split()) <= 10 for c in chunks), "strict phải trừ nhãn vào ngân sách"


def test_van_ban_rong_va_strategy_sai():
    assert chunk_document("   ") == []
    with pytest.raises(ValueError, match="strategy"):
        ChunkerConfig(strategy="theo-dieu")
