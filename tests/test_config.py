"""Kiểm `src/common/config.py` (kế thừa `extends`) và mọi config thật trong `configs/`."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import REPO, deep_merge, load_config
from src.retrieval.base import retriever_spec

CONFIGS = sorted(p for p in (REPO / "configs").glob("v*.yaml"))


def write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def test_deep_merge_gop_de_quy_va_null_la_gia_tri_that():
    base = {"a": {"x": 1, "y": 2}, "b": 1, "c": 2000}
    out = deep_merge(base, {"a": {"y": 3}, "c": None})
    assert out == {"a": {"x": 1, "y": 3}, "b": 1, "c": None}
    assert base["a"]["y"] == 2, "deep_merge không được sửa base tại chỗ"


def test_extends_ca_file_va_nhanh_con(tmp_path):
    write(tmp_path, "base.yaml", "seed: 1\nretrieval:\n  bm25: {k1: 1.5, b: 0.75}\n")
    write(tmp_path, "child.yaml", "extends: base.yaml\nretrieval:\n  bm25: {k1: 1.2}\n")
    write(tmp_path, "src.yaml", "x:\n  extends: child.yaml#retrieval.bm25\n  weight: 0.4\n")
    cfg = load_config(tmp_path / "src.yaml")
    assert cfg == {"x": {"k1": 1.2, "b": 0.75, "weight": 0.4}}


def test_extends_vong_tron_bao_loi(tmp_path):
    write(tmp_path, "a.yaml", "extends: b.yaml\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="vòng tròn"):
        load_config(tmp_path / "a.yaml")


def test_extends_nhanh_khong_ton_tai_bao_loi(tmp_path):
    write(tmp_path, "base.yaml", "retrieval: {}\n")
    write(tmp_path, "c.yaml", "x:\n  extends: base.yaml#retrieval.bm25\n")
    with pytest.raises(KeyError, match="retrieval.bm25"):
        load_config(tmp_path / "c.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_moi_config_that_doc_duoc_va_dung_duoc_spec(path):
    """Mọi config trong repo phải giải hết `extends` và chỉ ra đúng MỘT retriever."""
    cfg = load_config(path)
    assert "extends" not in str(cfg), "còn khoá `extends` chưa giải"
    assert cfg["exp_id"] == path.stem or path.stem == "v0.2_bm25_tokenizer"
    spec = retriever_spec(cfg, demo=True)
    assert spec["type"] in ("bm25", "dense", "hybrid")
    if spec["type"] == "hybrid":
        assert all("type" in s for s in spec["sources"].values()), "nguồn hybrid thiếu `type`"


def test_hybrid_v08_ke_thua_dung_khoi_chuan():
    """Nguồn của hybrid nộp bài phải TRÙNG khối chuẩn của v0.3 (BM25) và v0.4 (dense)."""
    hybrid = load_config("configs/v0.8_hybrid_rrf.yaml")["retrieval"]["hybrid"]["sources"]
    bm25 = load_config("configs/v0.3_bm25_best.yaml")["retrieval"]["bm25"]
    dense = load_config("configs/v0.4_dense.yaml")["retrieval"]["dense"]
    assert {k: v for k, v in hybrid["bm25"].items() if k not in ("type", "weight")} == bm25
    assert {k: v for k, v in hybrid["dense"].items() if k not in ("type", "weight")} == dense
    assert list(hybrid) == ["bm25", "dense"], "nguồn ĐẦU là nguồn tune_rrf quét trọng số"
