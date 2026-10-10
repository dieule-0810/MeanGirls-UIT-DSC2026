"""Đọc config YAML có kế thừa (`extends`) — nguồn sự thật duy nhất cho mọi script.

Trước đây 13 file trong `configs/` chép lại nguyên khối `paths` và khối `bm25`/`dense`, nhiều
file chỉ khác nhau ĐÚNG MỘT DÒNG. Chép tay như vậy là chỗ sinh lệch im lặng: sửa tham số ở một
file mà quên file kia, và đã có hai file hỏng vì conflict merge dán nhầm vào giữa khối.

Cú pháp: khoá `extends` được phép ở BẤT KỲ mức mapping nào.

    extends: base.yaml                                  # kế thừa cả file
    retrieval:
      hybrid:
        sources:
          bm25:
            extends: v0.3_bm25_best.yaml#retrieval.bm25  # kế thừa một nhánh con
            weight: 0.4

Đường dẫn trong `extends` tính tương đối so với file đang khai nó. Phần ghi đè gộp ĐỆ QUY vào
phần kế thừa: mapping gộp theo khoá, mọi giá trị khác (list, số, chuỗi, `null`) thay thế
nguyên khối. `null` là giá trị thật (vd `candidate_chunks: null` = không cắt ứng viên), không
phải "xoá khoá".

Typical usage example:

    cfg = load_config("configs/v0.8_hybrid_rrf.yaml")
    out_dir = resolve_out_dir(cfg)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
EXTENDS_KEY = "extends"


def deep_merge(base: dict, override: dict) -> dict:
    """Gộp đệ quy `override` vào bản sao của `base`.

    Args:
        base: Mapping được kế thừa. Không bị sửa tại chỗ.
        override: Mapping ghi đè. Khoá trùng mà cả hai bên là dict thì gộp tiếp; còn lại
            giá trị của `override` thay thế nguyên khối.

    Returns:
        Mapping mới đã gộp.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _subtree(tree: dict, dotted: str, source: Path) -> dict:
    """Lấy nhánh con theo đường dẫn chấm (`retrieval.bm25`), báo lỗi rõ nếu không có."""
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{source}: không có nhánh '{dotted}' (vấp ở '{part}').")
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"{source}#{dotted} không phải mapping, không kế thừa được.")
    return node


def _resolve_ref(ref: str, origin: Path, chain: tuple[Path, ...]) -> dict:
    """Giải một giá trị `extends` (`file.yaml` hoặc `file.yaml#a.b`) thành mapping đã giải."""
    file_part, _, dotted = ref.partition("#")
    target = (origin.parent / file_part).resolve()
    if target in chain:
        cycle = " → ".join(p.name for p in (*chain, target))
        raise ValueError(f"`extends` vòng tròn: {cycle}")
    tree = _load_resolved(target, chain + (target,))
    return copy.deepcopy(_subtree(tree, dotted, target) if dotted else tree)


def _resolve_node(node: Any, origin: Path, chain: tuple[Path, ...]) -> Any:
    """Duyệt đệ quy, thay mọi mapping có `extends` bằng bản đã gộp."""
    if isinstance(node, list):
        return [_resolve_node(v, origin, chain) for v in node]
    if not isinstance(node, dict):
        return node
    own = {k: _resolve_node(v, origin, chain) for k, v in node.items() if k != EXTENDS_KEY}
    ref = node.get(EXTENDS_KEY)
    if ref is None:
        return own
    if not isinstance(ref, str):
        raise TypeError(f"{origin}: `extends` phải là chuỗi, nhận {type(ref).__name__}.")
    return deep_merge(_resolve_ref(ref, origin, chain), own)


def _load_resolved(path: Path, chain: tuple[Path, ...]) -> dict:
    """Đọc một file YAML và giải đệ quy mọi `extends` trong nó."""
    if not path.exists():
        raise FileNotFoundError(f"Không thấy config {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config phải là mapping ở gốc.")
    return _resolve_node(raw, path, chain)


def load_config(path: str | Path) -> dict:
    """Đọc một file config và giải hết mọi `extends`.

    Args:
        path: Đường dẫn tuyệt đối, hoặc tương đối so với thư mục hiện hành; không thấy thì
            thử tương đối so với gốc repo (để script chạy được từ bất kỳ đâu).

    Returns:
        Config đã gộp, không còn khoá `extends` nào.

    Raises:
        FileNotFoundError: Không tìm thấy file config hoặc file được kế thừa.
        ValueError: `extends` tạo thành vòng.
        KeyError: `extends` trỏ tới một nhánh không tồn tại.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO / p
    p = p.resolve()
    return _load_resolved(p, (p,))


def repo_path(path: str | Path) -> Path:
    """Đường dẫn trong config (tương đối so với gốc repo) → đường dẫn tuyệt đối."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def resolve_out_dir(cfg: dict, override: str | None = None, fallback: str = "run") -> Path:
    """Thư mục ra của một lần chạy: `--out-dir` > `paths.out_dir` > `outputs/<exp_id>`.

    Args:
        cfg: Config đã đọc bằng `load_config`.
        override: Giá trị `--out-dir` từ dòng lệnh, nếu có.
        fallback: Tên dùng khi config thiếu cả `out_dir` lẫn `exp_id`.

    Returns:
        Đường dẫn tuyệt đối (chưa tạo thư mục).
    """
    default = f"outputs/{cfg.get('exp_id', fallback)}"
    return repo_path(override or cfg.get("paths", {}).get("out_dir", default))
