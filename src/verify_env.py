"""Kiểm tra môi trường chạy (phiên bản Python, gói bắt buộc).

Lý do tồn tại: BTC clone repo về tháng 10 và chạy. Nếu Python/thư viện lệch, ta muốn
họ biết sau 2 GIÂY với thông báo rõ ràng, chứ không phải sau 40 phút với một traceback
ở chỗ chẳng liên quan gì. Đội nào để BTC phải tự debug môi trường thì khả năng cao là
bị hủy kết quả chứ không phải được nhắn tin hỏi lại.
"""
from __future__ import annotations

import sys

REQUIRED_PYTHON = (3, 11)
REQUIRED_PACKAGES = {"numpy": None, "yaml": None}   # v0.1 cố ý giữ tối thiểu


def check(strict: bool = True) -> bool:
    """Kiểm Python 3.11.x và các gói bắt buộc; in thông tin torch nếu có.

    Args:
        strict: True thì lệch phiên bản Python là lỗi (❌), False thì chỉ cảnh báo.

    Returns:
        True nếu môi trường đạt.
    """
    problems: list[str] = []

    if sys.version_info[:2] != REQUIRED_PYTHON:
        msg = (f"Python {'.'.join(map(str, REQUIRED_PYTHON))}.x là bản đã kiểm chứng, "
               f"đang chạy {sys.version_info.major}.{sys.version_info.minor}")
        problems.append(("❌ " if strict else "⚠️  ") + msg)

    import importlib
    for mod in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod)
        except ImportError:
            problems.append(f"❌ Thiếu gói: {mod}  →  pip install -r requirements.txt")

    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"   torch {torch.__version__}, thiết bị: {dev}")
    except ImportError:
        print("   torch: chưa cài (v0.1 chưa cần)")

    if problems:
        print("\n".join(problems))
        return False
    print(f"✅ Môi trường OK — Python {sys.version.split()[0]}")
    return True


def require() -> None:
    """Gọi ở đầu script: dừng ngay nếu môi trường không đạt.

    Raises:
        SystemExit: Môi trường không đạt.
    """
    if not check(strict=True):
        raise SystemExit("Môi trường không đạt. Xem README mục 2 'Yêu cầu hệ thống'.")


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
