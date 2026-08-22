"""
scripts/verify_bge_m3.py — đếm tham số thật của BAAI/bge-m3.

BGE-M3 không có safetensors nên count_params.py không đọc được. Script này tải
pytorch_model.bin và đếm trực tiếp, đồng thời kiểm tra hai head phụ.

    pip install torch huggingface_hub
    python scripts/verify_bge_m3.py

Tải ~2.3GB. Chạy một lần, ghi kết quả vào docs/param_audit.md rồi đặt
params_verified: true trong configs/models.yaml.
"""

import torch
from huggingface_hub import hf_hub_download

REPO = "BAAI/bge-m3"
REVISION = "5617a9f61b02"  # phải khớp configs/models.yaml
DERIVED = 567_754_752      # con số suy diễn cần kiểm chứng


def count(filename: str) -> int:
    path = hf_hub_download(REPO, filename, revision=REVISION)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        return sum(p.numel() for p in sd.parameters())
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


def main():
    backbone = count("pytorch_model.bin")
    print(f"backbone (dense)      : {backbone:>13,}  = {backbone/1e6:.2f}M")

    extras = {}
    for name, fn in [("sparse", "sparse_linear.pt"), ("colbert", "colbert_linear.pt")]:
        try:
            extras[name] = count(fn)
            print(f"{name+' head':<22}: {extras[name]:>13,}")
        except Exception as e:
            print(f"{name+' head':<22}: không tải được ({type(e).__name__})")

    total = backbone + sum(extras.values())
    print(f"\nTổng cả 3 chế độ      : {total:>13,}  = {total/1e6:.2f}M")

    delta = backbone - DERIVED
    print(f"\nSuy diễn trong param_audit.md: {DERIVED:,}")
    print(f"Chênh lệch                    : {delta:+,}")
    if abs(delta) < 1_000_000:
        print("✅ Suy diễn ĐÚNG. Cập nhật params_verified: true trong configs/models.yaml.")
    else:
        print("🔴 Suy diễn SAI. Sửa docs/param_audit.md và configs/models.yaml theo số đếm thật,")
        print("   và kiểm tra lại giả định max_position_embeddings = 8194.")

    print(f"\nDùng dense-only, ngân sách khai: {backbone/1e6:.1f}M")
    print(f"Dùng cả 3 chế độ, khai         : {total/1e6:.1f}M")


if __name__ == "__main__":
    main()
