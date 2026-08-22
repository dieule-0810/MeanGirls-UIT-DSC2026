"""
Đếm số tham số thật + lấy revision hash của các model ứng viên.
Chỉ tải header safetensors (vài KB), KHÔNG tải trọng số.

    pip install huggingface_hub
    python count_params.py

Output: bảng markdown dán thẳng vào P4_TASKS.md + config pin revision.
"""

from huggingface_hub import HfApi, get_safetensors_metadata

RERANKER = [
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Reranker-0.6B",
    "AITeamVN/Vietnamese_Reranker",
    "namdp-ptit/ViRanker",
    # ứng viên nhỏ đề xuất đăng ký thêm (ô ablation "reranker nhỏ vs lớn"):
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "jinaai/jina-reranker-v2-base-multilingual",
]

BIENCODER = [
    "BAAI/bge-m3",
    "AITeamVN/Vietnamese_Embedding",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "Alibaba-NLP/gte-multilingual-base",
    "bkai-foundation-models/vietnamese-bi-encoder",
]

api = HfApi()


def probe(repo_id: str) -> dict:
    row = {"model": repo_id, "params": None, "dtypes": "", "sha": "", "license": "", "err": ""}
    try:
        info = api.model_info(repo_id)
        row["sha"] = (info.sha or "")[:12]
        row["license"] = (info.card_data or {}).get("license", "?") if info.card_data else "?"
    except Exception as e:
        row["err"] = f"info: {type(e).__name__}"
    try:
        meta = get_safetensors_metadata(repo_id)
        # parameter_count: dict dtype -> count
        row["params"] = sum(meta.parameter_count.values())
        row["dtypes"] = ",".join(sorted(meta.parameter_count))
    except Exception as e:
        row["err"] += f" | safetensors: {type(e).__name__}"
    return row


def render(title: str, repos: list) -> list:
    print(f"\n### {title}\n")
    print("| Model | Tham số | dtype | revision | license | lỗi |")
    print("|---|---:|---|---|---|---|")
    rows = []
    for r in repos:
        row = probe(r)
        rows.append(row)
        p = f"{row['params']/1e6:,.1f}M" if row["params"] else "—"
        print(f"| `{row['model']}` | {p} | {row['dtypes']} | `{row['sha']}` | {row['license']} | {row['err']} |")
    return rows


if __name__ == "__main__":
    a = render("Cross-encoder / reranker", RERANKER)
    b = render("Bi-encoder / embedding", BIENCODER)

    print("\n### Tổ hợp pipeline (ngân sách cứng 4.000M)\n")
    print("| Bi-encoder | Reranker | Tổng | Trạng thái |")
    print("|---|---|---:|---|")
    for be in b:
        for re_ in a:
            if not (be["params"] and re_["params"]):
                continue
            tot = be["params"] + re_["params"]
            flag = "OK" if tot < 4_000_000_000 else "VƯỢT"
            print(f"| {be['model'].split('/')[-1]} | {re_['model'].split('/')[-1]} "
                  f"| {tot/1e6:,.0f}M | {flag} |")

    print("\n### Dán vào configs/models.yaml (pin revision — mục 7 bảng rủi ro)\n")
    print("models:")
    for row in a + b:
        if row["sha"]:
            print(f"  - repo: {row['model']}\n    revision: {row['sha']}")
