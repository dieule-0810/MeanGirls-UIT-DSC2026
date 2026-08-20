
### Cross-encoder / reranker

| Model | Tham số | dtype | revision | license | lỗi |
|---|---:|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 567.8M | F32 | `953dc6f6f85a` | apache-2.0 |  |
| `Qwen/Qwen3-Reranker-0.6B` | 595.8M | BF16 | `e61197ed4502` | apache-2.0 |  |
| `AITeamVN/Vietnamese_Reranker` | 567.8M | F32 | `f53697624840` | apache-2.0 |  |
| `namdp-ptit/ViRanker` | 567.8M | F32 | `922bd0d698b1` | apache-2.0 |  |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 117.6M | F32,I64 | `1427fd652930` | apache-2.0 |  |
| `jinaai/jina-reranker-v2-base-multilingual` | 278.4M | BF16 | `9cfeff2df7d4` | cc-by-nc-4.0 |  |

### Bi-encoder / embedding

| Model | Tham số | dtype | revision | license | lỗi |
|---|---:|---|---|---|---|
| `BAAI/bge-m3` | — |  | `5617a9f61b02` | mit |  | safetensors: NotASafetensorsRepoError |
| `AITeamVN/Vietnamese_Embedding` | 567.8M | F32 | `dea33aa1ab33` | apache-2.0 |  |
| `Qwen/Qwen3-Embedding-0.6B` | 595.8M | BF16 | `97b0c614be4d` | apache-2.0 |  |
| `intfloat/multilingual-e5-base` | 278.0M | F32,I64 | `d12875059715` | mit |  |
| `intfloat/multilingual-e5-large` | 559.9M | F32,I64 | `3d7cfbdacd47` | mit |  |
| `Alibaba-NLP/gte-multilingual-base` | 305.4M | F16 | `9bbca17d9273` | apache-2.0 |  |
| `bkai-foundation-models/vietnamese-bi-encoder` | 135.0M | F32 | `84f9d9ada0d1` | apache-2.0 |  |

### Tổ hợp pipeline (ngân sách cứng 4.000M)

| Bi-encoder | Reranker | Tổng | Trạng thái |
|---|---|---:|---|
| Vietnamese_Embedding | bge-reranker-v2-m3 | 1,136M | OK |
| Vietnamese_Embedding | Qwen3-Reranker-0.6B | 1,164M | OK |
| Vietnamese_Embedding | Vietnamese_Reranker | 1,136M | OK |
| Vietnamese_Embedding | ViRanker | 1,136M | OK |
| Vietnamese_Embedding | mmarco-mMiniLMv2-L12-H384-v1 | 685M | OK |
| Vietnamese_Embedding | jina-reranker-v2-base-multilingual | 846M | OK |
| Qwen3-Embedding-0.6B | bge-reranker-v2-m3 | 1,164M | OK |
| Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B | 1,192M | OK |
| Qwen3-Embedding-0.6B | Vietnamese_Reranker | 1,164M | OK |
| Qwen3-Embedding-0.6B | ViRanker | 1,164M | OK |
| Qwen3-Embedding-0.6B | mmarco-mMiniLMv2-L12-H384-v1 | 713M | OK |
| Qwen3-Embedding-0.6B | jina-reranker-v2-base-multilingual | 874M | OK |
| multilingual-e5-base | bge-reranker-v2-m3 | 846M | OK |
| multilingual-e5-base | Qwen3-Reranker-0.6B | 874M | OK |
| multilingual-e5-base | Vietnamese_Reranker | 846M | OK |
| multilingual-e5-base | ViRanker | 846M | OK |
| multilingual-e5-base | mmarco-mMiniLMv2-L12-H384-v1 | 396M | OK |
| multilingual-e5-base | jina-reranker-v2-base-multilingual | 556M | OK |
| multilingual-e5-large | bge-reranker-v2-m3 | 1,128M | OK |
| multilingual-e5-large | Qwen3-Reranker-0.6B | 1,156M | OK |
| multilingual-e5-large | Vietnamese_Reranker | 1,128M | OK |
| multilingual-e5-large | ViRanker | 1,128M | OK |
| multilingual-e5-large | mmarco-mMiniLMv2-L12-H384-v1 | 678M | OK |
| multilingual-e5-large | jina-reranker-v2-base-multilingual | 838M | OK |
| gte-multilingual-base | bge-reranker-v2-m3 | 873M | OK |
| gte-multilingual-base | Qwen3-Reranker-0.6B | 901M | OK |
| gte-multilingual-base | Vietnamese_Reranker | 873M | OK |
| gte-multilingual-base | ViRanker | 873M | OK |
| gte-multilingual-base | mmarco-mMiniLMv2-L12-H384-v1 | 423M | OK |
| gte-multilingual-base | jina-reranker-v2-base-multilingual | 584M | OK |
| vietnamese-bi-encoder | bge-reranker-v2-m3 | 703M | OK |
| vietnamese-bi-encoder | Qwen3-Reranker-0.6B | 731M | OK |
| vietnamese-bi-encoder | Vietnamese_Reranker | 703M | OK |
| vietnamese-bi-encoder | ViRanker | 703M | OK |
| vietnamese-bi-encoder | mmarco-mMiniLMv2-L12-H384-v1 | 253M | OK |
| vietnamese-bi-encoder | jina-reranker-v2-base-multilingual | 413M | OK |

### Dán vào configs/models.yaml (pin revision — mục 7 bảng rủi ro)

models:
  - repo: BAAI/bge-reranker-v2-m3
    revision: 953dc6f6f85a
  - repo: Qwen/Qwen3-Reranker-0.6B
    revision: e61197ed4502
  - repo: AITeamVN/Vietnamese_Reranker
    revision: f53697624840
  - repo: namdp-ptit/ViRanker
    revision: 922bd0d698b1
  - repo: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    revision: 1427fd652930
  - repo: jinaai/jina-reranker-v2-base-multilingual
    revision: 9cfeff2df7d4
  - repo: BAAI/bge-m3
    revision: 5617a9f61b02
  - repo: AITeamVN/Vietnamese_Embedding
    revision: dea33aa1ab33
  - repo: Qwen/Qwen3-Embedding-0.6B
    revision: 97b0c614be4d
  - repo: intfloat/multilingual-e5-base
    revision: d12875059715
  - repo: intfloat/multilingual-e5-large
    revision: 3d7cfbdacd47
  - repo: Alibaba-NLP/gte-multilingual-base
    revision: 9bbca17d9273
  - repo: bkai-foundation-models/vietnamese-bi-encoder
    revision: 84f9d9ada0d1
