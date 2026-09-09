"""Xếp lại top-K của BM25 bằng cross-encoder — hai biến thể 1-chunk / n-chunk.

    # sàng lọc: 3 model trên dev-300
    python -m scripts.p4_rerank --model mminilm \
        --ranking outputs/p4_bm25_ranking/dev_rerank_pool.json \
        --questions data/dev_sub300.json \
        --out outputs/p4_rerank/devsub_mminilm_1chunk.json --rerank-top 50 --chunks 1

    # biến thể n-chunk (cần ranking sinh bằng --chunks-per-doc 3)
    python -m scripts.p4_rerank --model bge-m3 ... --chunks 3

VÌ SAO CÓ HAI BIẾN THỂ:
  1-chunk  chấm đúng một đoạn/văn bản — đoạn BM25 cho điểm cao nhất.
  n-chunk  chấm M đoạn rồi lấy max điểm reranker.
Cần vì 17/300 câu error_pool có ≥2 chunk cùng văn bản HOÀ ĐIỂM BM25 TUYỆT ĐỐI: BM25
tuyên bố nó không phân biệt được, nên ở 1-chunk thì đoạn đem đi rerank do quy ước phá
hoà chọn, không do độ liên quan. Chênh lệch giữa hai biến thể = CHI PHÍ CỦA LỐI TẮT,
đo được bằng số. Đúng loại phân tích BTC yêu cầu.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_chunk_texts(path: str, need: set[str]) -> dict[str, str]:
    """Đọc chunks.jsonl một lượt, chỉ giữ chunk_id cần. 576MB nên không nạp hết."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            i = line.find('"chunk_id"')
            if i < 0:
                continue
            d = json.loads(line)
            if d["chunk_id"] in need:
                out[d["chunk_id"]] = d["text"]
                if len(out) == len(need):
                    break
    missing = need - set(out)
    if missing:
        raise SystemExit(
            f"❌ Thiếu {len(missing)} chunk_id trong {path}, vd {sorted(missing)[:3]}. "
            "Ranking và chunks.jsonl không cùng một đợt sinh?"
        )
    return out


def recall_at(rank: dict, questions: dict, k: int) -> float:
    tot = 0.0
    for qid, v in questions.items():
        gold = {str(x) for x in v["answer"]}
        got = {r[0] for r in rank.get(str(qid), [])[:k]}
        tot += len(gold & got) / len(gold)
    return tot / len(questions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ranking", required=True, help="ranking BM25 đầu vào")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--chunks-file", default="data/chunks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerank-top", type=int, default=50,
                    help="xếp lại bao nhiêu doc đầu. Phần đuôi giữ nguyên thứ tự BM25.")
    ap.add_argument("--chunks", type=int, default=1,
                    help="số chunk/doc đem chấm. >1 cần ranking có --chunks-per-doc")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--prepend-name", action="store_true",
                    help="gắn tên văn bản vào đầu mỗi đoạn trước khi chấm. "
                         "configs ghi prepend_name: true nhưng chunks.jsonl thực tế "
                         "KHÔNG có (đo được 0,00%% trên 100.000 chunk đầu), nên "
                         "reranker đang chấm mẩu 180 từ không rõ thuộc văn bản nào. "
                         "1.100/8.507 văn bản (12,9%%) có name rỗng — những doc đó "
                         "không đổi gì.")
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    rank = json.load(open(a.ranking, encoding="utf-8"))
    missing = set(map(str, q)) - set(rank)
    if missing:
        raise SystemExit(f"❌ Ranking thiếu {len(missing)} câu hỏi, vd {sorted(missing)[:3]}")

    # gom chunk_id cần chấm
    def cids_of(row: list) -> list[str]:
        if a.chunks == 1:
            return [row[2]]
        if len(row) < 4:
            raise SystemExit(
                "❌ --chunks > 1 nhưng ranking chỉ có 3 phần tử/dòng. Sinh lại bằng "
                "p4_build_ranking.py --chunks-per-doc N."
            )
        return row[3][: a.chunks]

    need: set[str] = set()
    for qid in q:
        for row in rank[str(qid)][: a.rerank_top]:
            need.update(cids_of(row))
    print(f"Cần {len(need):,} chunk, đọc {a.chunks_file} ...", flush=True)
    t0 = time.time()
    texts = load_chunk_texts(a.chunks_file, need)
    print(f"  xong trong {time.time()-t0:.1f}s")

    names: dict[str, str] = {}
    if a.prepend_name:
        for line in open(a.corpus, encoding="utf-8"):
            d = json.loads(line)
            names[str(d["doc_id"])] = (d.get("name") or "").strip()
        n_empty = sum(1 for v in names.values() if not v)
        print(f"--prepend-name: {len(names)-n_empty:,}/{len(names):,} văn bản có tên; "
              f"{n_empty:,} rỗng nên giữ nguyên")

    def passage(doc_id: str, cid: str) -> str:
        t = texts[cid]
        if not a.prepend_name:
            return t
        n = names.get(doc_id, "")
        # tên là slug URL ("Thong-tu-17-2022-TT-BGTVT-..."): thay gạch bằng khoảng
        # trắng để tokenizer tách được từ. KHÔNG khôi phục dấu — làm thế cần từ điển
        # hoặc mô hình ngoài, tức dữ liệu ngoài, mà thể lệ cấm.
        return f"{n.replace('-', ' ')}\n{t}" if n else t

    # dựng danh sách cặp phẳng, nhớ đường về
    pairs: list[tuple[str, str]] = []
    back: list[tuple[str, int]] = []   # (qid, chỉ số doc trong top-rerank)
    for qid in q:
        question = q[qid]["question"]
        for di, row in enumerate(rank[str(qid)][: a.rerank_top]):
            for cid in cids_of(row):
                pairs.append((question, passage(row[0], cid)))
                back.append((str(qid), di))
    print(f"{len(pairs):,} cặp (câu hỏi, đoạn) — {a.chunks} chunk/doc, "
          f"top-{a.rerank_top}", flush=True)

    from src.rerank.cross_encoder import CrossEncoderReranker

    ce = CrossEncoderReranker(
        a.model, device=a.device, max_length=a.max_length,
        batch_size=a.batch_size, fp16=not a.no_fp16,
    )
    t0 = time.time()
    scores = ce.score(pairs)
    dt = time.time() - t0
    print(f"Chấm {len(pairs):,} cặp trong {dt:.1f}s ({1000*dt/len(pairs):.1f} ms/cặp)")

    # max-pool điểm reranker theo doc
    best: dict[str, dict[int, float]] = {str(x): {} for x in q}
    for (qid, di), s in zip(back, scores):
        cur = best[qid].get(di)
        if cur is None or s > cur:
            best[qid][di] = s

    out: dict[str, list] = {}
    for qid in q:
        head = rank[str(qid)][: a.rerank_top]
        tail = rank[str(qid)][a.rerank_top :]
        # hoà điểm reranker → giữ thứ tự BM25 (di tăng dần): sort ổn định theo (-score, di)
        order = sorted(range(len(head)), key=lambda i: (-best[qid][i], i))
        out[str(qid)] = [
            [head[i][0], float(best[qid][i]), *head[i][2:]] for i in order
        ] + tail

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ {a.out}")
    print(f"{'K':>4} {'BM25':>8} {'rerank':>8} {'Δ':>8}")
    for k in (1, 3, 5, 10, 20, a.rerank_top):
        b = recall_at(rank, q, k)
        r = recall_at(out, q, k)
        print(f"{k:>4} {b:>8.4f} {r:>8.4f} {r-b:>+8.4f}")
    print(f"\nTrần (R@{a.rerank_top} của BM25) = {recall_at(rank, q, a.rerank_top):.4f}"
          "  — reranker không thể vượt, chỉ tiến tới.")


if __name__ == "__main__":
    main()
