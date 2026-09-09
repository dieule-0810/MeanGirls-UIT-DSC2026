"""Phép thử chẩn đoán: reranker có HOẠT ĐỘNG không, trước khi hỏi nó có TỐT không.

Ba model đều làm tệ đi R@5 — trước khi kết luận "cross-encoder zero-shot không hợp
tiếng Việt pháp lý", phải loại trừ khả năng code/số học hỏng. Script này tách bạch:

  1. THỐNG KÊ ĐIỂM — NaN/inf làm sort ra thứ tự rác mà không báo lỗi. fp16 với
     XLM-R-large là chỗ hay tràn số. Nếu phương sai ~0 thì model trả điểm hằng số
     và mọi thứ hạng là ngẫu nhiên.
  2. PHÂN BIỆT GOLD vs NGẪU NHIÊN — chấm chunk của văn bản gold so với chunk của một
     văn bản lấy bừa trong corpus. Model lành mạnh phải thắng >90%. Quanh 50% nghĩa
     là hỏng (sai cột logits, đảo thứ tự cặp, tràn số), KHÔNG phải "model yếu".
  3. CÓ TIÊU ĐỀ vs KHÔNG — đo trực tiếp xem việc thiếu tên văn bản có phải nguyên
     nhân không. configs ghi prepend_name: true nhưng 0,00% chunk thật sự có tiêu đề.

    python -m scripts.p4_rerank_probe --model bge-m3 --n 60
"""
from __future__ import annotations

import argparse
import json
import random
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default="data/dev_sub300.json")
    ap.add_argument("--ranking", default="outputs/p4_bm25_ranking/devsub_pool.json")
    ap.add_argument("--chunks-file", default="data/chunks.jsonl")
    ap.add_argument("--corpus", default="data/corpus_clean.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    q = json.load(open(a.questions, encoding="utf-8"))
    rank = json.load(open(a.ranking, encoding="utf-8"))
    names = {}
    for line in open(a.corpus, encoding="utf-8"):
        d = json.loads(line)
        names[str(d["doc_id"])] = (d.get("name") or "").strip()

    rng = random.Random(a.seed)
    qids = sorted(q)
    rng.shuffle(qids)

    # chỉ giữ câu có gold nằm trong top-50 — nếu không thì không có chunk gold để chấm
    picked = []
    for qid in qids:
        gold = {str(x) for x in q[qid]["answer"]}
        hit = [r for r in rank[qid][:50] if r[0] in gold]
        other = [r for r in rank[qid][:50] if r[0] not in gold]
        if hit and other:
            picked.append((qid, hit[0], rng.choice(other)))
        if len(picked) == a.n:
            break
    print(f"{len(picked)} câu có cả chunk gold lẫn chunk không-gold trong top-50")

    need = {r[2] for _, g, o in picked for r in (g, o)}
    texts = {}
    for line in open(a.chunks_file, encoding="utf-8"):
        d = json.loads(line)
        if d["chunk_id"] in need:
            texts[d["chunk_id"]] = d["text"]
            if len(texts) == len(need):
                break

    def with_name(doc_id: str, txt: str) -> str:
        n = names.get(doc_id, "")
        return f"{n.replace('-', ' ')}\n{txt}" if n else txt

    plain, titled = [], []
    for qid, g, o in picked:
        question = q[qid]["question"]
        for row in (g, o):
            plain.append((question, texts[row[2]]))
            titled.append((question, with_name(row[0], texts[row[2]])))

    from src.rerank.cross_encoder import CrossEncoderReranker

    ce = CrossEncoderReranker(a.model, batch_size=a.batch_size)

    for label, pairs in (("KHÔNG tiêu đề", plain), ("CÓ tiêu đề", titled)):
        sc = ce.score(pairs, quiet=True)
        bad = [x for x in sc if x != x or abs(x) == float("inf")]
        gold_s = sc[0::2]
        other_s = sc[1::2]
        win = sum(1 for g, o in zip(gold_s, other_s) if g > o)
        print(f"\n--- {label} ---")
        print(f"  điểm: min {min(sc):+.3f}  max {max(sc):+.3f}  "
              f"trung bình {statistics.mean(sc):+.3f}  độ lệch chuẩn {statistics.pstdev(sc):.3f}")
        if bad:
            print(f"  ❌ {len(bad)} điểm NaN/inf — fp16 tràn số. Chạy lại với --no-fp16.")
        if statistics.pstdev(sc) < 1e-6:
            print("  ❌ Điểm gần như hằng số — model không phân biệt gì. Kiểm cột logits.")
        print(f"  gold > không-gold: {win}/{len(gold_s)} = {win/len(gold_s):.1%}")
        print(f"  điểm gold trung bình {statistics.mean(gold_s):+.3f}  "
              f"vs không-gold {statistics.mean(other_s):+.3f}")

    print("""
ĐỌC KẾT QUẢ:
  ~50%  → HỎNG (sai cột logits / đảo cặp / tràn số), không phải model yếu.
  >90%  → model lành mạnh; việc nó thua BM25 là kết quả THẬT, phải giải thích
          bằng bài toán chứ không bằng bug.
  CÓ tiêu đề hơn hẳn KHÔNG tiêu đề → thiếu tên văn bản là nguyên nhân chính.""")


if __name__ == "__main__":
    main()
