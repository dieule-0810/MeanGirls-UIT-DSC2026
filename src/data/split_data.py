"""
src/data/split_data.py - Bộ lọc bẫy "Vùng Chết", Chống rò rỉ Cụm trùng (Union-Find) và
                          Chia tách 4 đường + 3 mốc learning-curve (v6)

Công dụng:
    1. Đọc danh sách 11 qid "vùng chết" và 4 cụm trùng đã chốt từ docs/exclusion_decisions.json
    2. Loại bỏ hoàn toàn 11 câu hỏi "vùng chết" (unsolvable) khỏi train.json theo danh sách đã
       chốt ở bước 1. Còn lại 6.989 câu hợp lệ. (Có kèm 1 bước quét chéo corpus_clean.jsonl để
       cảnh báo nếu phát sinh doc rỗng MỚI ngoài danh sách đã biết - không dùng để tự động loại.)
    3. Chống rò rỉ cụm trùng bằng Union-Find: câu hỏi bắc cầu qua nhiều cụm -> gộp thành 1 siêu-cụm,
       đảm bảo luôn cùng 1 phía - áp dụng cho cả 4 tập.
    4. Sắp xếp và cố định seed = 42, chia 6.989 câu hợp lệ theo ĐÚNG 1 LẦN SHUFFLE, ưu tiên lấp đầy
       theo thứ tự holdout -> error_pool -> dev -> train_split. dev chen GIỮA error_pool và train_split 
       nên holdout + error_pool tái tạo GIỐNG HỆT v5 (bit-for-bit); chỉ train_split co lại.
       - data/holdout.json     : 1.000 câu - ĐO LẦN CUỐI, không đụng lúc thử nghiệm.
       - data/error_pool.json  :   300 câu - RIÊNG P4 đọc tay / phân tích lỗi.
       - data/dev.json         : 1.000 câu - ĐO LÚC THỬ NGHIỆM, dùng chung cả nhóm (INTERFACES §7).
       - data/train_split.json : 4.689 câu còn lại (giảm từ 5.689 vì 1.000 câu carve sang dev).
    5. Tạo 3 tập con LỒNG NHAU của train_split cho learning curve (plan.md mục 0.6), carve theo
       nhóm chống rò rỉ, seed 42: data/train_1000.json C train_2500.json C train_4689.json.
    6. Lưu vết câu hỏi bị loại vào <out_dir>/excluded_questions.md.
    7. PRE-WRITE VALIDATION: chạy đầy đủ trên dữ liệu TRONG BỘ NHỚ, trước khi ghi đĩa - gồm cả
       assert 6 cặp giao nhau = rỗng VÀ assert tổng 4 tập = 6.989.
       Fail-loud: raise ngay, không chỉ in warning. Không có trường hợp nào ghi file hỏng ra đĩa.
    8. POST-WRITE VALIDATOR: đọc ngược dữ liệu từ đĩa, chạy lại đúng bộ kiểm tra ở bước 7 (dùng
       chung hàm run_integrity_checks và check_nested_subsets). Nếu fail thì ROLLBACK (xoá cả 7 
       file + log vừa ghi) trước khi raise, không để lại dữ liệu hỏng trên đĩa.

Cách chạy:
    python -m src.data.split_data --corpus data/corpus_clean.jsonl --train data/train.json --out-dir data

Cấu trúc mã nguồn:
    ┌────────────────────────────────────────────────────────────────┐
    │  1. IMPORTS & CONSTANTS                                        │
    |     (EXPECTED_*, NESTED_SMALL_SIZES = [1000, 2500])            │
    ├────────────────────────────────────────────────────────────────┤
    │  2. UNION-FIND + build_anti_leak_groups                        │
    ├────────────────────────────────────────────────────────────────┤
    │  3. INTEGRITY CHECKS (N-way, dùng chung pre/post-write)        │
    │     ├─ _merge_dup_groups_via_data — tính lại cụm ĐỘC LẬP với   |
    │     |  bước gom nhóm                                           |
    │     ├─ run_integrity_checks (N tập RỜI nhau, cả pre/post-write)|
    │     │     • 1. không câu nào trỏ vào doc rỗng (vùng chết)      |
    │     │     • 2. mọi CẶP tập giao nhau = rỗng                    |
    │     │     • 3. tổng 4 tập = EXPECTED_TOTAL_VALID               |
    │     │     • 4. 1 cụm trùng chỉ nằm trong ĐÚNG 1 tập            |
    │     └─ check_nested_subsets (nested LỒNG nhau, kiểm RIÊNG)     |
    ├────────────────────────────────────────────────────────────────┤
    │  4. WRITERS + NESTED CARVER (save_excluded_log)                │
    │     ├─ save_excluded_log — ghi excluded_questions.md           │
    │     └─ carve_nested_by_group — 3 mốc nested, carve THEO nhóm   │
    │        (giữ anti-leak)                                         │
    ├────────────────────────────────────────────────────────────────┤
    │  5. CORE SPLITTER (split_data)                                 │
    │     ├── Nạp dup_groups + vung_chet_qids từ                     │
    │     │   exclusion_decisions.json                               │
    │     ├── Lọc vùng chết khỏi train.json -> Gom nhóm chống rò rỉ  │
    │     ├── 1 vòng shuffle (seed 42),thứ tự lấp: holdout           │
    │     │   -> error_pool -> dev -> train_split                    │
    │     ├── PRE-WRITE validate (in-memory) -> Ghi 7 file + log     │
    │     └── POST-WRITE validate (đọc lại đĩa) -> Rollback nếu fail │
    ├────────────────────────────────────────────────────────────────┤
    │  6. CLI ENTRYPOINT (main) - bắt exception -> sys.exit(1)       │
    └────────────────────────────────────────────────────────────────┘

Ghi chú (TODO):
    - dup_groups được nạp động từ docs/exclusion_decisions.json (do scripts/eda.py sinh ra) -
      không hard-code trong file này nữa. Chạy scripts/eda.py trước nếu file chưa tồn tại.
    - train_split.json co từ 5.689 xuống 4.689, mất 1.000 câu sang dev. Checkpoint đã fine-tune, 
      hoặc hard-negative đã mine, từ train_split cũ phải chạy LẠI - nếu không, dev không còn 
      "chưa từng thấy" với checkpoint đó và mọi lựa chọn siêu tham số trên dev sẽ lạc quan giả. 
      Cùng bài học đã ghi cho error_pool.
    - holdout.json và error_pool.json giữ nguyên bit-for-bit (dev chen sau) - điểm baseline cũ
      đo trên holdout vẫn so sánh được.
"""

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

# ===========================================================================
# 1. CONSTANTS
# ===========================================================================
EXPECTED_EXCLUDED = 11
EXPECTED_HOLDOUT = 1000
EXPECTED_DEV = 1000
EXPECTED_ERROR_POOL = 300
EXPECTED_TOTAL_VALID = None
EXPECTED_TRAIN_SPLIT = None
NESTED_SMALL_SIZES = [1000, 2500]


# ===========================================================================
# 2. UNION-FIND (chống rò rỉ cụm trùng bắc cầu qua nhiều cụm)
# ===========================================================================
class UnionFind:
    """Union-Find tối giản, đủ dùng cho vài chục cụm - không cần union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_anti_leak_groups(clean_train_data: dict, dup_groups: list) -> dict:
    """
    Gom các QID thành các nhóm sao cho MỌI câu hỏi liên quan đến cùng 1 (siêu) cụm trùng luôn nằm
    chung 1 nhóm - kể cả khi 1 câu hỏi bắc cầu qua 2+ cụm khác nhau (câu hỏi đa đáp án). Dùng
    Union-Find để gộp mọi cụm mà 1 câu hỏi chạm tới thành 1 siêu-cụm trước khi chia nhóm.
    """
    doc_to_cluster_idx = {}
    for idx, group in enumerate(dup_groups):
        for doc_id in group:
            doc_to_cluster_idx[str(doc_id)] = idx

    uf = UnionFind(len(dup_groups))

    qid_touched_clusters: dict = {}
    for qid, qdata in clean_train_data.items():
        answers = [str(a) for a in (qdata.get("answer") or [])]
        touched = {doc_to_cluster_idx[a] for a in answers if a in doc_to_cluster_idx}
        if touched:
            qid_touched_clusters[qid] = touched
            touched_list = list(touched)
            for i in range(1, len(touched_list)):
                uf.union(touched_list[0], touched_list[i])

    groups: dict = {}
    for qid in clean_train_data:
        touched = qid_touched_clusters.get(qid)
        if touched:
            root = uf.find(next(iter(touched)))
            key = f"cluster_{root}"
        else:
            key = qid
        groups.setdefault(key, []).append(qid)

    return groups


# ===========================================================================
# 3. INTEGRITY CHECKS - N-WAY, dùng chung cho cả PRE-WRITE và POST-WRITE
# ===========================================================================
def _merge_dup_groups_via_data(combined_data: dict, dup_groups: list) -> list:
    """
    Tính lại phần GỘP cụm (Union-Find) trực tiếp từ dữ liệu combined (tất cả các tập gộp lại),
    ĐỘC LẬP với build_anti_leak_groups - để validator không tin tưởng mù vào bước gom nhóm.
    """
    if not dup_groups:
        return []
    doc_to_cluster_idx = {}
    for idx, group in enumerate(dup_groups):
        for doc_id in group:
            doc_to_cluster_idx[str(doc_id)] = idx

    uf = UnionFind(len(dup_groups))
    for qdata in combined_data.values():
        answers = [str(a) for a in (qdata.get("answer") or [])]
        touched = {doc_to_cluster_idx[a] for a in answers if a in doc_to_cluster_idx}
        touched_list = list(touched)
        for i in range(1, len(touched_list)):
            uf.union(touched_list[0], touched_list[i])

    merged: dict = {}
    for idx, group in enumerate(dup_groups):
        root = uf.find(idx)
        merged.setdefault(root, set()).update(group)
    return list(merged.values())


def run_integrity_checks(splits: dict, empty_doc_ids: set, dup_groups: list) -> None:
    """
    Bộ kiểm tra lõi cho N tập (đặt tên qua dict, vd {"holdout": ..., "train_split": ...,
    "error_pool": ...}). Raise AssertionError NGAY khi phát hiện vi phạm - fail-loud, không chỉ
    in warning. Được gọi 2 lần (pre-write trên dữ liệu bộ nhớ, post-write trên dữ liệu đọc lại
    từ đĩa) để đóng vai trò round-trip check độc lập.
    """
    combined: dict = {}
    for data in splits.values():
        combined.update(data)

    # Check 1: không câu hỏi nào trỏ vào tài liệu rỗng (vùng chết)
    for qid, qdata in combined.items():
        answers = [str(a) for a in (qdata.get("answer") or [])]
        if any(a in empty_doc_ids for a in answers):
            raise AssertionError(f"QID {qid} dính bẫy vùng chết (trỏ vào tài liệu rỗng)!")

    # Check 2: MỌI CẶP tập không giao nhau QID (fail-loud, không chỉ warning)
    for (name_a, data_a), (name_b, data_b) in combinations(splits.items(), 2):
        overlap = set(data_a) & set(data_b)
        if overlap:
            raise AssertionError(
                f"Rò rỉ trực tiếp: {len(overlap)} QID xuất hiện ở CẢ '{name_a}' VÀ '{name_b}': {overlap}"
            )

    # Check 3: tổng số câu hỏi trên toàn bộ N tập phải khớp EXPECTED_TOTAL_VALID (fail-loud)
    total = len(combined)
    if total != EXPECTED_TOTAL_VALID:
        raise AssertionError(
            f"Tổng số câu hỏi trên {len(splits)} tập là {total}, không khớp kỳ vọng "
            f"{EXPECTED_TOTAL_VALID} (= 7000 câu gốc - {EXPECTED_EXCLUDED} câu vùng chết)!"
        )

    # Check 4: không rò rỉ cụm trùng - 1 (siêu) cụm chỉ được xuất hiện trong ĐÚNG 1 tập
    merged_groups = _merge_dup_groups_via_data(combined, dup_groups)
    docs_by_split = {
        name: {str(a) for qdata in data.values() for a in (qdata.get("answer") or [])}
        for name, data in splits.items()
    }
    for group in merged_groups:
        touched_splits = [name for name, docs in docs_by_split.items() if group & docs]
        if len(touched_splits) > 1:
            raise AssertionError(
                f"Rò rỉ cụm trùng! Cụm {group} bị xé lẻ giữa các tập: {touched_splits} - "
                f"sẽ làm số liệu đo trên các tập này bị thổi phồng/nhiễu giả tạo!"
            )


# ===========================================================================
# 4. EXCLUDED WRITER
# ===========================================================================
def save_excluded_log(excluded_list: list, log_path: Path) -> None:
    """Ghi nhận lịch sử các câu hỏi vùng chết bị loại bỏ để đảm bảo tính minh bạch."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Danh sách câu hỏi Vùng Chết bị loại bỏ (Excluded Questions)\n\n")
        f.write("> Tự động phát hiện và loại bỏ bởi `src/data/split_data.py`.\n")
        f.write(f"> Tổng số câu hỏi bị loại: **{len(excluded_list)} câu**.\n\n")
        f.write("## Chi tiết các câu hỏi bị loại:\n\n")
        for idx, item in enumerate(excluded_list):
            f.write(f"### {idx + 1}. QID: {item['qid']}\n")
            f.write(f"- **Câu hỏi:** {item['question']}\n")
            f.write(f"- **Đáp án lỗi (Gold ID):** {item['answer']}\n\n")


def carve_nested_by_group(groups: dict, train_split_qids: list, small_sizes: list, seed: int = 42) -> dict:
    """
    Tập con LỒNG NHAU của train_split cho learning curve (plan.md mục 0.6):
    train_{s0} C train_{s1} C ... C train_split. Carve THEO GROUP chống rò rỉ (giữ anti-leak cả
    trong nested) - điểm cộng từ carve_exact_nested (bản nháp P2). Đảm bảo lồng nhau bằng cách
    carve mốc nhỏ TỪ TRONG mốc lớn (pool thu hẹp dần).
    """
    train_set = set(train_split_qids)
    train_groups = {k: v for k, v in groups.items() if v[0] in train_set}  # group nguyên khối -> xét 1 qid đủ
    keys = sorted(train_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    full_n = len(train_set)

    def _carve(pool_keys: list, target: int) -> list:
        chosen, cur = [], 0
        for k in pool_keys:                       # pass 1: greedy theo nhóm, không vượt target
            g = train_groups[k]
            if cur + len(g) <= target:
                chosen.append(k); cur += len(g)
        if cur < target:                          # pass 2: top-up khít bằng nhóm size 1
            chosen_set = set(chosen)
            for k in pool_keys:
                if k in chosen_set:
                    continue
                if len(train_groups[k]) == 1 and cur + 1 <= target:
                    chosen.append(k); cur += 1
        if cur != target:
            raise ValueError(f"Không carve chính xác train_{target} theo nhóm (thiếu nhóm size-1 để lấp khít).")
        return chosen

    targets = sorted((s for s in small_sizes if s < full_n), reverse=True)
    subset_keys = {full_n: keys}
    pool = keys
    for t in targets:
        pool = _carve(pool, t)
        subset_keys[t] = pool
    return {n: sorted(q for k in ks for q in train_groups[k]) for n, ks in subset_keys.items()}


def check_nested_subsets(subsets: dict, train_split_json: dict) -> None:
    """Bất biến riêng cho nested (fail-loud): đúng kích thước, LỒNG NHAU, mốc lớn nhất trùng khít
    train_split. Không đưa nested vào run_integrity_checks vì chúng CỐ Ý giao nhau (lồng)."""
    sizes = sorted(subsets.keys())
    for n in sizes:
        if len(subsets[n]) != n:
            raise AssertionError(f"train_{n}: có {len(subsets[n])} câu, kỳ vọng {n}.")
    for smaller, larger in zip(sizes, sizes[1:]):
        if not set(subsets[smaller]) <= set(subsets[larger]):
            missing = set(subsets[smaller]) - set(subsets[larger])
            raise AssertionError(f"train_{smaller} KHÔNG phải tập con của train_{larger} ({len(missing)} qid lọt ra ngoài).")
    if set(subsets[sizes[-1]]) != set(train_split_json):
        raise AssertionError(f"train_{sizes[-1]} không trùng khít train_split ({len(subsets[sizes[-1]])} vs {len(train_split_json)}).")

# ===========================================================================
# 5. CORE SPLITTER
# ===========================================================================
def split_data(corpus_path: Path, train_path: Path, out_dir: Path) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file corpus sạch tại: {corpus_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file train.json gốc tại: {train_path}")

    decisions_path = Path("docs/exclusion_decisions.json")
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {decisions_path}. Chạy `python scripts/eda.py` trước để sinh file quyết định loại trừ."
        )
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    dup_groups = [set(str(i) for i in group) for group in decisions["dup_groups"]]
    print(f"Đã tải {len(dup_groups)} cụm trùng từ: {decisions_path}")

    # Qid vùng chết (docs/exclusion_decisions.json)
    vung_chet_qids = set(decisions["qids_exclude_vung_chet"])

    print(f"Đang đối chiếu tập câu hỏi train.json với {len(vung_chet_qids)} qid vùng chết đã chốt...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    global EXPECTED_TOTAL_VALID, EXPECTED_TRAIN_SPLIT
    EXPECTED_TOTAL_VALID = len(train_data) - len(vung_chet_qids)
    EXPECTED_TRAIN_SPLIT = EXPECTED_TOTAL_VALID - EXPECTED_HOLDOUT - EXPECTED_ERROR_POOL - EXPECTED_DEV

    clean_train_data, excluded_questions = {}, []
    for qid, qdata in train_data.items():
        if qid in vung_chet_qids:
            answers_str = [str(a) for a in (qdata.get("answer") or [])]
            excluded_questions.append({"qid": qid, "question": qdata.get("question", ""), "answer": answers_str})
        else:
            clean_train_data[qid] = qdata

    # Cảnh báo chéo: nếu corpus vẫn còn dòng "text" rỗng khác vung_chet_qids,
    # đó là dấu hiệu dữ liệu mới phát sinh chưa được đưa vào eda_notes.md - không tự động loại, chỉ báo.
    empty_doc_ids_now = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if not doc.get("text", "").strip():
                empty_doc_ids_now.add(str(doc["doc_id"]))
    if empty_doc_ids_now:
        print(f"  CẢNH BÁO: corpus vẫn còn {len(empty_doc_ids_now)} doc rỗng chưa nằm trong danh sách loại - kiểm tra lại eda_notes.md.")
    print(f"Phát hiện và loại bỏ thành công {len(excluded_questions)} câu hỏi Vùng Chết!")
    print(f"Còn lại {len(clean_train_data)} câu hợp lệ để chia 4 đường "
          f"({EXPECTED_HOLDOUT}/{EXPECTED_ERROR_POOL}/{EXPECTED_TRAIN_SPLIT}).")

    print("Đang gom nhóm các câu hỏi cùng cụm trùng lặp (Union-Find) để chống rò rỉ...")
    groups = build_anti_leak_groups(clean_train_data, dup_groups)
    print(f"  - Tổng số nhóm sau khi gom cụm chống rò rỉ: {len(groups)} nhóm.")

    sorted_group_keys = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(sorted_group_keys)

    # MỘT vòng lặp duy nhất, cùng thứ tự shuffle như bản trước (seed 42) - ưu tiên lấp đầy
    # holdout TRƯỚC (y hệt logic cũ -> holdout.json tái tạo giống hệt bản đã commit), rồi mới
    # đến error_pool, rồi dev (mới, chen giữa), phần còn lại là train_split.
    holdout_qids, error_pool_qids, dev_qids, train_split_qids = [], [], [], []
    for key in sorted_group_keys:
        group_qids = groups[key]
        if len(holdout_qids) + len(group_qids) <= EXPECTED_HOLDOUT:
            holdout_qids.extend(group_qids)
        elif len(error_pool_qids) + len(group_qids) <= EXPECTED_ERROR_POOL:
            error_pool_qids.extend(group_qids)
        elif len(dev_qids) + len(group_qids) <= EXPECTED_DEV:
            dev_qids.extend(group_qids)
        else:
            train_split_qids.extend(group_qids)
    holdout_qids.sort(); error_pool_qids.sort(); dev_qids.sort(); train_split_qids.sort()

    holdout_json = {qid: clean_train_data[qid] for qid in holdout_qids}
    error_pool_json = {qid: clean_train_data[qid] for qid in error_pool_qids}
    dev_json = {qid: clean_train_data[qid] for qid in dev_qids}
    train_split_json = {qid: clean_train_data[qid] for qid in train_split_qids}

    splits = {"holdout": holdout_json, "dev": dev_json,
              "error_pool": error_pool_json, "train_split": train_split_json}

    nested_qids = carve_nested_by_group(groups, train_split_qids, NESTED_SMALL_SIZES, seed=42)
    nested = {n: {qid: clean_train_data[qid] for qid in qids} for n, qids in nested_qids.items()}
    nested_sizes = sorted(nested.keys())

    # -----------------------------------------------------------------
    # PRE-WRITE VALIDATION (in-memory, đầy đủ) - KHÔNG ghi file nếu fail
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" PRE-WRITE VALIDATION (kiểm tra trong bộ nhớ, trước khi ghi đĩa)...")
    try:
        size_ok = (
            len(excluded_questions) == EXPECTED_EXCLUDED
            and len(holdout_json) == EXPECTED_HOLDOUT
            and len(error_pool_json) == EXPECTED_ERROR_POOL
            and len(train_split_json) == EXPECTED_TRAIN_SPLIT
            and len(dev_json) == EXPECTED_DEV
        )
        if not size_ok:
            raise AssertionError(
                f"Sai số lượng - loại trừ: {len(excluded_questions)} (kỳ vọng {EXPECTED_EXCLUDED}), "
                f"holdout: {len(holdout_json)} (kỳ vọng {EXPECTED_HOLDOUT}), "
                f"error_pool: {len(error_pool_json)} (kỳ vọng {EXPECTED_ERROR_POOL}), "
                f"train_split: {len(train_split_json)} (kỳ vọng {EXPECTED_TRAIN_SPLIT})"
            )
        run_integrity_checks(splits, empty_doc_ids_now, dup_groups)
        check_nested_subsets(nested, train_split_json)
    except AssertionError as e:
        print(f"{e}")
        print("  => ĐÃ CHẶN ĐỨNG việc ghi tệp hỏng ra đĩa để tránh lỗi im lặng (silent failure).")
        print("=" * 80)
        raise RuntimeError("Quy trình kiểm định trước khi ghi đĩa thất bại! Dữ liệu không đạt mốc kỳ vọng.")
    print(f"  - Số lượng, vùng chết, giao QID (6 cặp), rò rỉ cụm trùng, tổng = {EXPECTED_TOTAL_VALID}: Tất cả đạt.")
    print("=" * 80)

    # -----------------------------------------------------------------
    # GHI ĐĨA (chỉ chạy tới đây khi pre-write validation đã pass)
    # -----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_file = out_dir / "holdout.json"
    error_pool_file = out_dir / "error_pool.json"
    dev_file = out_dir / "dev.json"
    train_split_file = out_dir / "train_split.json"
    log_path = out_dir / "excluded_questions.md"
    nested_files = {n: out_dir / f"train_{n}.json" for n in nested_sizes}

    def _dump(obj, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)

    _dump(holdout_json, holdout_file)
    _dump(error_pool_json, error_pool_file)
    _dump(dev_json, dev_file)
    _dump(train_split_json, train_split_file)
    for n, path in nested_files.items():
        _dump(nested[n], path)
    save_excluded_log(excluded_questions, log_path)
    print(f"Đã ghi nhật ký loại bỏ vào: {log_path}")

    all_written = [holdout_file, error_pool_file, dev_file, train_split_file, log_path, *nested_files.values()]

    # -----------------------------------------------------------------
    # POST-WRITE VALIDATOR (đọc ngược từ đĩa) - rollback TOÀN BỘ nếu fail
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ĐANG CHẠY BỘ ĐỐI SOÁT ĐỘC LẬP (POST-WRITE INTEGRITY CHECK)...")
    try:
        reread = {name: json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
                  for name in ("holdout", "dev", "error_pool", "train_split")}
        nested_reread = {n: json.loads(path.read_text(encoding="utf-8")) for n, path in nested_files.items()}
        if (len(reread["holdout"]) != EXPECTED_HOLDOUT or len(reread["error_pool"]) != EXPECTED_ERROR_POOL
                or len(reread["dev"]) != EXPECTED_DEV or len(reread["train_split"]) != EXPECTED_TRAIN_SPLIT):
            raise AssertionError(
                f"Lệch số lượng sau khi đọc lại - holdout: {len(reread['holdout'])}, "
                f"dev: {len(reread['dev'])}, error_pool: {len(reread['error_pool'])}, "
                f"train_split: {len(reread['train_split'])}"
            )
        run_integrity_checks(reread, empty_doc_ids_now, dup_groups)
        check_nested_subsets(nested_reread, reread["train_split"])
    except AssertionError as e:
        print(f"{e}")
        print("  => ROLLBACK: xoá các file vừa ghi để không để lại dữ liệu hỏng trên đĩa...")
        for f in all_written:
            f.unlink(missing_ok=True)
        print("=" * 80)
        raise RuntimeError("Đối soát độc lập sau khi ghi thất bại. Đã rollback, không có file nào sót lại.")

    print("  - Đọc lại từ đĩa, đối soát toàn bộ điều kiện (gồm nested): Hoàn hảo 100% (CODA-READY)!")
    print("=" * 80)

    # -----------------------------------------------------------------
    # BÁO CÁO NGHIỆM THU
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BÁO CÁO NGHIỆM THU CHIA TÁCH DỮ LIỆU (P2):")
    print("=" * 80)
    print(f"  - train.json gốc: {len(train_data)} | Vùng chết loại: {len(excluded_questions)} "
          f"(kỳ vọng {EXPECTED_EXCLUDED}) | Hợp lệ: {len(clean_train_data)}")
    print(f"  - holdout ({holdout_file.name}): {len(holdout_json)}/{EXPECTED_HOLDOUT} - ĐO LẦN CUỐI")
    print(f"  - dev ({dev_file.name}): {len(dev_json)}/{EXPECTED_DEV} - ĐO LÚC THỬ NGHIỆM, dùng chung nhóm")
    print(f"  - error_pool ({error_pool_file.name}): {len(error_pool_json)}/{EXPECTED_ERROR_POOL} - P4 đọc tay")
    print(f"  - train_split ({train_split_file.name}): {len(train_split_json)}/{EXPECTED_TRAIN_SPLIT} - fine-tune")
    for n in nested_sizes:
        print(f"      └─ learning-curve {nested_files[n].name}: {len(nested[n])} câu")
    print("=" * 80)
    print(" train_split CO LẠI so với v5 (mất 1.000 câu sang dev). Checkpoint/hard-negative đã")
    print("    mine từ train_split cũ phải chạy LẠI - dev không còn 'chưa từng thấy' với chúng.")
    print("=" * 80)
    print("KẾT LUẬN: CHIA 4 ĐƯỜNG + 3 MỐC LEARNING-CURVE SẠCH (CODA-READY)!")
    print("=" * 80)

# ===========================================================================
# 6. CLI ENTRYPOINT
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Bộ lọc bẫy Vùng Chết và chia 4 đường dữ liệu cho P2")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus_clean.jsonl"),
                     help="Đường dẫn file corpus sạch")
    ap.add_argument("--train", type=Path, default=Path("data/train.json"),
                     help="Đường dẫn file train.json thô gốc")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                     help="Thư mục ghi kết quả")
    args = ap.parse_args()

    try:
        split_data(args.corpus, args.train, args.out_dir)
    except Exception as e:
        print(f"\nLỖI HỆ THỐNG: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
