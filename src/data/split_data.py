"""
src/data/split_data.py - Bộ lọc bẫy "Vùng Chết", Chống rò rỉ Cụm trùng (Union-Find) và
                          Chia tách 3 đường: train_split / holdout / error_pool (v5)

Công dụng:
    1. Đọc danh sách 11 qid "vùng chết" và 4 cụm trùng đã chốt từ docs/exclusion_decisions.json
       đã bị parse_corpus.py loại hẳn khỏi corpus, không còn dấu vết để quét động.
    2. Loại bỏ hoàn toàn 11 câu hỏi "vùng chết" (unsolvable) khỏi train.json theo danh sách đã
       chốt ở bước 1. Còn lại 6.989 câu hợp lệ. (Có kèm 1 bước quét chéo corpus_clean.jsonl để
       cảnh báo nếu phát sinh doc rỗng MỚI ngoài danh sách đã biết - không dùng để tự động loại.)
    3. Chống rò rỉ cụm trùng bằng Union-Find: nếu 1 câu hỏi có đáp án bắc cầu qua 2 cụm trùng khác
       nhau, 2 cụm đó được GỘP thành 1 siêu-cụm để đảm bảo luôn nằm cùng 1 phía - áp dụng cho CẢ 3
       tập, không chỉ 2 như bản trước.
    4. Sắp xếp và cố định seed = 42, chia 6.989 câu hợp lệ theo ĐÚNG 1 LẦN SHUFFLE, ưu tiên lấp đầy
       theo thứ tự holdout -> error_pool -> train_split (xem mục "QUAN TRỌNG" bên dưới):
       - data/holdout.json     : 1.000 câu kiểm thử nội bộ siêu sạch, chống rò rỉ cụm trùng.
       - data/error_pool.json  :   300 câu dành RIÊNG cho P4 đọc bằng mắt / phân tích lỗi thủ công.
                                  RỜI HOÀN TOÀN với train_split (mô hình fine-tune từ Tuần 3 không
                                  được thấy các câu này) và với holdout.
       - data/train_split.json: 5.689 câu còn lại, dùng để fine-tune (giảm từ 5.989 vì 300 câu đã
                                  bị carve sang error_pool).
    5. Lưu lại vết danh sách câu hỏi bị loại vào <out_dir>/excluded_questions.md.
    6. PRE-WRITE VALIDATION: chạy đầy đủ trên dữ liệu TRONG BỘ NHỚ, trước khi ghi đĩa - gồm cả
       assert 3 cặp giao nhau = rỗng VÀ assert tổng 3 tập = 6.989.
       Fail-loud: raise ngay, không chỉ in warning. Không có trường hợp nào ghi file hỏng ra đĩa.
    7. POST-WRITE VALIDATOR: đọc ngược dữ liệu từ đĩa, chạy lại đúng bộ kiểm tra ở bước 6 (dùng
       chung 1 hàm run_integrity_checks). Nếu fail thì ROLLBACK (xoá cả 4 file vừa ghi) trước khi
       raise, không để lại dữ liệu hỏng trên đĩa.

Cách chạy:
    python -m src.data.split_data --corpus data/corpus_clean.jsonl --train data/train.json --out-dir data

Cấu trúc mã nguồn:
    ┌────────────────────────────────────────────────────────────────┐
    │  1. IMPORTS & CONSTANTS                                        │
    ├────────────────────────────────────────────────────────────────┤
    │  2. UNION-FIND + build_anti_leak_groups                        │
    ├────────────────────────────────────────────────────────────────┤
    │  3. run_integrity_checks (N-way, dùng chung pre/post-write)    │
    │     ├── Check vùng chết (toàn bộ các tập gộp lại)              │
    │     ├── Check giao nhau từng CẶP tập (C(n,2) cặp)              │
    │     └── Check rò rỉ cụm trùng (1 cụm chỉ được ở 1 tập)         │
    ├────────────────────────────────────────────────────────────────┤
    │  4. EXCLUDED WRITER (save_excluded_log)                        │
    ├────────────────────────────────────────────────────────────────┤
    │  5. CORE SPLITTER (split_data)                                 │
    │     ├── Nạp dup_groups + vung_chet_qids từ                     │
    │     │   exclusion_decisions.json (không tự dò từ corpus nữa)   │
    │     ├── Lọc vùng chết khỏi train.json -> Gom nhóm chống rò rỉ  │
    │     ├── 1 vòng shuffle (seed 42), ưu tiên holdout->error_pool  │
    │     │   ->train_split -> giữ holdout y hệt bản trước           │
    │     ├── PRE-WRITE validate (in-memory) -> Ghi đĩa (4 file)     │
    │     └── POST-WRITE validate (đọc lại đĩa) -> Rollback nếu fail │
    ├────────────────────────────────────────────────────────────────┤
    │  6. CLI ENTRYPOINT (main) - bắt exception -> sys.exit(1)       │
    └────────────────────────────────────────────────────────────────┘

Ghi chú (TODO):
    - dup_groups được nạp động từ docs/exclusion_decisions.json (do scripts/eda.py sinh ra) -
      không hard-code trong file này nữa. Chạy scripts/eda.py trước nếu file chưa tồn tại.
    - Nếu P3 đã fine-tune thử trên train_split.json bản CŨ (5.989 câu, chưa trừ error_pool), phải
      chạy lại từ đầu sau khi có train_split.json mới (5.689 câu) - nếu không, 300 câu trong
      error_pool coi như KHÔNG còn "chưa từng thấy" đối với checkpoint cũ, làm hỏng mục đích ban đầu
      của error_pool (P4 phân tích lỗi trên dữ liệu mô hình chưa từng thấy).
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
EXPECTED_ERROR_POOL = 300
EXPECTED_TOTAL_VALID = None
EXPECTED_TRAIN_SPLIT = None


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
    EXPECTED_TRAIN_SPLIT = EXPECTED_TOTAL_VALID - EXPECTED_HOLDOUT - EXPECTED_ERROR_POOL

    clean_train_data, excluded_questions = {}, []
    for qid, qdata in train_data.items():
        if qid in vung_chet_qids:
            answers_str = [str(a) for a in (qdata.get("answer") or [])]
            excluded_questions.append({"qid": qid, "question": qdata.get("question", ""), "answer": answers_str})
        else:
            clean_train_data[qid] = qdata

    # Cảnh báo chéo: nếu corpus (dù đã bị loại 25 doc) vẫn còn dòng "text" rỗng khác vung_chet_qids,
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
    print(f"Còn lại {len(clean_train_data)} câu hợp lệ để chia 3 đường "
          f"({EXPECTED_HOLDOUT}/{EXPECTED_ERROR_POOL}/{EXPECTED_TRAIN_SPLIT}).")

    print("Đang gom nhóm các câu hỏi cùng cụm trùng lặp (Union-Find) để chống rò rỉ...")
    groups = build_anti_leak_groups(clean_train_data, dup_groups)
    print(f"  - Tổng số nhóm sau khi gom cụm chống rò rỉ: {len(groups)} nhóm.")

    sorted_group_keys = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(sorted_group_keys)

    # MỘT vòng lặp duy nhất, cùng thứ tự shuffle như bản trước (seed 42) - ưu tiên lấp đầy
    # holdout TRƯỚC (y hệt logic cũ -> holdout.json tái tạo giống hệt bản đã commit), rồi mới
    # đến error_pool (mới), phần còn lại là train_split.
    holdout_qids, error_pool_qids, train_split_qids = [], [], []
    for key in sorted_group_keys:
        group_qids = groups[key]
        if len(holdout_qids) + len(group_qids) <= EXPECTED_HOLDOUT:
            holdout_qids.extend(group_qids)
        elif len(error_pool_qids) + len(group_qids) <= EXPECTED_ERROR_POOL:
            error_pool_qids.extend(group_qids)
        else:
            train_split_qids.extend(group_qids)
    holdout_qids.sort()
    error_pool_qids.sort()
    train_split_qids.sort()

    holdout_json = {qid: clean_train_data[qid] for qid in holdout_qids}
    error_pool_json = {qid: clean_train_data[qid] for qid in error_pool_qids}
    train_split_json = {qid: clean_train_data[qid] for qid in train_split_qids}

    splits = {"holdout": holdout_json, "error_pool": error_pool_json, "train_split": train_split_json}

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
        )
        if not size_ok:
            raise AssertionError(
                f"Sai số lượng - loại trừ: {len(excluded_questions)} (kỳ vọng {EXPECTED_EXCLUDED}), "
                f"holdout: {len(holdout_json)} (kỳ vọng {EXPECTED_HOLDOUT}), "
                f"error_pool: {len(error_pool_json)} (kỳ vọng {EXPECTED_ERROR_POOL}), "
                f"train_split: {len(train_split_json)} (kỳ vọng {EXPECTED_TRAIN_SPLIT})"
            )
        run_integrity_checks(splits, empty_doc_ids_now, dup_groups)
    except AssertionError as e:
        print(f"{e}")
        print("  => ĐÃ CHẶN ĐỨNG việc ghi tệp hỏng ra đĩa để tránh lỗi im lặng (silent failure).")
        print("=" * 80)
        raise RuntimeError("Quy trình kiểm định trước khi ghi đĩa thất bại! Dữ liệu không đạt mốc kỳ vọng.")
    print("  - Số lượng, vùng chết, giao QID (3 cặp), rò rỉ cụm trùng, tổng = 6989: Tất cả đạt.")
    print("=" * 80)

    # -----------------------------------------------------------------
    # GHI ĐĨA (chỉ chạy tới đây khi pre-write validation đã pass)
    # -----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_file = out_dir / "holdout.json"
    error_pool_file = out_dir / "error_pool.json"
    train_split_file = out_dir / "train_split.json"
    log_path = out_dir / "excluded_questions.md"

    with open(holdout_file, "w", encoding="utf-8") as f:
        json.dump(holdout_json, f, ensure_ascii=False, indent=2)
    with open(error_pool_file, "w", encoding="utf-8") as f:
        json.dump(error_pool_json, f, ensure_ascii=False, indent=2)
    with open(train_split_file, "w", encoding="utf-8") as f:
        json.dump(train_split_json, f, ensure_ascii=False, indent=2)
    save_excluded_log(excluded_questions, log_path)
    print(f"Đã ghi nhật ký loại bỏ vào: {log_path}")

    # -----------------------------------------------------------------
    # POST-WRITE VALIDATOR (đọc ngược từ đĩa) - round-trip check độc lập
    # Rollback (xoá cả 4 file) nếu vì lý do bất ngờ nào đó vẫn fail.
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ĐANG CHẠY BỘ ĐỐI SOÁT ĐỘC LẬP (POST-WRITE INTEGRITY CHECK)...")
    try:
        with open(holdout_file, "r", encoding="utf-8") as f:
            holdout_reread = json.load(f)
        with open(error_pool_file, "r", encoding="utf-8") as f:
            error_pool_reread = json.load(f)
        with open(train_split_file, "r", encoding="utf-8") as f:
            train_split_reread = json.load(f)

        reread_splits = {"holdout": holdout_reread, "error_pool": error_pool_reread, "train_split": train_split_reread}
        if (len(holdout_reread) != EXPECTED_HOLDOUT or len(error_pool_reread) != EXPECTED_ERROR_POOL
                or len(train_split_reread) != EXPECTED_TRAIN_SPLIT):
            raise AssertionError(
                f"Lệch số lượng sau khi đọc lại từ đĩa - holdout: {len(holdout_reread)}, "
                f"error_pool: {len(error_pool_reread)}, train_split: {len(train_split_reread)}"
            )
        run_integrity_checks(reread_splits, empty_doc_ids_now, dup_groups)
    except AssertionError as e:
        print(f"{e}")
        print("  => Kết quả trên đĩa KHÔNG khớp dữ liệu đã validate trong bộ nhớ (lỗi round-trip JSON).")
        print("  => ROLLBACK: đang xoá các file vừa ghi để không để lại dữ liệu hỏng trên đĩa...")
        for f in (holdout_file, error_pool_file, train_split_file, log_path):
            f.unlink(missing_ok=True)
        print("=" * 80)
        raise RuntimeError("Đối soát độc lập sau khi ghi thất bại! Đã rollback, không có file nào bị hỏng sót lại.")

    print("  - Đọc lại từ đĩa, đối soát toàn bộ điều kiện: Hoàn hảo 100% (CODA-READY)!")
    print("=" * 80)

    # -----------------------------------------------------------------
    # BÁO CÁO NGHIỆM THU
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BÁO CÁO NGHIỆM THU CHIA TÁCH DỮ LIỆU (P2):")
    print("=" * 80)
    print(f"  - Số lượng câu hỏi train.json gốc: {len(train_data)} câu")
    print(f"  - Số lượng câu hỏi Vùng Chết bị lọc: {len(excluded_questions)} câu (Kỳ vọng: {EXPECTED_EXCLUDED})")
    print(f"  - Tổng số câu hỏi hợp lệ (7000 - {EXPECTED_EXCLUDED}): {len(clean_train_data)} câu")
    print(f"  - Tập kiểm thử nội bộ ({holdout_file.name}): {EXPECTED_HOLDOUT}/{len(holdout_json)} câu")
    print(f"  - Tập error_pool cho P4 ({error_pool_file.name}): {EXPECTED_ERROR_POOL}/{len(error_pool_json)} câu")
    print(f"  - Tập huấn luyện mới ({train_split_file.name}): {EXPECTED_TRAIN_SPLIT}/{len(train_split_json)} câu")
    print("=" * 80)
    print("KẾT LUẬN: CHIA 3 ĐƯỜNG SẠCH (CODA-READY)!")
    print("=" * 80)


# ===========================================================================
# 6. CLI ENTRYPOINT
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Bộ lọc bẫy Vùng Chết và chia 3 đường dữ liệu cho P2")
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
