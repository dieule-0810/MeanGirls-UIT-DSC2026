"""Loại câu "vùng chết", chống rò rỉ cụm trùng và chia 4 tập + 3 mốc learning-curve. P2.

Quy trình (seed 42, một lần shuffle duy nhất):

1. Đọc 11 qid vùng chết và 4 cụm trùng từ `docs/exclusion_decisions.json` (do
   `scripts/eda.py` sinh), loại 11 câu khỏi `train.json` ⇒ 6.989 câu hợp lệ.
2. Union-Find gộp các câu hỏi chạm cùng một (siêu) cụm văn bản trùng thành một nhóm, để một
   cụm không bao giờ bị xé ra hai tập.
3. Lấp đầy theo thứ tự holdout (1.000) → error_pool (300) → dev (1.000) → train_split (4.689).
   dev chen GIỮA nên holdout + error_pool tái tạo bit-for-bit như bản chia v5.
4. Tập con LỒNG NHAU của train_split cho learning curve: train_1000 ⊂ train_2500 ⊂ train_4689.
5. Kiểm toàn vẹn TRƯỚC khi ghi (trong RAM) và SAU khi ghi (đọc lại từ đĩa); hỏng sau khi ghi
   thì xoá mọi file vừa ghi rồi mới raise.

* `holdout.json` — ĐO LẦN CUỐI, mọi script chặn bằng `--allow-holdout`.
* `dev.json` — đo lúc thử nghiệm, dùng chung cả nhóm.
* `error_pool.json` — P4 đọc tay.
* `train_split.json` — tập fit / chọn siêu tham số.

Typical usage example:

    python -m src.data.split_data --corpus data/corpus_clean.jsonl --train data/train.json --out-dir data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

DECISIONS_PATH = Path("docs/exclusion_decisions.json")
SPLIT_NAMES = ("holdout", "dev", "error_pool", "train_split")
SEED = 42


@dataclass(frozen=True)
class SplitPlan:
    """Kích thước kỳ vọng của từng tập.

    Attributes:
        n_raw: Số câu trong `train.json` gốc.
        excluded: Số câu vùng chết phải loại.
        holdout: Kích thước holdout.
        dev: Kích thước dev.
        error_pool: Kích thước error_pool.
        nested: Các mốc learning-curve nhỏ hơn train_split.
    """

    n_raw: int
    excluded: int = 11
    holdout: int = 1000
    dev: int = 1000
    error_pool: int = 300
    nested: tuple[int, ...] = (1000, 2500)

    @property
    def total_valid(self) -> int:
        """Số câu hợp lệ sau khi loại vùng chết."""
        return self.n_raw - self.excluded

    @property
    def train_split(self) -> int:
        """Phần còn lại sau holdout, error_pool và dev."""
        return self.total_valid - self.holdout - self.error_pool - self.dev

    def size_of(self, name: str) -> int:
        """Kích thước kỳ vọng của tập `name`."""
        return getattr(self, name)


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find và gom nhóm chống rò rỉ
# ─────────────────────────────────────────────────────────────────────────────
class UnionFind:
    """Union-Find tối giản (nén đường đi, không union by rank) — đủ cho vài chục cụm."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Gốc của `x`, có nén đường đi."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Gộp hai tập; gốc mới là chỉ số nhỏ hơn (xác định)."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _cluster_index(dup_groups: list) -> dict[str, int]:
    """`{doc_id: chỉ số cụm}` của mọi văn bản trong các cụm trùng."""
    return {str(doc_id): idx for idx, group in enumerate(dup_groups) for doc_id in group}


def _union_touched(data: dict, doc_to_cluster: dict[str, int], uf: UnionFind) -> dict[str, set[int]]:
    """Gộp mọi cụm mà một câu hỏi chạm tới; trả `{qid: các cụm đã chạm}`."""
    touched_by_qid: dict[str, set[int]] = {}
    for qid, qdata in data.items():
        answers = [str(a) for a in (qdata.get("answer") or [])]
        touched = {doc_to_cluster[a] for a in answers if a in doc_to_cluster}
        if touched:
            touched_by_qid[qid] = touched
            touched_list = list(touched)
            for i in range(1, len(touched_list)):
                uf.union(touched_list[0], touched_list[i])
    return touched_by_qid


def build_anti_leak_groups(clean_train_data: dict, dup_groups: list) -> dict[str, list[str]]:
    """Gom qid thành nhóm sao cho mọi câu chạm cùng một (siêu) cụm trùng nằm chung một nhóm.

    Câu hỏi đa đáp án có thể bắc cầu qua 2+ cụm — Union-Find gộp các cụm đó thành siêu-cụm trước.

    Args:
        clean_train_data: `{qid: {"question", "answer"}}` đã loại vùng chết.
        dup_groups: Danh sách cụm văn bản trùng.

    Returns:
        `{khoá nhóm: [qid, ...]}`; câu không chạm cụm nào tự thành một nhóm.
    """
    uf = UnionFind(len(dup_groups))
    touched_by_qid = _union_touched(clean_train_data, _cluster_index(dup_groups), uf)
    groups: dict[str, list[str]] = {}
    for qid in clean_train_data:
        touched = touched_by_qid.get(qid)
        key = f"cluster_{uf.find(next(iter(touched)))}" if touched else qid
        groups.setdefault(key, []).append(qid)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm toàn vẹn (dùng chung trước và sau khi ghi)
# ─────────────────────────────────────────────────────────────────────────────
def _merge_dup_groups_via_data(combined_data: dict, dup_groups: list) -> list[set[str]]:
    """Tính lại siêu-cụm trực tiếp từ dữ liệu, ĐỘC LẬP với `build_anti_leak_groups`."""
    if not dup_groups:
        return []
    uf = UnionFind(len(dup_groups))
    _union_touched(combined_data, _cluster_index(dup_groups), uf)
    merged: dict[int, set[str]] = {}
    for idx, group in enumerate(dup_groups):
        merged.setdefault(uf.find(idx), set()).update(group)
    return list(merged.values())


def run_integrity_checks(splits: dict, empty_doc_ids: set, dup_groups: list, expected_total: int) -> None:
    """Bốn bất biến của N tập rời nhau. Fail-loud, không chỉ cảnh báo.

    1. Không câu nào trỏ vào văn bản rỗng (vùng chết).
    2. Mọi CẶP tập không giao nhau về qid.
    3. Tổng số câu bằng `expected_total`.
    4. Một (siêu) cụm trùng chỉ xuất hiện trong ĐÚNG một tập.

    Args:
        splits: `{tên tập: {qid: ...}}`.
        empty_doc_ids: Văn bản rỗng còn trong corpus.
        dup_groups: Cụm văn bản trùng.
        expected_total: Tổng số câu hợp lệ kỳ vọng.

    Raises:
        AssertionError: Vi phạm bất kỳ bất biến nào.
    """
    combined: dict = {}
    for data in splits.values():
        combined.update(data)
    for qid, qdata in combined.items():
        if any(str(a) in empty_doc_ids for a in (qdata.get("answer") or [])):
            raise AssertionError(f"QID {qid} dính bẫy vùng chết (trỏ vào tài liệu rỗng)!")
    for (name_a, data_a), (name_b, data_b) in combinations(splits.items(), 2):
        overlap = set(data_a) & set(data_b)
        if overlap:
            raise AssertionError(f"Rò rỉ trực tiếp: {len(overlap)} QID xuất hiện ở CẢ '{name_a}' VÀ '{name_b}': {overlap}")
    if len(combined) != expected_total:
        raise AssertionError(f"Tổng số câu hỏi trên {len(splits)} tập là {len(combined)}, không khớp kỳ vọng {expected_total}!")
    docs_by_split = {
        name: {str(a) for qdata in data.values() for a in (qdata.get("answer") or [])}
        for name, data in splits.items()
    }
    for group in _merge_dup_groups_via_data(combined, dup_groups):
        touched = [name for name, docs in docs_by_split.items() if group & docs]
        if len(touched) > 1:
            raise AssertionError(f"Rò rỉ cụm trùng! Cụm {group} bị xé lẻ giữa các tập: {touched}")


def check_sizes(splits: dict, plan: SplitPlan) -> None:
    """Đúng kích thước từng tập.

    Raises:
        AssertionError: Một tập lệch kích thước.
    """
    wrong = {n: (len(splits[n]), plan.size_of(n)) for n in SPLIT_NAMES if len(splits[n]) != plan.size_of(n)}
    if wrong:
        raise AssertionError("Sai số lượng (thực tế, kỳ vọng): " + ", ".join(f"{n}={v}" for n, v in wrong.items()))


def check_nested_subsets(subsets: dict, train_split_json: dict) -> None:
    """Bất biến của các mốc learning-curve: đúng kích thước, LỒNG nhau, mốc lớn nhất = train_split.

    Không đưa vào `run_integrity_checks` vì các mốc CỐ Ý giao nhau.

    Raises:
        AssertionError: Vi phạm bất kỳ điều nào.
    """
    sizes = sorted(subsets)
    for n in sizes:
        if len(subsets[n]) != n:
            raise AssertionError(f"train_{n}: có {len(subsets[n])} câu, kỳ vọng {n}.")
    for smaller, larger in zip(sizes, sizes[1:]):
        if not set(subsets[smaller]) <= set(subsets[larger]):
            missing = set(subsets[smaller]) - set(subsets[larger])
            raise AssertionError(f"train_{smaller} KHÔNG phải tập con của train_{larger} ({len(missing)} qid lọt ra ngoài).")
    if set(subsets[sizes[-1]]) != set(train_split_json):
        raise AssertionError(f"train_{sizes[-1]} không trùng khít train_split ({len(subsets[sizes[-1]])} vs {len(train_split_json)}).")


def validate(splits: dict, nested: dict, empty_doc_ids: set, dup_groups: list, plan: SplitPlan) -> None:
    """Chạy đủ bộ kiểm (kích thước, bốn bất biến, mốc lồng nhau)."""
    check_sizes(splits, plan)
    run_integrity_checks(splits, empty_doc_ids, dup_groups, plan.total_valid)
    check_nested_subsets(nested, splits["train_split"])


# ─────────────────────────────────────────────────────────────────────────────
# Chia
# ─────────────────────────────────────────────────────────────────────────────
def exclude_dead_zone(train_data: dict, dead_qids: set[str]) -> tuple[dict, list[dict]]:
    """Tách câu vùng chết ra khỏi tập train.

    Returns:
        `(câu hợp lệ, danh sách câu bị loại {qid, question, answer})`.
    """
    clean, excluded = {}, []
    for qid, qdata in train_data.items():
        if qid in dead_qids:
            excluded.append({"qid": qid, "question": qdata.get("question", ""),
                             "answer": [str(a) for a in (qdata.get("answer") or [])]})
        else:
            clean[qid] = qdata
    return clean, excluded


def find_empty_docs(corpus_path: Path) -> set[str]:
    """Văn bản `text` rỗng còn sót trong corpus — chỉ để cảnh báo, không tự động loại."""
    empty = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if not doc.get("text", "").strip():
                empty.add(str(doc["doc_id"]))
    return empty


def assign_groups(groups: dict[str, list[str]], plan: SplitPlan) -> dict[str, list[str]]:
    """Một lần shuffle (seed 42), lấp đầy holdout → error_pool → dev → train_split.

    Thứ tự lấp đầy giữ nguyên bản v5 để holdout và error_pool tái tạo bit-for-bit.

    Returns:
        `{tên tập: [qid đã sort]}`.
    """
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for key in keys:
        g = groups[key]
        for name in ("holdout", "error_pool", "dev"):
            if len(out[name]) + len(g) <= plan.size_of(name):
                out[name].extend(g)
                break
        else:
            out["train_split"].extend(g)
    return {name: sorted(qids) for name, qids in out.items()}


def carve_nested_by_group(groups: dict, train_split_qids: list, small_sizes: list, seed: int = SEED) -> dict:
    """Các tập con LỒNG NHAU của train_split, carve theo nhóm chống rò rỉ.

    Mốc nhỏ carve TỪ TRONG mốc lớn (pool thu hẹp dần) nên lồng nhau theo cấu trúc.

    Args:
        groups: Nhóm từ `build_anti_leak_groups`.
        train_split_qids: qid của train_split.
        small_sizes: Các mốc nhỏ hơn train_split.
        seed: Hạt giống shuffle.

    Returns:
        `{kích thước: [qid đã sort]}`, gồm cả mốc đầy đủ.

    Raises:
        ValueError: Không lấp khít được một mốc bằng nhóm cỡ 1.
    """
    train_set = set(train_split_qids)
    train_groups = {k: v for k, v in groups.items() if v[0] in train_set}  # nhóm nguyên khối → xét 1 qid đủ
    keys = sorted(train_groups)
    random.Random(seed).shuffle(keys)
    full_n = len(train_set)

    def _carve(pool_keys: list, target: int) -> list:
        chosen, cur = [], 0
        for k in pool_keys:  # lượt 1: tham lam theo nhóm, không vượt target
            if cur + len(train_groups[k]) <= target:
                chosen.append(k)
                cur += len(train_groups[k])
        chosen_set = set(chosen)
        for k in pool_keys:  # lượt 2: lấp khít bằng nhóm cỡ 1
            if cur >= target:
                break
            if k not in chosen_set and len(train_groups[k]) == 1:
                chosen.append(k)
                cur += 1
        if cur != target:
            raise ValueError(f"Không carve chính xác train_{target} theo nhóm (thiếu nhóm size-1 để lấp khít).")
        return chosen

    subset_keys = {full_n: keys}
    pool = keys
    for t in sorted((s for s in small_sizes if s < full_n), reverse=True):
        pool = _carve(pool, t)
        subset_keys[t] = pool
    return {n: sorted(q for k in ks for q in train_groups[k]) for n, ks in subset_keys.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Ghi đĩa
# ─────────────────────────────────────────────────────────────────────────────
def save_excluded_log(excluded_list: list, log_path: Path) -> None:
    """Ghi danh sách câu vùng chết bị loại ra markdown, để minh bạch."""
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


def _dump(obj: dict, path: Path) -> None:
    """Ghi JSON (UTF-8, indent 2) — định dạng phải giữ nguyên để tái lập bit-for-bit."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def write_all(out_dir: Path, splits: dict, nested: dict, excluded: list) -> dict[str, Path]:
    """Ghi 4 tập, các mốc learning-curve và log loại trừ.

    Returns:
        `{nhãn: đường dẫn}` của mọi file đã ghi (để rollback).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in ("holdout", "error_pool", "dev", "train_split"):
        written[name] = out_dir / f"{name}.json"
        _dump(splits[name], written[name])
    for n in sorted(nested):
        written[f"train_{n}"] = out_dir / f"train_{n}.json"
        _dump(nested[n], written[f"train_{n}"])
    written["log"] = out_dir / "excluded_questions.md"
    save_excluded_log(excluded, written["log"])
    return written


def reread(written: dict[str, Path], nested_sizes: list[int]) -> tuple[dict, dict]:
    """Đọc lại các tập vừa ghi từ đĩa — để kiểm độc lập với dữ liệu trong RAM."""
    def load(key: str) -> dict:
        return json.loads(written[key].read_text(encoding="utf-8"))

    return {n: load(n) for n in SPLIT_NAMES}, {n: load(f"train_{n}") for n in nested_sizes}


def _print_report(n_raw: int, excluded: list, splits: dict, nested: dict, plan: SplitPlan) -> None:
    """In báo cáo nghiệm thu chia tập."""
    print("\n" + "=" * 80 + "\nBÁO CÁO CHIA TÁCH DỮ LIỆU (P2)\n" + "=" * 80)
    print(f"  - train.json gốc: {n_raw} | vùng chết loại: {len(excluded)} (kỳ vọng {plan.excluded}) | hợp lệ: {plan.total_valid}")
    notes = {"holdout": "ĐO LẦN CUỐI", "dev": "đo lúc thử nghiệm", "error_pool": "P4 đọc tay", "train_split": "fit"}
    for name in SPLIT_NAMES:
        print(f"  - {name}: {len(splits[name])}/{plan.size_of(name)} — {notes[name]}")
    for n in sorted(nested):
        print(f"      └─ learning-curve train_{n}.json: {len(nested[n])} câu")
    print("  Checkpoint/hard-negative mine từ train_split cũ (5.689 câu) phải chạy LẠI.")


def split_data(corpus_path: Path, train_path: Path, out_dir: Path, decisions_path: Path = DECISIONS_PATH) -> None:
    """Toàn bộ quy trình chia: loại vùng chết → gom nhóm → chia → kiểm → ghi → kiểm lại.

    Args:
        corpus_path: `corpus_clean.jsonl` (để cảnh báo văn bản rỗng còn sót).
        train_path: `train.json` gốc của BTC.
        out_dir: Thư mục ghi các tập.
        decisions_path: File quyết định loại trừ (do `scripts/eda.py` sinh).

    Raises:
        FileNotFoundError: Thiếu file đầu vào.
        RuntimeError: Kiểm toàn vẹn thất bại (trước khi ghi: không ghi gì; sau khi ghi: đã rollback).
    """
    for p, hint in ((corpus_path, "corpus sạch"), (train_path, "train.json gốc"),
                    (decisions_path, "quyết định loại trừ — chạy `python scripts/eda.py` trước")):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy {p} ({hint}).")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    dup_groups = [set(str(i) for i in group) for group in decisions["dup_groups"]]
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    plan = SplitPlan(n_raw=len(train_data))

    clean, excluded = exclude_dead_zone(train_data, set(decisions["qids_exclude_vung_chet"]))
    empty_docs = find_empty_docs(corpus_path)
    if empty_docs:
        print(f"  CẢNH BÁO: corpus còn {len(empty_docs)} văn bản rỗng ngoài danh sách loại — kiểm tra eda_notes.md.")
    if len(excluded) != plan.excluded:
        raise RuntimeError(f"Loại {len(excluded)} câu vùng chết, kỳ vọng {plan.excluded}.")

    groups = build_anti_leak_groups(clean, dup_groups)
    qids = assign_groups(groups, plan)
    splits = {name: {q: clean[q] for q in qids[name]} for name in SPLIT_NAMES}
    nested_qids = carve_nested_by_group(groups, qids["train_split"], list(plan.nested))
    nested = {n: {q: clean[q] for q in ids} for n, ids in nested_qids.items()}
    try:
        validate(splits, nested, empty_docs, dup_groups, plan)
    except AssertionError as e:
        raise RuntimeError(f"Kiểm TRƯỚC khi ghi thất bại, không ghi file nào: {e}") from e

    written = write_all(out_dir, splits, nested, excluded)
    try:
        validate(*reread(written, sorted(nested)), empty_docs, dup_groups, plan)
    except AssertionError as e:
        for f in written.values():
            f.unlink(missing_ok=True)
        raise RuntimeError(f"Kiểm SAU khi ghi thất bại, đã xoá mọi file vừa ghi: {e}") from e
    _print_report(len(train_data), excluded, splits, nested, plan)


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description="Loại câu vùng chết và chia 4 tập + learning-curve (P2)")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus_clean.jsonl"))
    ap.add_argument("--train", type=Path, default=Path("data/train.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--decisions", type=Path, default=DECISIONS_PATH, help="sinh bởi scripts/eda.py")
    args = ap.parse_args()
    try:
        split_data(args.corpus, args.train, args.out_dir, args.decisions)
    except Exception as e:
        print(f"\nLỖI: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
