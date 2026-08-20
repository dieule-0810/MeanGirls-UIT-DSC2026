"""
scripts/eda.py - Khám phá dữ liệu trước khi build (xem plan.md mục 0.5).

Chạy trên corpus + train.json GỐC, trước khi chunker/parser bị chỉnh lần cuối.
Không phụ thuộc corpus_clean.jsonl / chunks.jsonl vì mục đích là phát hiện vấn đề
TRƯỚC bước tiền xử lý, không phải kiểm tra lại sau đó.

Cách chạy:
    python scripts/eda.py \
        --corpus-dir data/selected-contexts \
        --train data/train.json \
        --public data/public-official.json \
        --out docs/eda_notes.md

Output: in tóm tắt ra stdout + ghi báo cáo markdown vào --out.

TODO cho P2: các hàm dưới đây là khung - điền phần tokenize tiếng Việt
(underthesea/pyvi) nếu muốn số liệu "từ" chính xác hơn whitespace-split.
Whitespace-split hiện tại đủ để ra thứ tự độ lớn (order of magnitude),
đừng chặn tiến độ chỉ vì thiếu word-segmenter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

DIEU_PATTERN = re.compile(r"\bĐiều\s+\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: Path) -> list[dict]:
    """Đọc toàn bộ context_*.json. Không ép str(id) ở đây - mục đích EDA là
    NHÌN THẤY dữ liệu thô, kể cả bẫy kiểu dữ liệu, không phải sửa nó."""
    docs = []
    files = sorted(corpus_dir.glob("context_*.json"))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        raw["_source_file"] = fp.name
        docs.append(raw)
    return docs


def load_train(train_path: Path) -> dict[str, dict]:
    """train.json / public-official.json thật là dict {qid: {"question": ..., "answer": [...]}}
    - ĐÃ kiểm tra trực tiếp trên file thật, không phải list như nhiều format QA khác."""
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"{train_path} không phải dict như kỳ vọng - cấu trúc train.json đã đổi? Kiểm tra lại."
    )
    return data


# ---------------------------------------------------------------------------
# 1. Phân bố độ dài văn bản
# ---------------------------------------------------------------------------

def doc_length_stats(docs: list[dict]) -> dict:
    lengths_words = []
    lengths_chars = []
    for d in docs:
        text = d.get("passage") or ""
        lengths_chars.append(len(text))
        lengths_words.append(len(text.split()))

    def pct(vals, p):
        if not vals:
            return 0
        vals_sorted = sorted(vals)
        idx = min(len(vals_sorted) - 1, int(len(vals_sorted) * p))
        return vals_sorted[idx]

    return {
        "n_docs": len(docs),
        "words_mean": round(statistics.mean(lengths_words), 1) if lengths_words else 0,
        "words_median": statistics.median(lengths_words) if lengths_words else 0,
        "words_p90": pct(lengths_words, 0.90),
        "words_p99": pct(lengths_words, 0.99),
        "words_max": max(lengths_words) if lengths_words else 0,
        "chars_mean": round(statistics.mean(lengths_chars), 1) if lengths_chars else 0,
        "mean_to_median_ratio": round(statistics.mean(lengths_words) / statistics.median(lengths_words), 2) if lengths_words else 0,
        "max_to_p99_ratio": round(max(lengths_words) / pct(lengths_words, 0.99), 1) if lengths_words else 0,
    }


# ---------------------------------------------------------------------------
# 2. Tỉ lệ có cấu trúc "Điều N" vs cần fallback
# ---------------------------------------------------------------------------

def dieu_structure_stats(docs: list[dict]) -> dict:
    n_with_dieu = 0
    dieu_counts = []
    for d in docs:
        text = d.get("passage") or ""
        matches = DIEU_PATTERN.findall(text)
        if matches:
            n_with_dieu += 1
        dieu_counts.append(len(matches))

    n = len(docs) or 1
    return {
        "pct_with_dieu": round(100 * n_with_dieu / n, 1),
        "pct_needs_fallback": round(100 * (n - n_with_dieu) / n, 1),
        "avg_dieu_per_doc_when_present": (
            round(sum(c for c in dieu_counts if c > 0) / max(n_with_dieu, 1), 1)
        ),
    }


# ---------------------------------------------------------------------------
# 3. Trường thiếu (name / link)
# ---------------------------------------------------------------------------

def missing_field_stats(docs: list[dict]) -> dict:
    n = len(docs) or 1
    
    # Thống kê khuyết KEY trong JSON
    key_missing_id = sum(1 for d in docs if "id" not in d)
    key_missing_name = sum(1 for d in docs if "name" not in d)
    key_missing_link = sum(1 for d in docs if "link" not in d)
    key_missing_passage = sum(1 for d in docs if "passage" not in d)
    
    # Có KEY nhưng giá trị rỗng/null/none
    val_empty_id = sum(1 for d in docs if "id" in d and (d.get("id") is None or str(d.get("id")).strip() == ""))
    val_empty_name = sum(1 for d in docs if "name" in d and not d.get("name"))
    val_empty_link = sum(1 for d in docs if "link" in d and not d.get("link"))
    val_empty_passage = sum(1 for d in docs if "passage" in d and not d.get("passage"))
    
    # Kiểm tra KEY lạ dư thừa (key chuẩn: id, name, link, passage)
    standard_keys = {"id", "name", "link", "passage"}
    files_with_extra_keys = {}
    for d in docs:
        actual_keys = set(d.keys()) - {"_source_file"}
        extra_keys = actual_keys - standard_keys
        if extra_keys:
            files_with_extra_keys[d["_source_file"]] = sorted(list(extra_keys))
            
    res = {}
        
    # Xử lý lỗi trường ID
    total_missing_id = key_missing_id + val_empty_id
    if total_missing_id > 0:
        missing_id_files = [d["_source_file"] for d in docs if "id" not in d or d.get("id") is None or str(d.get("id")).strip() == ""]
        res["ID_missing_summary"] = {
            "pct_missing": round(100 * total_missing_id / n, 1),
            "key_missing_count": key_missing_id,
            "val_empty_count": val_empty_id,
            "eg": missing_id_files[:10]
        }
    else:
        res["ID_missing_summary"] = {}
        
    # Xử lý lỗi trường NAME
    total_missing_name = key_missing_name + val_empty_name
    if total_missing_name > 0:
        missing_name_files = [d["_source_file"] for d in docs if "name" not in d or not d.get("name")]
        res["NAME_missing_summary"] = {
            "pct_missing": round(100 * total_missing_name / n, 1),
            "key_missing_count": key_missing_name,
            "val_empty_count": val_empty_name,
            "eg": missing_name_files[:10]
        }
    else:
        res["NAME_missing_summary"] = {}
        
    # Xử lý lỗi trường LINK
    total_missing_link = key_missing_link + val_empty_link
    if total_missing_link > 0:
        missing_link_files = [d["_source_file"] for d in docs if "link" not in d or not d.get("link")]
        res["LINK_missing_summary"] = {
            "pct_missing": round(100 * total_missing_link / n, 1),
            "key_missing_count": key_missing_link,
            "val_empty_count": val_empty_link,
            "eg": missing_link_files[:10]
        }
        
    else:
        res["LINK_missing_summary"] = {}
        
    # Xử lý lỗi trường PASSAGE
    total_missing_passage = key_missing_passage + val_empty_passage
    if total_missing_passage > 0:
        missing_passage_files = [d["_source_file"] for d in docs if "passage" not in d or not d.get("passage")]
        res["PASSAGE_missing_summary"] = {
            "pct_missing": round(100 * total_missing_passage / n, 1),
            "key_missing_count": key_missing_passage,
            "val_empty_count": val_empty_passage,
            "eg": missing_passage_files[:10]
        }
    else:
        res["PASSAGE_missing_summary"] = {}

    # Xử lý key lạ dư thừa
    if files_with_extra_keys:
        res["EXTRA_keys_analysis"] = {
            "count": len(files_with_extra_keys),
            "eg": {
                k: files_with_extra_keys[k] for k in list(files_with_extra_keys.keys())[:10]
            }
        }
    else:
        res["EXTRA_keys_analysis"] = {}
        
    return res

# ---------------------------------------------------------------------------
# 4. Phân bố số đáp án đúng / câu hỏi (toàn bộ train.json)
# ---------------------------------------------------------------------------

def answer_count_stats(train: dict[str, dict]) -> dict:
    counts = []
    empty_questions = 0
    for qid, item in train.items():
        answers = item.get("answer") or []
        counts.append(len(answers))
        if not (item.get("question") or "").strip():
            empty_questions += 1

    n = len(counts) or 1
    dist = Counter(counts)

    mean_gold = sum(counts) / n
    # Trần trên của Precision KHI LUÔN nộp đủ 5 doc/câu, giả định mọi gold lọt top-5.
    # precision mỗi câu = min(|gold|,5)/5  (min để không vượt 1 nếu |gold|>5).
    # Đây là CẬN TRÊN, không phải precision đo được - thực tế Recall@5<1 nên thấp hơn.
    precision_ceiling_always5 = sum(min(g, 5) for g in counts) / n / 5

    return {
        "n_questions": len(train),
        "pct_exactly_1_answer": round(100 * dist.get(1, 0) / n, 1),
        "pct_multi_answer": round(100 * sum(v for k, v in dist.items() if k >= 2) / n, 1),
        "distribution": dict(sorted(dist.items())),
        "n_empty_question_text": empty_questions,
        "n_zero_answer": dist.get(0, 0),
        "mean_gold_per_q": round(mean_gold, 3),
        "precision_ceiling_if_always_5": round(precision_ceiling_always5, 3),
    }


# ---------------------------------------------------------------------------
# 5. Trùng / gần trùng văn bản (kiểm tra thô bằng hash sau khi chuẩn hoá)
# ---------------------------------------------------------------------------

def near_duplicate_stats(docs: list[dict], train: dict[str, dict]) -> dict:
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()
        
    passage_hashes = {}
    link_groups = {}
    
    # Map từ doc_id (str) sang link gốc
    id_to_link = {str(d.get("id")): (d.get("link") or "") for d in docs}

    train_referenced_ids = {str(a) for item in train.values() for a in (item.get("answer") or [])}
    
    for d in docs:
        doc_id = d.get("id")
        doc_id_str = str(doc_id) if doc_id is not None else ""
        
        # Xử lý trùng lặp nội dung (passage)
        text = d.get("passage") or ""
        norm = normalize(text)
        
        if norm:
            h = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if h in passage_hashes:
                passage_hashes[h].append(doc_id_str)
            else:
                passage_hashes[h] = [doc_id_str]
                
        # Xử lý trùng lặp đường dẫn (link)
        link = (d.get("link") or "").strip()
        if link:
            if link in link_groups:
                link_groups[link].append(doc_id_str)
            else:
                link_groups[link] = [doc_id_str]
                
    # Lọc ra các nhóm thực sự bị trùng
    dup_passage_groups = [ids for ids in passage_hashes.values() if len(ids) > 1]
    dup_link_groups = [ids for ids in link_groups.values() if len(ids) > 1]
    
    dup_passage_sets = [set(group) for group in dup_passage_groups]
    
    # Join #1: Nhóm ID trùng có gán làm đáp án hợp lệ (đa đáp án)?
    multi_answer_due_to_duplicates = []
    
    # Join #2: Label Noise
    questions_at_risk_of_label_noise = []
    risk_by_answer_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for qid, item in train.items():
        answers = [str(a) for a in (item.get("answer") or [])]
        answers_set = set(answers)
        n_ans = len(answers)
        
        # Join #1
        if n_ans >= 2:
            for group in dup_passage_sets:
                intersect = answers_set & group
                if len(intersect) >= 2:
                    multi_answer_due_to_duplicates.append({
                        "qid": qid,
                        "question": (item.get("question") or "")[:60] + "...",
                        "answers_in_train": answers,
                        "intersecting_duplicates": sorted(list(intersect))
                    })
                    break
                    
        # Join #2
        for group in dup_passage_sets:
            intersect = answers_set & group
            difference = group - answers_set
            # Điều kiện rủi ro: Có chứa ít nhất 1 đáp án trong cụm trùng, VÀ cụm trùng còn ID bị bỏ sót
            if len(intersect) >= 1 and len(difference) >= 1:
                questions_at_risk_of_label_noise.append({
                    "qid": qid,
                    "answer_count": n_ans,
                    "question": (item.get("question") or "")[:60] + "...",
                    "answers_in_train": answers,
                    "duplicate_missing_from_gold": sorted(list(difference))
                })
                if n_ans in risk_by_answer_count:
                    risk_by_answer_count[n_ans] += 1
                break
                
    res = {}
    
    # Thống kê trùng lặp PASSAGE
    n_passage_groups = len(dup_passage_groups)
    if n_passage_groups > 0:
        res["PASSAGE_duplicate_summary"] = {
            "n_groups": n_passage_groups,
            "total_duplicate_files": sum(len(ids) for ids in dup_passage_groups),
            "eg": [
                {
                    "ids": ids,
                    "links": [id_to_link.get(i, "") for i in ids],
                    "members_ever_gold": sorted(list(set(ids) & train_referenced_ids))
                }
                for ids in dup_passage_groups[:10]
            ]
        }
    else:
        res["PASSAGE_duplicate_summary"] = {}
        
    # Thống kê trùng lặp LINK
    n_link_groups = len(dup_link_groups)
    if n_link_groups > 0:
        res["LINK_duplicate_summary"] = {
            "n_groups": n_link_groups,
            "total_duplicate_files": sum(len(ids) for ids in dup_link_groups),
            "eg": dup_link_groups[:10]  
        }
    else:
        res["LINK_duplicate_summary"] = {}
        
    # Thống kê kết quả Join #1
    res["Join_1"] = {
        "total_multi_answer_questions": sum(1 for item in train.values() if len(item.get("answer") or []) >= 2),
        "count": len(multi_answer_due_to_duplicates),
        "eg": multi_answer_due_to_duplicates[:5]
    }
    
    # Thống kê kết quả Join #2
    res["Join_2"] = {
        "total_questions_at_risk": len(questions_at_risk_of_label_noise),
        "pct_of_total_train": round(100 * len(questions_at_risk_of_label_noise) / (len(train) or 1), 2),
        "distribution_by_answer_count": risk_by_answer_count,
        "eg": questions_at_risk_of_label_noise[:10]
    }
    
    res["dup_passage_groups"] = dup_passage_groups
    
    return res

# ---------------------------------------------------------------------------
# 6. Độ dài câu hỏi / câu hỏi lỗi
# ---------------------------------------------------------------------------

def question_length_stats(train: dict[str, dict]) -> dict:
    items = list(train.values())
    lengths = [len((item.get("question") or "").split()) for item in items]
    lengths = [l for l in lengths if l > 0]
    suspicious = [
        item.get("question", "")[:80]
        for item in items
        if len((item.get("question") or "").split()) <= 2
    ][:10]
    return {
        "words_mean": round(statistics.mean(lengths), 1) if lengths else 0,
        "words_min": min(lengths) if lengths else 0,
        "words_max": max(lengths) if lengths else 0,
        "n_suspiciously_short": len(suspicious),
        "examples_suspiciously_short": suspicious,
    }


# ---------------------------------------------------------------------------
# 7. doc_id trong train.json có phủ hết corpus không / có id mồ côi không
# ---------------------------------------------------------------------------

def doc_id_coverage_stats(docs: list[dict], train: dict[str, dict]) -> dict:
    corpus_ids = {str(d.get("id")) for d in docs}
    train_ids = set()
    for item in train.values():
        answers = item.get("answer") or []
        train_ids.update(str(a) for a in answers)

    orphan_ids = train_ids - corpus_ids  # xuất hiện trong train nhưng KHÔNG có trong corpus
    unused_ids = corpus_ids - train_ids  # có trong corpus nhưng chưa từng là đáp án train

    return {
        "n_corpus_ids": len(corpus_ids),
        "n_train_referenced_ids": len(train_ids),
        "n_orphan_ids_in_train": len(orphan_ids),  # phải bằng 0, nếu không → BUG NGHIÊM TRỌNG
        "orphan_examples": sorted(orphan_ids)[:10],
        "n_corpus_ids_never_answer_in_train": len(unused_ids),
        "pct_corpus_never_answer_in_train": round(100 * len(unused_ids) / (len(corpus_ids) or 1), 1),
    }

# ---------------------------------------------------------------------------
# 8. Mối quan hệ lỗi (3.1)
# ---------------------------------------------------------------------------

def check_name_passage_overlap(docs: list[dict]) -> dict:
    missing_name_files = {d["_source_file"] for d in docs if "name" not in d}
    empty_passage_files = {d["_source_file"] for d in docs if "passage" in d and not d.get("passage")}
    
    # Có file nào bị cả hai lỗi không
    overlap_files = missing_name_files & empty_passage_files
    
    return {
        "total_missing_name_keys": len(missing_name_files),
        "total_empty_passages": len(empty_passage_files),
        "both_errors_overlap_count": len(overlap_files),
        "overlap_percentage_of_empty_passage": round(100 * len(overlap_files) / (len(empty_passage_files) or 1), 1),
        "eg_overlap_files": sorted(list(overlap_files))[:10]
    }
 
# ---------------------------------------------------------------------------
# 9. Kiểm chứng Gold ID trỏ vào file lỗi
# ---------------------------------------------------------------------------

def verify_gold_with_missing_fields(docs: list[dict], train: dict[str, dict]) -> dict:
    missing_name_ids = {str(d.get("id")) for d in docs if "name" not in d}
    empty_passage_ids = {str(d.get("id")) for d in docs if "passage" in d and not d.get("passage")}

    mismatch_name_cases = []
    mismatch_passage_cases = []
    doc_hit_count_name = Counter()
    doc_hit_count_passage = Counter()

    for qid, item in train.items():
        answers = [str(a) for a in (item.get("answer") or [])]
        question_text = item.get("question") or ""
        for ans_id in answers:
            if ans_id in missing_name_ids:
                mismatch_name_cases.append({"qid": qid, "gold_id": ans_id,
                    "question": question_text[:80] + "..." if len(question_text) > 80 else question_text})
                doc_hit_count_name[ans_id] += 1
            if ans_id in empty_passage_ids:
                mismatch_passage_cases.append({"qid": qid, "gold_id": ans_id,
                    "question": question_text[:80] + "..." if len(question_text) > 80 else question_text})
                doc_hit_count_passage[ans_id] += 1

    n_name_live = len(doc_hit_count_name)
    n_passage_live = len(doc_hit_count_passage)

    return {
        "gold_missing_name_summary": {
            "count_questions": len(mismatch_name_cases),
            "pct_of_train_questions": round(100 * len(mismatch_name_cases) / (len(train) or 1), 2),
            "n_unique_docs_affected": n_name_live,
            "pct_of_error_set_that_matters": round(100 * n_name_live / (len(missing_name_ids) or 1), 1),
            "top_offenders": doc_hit_count_name.most_common(5),
            "top_offender_pct_of_cases": round(100 * doc_hit_count_name.most_common(1)[0][1] / (len(mismatch_name_cases) or 1), 1) if mismatch_name_cases else 0,
            "total_docs_with_missing_name": len(missing_name_ids),
            "cases": mismatch_name_cases[:20],
        },
        "gold_empty_passage_VÙNG_CHẾT_summary": {
            "count_questions": len(mismatch_passage_cases),
            "pct_of_train_questions": round(100 * len(mismatch_passage_cases) / (len(train) or 1), 2),
            "n_unique_docs_affected": n_passage_live,
            "pct_of_error_set_that_matters": round(100 * n_passage_live / (len(empty_passage_ids) or 1), 1),
            "top_offenders": doc_hit_count_passage.most_common(5),
            "top_offender_pct_of_cases": round(100 * doc_hit_count_passage.most_common(1)[0][1] / (len(mismatch_passage_cases) or 1), 1) if mismatch_passage_cases else 0,
            "total_docs_with_empty_passage": len(empty_passage_ids),
            "cases": mismatch_passage_cases[:20],
        },
    }

# ---------------------------------------------------------------------------
# 10. Đối chiếu Train vs Public - kiểm tra qua trùng qid
# ---------------------------------------------------------------------------

def compare_train_public_stats(docs: list[dict], train: dict[str, dict], public_path: Path) -> dict:
    if not public_path.exists():
        return {"status": f"Chưa nạp {public_path.name}. Đặt file vào data/ để chạy đối chiếu."}

    try:
        with open(public_path, "r", encoding="utf-8") as f:
            public_data = json.load(f)
    except Exception as e:
        return {"status": f"Lỗi đọc {public_path.name}: {e}"}

    train_qids = set(train.keys())
    public_qids = set(public_data.keys())
    overlap_qids = train_qids & public_qids

    empty_passage_ids = {str(d.get("id")) for d in docs if "passage" in d and not d.get("passage")}

    # Nếu qid trùng, gold thật lấy từ train (vì public.answer luôn null)
    cases_pointing_to_broken = []
    for qid in overlap_qids:
        train_answers = [str(a) for a in (train[qid].get("answer") or [])]
        broken = [a for a in train_answers if a in empty_passage_ids]
        if broken:
            cases_pointing_to_broken.append({
                "qid": qid,
                "gold_from_train": train_answers,
                "broken_ids": broken,
            })

    return {
        "n_train_qids": len(train_qids),
        "n_public_qids": len(public_qids),
        "n_overlap_qids": len(overlap_qids),
        "overlap_qids_sample": sorted(overlap_qids)[:10],
        "interpretation": (
            "0 overlap KHÔNG bác bỏ giả thuyết 5.427-tài-liệu - chỉ nghĩa là train/public "
            "dùng không gian qid tách biệt, không thể kiểm chứng bằng cách này."
            if not overlap_qids else
            "Có overlap qid - dùng gold thật từ train để soi câu hỏi đó trong public."
        ),
        "overlap_pointing_to_broken_doc": {
            "count": len(cases_pointing_to_broken),
            "cases": cases_pointing_to_broken[:10],
        },
    }

# ---------------------------------------------------------------------------
# 11. Chi tiết outlier độ dài (tra tay xác định bug hay dữ liệu thật)
# ---------------------------------------------------------------------------

def top_length_outliers(docs: list[dict], top_n: int = 5) -> dict:
    """Không kết luận đúng/sai - chỉ trích đủ thông tin để tra tay nhanh,
    tránh phải tự grep tìm outlier mỗi lần chạy lại EDA."""
    enriched = []
    for d in docs:
        text = d.get("passage") or ""
        n_words = len(text.split())
        dieu_matches = DIEU_PATTERN.findall(text)
        enriched.append({
            "id": str(d.get("id")),
            "source_file": d.get("_source_file"),
            "n_words": n_words,
            "has_dieu": bool(dieu_matches),
            "n_dieu": len(dieu_matches),
            "words_per_dieu_gap": round(n_words / max(len(dieu_matches) - 1, 1), 1) if dieu_matches else None,
            "link": d.get("link") or "",
            "text_head": text[:200],
            "text_tail": text[-200:] if len(text) > 200 else "",
        })
    enriched.sort(key=lambda x: x["n_words"], reverse=True)
    top = enriched[:top_n]
    return {
        "top_n": top_n,
        "outliers": top,
        "note": (
            "text_head/text_tail để soi nhanh dấu hiệu lỗi parser (nội dung lặp, "
            "nhiều văn bản dính liền) mà không cần mở file gốc. has_dieu/n_dieu "
            "để biết outlier rơi vào nhóm nào ở mục 2 (ảnh hưởng ước lượng "
            "sliding-window cho 8.7% văn bản fallback)."
        ),
    }

### ---------------------------------------------------------------------------
### 12. Kiểm tra ô nhiễm dữ liệu do lỗi Crawler (Tường bảo mật)
### ---------------------------------------------------------------------------

def check_crawler_pollution(docs: list[dict]) -> dict:
    """
    Rà soát và đếm số lượng file dính lỗi cào rác (Crawler Pollution) chứa thông báo bảo mật.
    """
    infected_files = []
    keywords = ["rò rỉ mật khẩu", "đăng nhập", "quý khách"]
    
    for d in docs:
        text = (d.get("passage") or "").strip().lower()
        if any(kw in text for kw in keywords):
            infected_files.append(d.get("_source_file"))
            
    return {
        "total_infected_files": len(infected_files),
        "percentage_infected": round(100 * len(infected_files) / (len(docs) or 1), 2),
        "eg_first_10": sorted(infected_files)[:10]
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_markdown(results: dict) -> str:
    lines = ["# EDA Notes - chạy trước khi chunker/parser bị chỉnh lần cuối", ""]
    lines.append("> Sinh bởi `scripts/eda.py`. Điền thủ công phần nhận xét sau mỗi mục.")
    lines.append("")
    for section, data in results.items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("**Nhận xét:** _(điền)_")
        lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/selected-contexts"))
    ap.add_argument("--train", type=Path, default=Path("data/train.json"))
    ap.add_argument("--public", type=Path, default=Path("data/public-official.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/eda_notes.md"))
    args = ap.parse_args()

    print(f"Đang đọc corpus từ {args.corpus_dir} ...")
    docs = load_corpus(args.corpus_dir)
    print(f"  → {len(docs)} văn bản")

    print(f"Đang đọc {args.train} ...")
    train = load_train(args.train)
    print(f"  → {len(train)} câu hỏi")

    results = {
        "1. Phân bố độ dài văn bản": doc_length_stats(docs),
        "2. Cấu trúc Điều N vs fallback": dieu_structure_stats(docs),
        "3. Trường thiếu (id/name/link/text)": missing_field_stats(docs),
        "4. Phân bố số đáp án / câu hỏi": answer_count_stats(train),
        "5. Trùng / gần trùng văn bản": near_duplicate_stats(docs, train),
        "6. Độ dài câu hỏi": question_length_stats(train),
        "7. Độ phủ doc_id (train vs corpus)": doc_id_coverage_stats(docs, train),
        "8. Mối quan hệ lỗi (3.1)": check_name_passage_overlap(docs),
        "9. Kiểm chứng Gold ID trỏ vào file lỗi": verify_gold_with_missing_fields(docs, train),
        "10. Đối chiếu Train vs Public": compare_train_public_stats(docs, train, args.public),
        "11. Chi tiết outlier độ dài (Tra cứu thủ công)": top_length_outliers(docs, top_n=5),
        "12. Kiểm tra ô nhiễm dữ liệu do lỗi Crawler": check_crawler_pollution(docs),
    }

    print("\n=== TÓM TẮT ===")
    for section, data in results.items():
        print(f"\n{section}")
        for k, v in data.items():
            if isinstance(v, list):
                continue  # danh sách ví dụ chỉ in trong file markdown, không spam stdout
            print(f"  {k}: {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(results), encoding="utf-8")
    print(f"\nĐã ghi báo cáo: {args.out}")

    # Xuất cụm trùng ra file riêng để split_data.py đọc lại
    dup_clusters = results["5. Trùng / gần trùng văn bản"]["dup_passage_groups"]
    dup_clusters_out = Path("data/dup_clusters.json")
    dup_clusters_out.parent.mkdir(parents=True, exist_ok=True)
    with open(dup_clusters_out, "w", encoding="utf-8") as f:
        json.dump(dup_clusters, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {len(dup_clusters)} cụm trùng ra: {dup_clusters_out}")

    n_orphan = results["7. Độ phủ doc_id (train vs corpus)"]["n_orphan_ids_in_train"]
    if n_orphan > 0:
        print(
            f"\n  CẢNH BÁO: {n_orphan} doc_id trong train.json không tồn tại trong corpus. "
            "Kiểm tra ngay - có thể là bẫy str/int (xem INTERFACES.md mục 0) hoặc thiếu file corpus."
        )


if __name__ == "__main__":
    main()
