"""
Gắn tiêu đề văn bản vào chunk — CHỦ SỞ HỮU: P3. Làm ở thời điểm index, KHÔNG đụng chunker của P2.

Giả thuyết H6: trung vị 36 chunk/văn bản, mà chỉ chunk đầu chứa tiêu đề ⇒ ~97% chunk không biết
mình thuộc văn bản nào. Trả lại danh tính đó cho mọi chunk.

Cơ chế phản tác dụng phải theo dõi cùng lúc: thêm CÙNG một chuỗi vào 36 chunk làm df của các term
đó tăng vọt ⇒ IDF sụp, và khớp cấp CHỦ ĐỀ lấn át khớp cấp ĐIỀU KHOẢN (văn bản đúng chủ đề nhưng
không chứa câu trả lời vẫn được `mean_topN` đẩy lên). Dấu của hiệu ứng không đoán được — phải đo
bằng `scripts/ab_prepend_title.py`.

KHÔNG dùng trường `name`: đo trên 800 văn bản thì 0,0% có dấu tiếng Việt (nó là slug URL,
`Thong-tu-17-2022-TT-BGTVT-...`) và 12,5% rỗng. Câu hỏi viết "thông tư", slug viết "thong-tu" —
hai token khác nhau, không bao giờ khớp. Tiêu đề thật nằm trong chính `text` (85% văn bản có).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.common.io import read_jsonl

# Loại văn bản, viết hoa trong bản gốc. Đặt cụm dài trước để không khớp nhầm phần đầu của nó.
DOC_TYPES = (
    "THÔNG TƯ LIÊN TỊCH", "THÔNG TƯ", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "QUYẾT ĐỊNH",
    "CHỈ THỊ", "PHÁP LỆNH", "LUẬT", "BỘ LUẬT", "HIẾN PHÁP",
    "QUY CHUẨN KỸ THUẬT QUỐC GIA", "TIÊU CHUẨN QUỐC GIA", "CÔNG VĂN", "KẾ HOẠCH",
)
_TYPE_RE = re.compile("|".join(re.escape(t) for t in DOC_TYPES))
# [A-ZĐ]{2,}[a-z]* — hậu tố chữ thường có thật và hay gặp: QĐ-TTg, NQ-HĐND.
_CODE_RE = re.compile(
    r"Số:\s*([0-9]{1,5}[A-ZĐ]*\s*/\s*(?:[0-9]{4}\s*/\s*)?[A-ZĐ]{2,}[a-z]*(?:\s*-\s*[A-ZĐ]+[a-z]*)*)"
)
# Điểm dừng của tiêu đề: phần căn cứ pháp lý, điều khoản, hoặc chương đầu tiên.
_STOP_RE = re.compile(r"\b(Căn cứ|CĂN CỨ|Điều\s+1\b|ĐIỀU\s+1\b|Chương\s+I\b|CHƯƠNG\s+I\b|Xét đề nghị)")
# Quốc hiệu + tiêu ngữ: có ở MỌI văn bản ⇒ thêm vào mọi chunk là phá IDF toàn corpus.
_BOILER_RE = re.compile(
    r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM|Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc|-{3,}|_{3,}"
)

HEAD_CHARS = 1500   # tiêu đề luôn nằm ở phần đầu; quét xa hơn chỉ tăng dương tính giả
MAX_WORDS = 30      # cắt ngắn để hạn chế phình độ dài chunk và hạn chế phá IDF


def extract_title(text: str, max_words: int = MAX_WORDS) -> str | None:
    """
    Trích `"<SỐ HIỆU> <LOẠI VĂN BẢN> <tên>"` từ phần đầu văn bản. Không thấy → None.

    Số hiệu đứng trước vì nó là phần PHÂN BIỆT nhất (IDF cao nhất); tên văn bản dùng chung
    rất nhiều cụm khuôn mẫu giữa các văn bản.
    """
    head = text[:HEAD_CHARS]
    code_m = _CODE_RE.search(head)
    # Loại văn bản đứng SAU dòng "Số:" — trước đó là tên cơ quan ban hành, không phải tiêu đề.
    search_from = code_m.end() if code_m else 0
    type_m = _TYPE_RE.search(head, search_from)
    if type_m is None:
        return None

    tail = head[type_m.start():]
    stop = _STOP_RE.search(tail)
    if stop:
        tail = tail[: stop.start()]
    tail = _BOILER_RE.sub(" ", tail)
    words = tail.split()[:max_words]
    if not words:
        return None

    code = re.sub(r"\s+", "", code_m.group(1)) if code_m else ""
    title = " ".join(words).strip(" .,;:-")
    return f"{code} {title}".strip() if code else title


def title_map(corpus_path: str | Path, max_words: int = MAX_WORDS) -> dict[str, str]:
    """`{doc_id: tiêu đề}` cho các văn bản trích được. Văn bản không trích được thì vắng mặt."""
    out: dict[str, str] = {}
    for d in read_jsonl(corpus_path):
        doc_id = str(d.get("doc_id", d.get("id")))
        t = extract_title(d.get("text", d.get("passage", "")) or "", max_words)
        if t:
            out[doc_id] = t
    return out


def prepend_titles(
    chunks: list[dict], titles: dict[str, str], skip_if_present: bool = True
) -> tuple[list[dict], dict]:
    """
    Trả về (chunk mới, thống kê). KHÔNG sửa tại chỗ — người gọi giữ nguyên bản gốc để A/B.

    `skip_if_present`: chunk đã chứa sẵn tiêu đề (thường là chunk đầu văn bản) thì không gắn lại,
    tránh nhân đôi tf của chính những term ta đang lo là sẽ mất giá trị.
    """
    out, n_add, n_skip, n_no_title = [], 0, 0, 0
    for c in chunks:
        t = titles.get(str(c["doc_id"]))
        if not t:
            n_no_title += 1
            out.append(c)
            continue
        if skip_if_present and t[:40] in c["text"][:300]:
            n_skip += 1
            out.append(c)
            continue
        n_add += 1
        out.append({**c, "text": f"{t}. {c['text']}"})
    stats = {
        "n_chunks": len(chunks),
        "n_prepended": n_add,
        "n_already_had": n_skip,
        "n_no_title": n_no_title,
        "pct_prepended": round(n_add / max(len(chunks), 1), 4),
        "n_docs_with_title": len(titles),
    }
    return out, stats
