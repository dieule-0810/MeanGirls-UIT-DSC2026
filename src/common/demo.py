"""Corpus giả cho chế độ `--demo` của các script — chạy được trên máy sạch, không cần `data/`.

KHÔNG phải benchmark: con số sinh ra từ đây chỉ chứng minh đường ống chạy đúng định dạng,
không được ghi vào `experiments.csv`.
"""
from __future__ import annotations

import random

_LINH_VUC = [
    ("giấy phép lái xe", "sở giao thông vận tải", "cấp đổi"),
    ("hoá đơn điện tử", "cơ quan thuế", "huỷ bỏ"),
    ("giấy chứng nhận quyền sử dụng đất", "uỷ ban nhân dân cấp huyện", "thu hồi"),
    ("đăng ký kinh doanh", "phòng đăng ký kinh doanh", "đình chỉ"),
    ("an toàn thực phẩm", "bộ y tế", "kiểm tra"),
    ("bảo hiểm xã hội bắt buộc", "cơ quan bảo hiểm xã hội", "truy thu"),
    ("phòng cháy chữa cháy", "cơ quan công an", "thẩm duyệt"),
    ("xử phạt vi phạm hành chính", "chủ tịch uỷ ban nhân dân", "ra quyết định"),
    ("đấu thầu qua mạng", "bên mời thầu", "huỷ thầu"),
    ("chứng chỉ hành nghề xây dựng", "sở xây dựng", "cấp lại"),
]
_BOILERPLATE = [
    "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015",
    "Thông tư này quy định chi tiết một số điều của Nghị định số 15/2020/NĐ-CP",
    "Các quy định trước đây trái với Thông tư này đều bị bãi bỏ",
    "Trong quá trình thực hiện, nếu có vướng mắc, đề nghị phản ánh về Bộ để xem xét",
    "Thông tư này có hiệu lực thi hành kể từ ngày ký ban hành",
]
# Chứa đúng âm tiết câu hỏi nhưng ở từ ghép khác ("cơ sở"+"quan hệ" ≠ "cơ quan") — kịch bản
# word-segment/bigram thắng âm tiết thuần, và pool=sum thua vì thiên vị văn bản dài.
_NHIEU = [
    "cơ sở dữ liệu; quan hệ lao động; giấy tờ tuỳ thân; phép đo lường",
    "thẩm định giá; quyền sở hữu; hoá chất công nghiệp; đơn vị sự nghiệp",
    "lái tàu đường sắt; xe máy chuyên dùng; điện lực; tử tuất",
    "uỷ thác đầu tư; ban quản lý dự án; nhân sự; dân sinh",
    "thu nhập cá nhân; hồi tố; kinh tế tập thể; doanh trại",
]


def tiny_corpus() -> tuple[list[dict], dict]:
    """Ba văn bản, ba câu hỏi — đủ để chạy hết đường ống trong một giây.

    Returns:
        `(chunks, questions)` với `questions = {qid: {"question", "answer"}}`.
    """
    docs = {
        "740": ["Điều 1. Cơ quan thuế huỷ bỏ hoá đơn điện tử đã lập sai.",
                "Điều 2. Việc huỷ hoá đơn điện tử phải lập biên bản theo Thông tư 78/2021/TT-BTC."],
        "812": ["Điều 1. Cơ sở dữ liệu quốc gia về dân cư do Bộ Công an quản lý.",
                "Điều 2. Quan hệ lao động giữa người sử dụng lao động và người lao động."],
        "915": ["Điều 1. Mức đóng bảo hiểm xã hội bắt buộc của người lao động.",
                "Điều 2. Thời gian hưởng chế độ thai sản theo Luật Bảo hiểm xã hội."],
    }
    chunks = [
        {"chunk_id": f"{d}::{i:04d}", "doc_id": d, "position": i, "text": t}
        for d, texts in docs.items()
        for i, t in enumerate(texts)
    ]
    questions = {
        "1": {"question": "Huỷ hoá đơn điện tử lập sai thì làm thế nào?", "answer": ["740"]},
        "2": {"question": "Cơ sở dữ liệu quốc gia về dân cư do ai quản lý?", "answer": ["812"]},
        "3": {"question": "Thời gian hưởng chế độ thai sản là bao lâu?", "answer": ["915"]},
    }
    return chunks, questions


def _relevant_doc(rng: random.Random, d: int) -> tuple[list[dict], str, str]:
    """Một văn bản có câu hỏi: trả `(chunks, doc_id, câu hỏi)`."""
    doc_id = str(100000 + d * 7)
    chu_de, co_quan, hanh_vi = _LINH_VUC[d % len(_LINH_VUC)]
    bien_the = f"{chu_de} thuộc nhóm {d % 5 + 1}"
    chunks = []
    for pos in range(rng.randint(2, 6)):
        body = rng.choice(_BOILERPLATE)
        if pos == 0:
            body = f"{co_quan} có thẩm quyền {hanh_vi} đối với {bien_the} theo quy định tại Điều {pos + 1}. {body}"
        chunks.append({
            "chunk_id": f"{doc_id}::{pos:04d}", "doc_id": doc_id, "position": pos,
            "text": f"Điều {pos + 1}. Quy định về {chu_de}. {body}",
        })
    # Dấu thanh đặt kiểu khác văn bản ("uỷ" ↔ "ủy") để phần chuẩn hoá có việc thật mà làm.
    question = (
        f"Cơ quan nào có thẩm quyền {hanh_vi.replace('uỷ', 'ủy')} đối với "
        f"{bien_the.replace('uỷ', 'ủy').replace('hoá', 'hóa')}?"
    )
    return chunks, doc_id, question


def _noise_doc(rng: random.Random, j: int) -> list[dict]:
    """Một văn bản nhiễu dài, không ứng với câu hỏi nào."""
    doc_id = str(900000 + j)
    return [
        {"chunk_id": f"{doc_id}::{pos:04d}", "doc_id": doc_id, "position": pos,
         "text": f"Điều {pos + 1}. {rng.choice(_NHIEU)}. {rng.choice(_BOILERPLATE)}"}
        for pos in range(rng.randint(8, 14))
    ]


def synthetic_corpus(n_docs: int = 60, seed: int = 42) -> tuple[list[dict], dict[str, str], dict[str, list[str]]]:
    """Corpus giả có nhiễu, đủ để lưới bench không toàn số 1,000.

    Args:
        n_docs: Số văn bản có câu hỏi; thêm `n_docs // 3` văn bản nhiễu.
        seed: Hạt giống ngẫu nhiên.

    Returns:
        `(chunks, questions, gold)` — `questions = {qid: câu hỏi}`, `gold = {qid: [doc_id]}`.
    """
    rng = random.Random(seed)
    chunks: list[dict] = []
    questions: dict[str, str] = {}
    gold: dict[str, list[str]] = {}
    for d in range(n_docs):
        doc_chunks, doc_id, question = _relevant_doc(rng, d)
        chunks.extend(doc_chunks)
        questions[f"demo_{d:03d}"] = question
        gold[f"demo_{d:03d}"] = [doc_id]
    for j in range(n_docs // 3):
        chunks.extend(_noise_doc(rng, j))
    return chunks, questions, gold
