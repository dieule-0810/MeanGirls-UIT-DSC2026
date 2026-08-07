# Dockerfile — KHÔNG phải để gửi image cho BTC (đã chốt: nộp GitHub + README theo Q&A).
# Lý do tồn tại: đây là cách ĐẶC TẢ môi trường chính xác mà requirements.txt không diễn đạt được
# (base image, phiên bản Python, thư viện hệ thống). README có chỗ mơ hồ thì file này là bản mô tả không mơ hồ.
#
# ⚠️ BẪY ARM64 — máy M4 Pro build mặc định ra linux/arm64, BTC chạy x86_64.
#    Image sẽ không khởi động ở phía họ mà team KHÔNG phát hiện ra vì máy mình chạy hoàn hảo.
#    BẮT BUỘC build bằng:
#        docker buildx build --platform linux/amd64 -f docker/Dockerfile -t dsc2026:v0.1 .
#    Kiểm chứng ngay ở v0.1 khi image còn nhỏ, không để đến v1.0.

FROM --platform=linux/amd64 python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Thư viện hệ thống trước, để tận dụng cache layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# data/ và checkpoints/ được mount lúc chạy, KHÔNG bake vào image
# (dữ liệu là của BTC; trọng số làm image phình lên nhiều GB)
VOLUME ["/app/data", "/app/outputs"]

# Smoke test chạy được ngay lúc build → image hỏng thì biết tại chỗ
RUN python -m src.verify_env && python -m pytest tests/ -q

CMD ["bash", "scripts/run_v0.1.sh"]
