#!/usr/bin/env bash
# Smoke test 2 phút. BTC chạy cái này TRƯỚC khi chạy full 1000 câu.
# Môi trường sai thì biết sau 2 phút, không phải sau 3 tiếng.
set -euo pipefail

echo "══ Smoke test ══"
python -m src.verify_env
python -m pytest tests/ -q
echo -e "\n✅ PASS — môi trường và logic chấm điểm đều đúng."
