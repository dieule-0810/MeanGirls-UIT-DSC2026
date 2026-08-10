### Hướng dẫn Đóng góp (Contributing Guide)

Để giữ code sạch, dễ bảo trì và tránh xung đột khi 4 người làm song song, vui lòng tuân thủ các quy định dưới đây. Đi cùng `INTERFACES.md` (ai chạm file nào).

### 1. Quy trình Làm việc với Nhánh (Git Workflow)

Dự án áp dụng quy trình **GitHub Flow**.

1. Luôn cập nhật code mới nhất từ nhánh `main` trước khi làm việc:

```bash
git checkout main
git pull origin main
```

2. Tạo nhánh mới từ `main` theo cú pháp `<role>/<mo-ta-ngan>` (role = `p1`/`p2`/`p3`/`p4`, khớp vai trò trong `plan.md`):

  * Tính năng mới: `p3/feature-ten-tinh-nang` (Ví dụ: `p3/feature-bi-encoder-scratch`)
  * Sửa lỗi: `p1/bugfix-ten-loi` (Ví dụ: `p1/bugfix-docid-int`)
  * Thí nghiệm: `p4/exp-ten-thi-nghiem` (Ví dụ: `p4/exp-reranker-zeroshot`)
3. Sau khi hoàn thành, đẩy nhánh lên remote và tạo **Pull Request (PR)** về nhánh `main`. Không push thẳng lên `main`.
4. Gắn thẻ ít nhất 1 thành viên khác vào để **Review Code**. Nếu PR đụng vào `evaluate.py`, `make_submission.py`, `verify_env.py` thì bắt buộc P1 review. Chỉ khi được Approve mới được Merge.
5. Xoá nhánh sau khi merge.

### 2. Quy chuẩn Commit Message

Dùng chuẩn **Conventional Commits**. Định dạng bắt buộc:

```text
<type>(<scope>): <subject>
```

### Các loại Commit (type)

* **feat**: Thêm một tính năng mới cho hệ thống.
* **fix**: Sửa một lỗi (bug) vừa phát hiện.
* **exp**: Thêm/chạy một thí nghiệm, đổi config, cập nhật `experiments.csv`.
* **docs**: Thay đổi tài liệu (`plan.md`, `README.md`, `INTERFACES.md`, notes tuần).
* **style**: Thay đổi định dạng code, không đổi logic.
* **refactor**: Sửa cấu trúc code, không thêm tính năng hay sửa lỗi.
* **test**: Thêm/sửa test tự động.
* **chore**: Cập nhật cấu hình, cài thư viện, dọn file.

### Scope gợi ý

`data`, `retrieval`, `rerank`, `eval`, `submission`, `docker`, `docs`

### Ví dụ hợp lệ

* `feat(retrieval): thêm hybrid RRF gộp BM25 và dense`
* `fix(submission): ép str() cho doc_id tại điểm đọc corpus`
* `exp(retrieval): benchmark zero-shot bge-m3, exp_id=v0.2_bge_m3`
* `docs(plan): cập nhật lịch tuần 3 sau review hard negative`

### 3. Lưu ý quan trọng để tránh Xung đột (Conflict)

* **Commit nhỏ và thường xuyên:** đừng gom nhiều việc vào 1 commit/PR lớn.
* **Sửa file thư mục người khác** (`src/data/`, `src/retrieval/`, `src/rerank/`) → luôn qua PR, không tự merge.
* **`experiments.csv` hay conflict nhất** vì nhiều người cùng thêm dòng: `pull --rebase` trước khi thêm; nếu vẫn conflict, giữ cả hai dòng, không xoá dòng người khác.
* **`plan.md`/`INTERFACES.md` conflict** → mang ra standup, không tự ý resolve một mình — đây là hợp đồng chung cả team.
* Nếu xảy ra conflict, tuyệt đối không tự ý xóa code người khác nếu chưa thảo luận.

### 4. Tag & Submission

* Mỗi mốc release theo `plan.md` gắn tag `v0.1` → `v1.0` khi nghiệm thu.
* Mỗi lần nộp submission thật gắn tag riêng: `submission/<ngay>-<mo-ta>`, ghi cặp tag → điểm vào `experiments.csv`.
* Không sửa code sau khi tag trừ khi crash; không `git tag -f` đè tag cũ.

### 5. Không commit vào Git

Checkpoint model, `data/embeddings.npy`, corpus thô, `submission.zip`, API key — thêm vào `.gitignore`.