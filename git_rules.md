### Hướng dẫn Đóng góp (Contributing Guide)

Chào mừng bạn tham gia phát triển dự án! Để giữ cho mã nguồn luôn sạch sẽ, dễ bảo trì và tránh xung đột khi làm việc nhóm, vui lòng tuân thủ các quy định dưới đây. 

### 1. Quy trình Làm việc với Nhánh (Git Workflow)

Dự án áp dụng quy trình **GitHub Flow**. 

1. Luôn cập nhật code mới nhất từ nhánh main trước khi làm việc: 

bash

git checkout main
git pull origin main

Use code with caution.
2. Tạo nhánh mới từ main theo cú pháp: 

  * Tính năng mới: feature/ten-tinh-nang (Ví dụ: feature/shopping-cart)
  * Sửa lỗi: bugfix/ten-loi (Ví dụ: bugfix/login-crash)
3. Sau khi hoàn thành, đẩy nhánh lên remote và tạo **Pull Request (PR)** về nhánh main.
4. Gắn thẻ ít nhất 1 thành viên khác vào để **Review Code**. Chỉ khi được chấp thuận (Approve) mới được phép Merge.

### 2. Quy chuẩn Commit Message

Chúng ta sử dụng chuẩn **Conventional Commits**. Định dạng bắt buộc: 

text

<type>(<scope>): <subject>

Use code with caution.

### Các Loại Commit (type)

* **feat**: Thêm một tính năng mới cho hệ thống.
* **fix**: Sửa một lỗi (bug) vừa phát hiện.
* **docs**: Thay đổi liên quan đến tài liệu (README, API doc, comment...).
* **style**: Thay đổi về định dạng code (khoảng trắng, format, dấu chấm phẩy) - không đổi logic.
* **refactor**: Sửa đổi code để tối ưu cấu trúc nhưng không thêm tính năng hay sửa lỗi.
* **test**: Thêm code kiểm thử tự động (Unit test, Integration test).
* **chore**: Cập nhật các tác vụ nhỏ, file cấu hình hệ thống, cài thêm thư viện.

### Ví dụ Hợp lệ

* feat(auth): thêm chức năng đăng nhập bằng Google
* fix(cart): sửa lỗi không cập nhật số lượng khi bấm tăng
* docs(readme): cập nhật hướng dẫn cài đặt môi trường chạy local
* style(ui): định dạng lại căn lề cho trang chủ

### 3. Lưu ý quan trọng để tránh Xung đột (Conflict)

* **Commit nhỏ và thường xuyên:** Đừng gom quá nhiều tính năng vào 1 commit lớn hoặc 1 PR lớn. Hãy chia nhỏ tác vụ để dễ quản lý và dễ review.
* **Giải quyết xung đột:** Nếu xảy ra conflict khi merge, tuyệt đối không tự ý xóa code của người khác nếu chưa thảo luận với họ.
* **Giữ code sạch:** Chạy công cụ linter/formatter trước khi commit để đảm bảo định dạng nhất quán với toàn đội.