# Question & Answer

## 1. Về chỉ số đánh giá

BTC xác nhận lại đối với task LegalIR, Recall là metric chính. Nếu các đội có Recall bằng nhau, thứ hạng sẽ được xác định thông qua Precision.

Tuy nhiên, sau khi kiểm tra và thẩm định hệ thống, BTC nhận thấy có khả năng submit toàn bộ document-id cho tất cả các câu hỏi để lấy điểm Recall cao nhất. Khi đó, những nhóm thực sự chạy các thuật toán truy vấn văn bản nhưng có Recall thấp hơn, dù Precision có cao hơn, vẫn bị xếp thứ hạng thấp hơn.

Dựa trên đặc điểm của dữ liệu, BTC quyết định thêm ràng buộc vào mã nguồn của chương trình đánh giá kết quả: những submission nào có bất kỳ question nào nhiều hơn 5 answer thì kết quả của submission sẽ là 0 cho cả Recall lẫn Precision. Tức là một question chỉ được có tối đa 5 answer.

## 2. Về giới hạn số lượng tham số

BTC đặt ra quy định tất cả các mô hình/hệ thống dự thi phải có số lượng tham số dưới 4 tỷ. Điều này đồng nghĩa với việc giới hạn 4 tỷ tham số được tính cho tất cả các thành phần trong mô hình/hệ thống, dĩ nhiên bao gồm cả lớp embedding.

Quy định này áp dụng cho cả hai task, tức là trong từng task, mọi mô hình/hệ thống phải có tổng số tham số dưới 4 tỷ.

Các nhóm cần phân biệt rõ một khái niệm: các mô hình kích thước nhỏ được tạo ra từ kỹ thuật chuyển giao thông tin từ các mô hình kích thước lớn hơn (distillation), nếu bản thân mô hình sau khi distillation có tổng số tham số dưới 4 tỷ thì vẫn được phép sử dụng.

Ngoài ra, một số nhóm có thắc mắc về việc sử dụng các kỹ thuật tối ưu không gian bộ nhớ như LoRA, quantization, ... để các mô hình lớn hơn 4 tỷ tham số vận hành tương đương mô hình ít hơn 4 tỷ tham số về mặt lưu trữ và vận hành.

Các mô hình này về bản chất vẫn có hơn 4 tỷ tham số. Việc sau khi tối ưu không gian mà mô hình chạy chiếm tài nguyên tương đương mô hình ít hơn 4 tỷ tham số là do các kỹ thuật giảm số bit lưu trữ cho từng tham số, vốn không làm thay đổi số lượng tham số của bản thân mô hình.

Việc sử dụng mô hình nhiều hơn 4 tỷ tham số đồng nghĩa với việc sử dụng lượng thông tin pretrained nhiều hơn, gây bất công bằng giữa các đội thi và không phù hợp với mục tiêu của BTC.

## 3. Về API thương mại

Cuộc thi KHDL UIT 2026 là cuộc thi học thuật, sử dụng các thực nghiệm của các đội dự thi để kiểm nghiệm các câu hỏi mà BTC đặt ra, phục vụ hoàn toàn cho khoa học và giáo dục.

Do đó, mọi hình thức sử dụng sản phẩm thương mại đều không được phép trong khuôn khổ cuộc thi. Tất cả các mô hình/hệ thống có giấy phép sử dụng phi thương mại, sử dụng cho mục đích giáo dục/khoa học phi lợi nhuận đều được chấp nhận.

Về API, tất cả các đội thi không được sử dụng API, kể cả API phi lợi nhuận, trong quá trình xây dựng phương pháp.

BTC chỉ chấp nhận các hệ thống/mô hình mã nguồn mở mà đội thi có thể tải về, vận hành và kiểm soát trực tiếp, không thông qua một bên trung gian.

## 4. Về dữ liệu

BTC sẽ cung cấp dữ liệu huấn luyện cho cả 2 tác vụ. Tất cả các đội dự thi chỉ được sử dụng dữ liệu do BTC cung cấp, không được sử dụng dữ liệu từ nguồn khác, cũng như không được sử dụng các kỹ thuật tăng cường dữ liệu.

Bên cạnh đó, các đội cần phân định rõ rằng dữ liệu được sử dụng để xây dựng pretrained models/LLMs không được xem là dữ liệu từ nguồn khác. Các đội sử dụng pretrained models/LLMs chỉ đang sử dụng các mô hình ước lượng được huấn luyện để xấp xỉ các phân phối thể hiện thông qua dữ liệu, chứ các đội không trực tiếp sử dụng những bộ ngữ liệu đó.

## 5. Về đóng gói mô hình và tái lập thực nghiệm

Trong quy định BTC có đề cập sử dụng Docker. Tuy nhiên, ngoài Docker, BTC cho phép các nhóm sử dụng nhiều hình thức khác như commit và push code lên GitHub, nén mã nguồn/trọng số thành file ZIP và gửi về cho BTC, ...

Bất kể hình thức nào đều được chấp nhận miễn là có thể đóng gói và gửi cho BTC thẩm định.

Về việc trọng số mô hình phải được nén để khởi chạy thực nghiệm trực tiếp và offline hay có thể download từ Internet, cũng như việc môi trường và cấu hình khi sử dụng Docker có cần theo một quy định nhất quán hay không: tất cả các đội thi được tự do quyết định cách đóng gói.

Điều kiện là trong README hoặc tài liệu phải trình bày chi tiết từng bước tái lập thực nghiệm để BTC có thể thực thi theo và trả về kết quả.

Việc truy cập Internet để tải trọng số vẫn hợp lệ, miễn trọng số thuộc các mô hình/hệ thống mã nguồn mở, phi thương mại hoặc được sử dụng cho mục đích giáo dục/nghiên cứu.

## 6. Về đăng ký mô hình tiền huấn luyện

Danh sách các mô hình đã được BTC phê duyệt sẽ được công khai để tất cả các đội có thể theo dõi.

Các đội chỉ được phép sử dụng các mô hình đã đăng ký và được BTC phê duyệt.

BTC sẽ cập nhật danh sách này thường xuyên sau khi xét duyệt. Các đội vui lòng theo dõi định kỳ để đảm bảo thông tin luôn được cập nhật và không đăng ký trùng các mô hình đã được duyệt.

## 7. Về quy định mô hình

Các mô hình được đăng ký và sử dụng trong cuộc thi phải tuân thủ đầy đủ các quy định sau:

- Tổng số lượng tham số của toàn bộ hệ thống sử dụng trong mỗi bài dự thi phải dưới 4 tỷ tham số.
- Giới hạn này được tính trên tất cả các thành phần của hệ thống, bao gồm nhưng không giới hạn ở mô hình sinh, mô hình embedding, reranker hoặc các mô hình khác nếu được sử dụng trong pipeline.
- Các mô hình được tạo bằng kỹ thuật distillation vẫn được phép sử dụng nếu bản thân mô hình/hệ thống sau khi distillation có tổng số tham số dưới 4 tỷ.
- Các kỹ thuật tối ưu bộ nhớ như LoRA, Quantization, GPTQ, AWQ, GGUF hoặc các kỹ thuật tương tự không làm thay đổi số lượng tham số của mô hình.
- Vì vậy, các mô hình/hệ thống có trên 4 tỷ tham số, dù đã được quantize hoặc tối ưu để giảm dung lượng lưu trữ hay tài nguyên vận hành, vẫn không được phép sử dụng.

## 8. Về quy định API và giấy phép sử dụng

UIT Data Science Challenge 2026 là cuộc thi học thuật phục vụ mục tiêu nghiên cứu và giáo dục. Do đó:

- Không được phép sử dụng bất kỳ API nào trong quá trình xây dựng và phát triển hệ thống, bao gồm cả API thương mại và API phi thương mại.
- BTC chỉ chấp nhận các mô hình/hệ thống mã nguồn mở mà đội thi có thể tải về, vận hành và kiểm soát trực tiếp, không thông qua dịch vụ của bên thứ ba.
- Các mô hình có giấy phép sử dụng cho mục đích nghiên cứu, giáo dục hoặc phi thương mại được phép sử dụng nếu đáp ứng đầy đủ các quy định khác của cuộc thi.

## 9. Về lưu ý khi sử dụng mô hình

Mọi mô hình sử dụng trong cuộc thi phải tuân thủ đầy đủ Thể lệ cuộc thi.

Chỉ các mô hình đã đăng ký và được BTC phê duyệt mới được sử dụng hợp lệ.

Các bài nộp sử dụng mô hình chưa được đăng ký hoặc chưa được BTC phê duyệt sẽ không được công nhận.

Các mô hình đã được BTC phê duyệt không cần đăng ký lại. Đội thi chỉ cần theo dõi danh sách để xác nhận trạng thái phê duyệt.

Link danh sách mô hình đã được BTC phê duyệt (do nhóm trưởng đăng ký): [DSC@UIT 2026] Danh sách mô hình

## 10. Về công bố khoa học

Như BTC đã công bố, các đội có thứ hạng trong cuộc thi sẽ được mời viết bài nghiên cứu công bố phương pháp của đội trong cuộc thi. Bài báo sẽ được phản biện và được đăng, nếu được chấp nhận đăng, trong Tạp chí Phát triển Khoa học và Công nghệ ĐHQG-HCM.

Do đây sẽ là một bài nghiên cứu được công bố ở tạp chí, BTC cần các nhóm lưu ý những điều sau:

- Nên có một giả thiết/giả định cho việc giải quyết tác vụ của cuộc thi. Sau đó sử dụng thực nghiệm để kiểm nghiệm giả thiết/giả định đó.
- Quá trình thực nghiệm nên có đầy đủ các kịch bản để kiểm nghiệm giả thiết/giả định mà các đội đang theo đuổi.
- Nên có những số liệu/phân tích cho kết quả của các phương pháp đã thử.
- Cần phân tích vì sao một phương pháp chưa tốt, chưa tốt ở điểm nào, và phương pháp sau tốt hơn vì đã khắc phục được điểm yếu gì của phương pháp trước.

Câu hỏi mà BTC đặt ra cho các tác vụ của cuộc thi năm nay đang chờ câu trả lời là: với nguồn tài nguyên tính toán hạn chế (chỉ vận hành được hệ thống có dưới 4 tỷ tham số), và lượng dữ liệu khiêm tốn (khoảng 10k điểm dữ liệu cho mỗi tác vụ), các phương pháp deep learning thuần túy có hiệu quả ra sao so với các phương pháp tận dụng sức mạnh của các mô hình ngôn ngữ lớn?

Hướng tiếp cận không giới hạn ở mô hình/hệ thống mà còn có thể mở rộng ra cho các chiến lược xử lý dữ liệu, trong khuôn khổ quy định mà BTC đã ghi, bao gồm không data augmentation và không sử dụng dữ liệu ngoài.

## 11. Về mô hình Deep Learning train from scratch

BTC xác nhận rằng mô hình tự xây dựng và train from scratch không được xem là pretrained model, do đó không thuộc diện phải đăng ký và chờ phê duyệt như các pretrained model có sẵn.

Đội được tự do thiết kế và phát triển kiến trúc Deep Learning, khởi tạo trọng số ngẫu nhiên và huấn luyện mô hình từ đầu để phục vụ thực nghiệm.

Vì vậy, nhóm có thể triển khai baseline DL train from scratch mà không cần đăng ký mô hình trước với BTC.

## 12. Về context trùng/rỗng passage

BTC đã kiểm tra qua các tập train, public test và private test.

Hiện đúng là trên tập train đang có một số câu hỏi có passage rỗng hoặc trùng.

Trên tập public test và private test, đáp án không có hiện tượng context trùng hay rỗng.

Các nhóm nên tiến hành tiền xử lý kỹ trên tập train.
