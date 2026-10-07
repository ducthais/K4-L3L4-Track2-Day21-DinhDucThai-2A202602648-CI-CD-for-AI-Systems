# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Đinh Đức Thái |
| MSSV | 2A202602648 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/ducthais/K4-L3L4-Track2-Day21-DinhDucThai-2A202602648-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số ở Lần 3 đạt điểm f1_score cao nhất (0.7149), vượt ngưỡng chất lượng 0.65 và vượt trội hơn so với Lần 1 (0.7109) cùng Lần 2 (0.6051). Đáng chú ý, lần chạy có accuracy cao nhất là Lần 1 (0.8780) chứ không phải Lần 3 (0.8740). Điều này phản ánh rõ ràng việc accuracy bị chi phối bởi lớp đa số (thu nhập thấp), trong khi f1_score đo lường chính xác hiệu quả phát hiện lớp dương. Khi giảm learning_rate xuống 0.05 và max_depth xuống 2 ở Lần 2, mô hình học không đủ sâu khiến f1_score giảm mạnh xuống 0.6051. Do đó, việc tăng số cây n_estimators lên 200 kết hợp max_depth=5 đem lại khả năng phân loại tốt nhất cho bài toán.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng lớn khi lớp thu nhập cao (>50K) chỉ chiếm khoảng 24.8% số mẫu. Một mô hình ngây thơ luôn dự đoán "thu nhập thấp" cho mọi trường hợp vẫn đạt accuracy lên tới 75.2%, tạo ra con số gây hiểu nhầm dù mô hình không học được bất kỳ thông tin nào. F1-score của lớp dương là trung bình điều hòa giữa precision và recall, đánh giá trực tiếp độ chính xác và mức độ bao phủ của mô hình trên lớp thiểu số quan trọng. Chúng ta không truyền average="macro" hay average="weighted" khi gọi f1_score vì các tùy chọn này sẽ tính trung bình có trọng số với lớp đa số, làm tăng điểm số ảo và che giấu sai số thực tế trên lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi unpickle mô hình `_loss` trên Cloud VM | Máy ảo EC2 chạy Python 3.14 cài scikit-learn 1.9.1 lệch cấu trúc nhị phân với scikit-learn 1.4.2 trên CI. | Cài đặt Python 3.12 từ PPA deadsnakes và tạo virtualenv chứa chính xác `scikit-learn==1.4.2`. |
| IAM user bị từ chối truy cập tạo S3 và quản lý EC2 | Tài khoản IAM ban đầu chưa được gán chính sách phân quyền cho dịch vụ S3 và EC2. | Vào AWS IAM Console và gán các policy `AmazonS3FullAccess` cùng `AmazonEC2FullAccess`. |
| Lỗi cú pháp JSON khi gọi API bằng curl trên Windows | PowerShell tự động phân tích và loại bỏ dấu ngoặc kép trong tham số `-d`. | Sử dụng cờ `--%` (stop-parsing) trước các tham số của `curl.exe`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới, f1_score tăng nhẹ khoảng 0.02 (từ 0.7149 lên 0.7354) và accuracy tăng từ 0.8740 lên 0.8820. Mức tăng trưởng ổn định nhưng không đột biến vì tập dữ liệu mới có cùng nguồn gốc và phân phối với tập ban đầu. Điều quan trọng nhất là pipeline CI/CD đã tự động kích hoạt hoàn toàn từ một commit dữ liệu DVC, thực hiện huấn luyện và cập nhật API trên VM mà không cần can thiệp thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng xác suất từ 0.1 đến 0.9 (bước 0.05), xác định ngưỡng tối ưu là 0.30 giúp f1_score tăng từ 0.7354 lên 0.7537 so với ngưỡng mặc định 0.5.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tự động tính confusion matrix và classification report lưu vào `outputs/detail.txt` và upload làm artifact. Với bài toán này, bỏ sót người thu nhập cao (recall thấp) tốn kém hơn vì làm mất cơ hội tiếp cận khách hàng mục tiêu giá trị.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Kiểm tra tỷ lệ lớp dương trong tập huấn luyện (24.78%), cảnh báo nếu độ lệch vượt quá 5% so với mốc 24.8% và ghi nhận vào `report.json`.

