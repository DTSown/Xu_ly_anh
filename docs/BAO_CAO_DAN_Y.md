# Dàn ý báo cáo

1. **Giới thiệu**: bài toán phân đoạn khung truyện (webtoon/manga viewer, đọc trên điện thoại, OCR theo khung); phát biểu input/output; ràng buộc "from scratch".
2. **Cơ sở lý thuyết & cài đặt**: mỗi thuật toán gồm công thức, ý tưởng, file/hàm tương ứng, độ phức tạp.
   - Gray, Gauss (tách 1D), tích chập; Otsu; Dilate/Erode/Closing (Hit/Fit); CCL two-pass.
   - Giải thích vì sao **đảo ảnh** (nét = foreground) rồi mới Closing + CCL.
   - Kỹ thuật tối ưu NumPy: dịch mảng thay vòng lặp pixel, thu nhỏ ảnh, quy đổi tọa độ.
3. **Chuyên đề CV gần đây** (tự tra và kiểm chứng trích dẫn trước khi ghi):
   - Mask R-CNN (ICCV 2017); Segment Anything (ICCV 2023).
   - Manga109 và các công trình về hiểu truyện tranh (ví dụ hướng "Magi", Sachdeva & Zisserman 2024) — cần xác minh tên bài, năm, nơi đăng.
   - So sánh truyền thống vs học sâu; nếu có thể, chạy SAM trên 3-5 failure case để có bằng chứng thực nghiệm.
4. **Thực nghiệm**: định nghĩa chỉ số (IoU>=0.5 / 0.8, P/R/F1, thời gian); bảng theo từng case (`docs/ket_qua_thuc_nghiem.txt`); ablation (Gauss, Closing, lọc diện tích, kích thước SE); phân tích failure case có ảnh.
5. **Kết luận & hướng phát triển**
   - Tóm tắt đóng góp và kết quả chính.
   - Hạn chế: khung không viền/chéo, dính khung, screentone nặng.
   - Hướng mở: projection profile/XY-cut, ngưỡng cục bộ, kết hợp mô hình học sâu (SAM) cho ca khó.

# Phân công gợi ý (nhóm 3-4 người)
| Thành viên | Việc |
|---|---|
| A | `color`, `filters`, `threshold` + kiểm tra đơn vị |
| B | `morphology`, `segmentation`, `pipeline` |
| C | Thu thập/gán nhãn ảnh thật, `evaluate`, `experiments` |
| D | Đọc bài báo (từ giai đoạn 2), phần chuyên đề, slide |
