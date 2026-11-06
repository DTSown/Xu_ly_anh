# Comic Panel Segmentation

Ứng dụng Python tự động phát hiện và trích xuất các khung truyện tranh bằng các kỹ thuật xử lý ảnh số cơ bản được cài đặt từ đầu với NumPy.

## Chức năng

- Chuyển ảnh BGR sang ảnh xám.
- Thu nhỏ ảnh bằng trung bình khối.
- Lọc Gaussian bằng tích chập tự cài đặt.
- Phân ngưỡng Otsu.
- Closing hình thái học để nối các đường viền bị đứt.
- Gán nhãn thành phần liên thông 8 bằng thuật toán two-pass và union-find.
- Lọc bounding box, loại box lồng nhau và sắp xếp theo thứ tự đọc.
- Nhận diện bổ sung gutter ngang trắng/đen và gutter trắng chéo trên các trang có bố cục khó.
- Giao diện Flask để tải ảnh lên và xem từng bước xử lý.

Trong `core/` và `pipeline.py`, thuật toán chỉ sử dụng NumPy. OpenCV chỉ được dùng để đọc và ghi ảnh trong `main.py` và `app.py`.

## Cấu trúc chính

```text
core/
  color.py          Chuyển ảnh xám và thu nhỏ
  filters.py        Kernel Gaussian và tích chập
  threshold.py      Histogram và Otsu
  morphology.py     Dilate, erode, opening, closing
  segmentation.py   CCL, bounding box, lọc box và gutter chéo
pipeline.py         Ghép toàn bộ các bước xử lý
main.py             Chạy bằng dòng lệnh và lưu kết quả
app.py              Web demo Flask
templates/          Giao diện web
requirements.txt    Thư viện cần cài
```

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho phép kích hoạt môi trường, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Chạy giao diện web

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Hoặc mở tự động:

```powershell
Start-Process "http://127.0.0.1:5000"
```

Web hỗ trợ PNG, JPG, JPEG và BMP, giới hạn 20 MB. Kết quả được lưu trong `web_output/`; hệ thống chỉ giữ 20 lần xử lý gần nhất.

## Cách sử dụng các thông số trên giao diện

1. Chọn một ảnh trang truyện tại mục **Trang truyện**.
2. Điều chỉnh **Sigma Gauss** và **Kích thước Closing** nếu cần.
3. Chọn bật hoặc tắt bộ lọc Gauss và Closing.
4. Nhấn **Tách khung** để chạy pipeline và xem các ảnh trung gian cùng những panel đã cắt.

### Sigma Gauss

`Sigma Gauss` quyết định mức làm mượt ảnh trước khi phân ngưỡng Otsu:

- `0.5–1.2`: giữ nhiều chi tiết, phù hợp với ảnh rõ và có đường viền mảnh.
- `1.3–2.0`: mức phù hợp với phần lớn ảnh; giá trị mặc định là `1.5`.
- `2.1–4.0`: giảm nhiễu mạnh hơn nhưng có thể làm mất các nét viền mảnh.

Khi bỏ chọn **Bật lọc Gauss**, giá trị Sigma không còn tác dụng.

### Kích thước Closing

`Kích thước Closing (px)` dùng để nối các đoạn viền bị đứt và lấp các khe nhỏ:

- `3–10 px`: dành cho khung rõ hoặc khoảng cách giữa các panel hẹp.
- `11–25 px`: phù hợp với phần lớn trang truyện; giá trị mặc định là `20 px`.
- `26–60 px`: hữu ích khi đường viền bị đứt nhiều, nhưng có thể làm các panel dính vào nhau.

Khi bỏ chọn **Bật Closing**, giá trị kích thước Closing không còn tác dụng.

### Điều chỉnh theo kết quả

| Hiện tượng | Cách điều chỉnh |
|---|---|
| Phát hiện quá nhiều vùng nhỏ | Tăng Sigma hoặc tăng Closing nhẹ |
| Thiếu khung do đường viền bị đứt | Tăng Closing |
| Nhiều panel bị nhập thành một | Giảm Closing |
| Nét viền mảnh bị mất | Giảm Sigma |
| Ảnh có nhiều nhiễu hoặc họa tiết vụn | Tăng Sigma |
| Kết quả xấu hơn sau bước Closing | Bỏ chọn **Bật Closing** để so sánh |

Nên chạy lần đầu với `Sigma = 1.5`, `Closing = 20 px` và bật cả hai bộ lọc. Nếu các panel bị dính, giảm Closing xuống khoảng `8–15 px`; nếu đường viền không khép kín, tăng lên khoảng `25–35 px`.

## Xử lý một ảnh bằng dòng lệnh

```powershell
.\.venv\Scripts\python.exe main.py --input "C:\duong-dan\trang-truyen.jpg" --out output
```

Ví dụ:

```powershell
.\.venv\Scripts\python.exe main.py --input "C:\Images\l001.jpg" --out output
```

Kết quả được tạo tại:

```text
output/<tên-ảnh>/
  1_gray.png
  2_blur.png
  3_binary_otsu_inverted.png
  4_closing.png
  5_result.png
  panels/
```

## Pipeline

```text
Ảnh màu
→ ảnh xám
→ thu nhỏ
→ Gaussian blur
→ Otsu
→ đảo ảnh để nét vẽ là foreground
→ Closing
→ connected components
→ bounding box
→ lọc diện tích
→ loại box lồng nhau
→ sắp thứ tự đọc
→ cắt panel từ ảnh gốc
```

Khi các box từ connected components bị thiếu hoặc chồng lấn, hệ thống có thể kích hoạt detector gutter. Detector ngang nhận diện cả dải trắng có đường biên và dải đen dày; detector chéo tìm các dải trắng dài theo mô hình `y = ax + b` và tạo mask polygon cho từng panel.

## Tham số mặc định

```text
max_side = 700
sigma = 1.5
close_px = 20
min_frac = 0.03
max_frac = 0.95
```

Các tham số `sigma`, `close_px`, Gaussian và Closing có thể thay đổi trực tiếp trên giao diện web.

## Giới hạn

- Kết quả phụ thuộc vào độ rõ của đường viền và gutter.
- Closing quá lớn có thể làm nhiều vùng dính vào nhau.
- Trang không có gutter hoặc có nền quá phức tạp vẫn có thể nhận sai.
- Detector gutter chéo chỉ là fallback và không thay thế hoàn toàn phân đoạn polygon tổng quát.
- Bong bóng thoại nằm trên gutter có thể bị chia theo ranh giới panel.

## Dừng web server

Trong cửa sổ PowerShell đang chạy Flask, nhấn `Ctrl+C`.
