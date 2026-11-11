# Phát hiện phương tiện và cảnh báo khoảng cách an toàn

## Mục tiêu
Dự án phát hiện phương tiện giao thông trên video và cảnh báo khi có phương tiện nằm trong vùng an toàn phía dưới khung hình.

## Nội dung chính
- Tiền xử lý ảnh: grayscale + Gaussian Blur
- Phát hiện lane: Canny Edge Detection + Hough Line Transform
- Nhận dạng đối tượng: YOLOv8n-seg
- Logic cảnh báo: phương tiện ở vùng an toàn hoặc quá gần

## Yêu cầu
- Python 3.10+ hoặc 3.11
- Gói Python:
  - `opencv-python`
  - `numpy`
  - `ultralytics`

## Cài đặt
1. Mở PowerShell trong thư mục dự án.
2. Tạo môi trường ảo (tuỳ chọn):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Cài đặt thư viện:
   ```powershell
   pip install opencv-python numpy ultralytics
   ```

## Dữ liệu mẫu
Trong thư mục có sẵn các file mẫu:
- `vd1.mp4`
- `vd2.mp4`
- `vd3.mp4`
- `yolov8n-seg.pt`


## Chạy pipeline
1. Chạy với video mẫu mặc định:
   ```powershell
   python phathienphuongtien.py
   ```
2. Chạy với video khác nếu muốn:
   ```powershell
   python phathienphuongtien.py vd2.mp4
   ```
3. Chạy với video đầu vào và tên file output tuỳ chọn:
   ```powershell
   python phathienphuongtien.py vd3.mp4 output_video_custom.mp4
   ```

## Kết quả
- Video xuất ra: `output_video_explained.mp4` hoặc tên file output tuỳ chọn.
- Nhấn `q` để dừng hiển thị và kết thúc chương trình.

## Lưu ý
- Đảm bảo file `yolov8n-seg.pt` nằm cùng thư mục với script.
- Nếu không mở được video, kiểm tra lại đường dẫn và định dạng file video.
