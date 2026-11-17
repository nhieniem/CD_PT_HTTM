# Smart ADAS Web – Video-based prototype

Ứng dụng web mẫu phân tích video giao thông bằng OpenCV + YOLOv8 + ByteTrack.
Đây là prototype học tập, không dùng để hỗ trợ lái xe ngoài đời thực.

## 1. Yêu cầu
- Windows 10/11
- Python 3.10 hoặc 3.11 (khuyến nghị)
- VS Code hoặc CMD
- Video MP4/AVI/MOV/MKV

## 2. Cài đặt trên Windows CMD

Mở CMD tại thư mục `SMART_ADAS_WEB`:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Trọng số YOLO

Tạo thư mục `backend\models` và tải trọng số YOLOv8n-seg chính thức:

```cmd
mkdir backend\models
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
```

Lệnh trên tải model vào thư mục làm việc theo cơ chế Ultralytics. Nếu file `yolov8n-seg.pt` nằm tại thư mục gốc dự án, hãy sao chép nó:

```cmd
copy yolov8n-seg.pt backend\models\yolov8n-seg.pt
```

Có thể bỏ qua bước sao chép nếu `backend\models\yolov8n-seg.pt` đã tồn tại. Khi chưa có file trong thư mục models, engine cũng thử tải model theo tên `yolov8n-seg.pt`.

## 4. Chạy ứng dụng

Tại thư mục gốc dự án:

```cmd
uvicorn backend.main:app --reload
```

Mở trình duyệt tại:

http://127.0.0.1:8000

## 5. Cách sử dụng
1. Chọn video MP4/AVI/MOV/MKV.
2. Nhấn **Tải video lên**.
3. Nhấn **Bắt đầu phân tích**.
4. Theo dõi luồng video đã xử lý, trạng thái rủi ro và lịch sử cảnh báo.
5. Có thể tạm dừng, tiếp tục hoặc dừng.
6. Khi phân tích hoàn tất, nhấn **Tải video kết quả**.

## 6. API
- `POST /api/upload`: tải video lên
- `POST /api/start`: bắt đầu phân tích
- `POST /api/pause`: tạm dừng
- `POST /api/resume`: tiếp tục
- `POST /api/stop`: dừng
- `GET /api/status`: trạng thái
- `GET /api/events`: lịch sử cảnh báo
- `GET /api/stream`: luồng MJPEG
- `GET /api/download`: tải video kết quả

## 7. Ghi chú thuật toán
- YOLOv8n-seg và ByteTrack được giữ từ bài tập gốc.
- Lane detection giữ cách làm Canny + HoughLinesP.
- TTC* là ước lượng heuristic dựa trên tốc độ tăng chiều cao bounding box, không phải khoảng cách vật lý hoặc TTC đã hiệu chuẩn.
- Quy tắc SAFE/WARNING/DANGER ban đầu được kế thừa từ bài tập gốc. Cần hiệu chỉnh và đánh giá bằng dữ liệu có ground truth trước khi đưa ra kết luận về độ an toàn.
- Phiên bản này xử lý một video tại một thời điểm, chạy một worker AI. Không chạy nhiều upload/phân tích đồng thời.
