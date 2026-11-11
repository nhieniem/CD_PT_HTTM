import sys
from pathlib import Path
import cv2
import numpy as np
import time
from ultralytics import YOLO
# Kiểm tra thư viện phát âm thanh trên Windows
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# ------------------------------------------------------------------
# 1. KHỞI TẠO ĐƯỜNG DẪN VÀ MÔ HÌNH
# ------------------------------------------------------------------
project_dir = Path(__file__).resolve().parent
video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_dir / 'video 1.mp4'
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else project_dir / 'output_smart_adas.mp4'
model_path = project_dir / 'yolov8n-seg.pt'

if not video_path.exists():
    raise FileNotFoundError(f"Video không tồn tại: {video_path}")

print(f"[ADAS SYSTEM] Đang tải video: {video_path}")
print(f"[ADAS SYSTEM] Ghi video đầu ra: {output_path}")

model = YOLO(str(model_path))
cap = cv2.VideoCapture(str(video_path))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps_input, (frame_width, frame_height))

traffic_objects = ["car", "bus", "motorbike", "bicycle", "truck", "person"]
safe_zone_y = int(frame_height * 0.65) # Vùng an toàn ở 65% chiều cao ảnh

# Bộ lưu trữ lịch sử để tính toán vận tốc tương đối & TTC
track_history = {}
danger_counter = 0
frame_count = 0

# ------------------------------------------------------------------
# 2. XỬ LÝ LÀN ĐƯỜNG (OPENCV)
# ------------------------------------------------------------------
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 1.5)

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([
        [(0, height), (0, int(height * 0.6)), (width, int(height * 0.6)), (width, height)]
    ], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lane_lines(frame):
    preprocessed = preprocess_frame(frame)
    edges = cv2.Canny(preprocessed, 50, 150)
    roi_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=40)
    line_img = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            # Ép kiểu và lấy chính xác 4 tọa độ x1, y1, x2, y2 để tránh lỗi numpy.int32
            coords = np.array(line).flatten()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
                
    return cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)

# ------------------------------------------------------------------
# 3. SUY LUẬN THÔNG MINH & CẢNH BÁO ÂM THANH
# ------------------------------------------------------------------
def evaluate_risk_level(y2, box_height, ttc):
    """Đánh giá rủi ro dựa trên vị trí, kích thước khung bao và chỉ số TTC."""
    # Trạng thái DANGER: Vượt vạch Safe Zone HOẶC TTC < 1.5s
    if y2 > safe_zone_y or (ttc is not None and ttc < 1.5) or box_height > (frame_height * 0.35):
        return "DANGER", (0, 0, 255)
    
    # Trạng thái WARNING: Tiến cận vùng Safe Zone HOẶC TTC < 3.0s
    elif y2 > (safe_zone_y - 50) or (ttc is not None and ttc < 3.0) or box_height > (frame_height * 0.25):
        return "WARNING", (0, 255, 255)
    
    # Trạng thái SAFE
    return "SAFE", (0, 255, 0)

def trigger_audio_alert():
    """Phát tiếng bíp cảnh báo khi gặp nguy hiểm."""
    if HAS_SOUND:
        winsound.Beep(2000, 80)

# ------------------------------------------------------------------
# 4. VÒNG LẶP XỬ LÝ CHÍNH
# ------------------------------------------------------------------
print("\n[SYSTEM RUNNING] Đang chạy Hệ thống ADAS... Nhấn phím 'Q' trên màn hình hiển thị để dừng.")

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # A. Vẽ làn đường và đường ranh giới Safe Zone
    frame_display = detect_lane_lines(frame)
    cv2.rectangle(frame_display, (0, safe_zone_y), (frame_width, frame_height), (0, 0, 180), 1)
    cv2.putText(frame_display, 'SAFE ZONE BOUNDARY', (10, safe_zone_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # B. Tích hợp Tracking (ByteTrack): Cấp ID cố định cho từng đối tượng
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    has_danger_in_frame = False
    active_vehicles_count = 0

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
            label = model.names[cls_id]
            if label not in traffic_objects:
                continue

            active_vehicles_count += 1
            x1, y1, x2, y2 = box
            box_height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # C. Tính toán chỉ số TTC (Time-To-Collision)
            ttc = None
            if track_id not in track_history:
                track_history[track_id] = []
            
            track_history[track_id].append((center_x, center_y, box_height, current_time))
            if len(track_history[track_id]) > 5:
                track_history[track_id].pop(0)

            if len(track_history[track_id]) >= 2:
                prev_height = track_history[track_id][0][2]
                prev_time = track_history[track_id][0][3]
                time_diff = current_time - prev_time

                delta_height = box_height - prev_height
                if delta_height > 0 and time_diff > 0:
                    approach_rate = delta_height / time_diff
                    remaining_growth = (frame_height * 0.5) - box_height
                    if remaining_growth > 0:
                        ttc = remaining_growth / approach_rate

            # D. Đánh giá cấp độ rủi ro
            status, color = evaluate_risk_level(y2, box_height, ttc)
            if status == "DANGER":
                has_danger_in_frame = True

            # E. Vẽ ô nhận diện kèm ID và chỉ số TTC
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"ID:{track_id} {label} {conf:.2f}"
            if ttc is not None:
                label_text += f" | TTC:{ttc:.1f}s"

            cv2.putText(frame_display, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # F. Xử lý Cảnh báo Đa phương thức (Banner & Âm thanh)
    if has_danger_in_frame:
        danger_counter += 1
        if (frame_count // 5) % 2 == 0:
            cv2.rectangle(frame_display, (40, 15), (frame_width - 40, 65), (0, 0, 255), -1)
            cv2.putText(frame_display, "CANH BAO: NGUY CO VA CHAM KHAN CAP!", (60, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            trigger_audio_alert()

    # G. Bảng điều khiển HUD Dashboard góc trái
    elapsed_time = time.time() - start_time
    calc_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

    cv2.rectangle(frame_display, (10, 10), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(frame_display, (10, 10), (280, 110), (255, 255, 255), 1)
    cv2.putText(frame_display, "SYSTEM: ADAS SMART V2.0", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame_display, f"FPS: {calc_fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_display, f"Vehicles Tracked: {active_vehicles_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_display, f"Total Alerts: {danger_counter}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if danger_counter > 0 else (0, 255, 0), 1)

    # Ghi file kết quả & Hiển thị màn hình
    out.write(frame_display)
    cv2.imshow("HE THONG ADAS THONG MINH - CANH BAO VA CHAM THOI GIAN THUC", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[SYSTEM INFO] Người dùng chủ động dừng hệ thống.")
        break

# ------------------------------------------------------------------
# 5. GIẢI PHÓNG TÀI NGUYÊN
# ------------------------------------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[SYSTEM COMPLETE] Đã lưu video kết quả tại: {output_path}")
print(f"[SYSTEM COMPLETE] Tổng số mốc ghi nhận cảnh báo nguy hiểm: {danger_counter}")
