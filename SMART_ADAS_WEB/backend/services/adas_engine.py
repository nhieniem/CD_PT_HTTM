from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import time
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "backend" / "models" / "yolov8n-seg.pt"
MODEL_NAME = "yolov8n-seg.pt"

TRAFFIC_OBJECTS = {"car", "bus", "motorbike", "bicycle", "truck", "person"}


class ADASProcessor:
    """
    Video-based ADAS processor.
    Reuses the original project's YOLO tracking, ByteTrack, lane detection,
    bounding-box growth history and SAFE/WARNING/DANGER rules.

    TTC in this educational prototype is a heuristic based on bounding-box
    growth, NOT a calibrated physical time-to-collision measurement.
    """

    def __init__(
        self, input_path, output_path, stop_event, pause_event,
        publish_frame, publish_metrics, publish_event
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.publish_frame = publish_frame
        self.publish_metrics = publish_metrics
        self.publish_event = publish_event
        self.track_history = defaultdict(lambda: deque(maxlen=5))
        self.last_status = {}
        self.alert_count = 0

    def load_model(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            return YOLO(str(MODEL_PATH))
        # Ultralytics downloads the official weights if they are not present.
        return YOLO(MODEL_NAME)

    @staticmethod
    def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 1.5)

    @staticmethod
    def sensor_polygon(width, height):
        """Perspective trapezoid for the forward driving corridor.

        Coordinates are proportional to the frame, so the region also works
        with videos of different resolutions. This is a visual starting
        calibration for the sample dashcam view, not a universal camera calibration.
        """
        # Calibrated visually for the sample dashcam perspective:
        # start around the road horizon and end just above the hood.
        # Re-tune these ratios if the camera mounting angle changes.
        return np.array([
            (int(width * 0.455), int(height * 0.31)),  # far-left
            (int(width * 0.545), int(height * 0.31)),  # far-right
            (int(width * 0.78), int(height * 0.73)),   # near-right
            (int(width * 0.22), int(height * 0.73)),   # near-left
        ], dtype=np.int32)

    @classmethod
    def region_of_interest(cls, img):
        height, width = img.shape
        mask = np.zeros_like(img)
        polygon = cls.sensor_polygon(width, height)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(img, mask)

    @classmethod
    def draw_sensor_region(cls, frame):
        """Draw a translucent trapezoid and its two perspective boundaries."""
        height, width = frame.shape[:2]
        polygon = cls.sensor_polygon(width, height)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 0, 180))
        frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0)
        cv2.polylines(frame, [polygon], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            frame, "FORWARD SENSOR ZONE",
            (int(width * 0.30), int(height * 0.71)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1500),
            (0, 0, 255), 2, cv2.LINE_AA
        )
        return frame

    @classmethod
    def point_in_sensor_zone(cls, x, y, width, height):
        polygon = cls.sensor_polygon(width, height)
        return cv2.pointPolygonTest(
            polygon, (float(x), float(y)), False
        ) >= 0

    def detect_lane_lines(self, frame):
        edges = cv2.Canny(self.preprocess_frame(frame), 50, 150)
        roi_edges = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, threshold=50,
            minLineLength=50, maxLineGap=40
        )
        line_img = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.array(line).flatten()
                cv2.line(
                    line_img, (int(x1), int(y1)), (int(x2), int(y2)),
                    (255, 200, 0), 2
                )
        return cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)

    @staticmethod
    def risk_level(y2, box_height, ttc, frame_height, in_sensor_zone=True):
        # Sensor-zone membership and collision severity are separate:
        # every tracked vehicle in the forward corridor is at least WARNING.
        if not in_sensor_zone:
            return "SAFE", (0, 255, 0)

        # DANGER uses conservative image-space proximity heuristics.
        # TTC is supplementary because bbox-growth TTC can be unstable.
        if (
            y2 >= frame_height * 0.62
            or box_height >= frame_height * 0.22
            or (ttc is not None and ttc <= 3.0)
        ):
            return "DANGER", (0, 0, 255)

        return "WARNING", (0, 255, 255)

    def estimate_ttc(self, track_id, box_height, video_time, frame_height):
        history = self.track_history[track_id]
        history.append((box_height, video_time))
        if len(history) < 2:
            return None

        previous_height, previous_time = history[0]
        delta_time = video_time - previous_time
        delta_height = box_height - previous_height

        if delta_time <= 0 or delta_height <= 0:
            return None

        growth_rate = delta_height / delta_time
        target_height = frame_height * 0.5
        remaining_growth = target_height - box_height
        if growth_rate <= 0 or remaining_growth <= 0:
            return None
        return max(0.0, remaining_growth / growth_rate)

    def run(self):
        model = self.load_model()
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {self.input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Không tạo được file video đầu ra.")

        frame_index = 0
        processing_started = time.time()
        try:
            while not self.stop_event.is_set():
                if not self.pause_event.wait(timeout=0.1):
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                video_time = frame_index / source_fps
                display = self.detect_lane_lines(frame)
                display = self.draw_sensor_region(display)

                result = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    verbose=False
                )[0]

                active_count = 0
                frame_risk = "SAFE"
                frame_ttc = None

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()

                    for box, track_id, class_id, confidence in zip(
                        boxes, track_ids, class_ids, confidences
                    ):
                        label = model.names[int(class_id)]
                        if label not in TRAFFIC_OBJECTS:
                            continue

                        active_count += 1
                        x1, y1, x2, y2 = map(int, box)
                        box_height = max(1, y2 - y1)

                        ttc = self.estimate_ttc(
                            int(track_id), box_height, video_time, height
                        )
                        # Use the bottom-centre of the detected object as the
                        # contact point with the road plane for zone membership.
                        bottom_center_x = (x1 + x2) / 2.0
                        # Test the bottom-centre and both lower box corners.
                        # This is more tolerant when a vehicle box straddles a
                        # sensor-zone boundary than testing only one point.
                        in_sensor_zone = any(
                            self.point_in_sensor_zone(px, y2, width, height)
                            for px in (x1 + 0.15 * (x2 - x1),
                                       bottom_center_x,
                                       x2 - 0.15 * (x2 - x1))
                        )
                        status, color = self.risk_level(
                            y2, box_height, ttc, height, in_sensor_zone
                        )

                        if status == "DANGER":
                            frame_risk = "DANGER"
                        elif status == "WARNING" and frame_risk == "SAFE":
                            frame_risk = "WARNING"

                        if ttc is not None:
                            frame_ttc = ttc if frame_ttc is None else min(frame_ttc, ttc)

                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        zone_label = "IN-ZONE" if in_sensor_zone else "OUTSIDE"
                        label_text = f"ID:{track_id} {label} {confidence:.2f} {zone_label}"
                        if ttc is not None:
                            label_text += f" TTC*:{ttc:.1f}s"
                        cv2.putText(
                            display, label_text, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                        previous = self.last_status.get(int(track_id))
                        if status in {"WARNING", "DANGER"} and status != previous:
                            event = {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "video_time": round(video_time, 2),
                                "track_id": int(track_id),
                                "object": label,
                                "risk": status,
                                "ttc": round(ttc, 2) if ttc is not None else None,
                                "description": (
                                    "Nguy cơ cao – cần chú ý"
                                    if status == "DANGER"
                                    else "Đối tượng tiến gần vùng cảnh báo"
                                ),
                            }
                            self.publish_event(event)
                        self.last_status[int(track_id)] = status

                if frame_risk == "DANGER":
                    cv2.rectangle(display, (20, 15), (width - 20, 62), (0, 0, 220), -1)
                    cv2.putText(
                        display, "WARNING: HIGH COLLISION RISK",
                        (35, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2
                    )
                elif frame_risk == "WARNING":
                    cv2.rectangle(display, (20, 15), (width - 20, 62), (0, 180, 230), -1)
                    cv2.putText(
                        display, "CAUTION: VEHICLE APPROACHING",
                        (35, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2
                    )

                elapsed = max(0.001, time.time() - processing_started)
                processing_fps = frame_index / elapsed
                cv2.putText(
                    display,
                    f"SMART ADAS | FPS: {processing_fps:.1f} | Vehicles: {active_count}",
                    (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                writer.write(display)
                ok_jpg, encoded = cv2.imencode(
                    ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if ok_jpg:
                    self.publish_frame(encoded.tobytes())

                progress = (frame_index / total_frames * 100) if total_frames > 0 else 0
                self.publish_metrics({
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "progress": progress,
                    "fps": processing_fps,
                    "vehicles": active_count,
                    "risk": frame_risk,
                    "ttc": round(frame_ttc, 2) if frame_ttc is not None else None,
                    "message": "Đang phân tích video...",
                })
        finally:
            cap.release()
            writer.release()
