from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import time
import cv2
import numpy as np
import onnxruntime as ort
from types import SimpleNamespace
import inspect
from ultralytics.trackers.byte_tracker import BYTETracker

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "backend" / "models" / "yolov8n-seg.onnx"
# COCO categories used by the exported YOLOv8n-seg model.
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
CLASS_IDS = set(CLASS_NAMES)
CONF_THRESHOLD = 0.30
NMS_IOU = 0.50

MAX_PROCESS_WIDTH = 960  # Lower to 768 for slower CPUs; 0 disables resize
YOLO_IMAGE_SIZE = 416  # Increase to 512/640 if distant vehicles are missed
JPEG_QUALITY = 70

TRAFFIC_OBJECTS = {"car", "bus", "motorbike", "bicycle", "truck", "person"}


class TrackingDetections:
    """Ultralytics Results-like detection view for BYTETracker.

    Supports .xywh/.xyxy/.conf/.cls, slicing with [indices], and len().
    In recent ByteTrack versions, results[indices] is used to select
    high/low confidence detections, which SimpleNamespace cannot support.
    """

    def __init__(self, boxes, scores, classes):
        self.xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        self.xywh = self.xyxy.copy()
        if len(self.xywh):
            self.xywh[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) * 0.5
            self.xywh[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) * 0.5
            self.xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
            self.xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        # Newer ByteTrack versions use results[mask] or results[indices].
        return TrackingDetections(self.xyxy[index], self.conf[index], self.cls[index])


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
        publish_frame, publish_metrics, publish_event, zone_config=None
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
        self.zone_config = zone_config

    def load_model(self):
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Khong tim thay ONNX model: {MODEL_PATH}")
        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(
                "DirectML khong kha dung. Kiem tra onnxruntime-directml trong .venv. "
                f"Providers: {available}"
            )
        options = ort.SessionOptions()
        options.enable_mem_pattern = False  # Required by DirectML.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options,
            providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        )
        if "DmlExecutionProvider" not in session.get_providers():
            raise RuntimeError("Khong tao duoc DirectML inference session")
        print("Smart ADAS ONNX providers:", session.get_providers())
        return session

    @staticmethod
    def create_tracker(frame_rate):
        # Uses Ultralytics ByteTrack, not an ad-hoc replacement tracker.
        args = SimpleNamespace(
            track_high_thresh=0.35, track_low_thresh=0.10,
            new_track_thresh=0.35, track_buffer=30,
            match_thresh=0.80, fuse_score=True
        )
        # Ultralytics 8.4.x accepts BYTETracker(args), while older versions
        # may also accept frame_rate. Inspect the installed API once.
        parameters = inspect.signature(BYTETracker.__init__).parameters
        if "frame_rate" in parameters:
            return BYTETracker(args=args, frame_rate=max(1, int(frame_rate)))
        return BYTETracker(args=args)

    @staticmethod
    def infer(session, frame):
        """Run YOLOv8-seg ONNX via DirectML; use detection head only.

        Export was made with imgsz=416, dynamic=False, no integrated NMS.
        Masks are unused, matching the original application behavior.
        """
        h, w = frame.shape[:2]
        size = YOLO_IMAGE_SIZE
        ratio = min(size / w, size / h)
        nw, nh = round(w * ratio), round(h * ratio)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        predictions = raw[0].T  # (anchors, 116): xywh + 80 classes + 32 mask coefficients
        scores_by_class = predictions[:, 4:84]
        cls_ids = scores_by_class.argmax(axis=1)
        scores = scores_by_class[np.arange(len(cls_ids)), cls_ids]
        valid = (scores >= CONF_THRESHOLD) & np.isin(cls_ids, list(CLASS_IDS))
        preds, cls_ids, scores = predictions[valid], cls_ids[valid], scores[valid]
        if len(preds) == 0:
            return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.float32)
        xywh = preds[:, :4].copy()
        boxes = np.empty_like(xywh)
        boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2 - pad_x) / ratio
        boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2 - pad_y) / ratio
        boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2 - pad_x) / ratio
        boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2 - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        # Class-aware NMS prevents boxes from different classes suppressing each other.
        keep = []
        for cls_id in np.unique(cls_ids):
            idx = np.where(cls_ids == cls_id)[0]
            b = boxes[idx]
            nms_boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                         for x1, y1, x2, y2 in b]
            selected = cv2.dnn.NMSBoxes(nms_boxes, scores[idx].tolist(),
                                       CONF_THRESHOLD, NMS_IOU)
            if len(selected):
                keep.extend(idx[np.asarray(selected).reshape(-1)].tolist())
        keep = np.asarray(keep, dtype=int)
        return boxes[keep], scores[keep].astype(np.float32), cls_ids[keep].astype(np.float32)

    @staticmethod
    def track_detections(tracker, boxes, scores, cls_ids, frame):
        detections = TrackingDetections(boxes, scores, cls_ids)
        # Expected tracks: x1,y1,x2,y2,track_id,score,class_id,det_index.
        return tracker.update(detections, frame)

    @staticmethod
    def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 1.5)

    def sensor_polygon(self, width, height):
        points = (self.zone_config or {}).get("warning")
        if not points:
            points = [[0.475, 0.52], [0.515, 0.52], [0.610, 0.76], [0.395, 0.76]]
        return np.array([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)

    def danger_polygon(self, width, height):
        points = (self.zone_config or {}).get("danger")
        if points:
            return np.array([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)
        outer = self.sensor_polygon(width, height)
        y_top = int(height * 0.65)
        y_far = int(outer[0, 1])
        y_near = int(outer[3, 1])
        fraction = np.clip((y_top - y_far) / max(1, y_near - y_far), 0, 1)
        left_x = int(outer[0, 0] + fraction * (outer[3, 0] - outer[0, 0]))
        right_x = int(outer[1, 0] + fraction * (outer[2, 0] - outer[1, 0]))
        return np.array([(left_x, y_top), (right_x, y_top), tuple(outer[2]), tuple(outer[3])], dtype=np.int32)

    def region_of_interest(self, img):
        height, width = img.shape
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [self.sensor_polygon(width, height)], 255)
        return cv2.bitwise_and(img, mask)

    def draw_sensor_region(self, frame):
        height, width = frame.shape[:2]
        warning = self.sensor_polygon(width, height)
        danger = self.danger_polygon(width, height)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [warning], (0, 185, 255))
        cv2.fillPoly(overlay, [danger], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.13, frame, 0.87, 0)
        cv2.polylines(frame, [warning], True, (0, 210, 255), 2, cv2.LINE_AA)
        cv2.polylines(frame, [danger], True, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def point_in_sensor_zone(self, x, y, width, height):
        return cv2.pointPolygonTest(self.sensor_polygon(width, height), (float(x), float(y)), False) >= 0

    def point_in_danger_zone(self, x, y, width, height):
        return cv2.pointPolygonTest(self.danger_polygon(width, height), (float(x), float(y)), False) >= 0

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
    def risk_level(y2, box_height, ttc, frame_height,
                   in_sensor_zone=True, in_danger_zone=False, danger_trigger_y=None):
        """Heuristic risk: outside=SAFE, corridor=WARNING, near=potential DANGER.

        A short TTC estimate alone cannot trigger DANGER for a distant object.
        """
        if not in_sensor_zone:
            return "SAFE", (0, 255, 0)
        if in_danger_zone and (
            y2 >= (danger_trigger_y if danger_trigger_y is not None else frame_height * 0.70)
            or (ttc is not None and 0 < ttc <= 2.5)
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

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if original_width <= 0 or original_height <= 0:
            cap.release()
            raise RuntimeError("Không đọc được kích thước video.")
        scale = min(1.0, MAX_PROCESS_WIDTH / original_width) if MAX_PROCESS_WIDTH > 0 else 1.0
        width = max(2, int(original_width * scale) // 2 * 2)
        height = max(2, int(original_height * scale) // 2 * 2)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = self.create_tracker(source_fps)
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

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                frame_index += 1
                video_time = frame_index / source_fps
                display = self.detect_lane_lines(frame)
                display = self.draw_sensor_region(display)

                boxes, confidences, class_ids = self.infer(model, frame)
                tracked = self.track_detections(tracker, boxes, confidences, class_ids, frame)

                active_count = 0
                frame_risk = "SAFE"
                frame_ttc = None
                risk_source_id = None
                risk_source_label = None

                if len(tracked):
                    for row in tracked:
                        x1, y1, x2, y2 = map(int, row[:4])
                        track_id = int(row[4])
                        confidence = float(row[5])
                        class_id = int(row[6])
                        label = CLASS_NAMES.get(class_id, "other")
                        if label not in TRAFFIC_OBJECTS:
                            continue

                        active_count += 1
                        box_height = max(1, y2 - y1)

                        ttc = self.estimate_ttc(
                            int(track_id), box_height, video_time, height
                        )
                        # Approximate the road contact point using the bottom
                        # center of the bounding box. Do NOT test side corners:
                        # wide trucks in adjacent lanes can overlap the polygon.
                        bottom_center_x = (x1 + x2) / 2.0
                        bottom_center_y = y2
                        in_sensor_zone = self.point_in_sensor_zone(
                            bottom_center_x, bottom_center_y, width, height
                        )
                        in_danger_zone = self.point_in_danger_zone(
                            bottom_center_x, bottom_center_y, width, height
                        )
                        danger_ys = self.danger_polygon(width, height)[:, 1]
                        danger_trigger_y = float(danger_ys.min() + 0.4 * (danger_ys.max() - danger_ys.min()))
                        status, color = self.risk_level(
                            y2, box_height, ttc, height,
                            in_sensor_zone, in_danger_zone, danger_trigger_y
                        )

                        # Select the highest-risk IN-ZONE vehicle for the banner.
                        # Never let an OUTSIDE vehicle trigger a global alert.
                        if in_sensor_zone and status == "DANGER":
                            if frame_risk != "DANGER":
                                risk_source_id = int(track_id)
                                risk_source_label = label
                            frame_risk = "DANGER"
                        elif in_sensor_zone and status == "WARNING" and frame_risk == "SAFE":
                            frame_risk = "WARNING"
                            risk_source_id = int(track_id)
                            risk_source_label = label

                        if ttc is not None and in_sensor_zone:
                            frame_ttc = ttc if frame_ttc is None else min(frame_ttc, ttc)

                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        # Mark the exact point used for the zone decision.
                        cv2.circle(display, (int(bottom_center_x), int(bottom_center_y)),
                                   5, color, -1, cv2.LINE_AA)
                        zone_label = ("DANGER-ZONE" if in_danger_zone else
                                      "IN-ZONE" if in_sensor_zone else "OUTSIDE")
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
                        display, f"DANGER: ID {risk_source_id} ({risk_source_label})",
                        (35, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2
                    )
                elif frame_risk == "WARNING":
                    cv2.rectangle(display, (20, 15), (width - 20, 62), (0, 180, 230), -1)
                    cv2.putText(
                        display, f"WARNING: ID {risk_source_id} ({risk_source_label})",
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
                    ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
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
