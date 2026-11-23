from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from threading import Thread, Event, Lock
from collections import deque
import shutil
import uuid
import json
import cv2
import numpy as np
from fastapi import Body

from backend.services.adas_engine import ADASProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"
ZONE_FILE = BASE_DIR / "zone_config.json"
DEFAULT_ZONES = {
    "warning": [[0.475, 0.52], [0.515, 0.52], [0.610, 0.76], [0.395, 0.76]],
    "danger": [[0.455, 0.65], [0.555, 0.65], [0.610, 0.76], [0.395, 0.76]],
}

def read_zones():
    try:
        return json.loads(ZONE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return json.loads(json.dumps(DEFAULT_ZONES))

def validate_zones(data):
    if not isinstance(data, dict) or set(data) != {"warning", "danger"}:
        raise HTTPException(422, "Cần hai vùng warning và danger.")
    for name in ("warning", "danger"):
        points = data[name]
        if not isinstance(points, list) or len(points) != 4:
            raise HTTPException(422, f"Vùng {name} phải có đúng 4 điểm.")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (int, float) or not 0 <= v <= 1 for v in point)):
                raise HTTPException(422, "Tọa độ phải nằm trong khoảng 0 đến 1.")
        polygon = np.array(points, dtype=np.float32)
        if abs(cv2.contourArea(polygon)) < 0.00015 or not cv2.isContourConvex(polygon):
            raise HTTPException(422, f"Vùng {name} phải là tứ giác lồi và đủ diện tích.")
    outer = np.array(data["warning"], dtype=np.float32)
    for point in data["danger"]:
        if cv2.pointPolygonTest(outer, tuple(map(float, point)), False) < 0:
            raise HTTPException(422, "Vùng DANGER phải nằm hoàn toàn trong vùng WARNING.")
    return data

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart ADAS Web", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="static")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = Lock()
stop_event = Event()
pause_event = Event()
pause_event.set()

state = {
    "status": "idle",
    "message": "Chưa có video. Hãy tải video lên.",
    "video_name": None,
    "frame": None,
    "fps": 0.0,
    "frame_index": 0,
    "total_frames": 0,
    "progress": 0.0,
    "vehicles": 0,
    "alerts": 0,
    "risk": "SAFE",
    "ttc": None,
    "events": deque(maxlen=100),
    "output_path": None,
    "error": None,
}
worker = None
processor = None
uploaded_path = None


def update_state(**kwargs):
    with lock:
        state.update(kwargs)


def get_state():
    with lock:
        return {
            "status": state["status"],
            "message": state["message"],
            "video_name": state["video_name"],
            "fps": round(state["fps"], 1),
            "frame_index": state["frame_index"],
            "total_frames": state["total_frames"],
            "progress": round(state["progress"], 1),
            "vehicles": state["vehicles"],
            "alerts": state["alerts"],
            "risk": state["risk"],
            "ttc": state["ttc"],
            "error": state["error"],
            "has_video": uploaded_path is not None,
        }


@app.get("/", response_class=HTMLResponse)
def home():
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "Không tìm thấy frontend/index.html")
    return page.read_text(encoding="utf-8")


@app.get("/api/zones")
def get_zones():
    return read_zones()

@app.put("/api/zones")
def save_zones(data: dict = Body(...)):
    config = validate_zones(data)
    with lock:
        if worker is not None and worker.is_alive() and state["status"] != "paused":
            raise HTTPException(409, "Hãy tạm dừng video trước khi lưu vùng cảm biến.")
        ZONE_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
        if processor is not None and state["status"] == "paused":
            processor.zone_config = json.loads(json.dumps(config))
    return {"ok": True, "zones": config}

@app.get("/api/preview")
def preview_video(current: bool = False):
    if uploaded_path is None or not uploaded_path.exists():
        raise HTTPException(404, "Hãy tải video lên trước.")
    cap = cv2.VideoCapture(str(uploaded_path))
    try:
        if current:
            with lock:
                index = state["frame_index"]
            if index > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise HTTPException(422, "Không đọc được khung hình xem trước.")
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, round(h * 960 / w)))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise HTTPException(500, "Không tạo được ảnh xem trước.")
        return StreamingResponse(iter([jpg.tobytes()]), media_type="image/jpeg", headers={"Cache-Control":"no-store"})
    finally:
        cap.release()


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    global uploaded_path, processor, worker

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Chỉ hỗ trợ MP4, AVI, MOV hoặc MKV.")

    with lock:
        if worker is not None and worker.is_alive():
            raise HTTPException(409, "Hãy dừng phân tích hiện tại trước khi tải video mới.")

    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    uploaded_path = destination
    processor = None
    update_state(
        status="uploaded",
        message="Video đã được tải lên. Nhấn Bắt đầu phân tích.",
        video_name=file.filename,
        frame=None,
        fps=0.0,
        frame_index=0,
        total_frames=0,
        progress=0.0,
        vehicles=0,
        alerts=0,
        risk="SAFE",
        ttc=None,
        events=deque(maxlen=100),
        output_path=None,
        error=None,
    )
    return {"ok": True, "filename": file.filename}


def run_processing(input_path: Path, output_path: Path):
    global processor
    try:
        processor = ADASProcessor(
            input_path=input_path,
            output_path=output_path,
            stop_event=stop_event,
            pause_event=pause_event,
            publish_frame=publish_frame,
            publish_metrics=publish_metrics,
            publish_event=publish_event,
            zone_config=read_zones(),
        )
        processor.run()
        if stop_event.is_set():
            update_state(status="stopped", message="Đã dừng phân tích.")
        else:
            update_state(status="completed", message="Đã phân tích xong video.")
    except Exception as exc:
        update_state(status="error", message="Có lỗi khi phân tích video.", error=str(exc))
    finally:
        update_state(output_path=str(output_path) if output_path.exists() else None)


def publish_frame(jpeg_bytes):
    update_state(frame=jpeg_bytes)


def publish_metrics(metrics):
    update_state(**metrics)


def publish_event(event):
    with lock:
        state["events"].appendleft(event)
        state["alerts"] = state["alerts"] + 1


@app.post("/api/start")
def start_analysis():
    global worker
    if uploaded_path is None or not uploaded_path.exists():
        raise HTTPException(400, "Bạn cần tải video lên trước.")

    with lock:
        if worker is not None and worker.is_alive():
            raise HTTPException(409, "Tiến trình đang chạy.")

    stop_event.clear()
    pause_event.set()
    output_path = OUTPUT_DIR / f"adas_result_{uuid.uuid4().hex[:8]}.mp4"
    update_state(
        status="processing",
        message="Đang khởi tạo mô hình và phân tích video...",
        frame=None,
        frame_index=0,
        progress=0.0,
        vehicles=0,
        alerts=0,
        risk="SAFE",
        ttc=None,
        events=deque(maxlen=100),
        output_path=None,
        error=None,
    )
    worker = Thread(target=run_processing, args=(uploaded_path, output_path), daemon=True)
    worker.start()
    return {"ok": True, "message": "Đã bắt đầu phân tích."}


@app.post("/api/pause")
def pause_analysis():
    if get_state()["status"] != "processing":
        raise HTTPException(409, "Không có tiến trình đang chạy.")
    pause_event.clear()
    update_state(status="paused", message="Video đang tạm dừng.")
    return {"ok": True}


@app.post("/api/resume")
def resume_analysis():
    if get_state()["status"] != "paused":
        raise HTTPException(409, "Video không ở trạng thái tạm dừng.")
    pause_event.set()
    update_state(status="processing", message="Tiếp tục phân tích video.")
    return {"ok": True}


@app.post("/api/stop")
def stop_analysis():
    stop_event.set()
    pause_event.set()
    return {"ok": True, "message": "Đã gửi yêu cầu dừng."}


@app.get("/api/status")
def status():
    return get_state()


@app.get("/api/events")
def events():
    with lock:
        return list(state["events"])


@app.get("/api/stream")
def stream():
    def generate():
        while True:
            with lock:
                frame = state["frame"]
                status_now = state["status"]
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            elif status_now in {"idle", "uploaded"}:
                # Keep connection alive until processing starts.
                pass
            import time
            time.sleep(0.04)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )


@app.get("/api/download")
def download_result():
    path = get_state()["output_path"]
    if not path or not Path(path).exists():
        raise HTTPException(404, "Chưa có video kết quả.")
    return FileResponse(path, media_type="video/mp4", filename="smart_adas_result.mp4")
