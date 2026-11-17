from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from threading import Thread, Event, Lock
from collections import deque
import shutil
import uuid

from backend.services.adas_engine import ADASProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"
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
