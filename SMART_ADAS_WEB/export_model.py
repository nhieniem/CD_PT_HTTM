from ultralytics import YOLO

model = YOLO("backend/models/yolov8n-seg.pt")

model.export(
    format="onnx",
    imgsz=416,
    dynamic=False,
    simplify=True,
    opset=17
)

print("Da chuyen YOLO sang ONNX thanh cong!")