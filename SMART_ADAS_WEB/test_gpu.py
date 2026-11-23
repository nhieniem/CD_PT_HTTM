import onnxruntime as ort

model_path = "backend/models/yolov8n-seg.onnx"

session = ort.InferenceSession(
    model_path,
    providers=[
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("Providers:", session.get_providers())
print("Input:", session.get_inputs()[0].shape)
print("GPU DirectML:", "DmlExecutionProvider" in session.get_providers())