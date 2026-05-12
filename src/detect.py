from ultralytics import YOLO
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = YOLO(config["detection"]["model_path"])


def detect_objects(frame):
    results = model(frame)
    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        if cls in config["detection"]["vehicle_classes"]:
            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": xyxy
            })
    return detections
