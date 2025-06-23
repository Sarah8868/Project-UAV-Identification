from ultralytics import YOLO
import numpy as np
from config import BEST_MODEL_PATH

model = YOLO(BEST_MODEL_PATH)
model.names[0] = "UAV"


def detect_uav(frame, confidence_threshold=0.6):
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls_id] == "UAV" and conf > confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(((x1, y1, x2, y2), conf))

    return detections, results[0]
