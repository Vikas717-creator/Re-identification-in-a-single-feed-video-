from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path="best.pt"):
        # Properly load the YOLOv8 model 
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Run prediction on CPU as there is constraint due to no gpu laptop
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=0.3,
            verbose=False,
            device='cpu'
        )[0]  # Get first (and only) result

        detections = []

        boxes = results.boxes
        print(f"Detected {len(boxes)} objects")  # checking for detection or not for debugging

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            print(f"Class: {cls}, Conf: {conf}")  # printing class and confidence score for each 

            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                "confidence": conf,
                "class_id": cls
            })

        return detections
