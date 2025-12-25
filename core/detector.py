class BirdDetector:
    def __init__(self):
        # Lazy import to satisfy Pylance and runtime
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")  # auto-downloads model

    def detect(self, frame):
        results = self.model(frame, conf=0.4, iou=0.5, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # COCO bird class ID = 14
                if cls == 14:
                    detections.append([x1, y1, x2, y2, conf])

        return detections
