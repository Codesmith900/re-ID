import cv2
from ultralytics import YOLO
import numpy as np
import os

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('mart.mp4')

unique_ids = set()
# Parent folder to save person images
parent_folder = 'persons'
os.makedirs(parent_folder, exist_ok=True)

while True:
    ret, frame = cap.read()
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for oid in ids:
            unique_ids.add(oid)
        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            person_folder = os.path.join(parent_folder, f"id_{oid}")
            os.makedirs(person_folder, exist_ok=True)
            crop = frame[y1:y2, x1:x2]
            # Save each crop with a unique filename (timestamp or frame number)
            filename = os.path.join(person_folder, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f}.jpg")
            cv2.imwrite(filename, crop)
        cv2.putText(annotated_frame, f'Count: {len(unique_ids)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    
  


