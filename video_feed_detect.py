import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('mart.mp4')

model = YOLO("yolov8n.pt")  

while True:
    ret, frame = cap.read()

    results = model(frame, classes = [0])  # Filter for class 0 (person)
    annotated_frame = results[0].plot()
    cv2.imshow("Video feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
