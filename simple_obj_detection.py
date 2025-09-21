import cv2
from ultralytics import YOLO    

model = YOLO("yolov8n.pt")  

image = cv2.imread("image.png")
results = model(image)

annotated_image = results[0].plot()
cv2.imshow("Annotated image:", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
