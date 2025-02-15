import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import json
from datetime import datetime
import pandas as pd

vertices = np.array([[[(2553, 438), (2560, 1428), (3, 1430), (1, 510), (315, 378), (496, 310)]]], dtype=np.int32)
cam = cv2.VideoCapture("/home/ekak-08/Downloads/task.mp4")
os.makedirs('datafinal1', exist_ok=True)
curr_frame = 0
model = YOLO("yolov8n.pt")
video_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))

# Initialize the data dictionary
all_data = {
    "object_data": []
}

while True:
    ret, frame = cam.read()
    if not ret:
        break
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    img = cv2.bitwise_and(frame, mask)
    name = './data5/frame' + str(curr_frame) + '.jpg'
    cv2.imwrite(name, img)
    results = model(img)
    timstmp = int(datetime.now().strftime("%H%M%S"))
    datstmp = datetime.now().strftime("%Y%m%d")
    if results:
        for i, box in enumerate(results[0].boxes.xyxy):
            xmin, ymin, xmax, ymax = box
            all_data["object_data"].append({
                "timeStamp": timstmp,
                "dateStamp": datstmp,
                "confidence": str(round(results[0].boxes.conf[i].item(), 2)),
                "object": model.names[results[0].boxes.cls[i].item()],
                "co_ordinates": [
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax)
                ],
                "frame_no": curr_frame,
            })

    curr_frame += 1

# Write the data to a single JSON file
with open('./datafinal/data1.json', 'w') as f:
    json.dump(all_data, f, indent=4)

cam.release()
cv2.destroyAllWindows()