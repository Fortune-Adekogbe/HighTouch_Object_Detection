import time
import cv2
from ultralytics.yolo.engine.model import YOLO
import imutils
from imutils.video import VideoStream
from helpers import *

door_model = YOLO("models/best.pt")
#chair_model = YOLO("models/yolov8m.pt")

def inference(img_path):

    dpred = door_model.predict(img_path, classes=1, conf=0.3)[0]
    #cpred = chair_model.predict(img_path, classes=[13, 56, 57, 60], conf=0.3)[0]

    return get_results(dpred), dpred #+ get_results(cpred), [dpred, cpred]

## Initializing video stream
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    predictions, _ = inference(frame)
    print(len(predictions))

    for pred in predictions:
        (startX, startY, endX, endY) = int(pred['x1']), int(pred['y1']), int(pred['x2']), int(pred['y2'])
        color = (0, 255, 0)
        label = f"{pred['class']} {float(pred['conf']):0.2f}"

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# clean up
cv2.destroyAllWindows()
vs.stop()
