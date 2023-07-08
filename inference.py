import time
import cv2
#from tflite_inference import main
from onnx_inference import *
import tensorflow as tf

import imutils
from imutils.video import VideoStream
from helpers import *

model = onnxruntime.InferenceSession("models/best.onnx") # tf.lite.Interpreter(model_path="models/best_float32.tflite")

## Initializing video stream
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # try tflite
    predictions = main(model, frame)
    print(len(predictions))

    for pred in predictions:
        (startX, startY, endX, endY) = round(pred['x1']), round(pred['y1']), round(pred['x2']), round(pred['y2'])
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
