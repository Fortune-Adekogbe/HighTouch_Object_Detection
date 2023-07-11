import cv2
import numpy as np
from ultralytics.yolo.engine.model import YOLO
from helpers import *

door_model_path = "models/best.pt"
#chair_model = YOLO("models/yolov8m.pt")

def main(model, frame):

    dpred = model.predict(frame, classes=1, conf=0.3)[0]
    #cpred = chair_model.predict(img_path, classes=[13, 56, 57, 60], conf=0.3)[0]
    try:
        return get_results(dpred)[0]
    except:
        return None

if __name__ == "__main__":
    model = YOLO(door_model_path)
    #image: np.ndarray = cv2.imread("samples/4952466327_b68537afd3_o.jpg")
    print(main(model, "samples/4952466327_b68537afd3_o.jpg"))