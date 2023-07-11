import onnxruntime
import cv2
import numpy as np

from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('models/best.yaml'))['names']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def main(onnx_model, frame):

    session = onnx_model
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name

    
    [height, width, _] = frame.shape

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    outputs = session.run([], {input_name: blob})

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        if class_ids[index] != 1:
            continue

        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class': CLASSES[class_ids[index]],
            'conf': scores[index],
        }
        
        detection['x1'] = box[0] * width/640
        detection['x2'] = (box[0] + box[2]) * width/640
        detection['y1'] = box[1]  * height/640
        detection['y2'] = (box[1] + box[3]) * height/640

        detections.append(detection)

    return detections


if __name__ == '__main__':
    model = onnxruntime.InferenceSession("models/best.onnx")
    original_image: np.ndarray = cv2.imread("samples/20485139e526d5b4.jpg")
    print(main(model, original_image))