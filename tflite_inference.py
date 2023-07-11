import numpy as np

from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.checks import check_yaml

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

def read_tensor_from_readed_frame(frame, input_height=640, input_width=640,
        input_mean=0, input_std=255):
  float_caster = tf.cast(frame, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize(dims_expander, [input_height, input_width])
  result = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  print(result.shape)
  return result

def inference(model, frame):
    ##Load tflite model and allocate tensors
    interpreter = model
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # cv_image = image = cv2.imread(input_path)

    ##Converting the readed frame to RGB as opencv reads frame in BGR
    image = Image.fromarray(frame).convert('RGB')

    ##Converting image into tensor
    image_tensor = read_tensor_from_readed_frame(image)

    ##Test model
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data # , cv_image

CLASSES = yaml_load(check_yaml('models/best.yaml'))['names']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def main(model, frame): 
    height, width, _ = (frame.shape)

    outputs = inference(model, frame)
    # original_image = cv2.resize(frame, (640, 640))

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], 
                outputs[0][i][3]
            ]
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
    model = tf.lite.Interpreter(model_path="models/best_float32.tflite")
    cv_image = image = cv2.imread("samples/20485139e526d5b4.jpg")
    print(main(model, cv_image))