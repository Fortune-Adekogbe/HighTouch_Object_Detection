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
  return result

def inference(Model_Path, input_path):
    ##Load tflite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=Model_Path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cv_image = image = cv2.imread(input_path)

    ##Converting the readed frame to RGB as opencv reads frame in BGR
    image = Image.fromarray(cv_image).convert('RGB')

    ##Converting image into tensor
    image_tensor = read_tensor_from_readed_frame(image)

    ##Test model
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data, cv_image

CLASSES = yaml_load(check_yaml('models/best.yaml'))['names']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(model, input_image): 

    outputs, original_image = inference(model, input_image)
    original_image = cv2.resize(original_image, (640, 640))

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
            'class_name': CLASSES[class_ids[index]],
            'conf': scores[index],
        }
        
        detection['x1'] = box[0]
        detection['x2'] = box[0]+box[2]
        detection['y1'] = box[1]
        detection['y2'] = box[1]+box[3]

        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0]), round(box[1]),
                          round((box[0] + box[2])), round((box[1] + box[3])))
    
    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == '__main__':
    main("models/best_float32.tflite", "samples/20485139e526d5b4.jpg")