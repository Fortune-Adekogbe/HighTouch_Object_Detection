
import supervision as sv

def get_results(pred):
    classes = [pred.names[int(i)].title() for i in pred.boxes.cls]
    bboxes_xyxy = pred.boxes.xyxy.cpu().numpy().tolist()
    confidences = pred.boxes.conf.cpu().numpy().tolist()
    output = []
    for i in range(len(confidences)):
        res = {
            'conf': confidences[i],
            'class': classes[i],

        }
        x1, y1, x2, y2 = bboxes_xyxy[i]
        res['x1'] = x1
        res['x2'] = x2
        res['y1'] = y1
        res['y2'] = y2
        output.append(res)
    return output

def visualize(dpred, cpred, image):
    detections = [sv.Detections.from_yolov8(dpred), sv.Detections.from_yolov8(cpred)]

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(text_scale=1.5)

    dlabels = [
        f"{dpred.names[int(class_id)]} {float(confidence):0.2f}"
        for _, _, confidence, class_id, _ in detections[0]
    ]
    clabels = [
        f"{cpred.names[int(class_id)]} {float(confidence):0.2f}"
        for _, _, confidence, class_id, _ in detections[1]
    ]

    annotated_frame_1 = box_annotator.annotate(
        scene=image.copy(), detections=detections[0], labels=dlabels
    )
    annotated_frame_2 = box_annotator.annotate(
        scene=annotated_frame_1, detections=detections[1], labels=clabels
    )

    sv.plot_image(annotated_frame_2, (16, 16))

# img_path = "samples/4803144044_fdaf3d855f_o.jpg" # "data/4803144044_fdaf3d855f_o.jpg"

# predictions, raw_pred = inference(img_path)
# print(predictions)
# visualize(raw_pred[0], raw_pred[1], cv2.imread(img_path))