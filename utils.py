import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict


def load_model(path: str):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_options, providers=providers)


def infer(session, inp: np.ndarray):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inp})


def preprocess_image(img: Image.Image, size: int = 640):
    """Resize image to 640x640, normalize, CHW."""
    arr = np.array(img, dtype=np.float32)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = arr / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr


def postprocess_detections(
    outputs, conf_thr: float, iou_thr: float, class_names: List[str]
):
    """
    Postprocess YOLOv8 outputs with pure NumPy NMS
    """
    preds = outputs.squeeze().T  # [N, 84]

    boxes = preds[:, :4]
    scores = preds[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Confidence filter
    mask = confidences >= conf_thr
    boxes = boxes[mask]
    scores = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert cxcywh → xyxy
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    final = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), cls_scores.tolist(), conf_thr, iou_thr
        )

        for idx in indices:
            box_idx = np.where(cls_mask)[0][idx]
            final.append(
                {
                    "bbox": [
                        float(xyxy[box_idx][0]),
                        float(xyxy[box_idx][1]),
                        float(xyxy[box_idx][2]),
                        float(xyxy[box_idx][3]),
                    ],
                    "confidence": float(scores[box_idx]),
                    "class_id": int(cls),
                    "class_name": class_names[int(cls)],
                }
            )

    return final


def draw_boxes(img: np.ndarray, dets: List[Dict], thickness=2):
    """Draw bounding boxes on image"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    colors = {
        0: (0, 255, 0),  # aim_cursor - green
        1: (255, 0, 0),  # entity - blue
        2: (0, 0, 255),  # obstacle - red
    }

    for det in dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls = det["class_id"]
        conf = det["confidence"]
        color = colors.get(cls, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['class_name']} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img
