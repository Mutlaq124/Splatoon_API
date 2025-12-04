import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict

def load_model(path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)

def infer(session, inp: np.ndarray):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inp})

def preprocess_image(img: Image.Image, size: int = 640):
    """Resize image to 640x640, normalize, CHW."""
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0  # normalize 0-1
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # BCHW
    return arr

def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """
    Pure NumPy Non-Maximum Suppression
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for suppression
    Returns:
        keep_indices: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by score descending
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

def postprocess_detections(outputs, conf_thr: float, iou_thr: float, class_names: List[str]):
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

    # Apply NMS per class
    final = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = xyxy[cls_mask]
        cls_scores = scores[cls_mask]
        
        # Apply NumPy NMS
        keep_indices = numpy_nms(cls_boxes, cls_scores, iou_thr)
        
        for idx in keep_indices:
            box_idx = np.where(cls_mask)[0][idx]
            final.append({
                "bbox": [float(xyxy[box_idx][0]), float(xyxy[box_idx][1]), 
                        float(xyxy[box_idx][2]), float(xyxy[box_idx][3])],
                "confidence": float(scores[box_idx]),
                "class_id": int(cls),
                "class_name": class_names[int(cls)]
            })

    return final

def draw_boxes(img: np.ndarray, dets: List[Dict], thickness=2):
    """Draw bounding boxes on image"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    colors = {
        0: (0, 255, 0),      # aim_cursor - green
        1: (255, 0, 0),      # entity - blue
        2: (0, 0, 255)       # obstacle - red
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
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img