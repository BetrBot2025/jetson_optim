import cv2
import numpy as np

def letterbox_bgr(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize with unchanged aspect ratio using padding (YOLO-style letterbox).
    Returns:
      padded_image, scale, (new_w, new_h)
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r, (nw, nh)

def xywh2xyxy(xywh):
    x, y, w, h = xywh.T
    return np.vstack([x - w/2, y - h/2, x + w/2, y + h/2]).T

def iou_xyxy(a, b):
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)

def nms(boxes, scores, iou_th=0.5):
    if len(boxes) == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return keep

def decode_yolov8(output, conf_th=0.25):
    """
    Decode YOLOv8-like output from TensorRT.
    Supports common layouts:
      (1, N, 85) or (1, 85, N) or (1, 84, N)
    Returns:
      boxes_xyxy: (M, 4)
      conf:       (M,)
      cls:        (M,)
    """
    out = output
    if out.ndim == 3:
        out = out[0]

    # Case A: (N, 85/84)
    if out.shape[-1] in (85, 84):
        xywh = out[:, :4]
        if out.shape[-1] == 85:
            obj = out[:, 4:5]
            cls_logits = out[:, 5:]
        else:  # 84
            obj = np.ones((out.shape[0], 1), dtype=out.dtype)
            cls_logits = out[:, 4:]
        cls_id = cls_logits.argmax(1)
        cls_conf = cls_logits.max(1)
        conf = obj[:, 0] * cls_conf
        keep = conf > conf_th
        boxes_xyxy = xywh2xyxy(xywh)[keep]
        return boxes_xyxy, conf[keep], cls_id[keep]

    # Case B: (85/84, N)
    if out.shape[0] in (85, 84):
        xywh = out[:4, :].T
        if out.shape[0] == 85:
            obj = out[4:5, :].T
            cls_logits = out[5:, :].T
        else:
            obj = np.ones((out.shape[1], 1), dtype=out.dtype)
            cls_logits = out[4:, :].T
        cls_id = cls_logits.argmax(1)
        cls_conf = cls_logits.max(1)
        conf = obj[:, 0] * cls_conf
        keep = conf > conf_th
        boxes_xyxy = xywh2xyxy(xywh)[keep]
        return boxes_xyxy, conf[keep], cls_id[keep]

    raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")
