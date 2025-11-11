import argparse
import time
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

from trt_yolo import TrtRunner
from postproc import letterbox_bgr, decode_yolov8, nms

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

app = FastAPI(title="YOLO TensorRT Service", version="1.0")

_runner: TrtRunner = None   # Will be set in main()

def draw_boxes(img_bgr, boxes, scores, cls_ids):
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, cls_ids):
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cname = COCO_NAMES[int(c)] if 0 <= int(c) < len(COCO_NAMES) else str(int(c))
        label = f"{cname}:{s:.2f}"
        ytxt = max(0, y1 - 6)
        cv2.putText(img_bgr, label, (x1, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_bgr

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Simple HTML page:
    - Upload an image
    - See annotated detections in an <img> frame
    """
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>YOLO TensorRT Visualizer</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 20px; }
      .row { display: flex; gap: 20px; align-items: flex-start; }
      .panel { border: 1px solid #ddd; padding: 12px; border-radius: 8px; }
      img { max-width: 640px; height: auto; border: 1px solid #ccc; }
      button { padding: 8px 12px; margin-top: 8px; }
      label { display:block; margin-top: 8px; }
    </style>
  </head>
  <body>
    <h2>YOLO TensorRT (Annotated Output)</h2>
    <div class="row">
      <div class="panel">
        <form id="f" enctype="multipart/form-data" method="post" action="/detect_visual" target="result_frame">
          <label>Image:
            <input type="file" name="image" accept="image/*" required/>
          </label>
          <label>Confidence threshold:
            <input name="conf" type="number" step="0.01" value="0.25"/>
          </label>
          <label>IoU threshold:
            <input name="iou" type="number" step="0.01" value="0.50"/>
          </label>
          <button type="submit">Run Detection</button>
        </form>
      </div>
      <div class="panel">
        <h4>Result</h4>
        <iframe name="result_frame" width="700" height="520" style="border:none;"></iframe>
      </div>
    </div>
    <hr/>
    <p>You can also call the JSON API at <code>/detect</code> or see docs at <a href="/docs">/docs</a>.</p>
  </body>
</html>
    """

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float  = Form(0.50),
):
    """
    JSON API: returns bbox coordinates, class ids, names, confidences.
    """
    raw = await image.read()
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if im is None:
        return JSONResponse({"error": "unable to decode image"}, status_code=400)

    h, w = im.shape[:2]
    lb, scale, _ = letterbox_bgr(im, (640, 640))
    rgb = lb[:, :, ::-1].astype(np.float32) / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None, ...]

    t0 = time.time()
    out = _runner.infer(nchw)
    boxes, scores, cls_ids = decode_yolov8(out, conf_th=conf)

    results = []
    if len(boxes) > 0:
        gain = scale
        boxes[:, [0, 2]] /= gain
        boxes[:, [1, 3]] /= gain
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        keep = nms(boxes, scores, iou_th=iou)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls_ids):
            results.append({
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(s),
                "class_id": int(c),
                "class_name": COCO_NAMES[int(c)] if 0 <= int(c) < len(COCO_NAMES) else str(int(c)),
            })

    latency_ms = round((time.time() - t0) * 1000.0, 2)
    return {
        "shape_hw": [int(h), int(w)],
        "count": len(results),
        "latency_ms": latency_ms,
        "detections": results,
    }

@app.post("/detect_visual")
async def detect_visual(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float  = Form(0.50),
):
    """
    Visual API: returns annotated JPEG with boxes drawn.
    Used by the HTML form at "/".
    """
    raw = await image.read()
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if im is None:
        return JSONResponse({"error": "unable to decode image"}, status_code=400)

    h, w = im.shape[:2]
    lb, scale, _ = letterbox_bgr(im, (640, 640))
    rgb = lb[:, :, ::-1].astype(np.float32) / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None, ...]

    out = _runner.infer(nchw)
    boxes, scores, cls_ids = decode_yolov8(out, conf_th=conf)

    if len(boxes) > 0:
        gain = scale
        boxes[:, [0, 2]] /= gain
        boxes[:, [1, 3]] /= gain
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        keep = nms(boxes, scores, iou_th=iou)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        im = draw_boxes(im, boxes, scores, cls_ids)

    ok, enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return JSONResponse({"error": "jpeg encode failed"}, status_code=500)

    return Response(content=enc.tobytes(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global _runner
    _runner = TrtRunner(args.engine)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
