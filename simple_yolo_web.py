import argparse
import cv2
import numpy as np
import time
import os
from flask import Flask, Response

app = Flask(__name__)

# 全局变量
args = None
model_wrapper = None

# COCO 类名
COCO_CLASSES = [
    'person'
]

# --- 辅助函数 (用于 RKNN 后处理) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_rknn_yolov5_output(outputs, input_shape=(640, 640)):
    # 处理 YOLOv5 原始输出 (3个 feature map)
    # anchors: list of 3 lists, e.g. [[10,13, 16,30, 33,23], ...]
    
    all_boxes = []
    
    for i, output in enumerate(outputs):
        # output shape: (1, 255, h, w)
        bs, c, h, w = output.shape
        stride = input_shape[0] // w
        
        # 对应的 anchors
        grid_anchors = anchors[i] # [w,h, w,h, w,h]
        grid_anchors = np.array(grid_anchors).reshape(3, 2)
        
        na = 3 # number of anchors
        no = c // na # 85
        
        # Reshape and transpose: (bs, na, no, h, w) -> (bs, na, h, w, no)
        output = output.reshape(bs, na, no, h, w).transpose(0, 1, 3, 4, 2)
        
        # Sigmoid
        output = sigmoid(output)
        
        # Grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, h, w, 2)
        
        # Anchors tensor
        anchors_tensor = grid_anchors.reshape(1, na, 1, 1, 2)
        
        # Decode xy, wh
        # xy = (xy * 2 - 0.5 + grid) * stride
        # wh = (wh * 2) ** 2 * anchors
        xy = (output[..., 0:2] * 2 - 0.5 + grid) * stride
        wh = (output[..., 2:4] * 2) ** 2 * anchors_tensor
        
        # Conf and Cls
        obj_conf = output[..., 4:5]
        cls_conf = output[..., 5:]
        
        # Combine scores: obj * cls
        scores = obj_conf * cls_conf
        
        # Flatten
        xy = xy.reshape(-1, 2)
        wh = wh.reshape(-1, 2)
        scores = scores.reshape(-1, scores.shape[-1])
        
        # Filter by conf threshold early to save time? 
        # For simplicity, just convert all and let NMS handle it, or filter here.
        
        # cxcy to xyxy
        x1 = xy[:, 0] - wh[:, 0] / 2
        y1 = xy[:, 1] - wh[:, 1] / 2
        x2 = xy[:, 0] + wh[:, 0] / 2
        y2 = xy[:, 1] + wh[:, 1] / 2
        
        boxes = np.stack((x1, y1, x2, y2), axis=1)
        
        # Concat boxes and scores
        # Output format for NMS: (x1, y1, x2, y2, score, cls_id)
        # But here we have scores for all classes.
        # We can pick best class.
        
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(scores.shape[0]), cls_ids]
        
        # Filter
        mask = max_scores > 0.25 # Hardcoded low threshold for pre-filtering
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        cls_ids = cls_ids[mask]
        
        dets = np.concatenate([
            boxes, 
            max_scores[:, None], 
            cls_ids[:, None].astype(np.float32)
        ], axis=1)
        
        all_boxes.append(dets)
        
    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)
        
    return np.concatenate(all_boxes, axis=0)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = shape
    h, w = new_shape[1], new_shape[0]

    r = min(w / w0, h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = w - new_unpad[0], h - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def nms(boxes, scores, iou_threshold=0.45):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def process_rknn_yolov5_output(outputs, conf_thres=0.25, input_shape=(640, 640)):
    # 定义 Anchors (YOLOv5s default)
    anchors_by_stride = {
        8:  [[10,13], [16,30], [33,23]],
        16: [[30,61], [62,45], [59,119]],
        32: [[116,90], [156,198], [373,326]]
    }
    
    all_boxes = []
    
    for output in outputs:
        # output shape: (1, 255, h, w)
        bs, c, h, w = output.shape
        stride = input_shape[0] // w
        
        # 根据 stride 获取对应的 anchors
        if stride not in anchors_by_stride:
            print(f"Warning: Unknown stride {stride}, skipping.")
            continue
            
        grid_anchors = anchors_by_stride[stride]
        grid_anchors = np.array(grid_anchors).reshape(3, 2)
        
        na = 3 # number of anchors
        no = c // na # 85
        
        # Reshape and transpose: (bs, na, no, h, w) -> (bs, na, h, w, no)
        output = output.reshape(bs, na, no, h, w).transpose(0, 1, 3, 4, 2)
        
        # Sigmoid
        output = sigmoid(output)
        
        # Grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, h, w, 2)
        
        # Anchors tensor
        anchors_tensor = grid_anchors.reshape(1, na, 1, 1, 2)
        
        # Decode xy, wh
        xy = (output[..., 0:2] * 2 - 0.5 + grid) * stride
        wh = (output[..., 2:4] * 2) ** 2 * anchors_tensor
        
        # Conf and Cls
        obj_conf = output[..., 4:5]
        cls_conf = output[..., 5:]
        
        # Combine scores: obj * cls
        scores = obj_conf * cls_conf
        
        # Flatten
        xy = xy.reshape(-1, 2)
        wh = wh.reshape(-1, 2)
        scores = scores.reshape(-1, scores.shape[-1])
        
        # Filter by conf threshold
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(scores.shape[0]), cls_ids]
        
        # Filter for person (class 0) and score
        mask = (max_scores > conf_thres) & (cls_ids == 0)
        boxes = np.stack((xy[:, 0] - wh[:, 0] / 2, 
                          xy[:, 1] - wh[:, 1] / 2, 
                          xy[:, 0] + wh[:, 0] / 2, 
                          xy[:, 1] + wh[:, 1] / 2), axis=1)
        
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        cls_ids = cls_ids[mask]
        
        if boxes.shape[0] > 0:
            dets = np.concatenate([
                boxes, 
                max_scores[:, None], 
                cls_ids[:, None].astype(np.float32)
            ], axis=1)
            all_boxes.append(dets)
        
    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)
        
    return np.concatenate(all_boxes, axis=0)

def postprocess_yolo(outputs, conf_thres=0.25, iou_thres=0.45):
    # Check if outputs are raw YOLOv5 feature maps (3 tensors, 255 channels)
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        # Check shape of first output
        if outputs[0].ndim == 4 and outputs[0].shape[1] == 255:
            dets = process_rknn_yolov5_output(outputs, conf_thres=conf_thres)
            # NMS
            if dets.shape[0] == 0:
                return dets
            
            boxes = dets[:, :4]
            scores = dets[:, 4]
            cls_ids = dets[:, 5]
            
            keep = nms(boxes, scores, iou_threshold=iou_thres)
            return dets[keep]

    if isinstance(outputs, (list, tuple)):
        outs = [o for o in outputs if isinstance(o, np.ndarray)]
        if not outs:
            return np.zeros((0, 6), dtype=np.float32)
        out = max(outs, key=lambda a: (a.shape[-1] if a.ndim >= 2 else 0))
    else:
        out = outputs

    out = np.squeeze(out)
    if out.ndim == 1:
        out = np.expand_dims(out, 0)

    if out.shape[-1] < 10:
        return np.zeros((0, 6), dtype=np.float32)

    boxes = out[:, :4]
    cls_scores = out[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    mask = (scores >= conf_thres) & (cls_ids == 0)
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if boxes.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    cxcywh_like = np.mean(boxes[:, 2] > 0) > 0.5 and np.mean(boxes[:, 3] > 0) > 0.5
    if cxcywh_like:
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, scores, iou_threshold=iou_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    dets = np.concatenate([
        boxes.astype(np.float32),
        scores[:, None].astype(np.float32),
        cls_ids[:, None].astype(np.float32)
    ], axis=1)
    return dets

def draw_detections(img, dets):
    h, w = img.shape[:2]
    for x1, y1, x2, y2, score, cls in dets:
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        cls = int(cls)
        label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
        
        color = (0, 255, 0)
        if label == 'person':
            color = (0, 0, 255) # Red for person

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f'{label} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# --- 模型封装类 ---

class YOLO_RKNN_Wrapper:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        print(f'--> Loading RKNN model: {model_path}')
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        self.model_wh = (640, 640) # 默认 640x640

    def detect(self, frame):
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb_img, ratio, (dw, dh) = letterbox(img_rgb, self.model_wh)
        input_tensor = np.expand_dims(lb_img, 0)
        
        # Inference
        outputs = self.rknn.inference(inputs=[input_tensor])
        
        # Postprocess
        dets = postprocess_yolo(outputs, conf_thres=0.5, iou_thres=0.25)
        
        # Map back to original image
        if dets.shape[0] > 0:
            dets_xyxy = dets[:, :4]
            dets_xyxy[:, [0, 2]] -= dw
            dets_xyxy[:, [1, 3]] -= dh
            dets_xyxy /= ratio
            dets[:, :4] = dets_xyxy
            
        return dets

class YOLO_CPU_Wrapper:
    def __init__(self, model_path):
        from ultralytics import YOLO
        print(f'--> Loading CPU model: {model_path}')
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        # Only detect class 0 (person)
        results = self.model(frame, device='cpu', verbose=False, classes=[0])
        
        det_list = []
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            det_list.append([x1, y1, x2, y2, conf, cls])
        
        if len(det_list) > 0:
            return np.array(det_list)
        else:
            return np.zeros((0, 6))

def generate_frames():
    # 尝试打开 USB 摄像头 /dev/video21 (HIK 2K Camera)
    # 如果失败，再尝试 /dev/video11 (ISP) 或 /dev/video0
    camera_indexes = [21, 11, 0]
    cap = None
    
    for idx in camera_indexes:
        print(f"尝试打开摄像头 index {idx} ...")
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            print(f"成功打开摄像头 index {idx}")
            cap = temp_cap
            break
        else:
            print(f"无法打开摄像头 index {idx}")
    
    if cap is None or not cap.isOpened():
        print("无法打开任何摄像头")
        return

    # 尝试设置分辨率为 640x480 以确保兼容性
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            print("无法读取帧 (Can't receive frame)")
            break
        
        # 推理
        dets = model_wrapper.detect(frame)
        
        # 查找第一个人物并打印坐标
        found_person = False
        for det in dets:
            # det: [x1, y1, x2, y2, score, cls]
            cls_id = int(det[5])
            if COCO_CLASSES[cls_id] == 'person':
                x1, y1, x2, y2 = det[:4]
                print(f"检测到人物: 坐标 (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})")
                found_person = True
                break # 只打印第一个
        
        # 绘制结果
        vis = frame.copy()
        vis = draw_detections(vis, dets)
        
        # 将图像编码为 JPEG
        ret, buffer = cv2.imencode('.jpg', vis)
        frame_bytes = buffer.tobytes()
        
        # 生成流数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>YOLO Web Stream</title>
      </head>
      <body>
        <h1>YOLO Video Stream</h1>
        <p>Engine: {} | Model: {}</p>
        <img src="/video_feed" width="100%">
      </body>
    </html>
    """.format(args.engine, args.model)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Web Stream (CPU/RKNN)')
    parser.add_argument('--engine', type=str, default='cpu', choices=['cpu', 'rknn'], help='Inference engine: cpu or rknn')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    
    args = parser.parse_args()
    
    # 设置默认模型路径
    if args.model is None:
        if args.engine == 'rknn':
            args.model = 'yolov5s-640-640.rknn'
        else:
            args.model = 'yolo11n.pt'
            
    # 初始化模型
    if args.engine == 'rknn':
        model_wrapper = YOLO_RKNN_Wrapper(args.model)
    else:
        model_wrapper = YOLO_CPU_Wrapper(args.model)

    # 监听所有 IP，端口 5000
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
