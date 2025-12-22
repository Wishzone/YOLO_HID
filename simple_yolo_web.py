import cv2
import numpy as np
import time
import os
import subprocess
import threading
from flask import Flask, Response

# ==========================================
#               配置参数区域
# ==========================================

# 运行引擎: 'cpu' 或 'rknn'
ENGINE = 'rknn'

# 模型路径
# 如果是 rknn 模式，建议使用 'yolov5s-640-640.rknn'
# 如果是 cpu 模式，建议使用 'yolo11n.pt'
# MODEL_PATH = 'yolov5s-640-640.rknn'
MODEL_PATH = 'yolov11s-1920-1080.rknn'

# 置信度阈值 (0.0 - 1.0)
# 只有得分高于此值的框才会被保留
CONF_THRES = 0.5

# NMS IoU 阈值 (0.0 - 1.0)
# 用于去除重叠框，值越小去除越严格
IOU_THRES = 0.30

# 摄像头索引列表 (按顺序尝试)
# 20: HDMI IN (rk_hdmirx)
# 21: USB Camera
CAMERA_INDEXES = [20, 21, 11, 0]

# 摄像头采集分辨率
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 模型输入尺寸 (RKNN 模型通常固定为 640x640)
MODEL_SIZE = (640, 640)

# Web 服务配置
HOST = '0.0.0.0'
PORT = 5000

# 目标类别 ID (0 代表 person)
TARGET_CLASS_ID = 0

# ==========================================

app = Flask(__name__)

# 全局变量
model_wrapper = None
output_frame = None
lock = threading.Lock()

# COCO 类名
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# --- 辅助函数 (用于 RKNN 后处理) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
        
        # Filter for target class and score
        mask = (max_scores > conf_thres) & (cls_ids == TARGET_CLASS_ID)
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

    mask = (scores >= conf_thres) & (cls_ids == TARGET_CLASS_ID)
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
        self.model_wh = MODEL_SIZE

    def detect(self, frame):
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb_img, ratio, (dw, dh) = letterbox(img_rgb, self.model_wh)
        input_tensor = np.expand_dims(lb_img, 0)
        
        # Inference
        outputs = self.rknn.inference(inputs=[input_tensor])
        
        # Postprocess
        dets = postprocess_yolo(outputs, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        
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
        # Only detect target class
        results = self.model(frame, device='cpu', verbose=False, classes=[TARGET_CLASS_ID], conf=CONF_THRES, iou=IOU_THRES)
        
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

def detection_loop():
    global output_frame, model_wrapper
    cap = None
    
    for idx in CAMERA_INDEXES:
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

    # 不强制设置分辨率，使用设备默认分辨率 (自适应)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # 获取实际分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前摄像头分辨率: {actual_width}x{actual_height}")

    # 鼠标当前位置 (初始化为画面中心)
    mouse_x = actual_width / 2
    mouse_y = actual_height / 2

    while True:
        success, frame = cap.read()
        if not success:
            print("无法读取帧 (Can't receive frame)")
            time.sleep(0.1)
            continue
        
        # 推理
        dets = model_wrapper.detect(frame)
        
        # 查找第一个人物并打印坐标
        found_person = False
        for det in dets:
            # det: [x1, y1, x2, y2, score, cls]
            cls_id = int(det[5])
            if cls_id == TARGET_CLASS_ID:
                x1, y1, x2, y2 = det[:4]
                print(f"检测到目标: 坐标 (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})")
                
                # 计算中心点
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 计算相对于当前鼠标位置的偏移量
                dx = cx - mouse_x
                dy = cy - mouse_y
                
                # 调用 hidtx 移动鼠标
                # 只有当偏移量足够大时才移动，避免抖动
                if abs(dx) >= 1 or abs(dy) >= 1:
                    try:
                        # 注意：这里假设 hidtx 在当前目录下，且有执行权限
                        subprocess.run(['sudo', '/home/pi/YOLO11n/hidtx', '/dev/hidg1', str(int(dx)), str(int(dy))], check=False)
                        # 更新鼠标位置
                        mouse_x = cx
                        mouse_y = cy
                    except Exception as e:
                        print(f"Error calling hidtx: {e}")

                found_person = True
                break # 只打印第一个
        
        # 绘制结果
        vis = frame.copy()
        vis = draw_detections(vis, dets)
        
        # 更新全局帧
        with lock:
            output_frame = vis.copy()

    cap.release()

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            # 将图像编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        
        # 生成流数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # 控制 Web 流的帧率

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
    """.format(ENGINE, MODEL_PATH)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 初始化模型
    if ENGINE == 'rknn':
        model_wrapper = YOLO_RKNN_Wrapper(MODEL_PATH)
    else:
        model_wrapper = YOLO_CPU_Wrapper(MODEL_PATH)

    # 启动检测线程
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    # 监听所有 IP
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
