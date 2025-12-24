import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import cv2
import numpy as np
import time
import threading
from flask import Flask, Response

# --- 配置参数 ---
MODEL_PATH = 'yolo11s-rk3588.rknn'
MODEL_SIZE = (640, 640)
CONF_THRES = 0.30
IOU_THRES = 0.45
CAMERA_INDEXES = [20, 1, 11, 20, 21] # 尝试不同的摄像头索引
HOST = '0.0.0.0'
PORT = 5000

# 类别名称 (根据模型实际训练的类别修改)
# 检测结果显示该模型有 2 个类别
CLASSES = ['class0', 'class1'] 

app = Flask(__name__)

# 全局变量
rknn_lite = None
output_frame = None
lock = threading.Lock()

# --- 图像预处理 ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# --- 后处理 (针对 YOLOv8/v11 单输出 (1, 4+nc, 8400)) ---
def postprocess(outputs, img_shape, ratio, dwdh):
    # 自动判断模型结构
    # YOLOv8/v11: outputs[0] shape (1, 4+nc, 8400)
    
    output = outputs[0]
    
    # 检查维度，确保是 (1, C, N) 格式
    if output.ndim != 3:
        return []

    # Transpose to (1, 8400, 6) -> (batch, anchors, channels)
    output = output.transpose(0, 2, 1)
    
    # Squeeze batch dimension -> (8400, 6)
    prediction = output[0]
    
    # 动态计算类别数
    nc = prediction.shape[1] - 4
    if nc < 1:
        return []

    # Split boxes and scores
    # boxes: cx, cy, w, h
    boxes_cxcywh = prediction[:, :4]
    
    # 强制只取第0类 (index 4)
    # 无论模型有多少类，我们只关心第一个类别
    scores = prediction[:, 4] # (8400,) - Class 0 score
    
    # Filter by confidence
    mask = scores > CONF_THRES
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    
    if len(boxes_cxcywh) == 0:
        return []
        
    # Convert cx, cy, w, h to x1, y1, x2, y2
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    # boxes for drawing (x1, y1, x2, y2)
    boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)
    
    # boxes for NMS (x, y, w, h) -> top-left x, top-left y, width, height
    boxes_xywh = np.stack((x1, y1, w, h), axis=1)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), CONF_THRES, IOU_THRES)
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes_xyxy[i]
            score = scores[i]
            cls_id = 0 # 强制设为 0
            
            # Rescale boxes to original image
            box[0] = (box[0] - dwdh[0]) / ratio[0]
            box[1] = (box[1] - dwdh[1]) / ratio[1]
            box[2] = (box[2] - dwdh[0]) / ratio[0]
            box[3] = (box[3] - dwdh[1]) / ratio[1]
            
            # Clip boxes to image bounds
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(img_shape[1], box[2])
            box[3] = min(img_shape[0], box[3])
            
            results.append((box, score, cls_id))
            
    return results

def draw_detections(img, detections):
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        
        # 通用标签显示
        label = f"Target: {score:.2f}"
        
        # 统一使用红色框
        color = (0, 0, 255)
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# --- 主循环 ---
def detection_loop():
    global output_frame, rknn_lite
    
    # Initialize RKNN
    from rknnlite.api import RKNNLite
    rknn_lite = RKNNLite()
    
    print(f"--> Loading RKNN model: {MODEL_PATH}")
    ret = rknn_lite.load_rknn(MODEL_PATH)
    if ret != 0:
        print("Load RKNN model failed")
        return
        
    print("--> Init runtime environment")
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        return
        
    # Open Camera
    cap = None
    for idx in CAMERA_INDEXES:
        print(f"Trying camera index {idx}...")
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            print(f"Opened camera index {idx}")
            cap = temp_cap
            break
            
    if cap is None:
        print("No camera found.")
        return

    print("Starting inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input, ratio, dwdh = letterbox(img_rgb, MODEL_SIZE)
        
        # Add batch dimension
        input_tensor = np.expand_dims(img_input, 0)
        
        # Inference
        outputs = rknn_lite.inference(inputs=[input_tensor])
        
        # Postprocess
        detections = postprocess(outputs, frame.shape, ratio, dwdh)
        
        # Draw
        frame_show = draw_detections(frame.copy(), detections)
        
        # Update global frame
        with lock:
            output_frame = frame_show

# --- Flask 路由 ---
@app.route('/')
def index():
    return """
    <html>
    <head><title>YOLOv11n-CF Stream</title></head>
    <body>
        <h1>YOLOv11n-CF Real-time Detection</h1>
        <img src="/video_feed" width="800">
    </body>
    </html>
    """

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, encoded_image = cv2.imencode('.jpg', output_frame)
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()
    
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
