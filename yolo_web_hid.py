import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import cv2
import numpy as np
import time
import subprocess
import threading
import os
import sys
from flask import Flask, Response
from queue import Queue

# 配置参数
MODEL_PATH = './Models/cf-11s-rk3588.rknn'
CONF_THRES = 0.7
IOU_THRES = 0.4
CAMERA_INDEXES = [20, 21, 11, 0]
PORT = 5000
TARGET_CLASS_ID = 0
ENABLE_HID = True # 是否启用HID控制
HID_MOVE_COOLDOWN = 0.0001 # HID移动冷却时间(秒) - 降低冷却时间以获得更平滑的移动
HID_DEVICE = '/dev/hidg1'
MOUSE_SENSITIVITY = 0.8 # 鼠标灵敏度系数

# 自动射击配置
AUTO_SHOOT = False # 是否启用自动射击
SHOOT_COOLDOWN = 0.2 # 射击冷却时间 (秒)
SHOOT_THRESHOLD = 15 # 射击触发范围 (像素距离)
SHOOT_DURATION = 0.02 # 点击持续时间 (秒)

# 准星偏移校准 (如果摄像头没有完全对准屏幕中心，调整这里)
AIM_OFFSET_X = 5 # 正数向右偏移，负数向左偏移
AIM_OFFSET_Y = 0 # 正数向下偏移，负数向上偏移
AIM_DEADZONE = 20 # 瞄准死区 (像素)，目标在此范围内不进行移动
AIM_HEIGHT_RATIO = 0.10 # 瞄准高度比例 (0.0=头顶, 0.5=中心, 1.0=脚底)

# PID 控制参数
PID_KP = 0.65  # 提高P值，追求"一次到位"的快速响应
PID_KI = 0.0   # 禁用I值，FPS场景下积分项容易导致滞后和震荡
PID_KD = 0.40  # 提高D值，提供强力刹车，抵消高P值带来的过冲风险
PID_MAX_INTEGRAL = 0 # 积分限幅

app = Flask(__name__)

# 全局变量
model_wrapper = None
output_frame = None
last_hid_move_time = 0
last_shoot_time = 0
hid_file = None # HID设备文件句柄
pid_state = {
    'last_error_x': 0,
    'last_error_y': 0,
    'integral_x': 0,
    'integral_y': 0
}
frame_id = 0
lock = threading.Lock()

# 初始化HID设备
try:
    # 尝试给予当前用户写权限
    subprocess.run(['sudo', 'chmod', '666', HID_DEVICE], check=False)
    # 使用 buffering=0 (无缓冲) 模式打开，提高写入速度
    hid_file = open(HID_DEVICE, 'wb', buffering=0)
    print(f"Successfully opened HID device: {HID_DEVICE}")
except Exception as e:
    print(f"Failed to open HID device: {e}")
    print("HID control will be disabled.")
    ENABLE_HID = False

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


class SilenceAndPrint:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.real_stdout = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None
        self.capture_path = "/tmp/rknn_capture.log"
        self.capture_file = None
        
    def __enter__(self):
        if self.verbose: 
            return sys.stdout
            
        sys.stdout.flush()
        sys.stderr.flush()
        
        self.saved_stdout_fd = os.dup(1)
        self.saved_stderr_fd = os.dup(2)
        
        self.real_stdout = os.fdopen(os.dup(self.saved_stdout_fd), 'w')
        
        self.capture_file = open(self.capture_path, 'w+')
        os.dup2(self.capture_file.fileno(), 1)
        os.dup2(self.capture_file.fileno(), 2)
        
        return self.real_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose: return
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        os.dup2(self.saved_stdout_fd, 1)
        os.dup2(self.saved_stderr_fd, 2)
        
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        
        if self.real_stdout:
            self.real_stdout.close()
        if self.capture_file:
            self.capture_file.close()

# --- 辅助函数 (用于 RKNN 后处理) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_rknn_yolov5_output(outputs, conf_thres=0.25, input_shape=(640, 640)):
    anchors_by_stride = {
        8:  [[10,13], [16,30], [33,23]],
        16: [[30,61], [62,45], [59,119]],
        32: [[116,90], [156,198], [373,326]]
    }
    
    all_boxes = []
    for output in outputs:
        bs, c, h, w = output.shape
        # 使用 input_shape 计算 stride
        stride = input_shape[0] // w
        if stride not in anchors_by_stride: continue
            
        grid_anchors = np.array(anchors_by_stride[stride]).reshape(3, 2)
        na, no = 3, c // 3
        
        output = output.reshape(bs, na, no, h, w).transpose(0, 1, 3, 4, 2)
        output = sigmoid(output)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, h, w, 2)
        anchors_tensor = grid_anchors.reshape(1, na, 1, 1, 2)
        
        xy = (output[..., 0:2] * 2 - 0.5 + grid) * stride
        wh = (output[..., 2:4] * 2) ** 2 * anchors_tensor
        scores = output[..., 4:5] * output[..., 5:]
        
        xy = xy.reshape(-1, 2)
        wh = wh.reshape(-1, 2)
        scores = scores.reshape(-1, scores.shape[-1])
        
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(scores.shape[0]), cls_ids]
        
        mask = (max_scores > conf_thres) & (cls_ids == TARGET_CLASS_ID)
        boxes = np.stack((xy[:, 0] - wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2, 
                          xy[:, 0] + wh[:, 0] / 2, xy[:, 1] + wh[:, 1] / 2), axis=1)
        
        if np.any(mask):
            all_boxes.append(np.concatenate([boxes[mask], max_scores[mask][:, None], cls_ids[mask][:, None].astype(np.float32)], axis=1))
        
    return np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 6), dtype=np.float32)

# Also compatible with YOLOv10 and YOLOv11
def process_rknn_yolov8_output(outputs, conf_thres=0.25, input_shape=(640, 640)):
    # YOLOv8/v10/v11 output shape is typically (1, 84, 8400)
    # 84 = 4 (box) + 80 (classes)
    output = outputs[0]
    
    # Remove batch dimension
    output = np.squeeze(output) # (84, 8400)
    
    # Transpose to (8400, 84) if needed
    if output.shape[0] < output.shape[1]:
        output = output.T
        
    # Split boxes and scores
    # YOLOv8/10/11 export usually gives cx, cy, w, h
    boxes = output[:, :4] 
    scores = output[:, 4:]
    
    # Get max score and class ID
    class_ids = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(scores.shape[0]), class_ids]
    
    # Filter
    mask = (max_scores > conf_thres) & (class_ids == TARGET_CLASS_ID)
    
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
        
    # Convert cx,cy,w,h to x1,y1,x2,y2
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    return np.concatenate([boxes_xyxy, max_scores[:, None], class_ids[:, None].astype(np.float32)], axis=1)

# Aliases for clarity
process_rknn_yolov10_output = process_rknn_yolov8_output
process_rknn_yolov11_output = process_rknn_yolov8_output

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = shape
    h, w = new_shape[1], new_shape[0]

    # 避免除以零错误
    if h0 == 0 or w0 == 0:
        return img, 1.0, (0, 0)

    r = min(w / w0, h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    
    # 确保 new_unpad 尺寸至少为 1x1
    new_unpad = (max(1, new_unpad[0]), max(1, new_unpad[1]))
    
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

def postprocess_yolo(outputs, conf_thres=0.25, iou_thres=0.45, input_shape=(640, 640)):
    dets = np.zeros((0, 6), dtype=np.float32)
    
    # Auto-detect output format
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 3 and outputs[0].ndim == 4:
             # Likely YOLOv5/v7 (3 output layers)
             dets = process_rknn_yolov5_output(outputs, conf_thres=conf_thres, input_shape=input_shape)
        elif len(outputs) == 1:
             # Likely YOLOv8/v10/v11 (1 output layer)
             dets = process_rknn_yolov8_output(outputs, conf_thres=conf_thres, input_shape=input_shape)
        else:
             # Try to handle generic case or warn
             try:
                 dets = process_rknn_yolov8_output(outputs, conf_thres=conf_thres, input_shape=input_shape)
             except:
                 print(f"Warning: Unknown output shape: {[o.shape for o in outputs]}")
                 return np.zeros((0, 6), dtype=np.float32)
    else:
        # Single tensor output
        dets = process_rknn_yolov8_output([outputs], conf_thres=conf_thres, input_shape=input_shape)

    if dets.shape[0] == 0: return dets
    
    # NMS
    keep = nms(dets[:, :4], dets[:, 4], iou_threshold=iou_thres)
    return dets[keep]

def draw_detections(img, dets):
    h, w = img.shape[:2]
    for x1, y1, x2, y2, score, cls in dets:
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1)))
        y2 = int(max(0, min(y2, h - 1)))

        cls = int(cls)
        label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
        color = (0, 0, 255) if label == 'person' else (0, 255, 0)

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
        
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            sys.exit(1)

        with SilenceAndPrint(verbose=False):
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                print('Load RKNN model failed')
                sys.exit(ret)
            self.rknn.init_runtime()
            
        self.model_wh = self._probe_model_size()
        # print(f"--> Detected RKNN model input size: {self.model_wh}")

    def _probe_model_size(self):
        # Probe with dummy inputs
        test_sizes = [320, 416, 512, 640, 736, 768, 960, 1280]
        
        with SilenceAndPrint(verbose=False) as out:
            for size in test_sizes:
                width, height = size, size
                try:
                    # 创建虚拟输入 (NHWC format usually for RKNN inputs via Python API)
                    img = np.zeros((1, height, width, 3), dtype=np.uint8)
                    outputs = self.rknn.inference(inputs=[img])
                    if outputs is not None:
                        if out: out.write(f"SUCCESS: Model accepts input size {width}x{height}\n")
                        return (width, height)
                except Exception:
                    pass
        
        print("Warning: Could not determine model size, defaulting to 640x640")
        return (640, 640)

    def detect(self, frame):
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb_img, ratio, (dw, dh) = letterbox(img_rgb, self.model_wh)
        input_tensor = np.expand_dims(lb_img, 0)
        
        # Inference
        outputs = self.rknn.inference(inputs=[input_tensor])

        # Postprocess
        dets = postprocess_yolo(outputs, conf_thres=CONF_THRES, iou_thres=IOU_THRES, input_shape=self.model_wh)
        
        # Map back to original image
        if dets.shape[0] > 0:
            dets_xyxy = dets[:, :4]
            dets_xyxy[:, [0, 2]] -= dw
            dets_xyxy[:, [1, 3]] -= dh
            dets_xyxy /= ratio
            dets[:, :4] = dets_xyxy
            
        return dets

class MultiYOLO_RKNN_Wrapper:
    def __init__(self, model_path, num_threads=3):
        self.wrappers = []
        # Load first one to probe size
        print(f"Loading NPU model 1/{num_threads}...")
        w1 = YOLO_RKNN_Wrapper(model_path)
        self.wrappers.append(w1)
        self.model_wh = w1.model_wh
        
        for i in range(num_threads - 1):
            print(f"Loading NPU model {i+2}/{num_threads}...")
            w = YOLO_RKNN_Wrapper(model_path)
            # Ensure all wrappers use the same size (avoid re-probing if possible, 
            # though YOLO_RKNN_Wrapper currently probes in __init__. 
            # We could optimize this but for now it's fine as it only happens on startup)
            self.wrappers.append(w)
            
        self.input_queue = Queue(maxsize=num_threads + 2)
        self.output_queue = Queue()
        self.running = True
        self.threads = []
        
    def start(self):
        for i, wrapper in enumerate(self.wrappers):
            t = threading.Thread(target=self._worker, args=(wrapper,))
            t.daemon = True
            t.start()
            self.threads.append(t)
            
    def _worker(self, wrapper):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
            except:
                continue
                
            frame = item
            # Detect
            try:
                dets = wrapper.detect(frame)
                self.output_queue.put((frame, dets))
            except Exception as e:
                print(f"Inference error: {e}")
            
    def detect_async(self, frame):
        if not self.input_queue.full():
            self.input_queue.put(frame)
            
    def get_result(self):
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None, None

class YOLO_CPU_Wrapper:
    def __init__(self, model_path):
        from ultralytics import YOLO
        # print(f'--> Loading CPU model: {model_path}')
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

def trigger_click():
    global hid_file
    if not hid_file: return
    try:
        # Press Left Button (0x01)
        hid_file.write(bytearray([0x01, 0, 0, 0]))
        # hid_file.flush() # Not needed with buffering=0
        
        # Short delay for click registration
        time.sleep(SHOOT_DURATION)
        
        # Release
        hid_file.write(bytearray([0, 0, 0, 0]))
        # hid_file.flush()
    except Exception as e:
        print(f"Click error: {e}")

def process_frame_result(frame, dets, center_x, center_y):
    global output_frame, last_hid_move_time, pid_state, last_shoot_time
    
    # Find target closest to center
    target = None
    min_dist = float('inf')
    
    for det in dets:
        if int(det[5]) == TARGET_CLASS_ID:
            x1, y1, x2, y2 = det[:4]
            cx = (x1 + x2) / 2
            # 计算瞄准点高度 (默认0.25即头部位置)
            cy = y1 + (y2 - y1) * AIM_HEIGHT_RATIO
            
            dist = (cx - center_x)**2 + (cy - center_y)**2
            if dist < min_dist:
                min_dist = dist
                target = (cx, cy)
    
    if target:
        cx, cy = target
        raw_dx = cx - center_x
        raw_dy = cy - center_y
        
        # Auto Shoot Logic
        current_time = time.time()
        if AUTO_SHOOT and ENABLE_HID and hid_file:
            # Check if target is within shooting threshold
            if abs(raw_dx) < SHOOT_THRESHOLD and abs(raw_dy) < SHOOT_THRESHOLD:
                if current_time - last_shoot_time > SHOOT_COOLDOWN:
                    # Trigger click in a separate thread to avoid blocking main loop
                    threading.Thread(target=trigger_click).start()
                    last_shoot_time = current_time

        # Apply Deadzone
        if abs(raw_dx) < AIM_DEADZONE: raw_dx = 0
        if abs(raw_dy) < AIM_DEADZONE: raw_dy = 0

        # PID Calculation
        # X Axis
        derivative_x = raw_dx - pid_state['last_error_x']
        move_x = (PID_KP * raw_dx) + (PID_KD * derivative_x)
        pid_state['last_error_x'] = raw_dx

        # Y Axis
        derivative_y = raw_dy - pid_state['last_error_y']
        move_y = (PID_KP * raw_dy) + (PID_KD * derivative_y)
        pid_state['last_error_y'] = raw_dy

        dx = int(move_x)
        dy = int(move_y)
        
        # Deadzone and sensitivity
        if ENABLE_HID and hid_file and (abs(dx) >= 1 or abs(dy) >= 1) and (current_time - last_hid_move_time > HID_MOVE_COOLDOWN):
            try:
                # 应用灵敏度系数
                final_dx = int(dx * MOUSE_SENSITIVITY)
                final_dy = int(dy * MOUSE_SENSITIVITY)
                
                # 限制单次移动最大值 (HID报告限制为 -127 到 127)
                final_dx = max(-127, min(127, final_dx))
                final_dy = max(-127, min(127, final_dy))
                
                if final_dx != 0 or final_dy != 0:
                    # 构建HID报告: [buttons, x, y, wheel]
                    # x, y 需要转换为有符号字节
                    buf = bytearray([0, final_dx & 0xFF, final_dy & 0xFF, 0])
                    hid_file.write(buf)
                    # hid_file.flush() # Not needed with buffering=0
                    last_hid_move_time = current_time
            except Exception as e:
                print(f"HID Write Error: {e}")
                pass
    else:
        # Reset PID integral when no target found to prevent windup
        pid_state['integral_x'] = 0
        pid_state['integral_y'] = 0
        pid_state['last_error_x'] = 0
        pid_state['last_error_y'] = 0
    
    vis = draw_detections(frame.copy(), dets)
    # Draw crosshair
    cv2.line(vis, (int(center_x)-10, int(center_y)), (int(center_x)+10, int(center_y)), (0,255,255), 1)
    cv2.line(vis, (int(center_x), int(center_y)-10), (int(center_x), int(center_y)+10), (0,255,255), 1)
    
    return vis

def detection_loop():
    global output_frame, model_wrapper
    cap = None
    
    for idx in CAMERA_INDEXES:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            # print(f"Opened camera index {idx}")
            cap = temp_cap
            break
    
    if cap is None or not cap.isOpened():
        print("No camera found")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = (actual_width / 2) + AIM_OFFSET_X
    center_y = (actual_height / 2) + AIM_OFFSET_Y
    
    # print(f"Camera resolution: {actual_width}x{actual_height}")

    # Start async wrapper if applicable
    if hasattr(model_wrapper, 'start'):
        model_wrapper.start()

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        # 修复某些特定 OpenCV 版本返回扁平化数组的问题
        if frame.ndim == 2 and frame.shape[0] == 1:
            expected_len = actual_width * actual_height * 3
            if frame.shape[1] == expected_len:
                try:
                    frame = frame.reshape((actual_height, actual_width, 3))
                except Exception as e:
                    # print(f"Reshape failed: {e}")
                    continue
        
        vis = None
        
        if hasattr(model_wrapper, 'detect_async'):
            model_wrapper.detect_async(frame.copy()) # Pass copy to avoid race condition if frame buffer reused
            
            # Process all available results
            while True:
                res_frame, dets = model_wrapper.get_result()
                if res_frame is None:
                    break
                
                vis = process_frame_result(res_frame, dets, center_x, center_y)
                
                # Calculate FPS (based on processed frames)
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
        else:
            dets = model_wrapper.detect(frame)
            vis = process_frame_result(frame, dets, center_x, center_y)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
        
        if vis is not None:
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Sanity check
            if vis.shape[0] > 4000 or vis.shape[1] > 4000:
                # print(f"Warning: Abnormal frame size detected: {vis.shape}, skipping...")
                continue

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
            
            # 检查图像尺寸
            h, w = output_frame.shape[:2]
            if h == 0 or w == 0 or h > 4000 or w > 4000:
                # print(f"Skipping invalid frame size: {w}x{h}")
                continue

            try:
                # 将图像编码为 JPEG
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"Encode error: {e}, shape={output_frame.shape}")
                continue
        
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
        <p>Model: {}</p>
        <img src="/video_feed" width="100%">
      </body>
    </html>
    """.format(MODEL_PATH)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 初始化模型
    if MODEL_PATH.endswith('.rknn'):
        # Use Multi-NPU wrapper for RKNN models
        model_wrapper = MultiYOLO_RKNN_Wrapper(MODEL_PATH, num_threads=3)
    elif MODEL_PATH.endswith('.pt'):
        model_wrapper = YOLO_CPU_Wrapper(MODEL_PATH)
    else:
        print(f"Error: Unsupported model format: {MODEL_PATH}")
        sys.exit(1)

    # 启动检测线程
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    # 监听所有 IP
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
