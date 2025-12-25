import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import cv2
import numpy as np
import time
import subprocess
import threading
import os
import sys
from flask import Flask, Response, request, jsonify
from queue import Queue

import atexit

# 配置参数
MODEL_PATH = './Models/cf-11n-rk3588.rknn'
# MODEL_PATH = './Models/cf-11s-rk3588-int8.rknn'
CONF_THRES = 0.5
IOU_THRES = 0.4
USE_RGB_INPUT = True # INT8 模型通常需要 BGR 输入，如果识别不准请尝试改为 True
CAMERA_INDEXES = [20] # 仅使用 HDMI IN
PORT = 5000
TARGET_CLASS_ID = 0
ENABLE_HID = True # 是否启用HID控制
HID_MOVE_COOLDOWN = 0.0005 # HID移动冷却时间(秒) - 降低冷却时间以获得更平滑的移动
HID_DEVICE = '/dev/hidg1'
MOUSE_SENSITIVITY = 0.4 # 鼠标灵敏度系数

# 自动射击配置
AUTO_SHOOT = True # 是否启用自动射击
SHOOT_COOLDOWN = 0.03 # 射击冷却时间 (秒) - 增加冷却时间以防止卡顿
SHOOT_THRESHOLD = 30 # 射击触发范围 (像素距离)
SHOOT_DURATION = 0.02 # 点击持续时间 (秒)
SHOOT_PREDICTION = 9 # 射击预测 (帧) - 提前多少帧进行射击判定
SHOOT_SUSTAIN_TIME = 0 # 射击信号保持时间 (秒)
RECOIL_STRENGTH = 3.0 # 压枪力度 (像素/次)

# 准星偏移校准 (如果摄像头没有完全对准屏幕中心，调整这里)
AIM_OFFSET_X = 0 # 正数向右偏移，负数向左偏移
AIM_OFFSET_Y = 0 # 正数向下偏移，负数向上偏移
AIM_DEADZONE = 0 # 瞄准死区 (像素)，目标在此范围内不进行移动
AIM_HEIGHT_RATIO = 0.10 # 瞄准高度比例 (0.0=头顶, 0.5=中心, 1.0=脚底)
HID_SMOOTH_FACTOR = 0.3 # 平滑移动系数 (0.1~1.0)，越小越平滑但越慢
HID_UPDATE_INTERVAL = 0.003 # HID更新间隔 (秒), 0.002 = 500Hz

# PID 控制参数
PID_KP = 0.25  # 降低P值，防止因延迟导致的过冲震荡
PID_KI = 0.001   # 禁用I值
PID_KD = 0.15  # 大幅提高D值，利用微分项预测趋势，提前刹车
PID_MAX_INTEGRAL = 0 # 积分限幅

app = Flask(__name__)

# 全局变量
model_wrapper = None
output_frame = None
last_hid_move_time = 0
should_shoot = False # 射击控制标志
last_shoot_signal_time = 0 # 上次触发射击信号的时间
hid_file = None # HID设备文件句柄
hid_buffer_x = 0.0
hid_buffer_y = 0.0
hid_buttons = 0 # 当前按键状态
hid_lock = threading.Lock()
hid_cond = threading.Condition(hid_lock) # Condition variable for immediate wake-up
pid_state = {
    'last_error_x': 0,
    'last_error_y': 0,
    'integral_x': 0,
    'integral_y': 0,
    'tracking_frames': 0
}
frame_id = 0
lock = threading.Lock()

def cleanup():
    global hid_file
    print("Cleaning up resources...")
    if hid_file:
        try:
            hid_file.close()
            print("HID device closed.")
        except:
            pass
    # Stop threads if possible (not strictly necessary as they are daemons)

atexit.register(cleanup)

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

import ctypes

# C Structure Definition
class CDetection(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int)
    ]

class YOLO_RKNN_C_Wrapper:
    def __init__(self, model_path, known_size=None):
        self.lib = ctypes.CDLL('./c_src/librknn_yolo.so')
        
        # Profiling
        self.t_pre = 0
        self.t_infer = 0
        self.count = 0
        
        # Define argument types
        self.lib.init_model.argtypes = [ctypes.c_char_p]
        self.lib.init_model.restype = ctypes.c_void_p
        
        self.lib.detect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(CDetection),
            ctypes.c_int
        ]
        self.lib.detect.restype = ctypes.c_int
        
        self.lib.release_model.argtypes = [ctypes.c_void_p]
        
        # Init model
        self.ctx = self.lib.init_model(model_path.encode('utf-8'))
        if not self.ctx:
            raise RuntimeError("Failed to init C RKNN model")
            
        if known_size:
            self.model_wh = known_size
        else:
            # Assume 640x640 if not provided, or we could add a C function to get it
            self.model_wh = (640, 640)
            
        self.results_buffer = (CDetection * 100)()

    def detect(self, frame):
        t0 = time.time()
        # Preprocess (Letterbox)
        if USE_RGB_INPUT:
            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            input_img = frame
            
        lb_img, ratio, (dw, dh) = letterbox(input_img, self.model_wh)
        
        # Get pointer to image data
        # Ensure contiguous array
        if not lb_img.flags['C_CONTIGUOUS']:
            lb_img = np.ascontiguousarray(lb_img)
            
        img_ptr = lb_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        t1 = time.time()
        
        # Run inference in C
        count = self.lib.detect(
            self.ctx,
            img_ptr,
            ctypes.c_float(CONF_THRES),
            ctypes.c_float(IOU_THRES),
            self.results_buffer,
            100
        )
        t2 = time.time()
        
        self.t_pre += (t1 - t0)
        self.t_infer += (t2 - t1)
        self.count += 1
        if self.count % 30 == 0:
             print(f"Avg Pre: {self.t_pre/self.count*1000:.2f}ms, Avg Infer: {self.t_infer/self.count*1000:.2f}ms")
             self.t_pre = 0
             self.t_infer = 0
             self.count = 0
        
        if count == 0:
            return np.zeros((0, 6), dtype=np.float32)
            
        # Convert results to numpy
        # We can optimize this by avoiding full copy if needed, but for <100 objs it's fast
        dets = np.zeros((count, 6), dtype=np.float32)
        for i in range(count):
            d = self.results_buffer[i]
            dets[i] = [d.x1, d.y1, d.x2, d.y2, d.score, float(d.class_id)]
            
        # Map back to original image
        dets[:, [0, 2]] -= dw
        dets[:, [1, 3]] -= dh
        dets[:, :4] /= ratio
        
        return dets

    def release(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.release_model(self.ctx)
            self.ctx = None

class YOLO_RKNN_Wrapper:
    def __init__(self, model_path, known_size=None):
        from rknnlite.api import RKNNLite
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")

        with SilenceAndPrint(verbose=False):
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError('Load RKNN model failed')
            self.rknn.init_runtime()
            
        if known_size:
            self.model_wh = known_size
        else:
            self.model_wh = self._probe_model_size()
        # print(f"--> Detected RKNN model input size: {self.model_wh}")

    def release(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()

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
        self.model_path = model_path
        self.num_threads = num_threads
        self.wrappers = [] # Keep references if needed, but mainly managed in threads
        
        # Load one instance temporarily to probe model size
        print(f"Probing model size with temporary instance...")
        temp_wrapper = YOLO_RKNN_Wrapper(model_path)
        self.model_wh = temp_wrapper.model_wh
        # Explicitly release resources of temp wrapper if possible, 
        # though Python's GC and RKNNLite's destructor should handle it.
        # In rknnlite2, we might need to be careful. 
        # For now, we assume letting it go out of scope is fine, 
        # or we could keep it as one of the workers.
        # Let's keep it simple: close it to be safe.
        if hasattr(temp_wrapper, 'rknn'):
            try:
                temp_wrapper.rknn.release()
            except:
                pass
        del temp_wrapper
            
        self.input_queue = Queue(maxsize=num_threads + 2)
        self.output_queue = Queue()
        self.running = True
        self.threads = []
        self.init_lock = threading.Lock()
        
    def start(self):
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker, args=(i,))
            t.daemon = True
            t.start()
            self.threads.append(t)
            # Give some time for thread to start and acquire lock if needed
            time.sleep(0.1)
            
    def _worker(self, thread_id):
        print(f"Loading NPU model {thread_id+1}/{self.num_threads} in thread (C Engine)...")
        try:
            # Initialize RKNN in the worker thread context
            # Use C Wrapper
            # Serialize initialization to prevent driver race conditions
            with self.init_lock:
                wrapper = YOLO_RKNN_C_Wrapper(self.model_path, known_size=self.model_wh)
                print(f"Thread {thread_id+1} model loaded.")
        except Exception as e:
            print(f"Failed to load NPU model in thread {thread_id}: {e}")
            # Fallback to Python wrapper if C fails?
            try:
                print(f"Fallback to Python wrapper for thread {thread_id}")
                wrapper = YOLO_RKNN_Wrapper(self.model_path, known_size=self.model_wh)
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                return

        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
            except:
                continue
                
            frame_id, frame = item
            # Detect
            try:
                dets = wrapper.detect(frame)
                self.output_queue.put((frame_id, frame, dets))
            except Exception as e:
                print(f"Inference error in thread {thread_id}: {e}")
            
    def detect_async(self, frame, frame_id):
        # Clear queue to ensure we only process the latest frame (LIFO behavior)
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                pass
        self.input_queue.put((frame_id, frame))
            
    def get_result(self):
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None, None, None

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

def shoot_worker():
    global should_shoot, hid_buffer_y, hid_buttons
    while True:
        if not ENABLE_HID or not hid_file:
            time.sleep(0.1)
            continue
            
        if should_shoot:
            try:
                # Press Left Button (0x01)
                with hid_lock:
                    hid_buttons = 0x01
                    hid_file.write(bytearray([hid_buttons, 0, 0, 0]))
                    # 压枪：增加 Y 轴向下的偏移量
                    if RECOIL_STRENGTH > 0:
                        hid_buffer_y += RECOIL_STRENGTH
                
                # Hold
                time.sleep(SHOOT_DURATION)
                
                # Release
                with hid_lock:
                    hid_buttons = 0
                    hid_file.write(bytearray([hid_buttons, 0, 0, 0]))
                
                # Cooldown
                time.sleep(SHOOT_COOLDOWN)
            except Exception as e:
                print(f"Shoot error: {e}")
                time.sleep(0.1)
        else:
            time.sleep(0.01)

def process_hid_logic(dets, center_x, center_y):
    global last_hid_move_time, pid_state, should_shoot, hid_buffer_x, hid_buffer_y, last_shoot_signal_time
    
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
        
        # Calculate derivatives for prediction and PID
        if pid_state['tracking_frames'] > 0:
            current_derivative_x = raw_dx - pid_state['last_error_x']
            current_derivative_y = raw_dy - pid_state['last_error_y']
        else:
            current_derivative_x = 0
            current_derivative_y = 0
            
        pid_state['tracking_frames'] += 1

        # Auto Shoot Logic
        if AUTO_SHOOT and ENABLE_HID:
            # Predict position in N frames
            pred_dx = raw_dx + (current_derivative_x * SHOOT_PREDICTION)
            pred_dy = raw_dy + (current_derivative_y * SHOOT_PREDICTION)
            
            # Check if current OR predicted position is within threshold
            # Using OR ensures we don't stop shooting if we are already there but derivative is fluctuating
            if (abs(raw_dx) < SHOOT_THRESHOLD and abs(raw_dy) < SHOOT_THRESHOLD) or \
               (abs(pred_dx) < SHOOT_THRESHOLD and abs(pred_dy) < SHOOT_THRESHOLD):
                last_shoot_signal_time = time.time()
                should_shoot = True
            else:
                if time.time() - last_shoot_signal_time < SHOOT_SUSTAIN_TIME:
                    should_shoot = True
                else:
                    should_shoot = False
        else:
            should_shoot = False

        # Apply Deadzone
        if abs(raw_dx) < AIM_DEADZONE: raw_dx = 0
        if abs(raw_dy) < AIM_DEADZONE: raw_dy = 0

        # PID Calculation
        # X Axis
        # derivative_x = raw_dx - pid_state['last_error_x'] # Already calculated
        move_x = (PID_KP * raw_dx) + (PID_KD * current_derivative_x)
        pid_state['last_error_x'] = raw_dx

        # Y Axis
        # derivative_y = raw_dy - pid_state['last_error_y'] # Already calculated
        move_y = (PID_KP * raw_dy) + (PID_KD * current_derivative_y)
        pid_state['last_error_y'] = raw_dy

        dx = int(move_x)
        dy = int(move_y)
        
        # Deadzone and sensitivity
        if ENABLE_HID and hid_file and (abs(dx) >= 1 or abs(dy) >= 1):
            # 应用灵敏度系数
            final_dx = dx * MOUSE_SENSITIVITY
            final_dy = dy * MOUSE_SENSITIVITY
            
            # 将移动量加入平滑缓冲区
            with hid_lock:
                hid_buffer_x += final_dx
                hid_buffer_y += final_dy
                hid_cond.notify() # Wake up HID worker immediately
    else:
        # Reset PID integral when no target found to prevent windup
        pid_state['integral_x'] = 0
        pid_state['integral_y'] = 0
        pid_state['last_error_x'] = 0
        pid_state['last_error_y'] = 0
        pid_state['tracking_frames'] = 0
        
        if AUTO_SHOOT and ENABLE_HID and (time.time() - last_shoot_signal_time < SHOOT_SUSTAIN_TIME):
             should_shoot = True
        else:
             should_shoot = False
        
        # Clear HID buffer to prevent drift/jitter when target is lost
        with hid_lock:
            hid_buffer_x = 0.0
            hid_buffer_y = 0.0

def hid_worker():
    global hid_buffer_x, hid_buffer_y, hid_buttons
    while True:
        if not ENABLE_HID or not hid_file:
            time.sleep(0.1)
            continue
            
        with hid_lock:
            # Wait for notification or timeout
            # If buffer is empty, wait longer (save CPU)
            # If buffer has data, process immediately
            if abs(hid_buffer_x) < 0.1 and abs(hid_buffer_y) < 0.1:
                 hid_cond.wait(timeout=0.01)
            
            # Simple Proportional smoothing
            # Move a fraction of the remaining distance
            step_x = hid_buffer_x * HID_SMOOTH_FACTOR
            step_y = hid_buffer_y * HID_SMOOTH_FACTOR
            
            # Minimum movement logic to ensure convergence
            # Only move if buffer has significant value (> 0.5) to prevent jitter
            if abs(hid_buffer_x) > 0.5 and abs(step_x) < 1:
                step_x = 1 if hid_buffer_x > 0 else -1
            elif abs(hid_buffer_x) <= 0.5:
                step_x = 0
                
            if abs(hid_buffer_y) > 0.5 and abs(step_y) < 1:
                step_y = 1 if hid_buffer_y > 0 else -1
            elif abs(hid_buffer_y) <= 0.5:
                step_y = 0
            
            # Clamp to max HID report size
            step_x = max(-127, min(127, step_x))
            step_y = max(-127, min(127, step_y))
            
            # Update buffer
            # We only subtract what we actually intend to send (integer part)
            dx = int(step_x)
            dy = int(step_y)
            
            hid_buffer_x -= dx
            hid_buffer_y -= dy
            
            current_buttons = hid_buttons
        
        if dx != 0 or dy != 0:
            try:
                buf = bytearray([current_buttons, dx & 0xFF, dy & 0xFF, 0])
                hid_file.write(buf)
                # hid_file.flush()
            except Exception:
                pass
        
        # Short sleep to limit polling rate (e.g. 1000Hz)
        time.sleep(0.001)

# --- 异步绘图模块 ---
vis_queue = Queue(maxsize=2)

def draw_final_frame(frame, dets, center_x, center_y, fps):
    vis = draw_detections(frame, dets)
    # Draw crosshair
    cv2.line(vis, (int(center_x)-10, int(center_y)), (int(center_x)+10, int(center_y)), (0,255,255), 1)
    cv2.line(vis, (int(center_x), int(center_y)-10), (int(center_x), int(center_y)+10), (0,255,255), 1)
    
    # Draw Shoot Threshold Circle (Visual Aid)
    if AUTO_SHOOT:
        # 画出射击触发范围，方便调试
        cv2.circle(vis, (int(center_x), int(center_y)), int(SHOOT_THRESHOLD), (0, 255, 255), 1)
    
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return vis

def vis_worker():
    global output_frame
    while True:
        try:
            # 获取最新的绘图任务
            frame, dets, center_x, center_y, fps = vis_queue.get()
            
            # 绘图
            vis = draw_final_frame(frame, dets, center_x, center_y, fps)
            
            # Sanity check
            if vis.shape[0] > 4000 or vis.shape[1] > 4000:
                continue
            
            # 更新 output_frame
            with lock:
                output_frame = vis
        except Exception as e:
            print(f"Vis error: {e}")

def detection_loop():
    global output_frame, model_wrapper
    cap = None
    
    for idx in CAMERA_INDEXES:
        # 针对 HDMI IN 优化：使用 V4L2 后端
        temp_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if temp_cap.isOpened():
            # print(f"Opened camera index {idx}")
            cap = temp_cap
            # 尝试设置缓冲区大小为1，减少摄像头内部延迟
            try:
                # HDMI IN 原生支持 BGR3 格式，直接采集 BGR 数据，避免 MJPG 解码和 YUV 转换，延迟最低
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B', 'G', 'R', '3'))
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                # cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
            break
    
    if cap is None or not cap.isOpened():
        print("No camera found")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = (actual_width / 2) + AIM_OFFSET_X
    center_y = (actual_height / 2) + AIM_OFFSET_Y
    
    print(f"Camera resolution: {actual_width}x{actual_height}")

    # Start async wrapper if applicable
    if hasattr(model_wrapper, 'start'):
        model_wrapper.start()

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    cam_fps_counter = 0
    cam_fps_start_time = time.time()
    
    frame_id_counter = 0
    last_processed_frame_id = -1
    
    fail_count = 0

    while True:
        if cap is None or not cap.isOpened():
            print("Camera disconnected, trying to reconnect...")
            time.sleep(1)
            for idx in CAMERA_INDEXES:
                temp_cap = cv2.VideoCapture(idx)
                if temp_cap.isOpened():
                    cap = temp_cap
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        pass
                    print(f"Reconnected to camera index {idx}")
                    break
            if cap is None or not cap.isOpened():
                continue

        success, frame = cap.read()
        if not success:
            fail_count += 1
            if fail_count > 10:
                print("Camera read failed, releasing...")
                cap.release()
                cap = None
                fail_count = 0
            time.sleep(0.01)
            continue
        
        fail_count = 0
        
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
        frame_id_counter += 1
        
        cam_fps_counter += 1
        if time.time() - cam_fps_start_time >= 1.0:
            cam_fps = cam_fps_counter / (time.time() - cam_fps_start_time)
            print(f"Camera FPS: {cam_fps:.2f}")
            cam_fps_counter = 0
            cam_fps_start_time = time.time()
        
        if hasattr(model_wrapper, 'detect_async'):
            model_wrapper.detect_async(frame, frame_id_counter) # Pass frame directly (assuming cap.read returns new buffer)
            
            # Process all available results, but only act on the latest one
            best_res = None
            while True:
                fid, res_frame, dets = model_wrapper.get_result()
                if res_frame is None:
                    break
                
                # Keep track of the latest frame result
                if fid > last_processed_frame_id:
                    if best_res is None or fid > best_res[0]:
                        best_res = (fid, res_frame, dets)
            
            # If we found a newer result, process it
            if best_res:
                fid, res_frame, dets = best_res
                
                # 1. HID Control (Critical Path)
                process_hid_logic(dets, center_x, center_y)
                last_processed_frame_id = fid
                
                # Calculate FPS (based on processed frames)
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                    print(f"Process FPS: {fps:.2f}")
                
                # 2. Visualization (Async)
                if not vis_queue.full():
                    vis_queue.put((res_frame, dets, center_x, center_y, fps))

        else:
            dets = model_wrapper.detect(frame)
            # 1. HID Control
            process_hid_logic(dets, center_x, center_y)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # 2. Visualization (Async)
            if not vis_queue.full():
                vis_queue.put((frame.copy(), dets, center_x, center_y, fps))
        
        # Old visualization logic removed

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
        <title>YOLO Web Stream & Config</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #333; color: #fff; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; }
            .video-box { flex: 1; min-width: 300px; }
            .controls { flex: 1; min-width: 300px; background: #444; padding: 20px; border-radius: 8px; }
            .control-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input[type=range] { width: 100%%; }
            input[type=number] { width: 80px; }
            .value-display { float: right; font-weight: bold; color: #0f0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
      </head>
      <body>
        <h1>YOLO Control Panel</h1>
        <div class="container">
            <div class="video-box">
                <img src="/video_feed" width="100%%" style="border: 2px solid #666;">
            </div>
            <div class="controls">
                <h2>Configuration</h2>
                
                <div class="control-group">
                    <label>Enable HID <input type="checkbox" id="enable_hid" onchange="updateConfig()"></label>
                </div>

                <div class="control-group">
                    <label>Smooth Factor (0.1=Slow, 1.0=Fast) <span id="val_smooth" class="value-display"></span></label>
                    <input type="range" id="hid_smooth" min="0.05" max="1.0" step="0.05" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>HID Interval (s) <span id="val_interval" class="value-display"></span></label>
                    <input type="range" id="hid_interval" min="0.001" max="0.02" step="0.001" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Recoil Strength (px/shot) <span id="val_recoil" class="value-display"></span></label>
                    <input type="range" id="recoil_strength" min="0" max="10" step="0.5" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Shoot Cooldown (s) <span id="val_cooldown" class="value-display"></span></label>
                    <input type="range" id="shoot_cooldown" min="0.01" max="0.5" step="0.01" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Shoot Threshold (px) <span id="val_thres" class="value-display"></span></label>
                    <input type="range" id="shoot_thres" min="10" max="200" step="5" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Shoot Prediction (frames) <span id="val_pred" class="value-display"></span></label>
                    <input type="range" id="shoot_pred" min="0" max="10" step="0.5" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>PID KP (Proportional) <span id="val_kp" class="value-display"></span></label>
                    <input type="range" id="pid_kp" min="0" max="2.0" step="0.01" oninput="updateConfig()">
                </div>
                <div class="control-group">
                    <label>PID KI (Integral) <span id="val_ki" class="value-display"></span></label>
                    <input type="range" id="pid_ki" min="0" max="0.1" step="0.001" oninput="updateConfig()">
                </div>
                <div class="control-group">
                    <label>PID KD (Derivative) <span id="val_kd" class="value-display"></span></label>
                    <input type="range" id="pid_kd" min="0" max="1.0" step="0.01" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Mouse Sensitivity <span id="val_sens" class="value-display"></span></label>
                    <input type="range" id="mouse_sens" min="0.1" max="2.0" step="0.1" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Aim Deadzone (px) <span id="val_dead" class="value-display"></span></label>
                    <input type="range" id="aim_dead" min="0" max="20" step="1" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Aim Height Ratio (0=Top, 1=Bottom) <span id="val_ratio" class="value-display"></span></label>
                    <input type="range" id="aim_ratio" min="0" max="1.0" step="0.05" oninput="updateConfig()">
                </div>

                <div class="control-group">
                    <label>Offset X <span id="val_offx" class="value-display"></span></label>
                    <input type="range" id="aim_offx" min="-50" max="50" step="1" oninput="updateConfig()">
                </div>
                <div class="control-group">
                    <label>Offset Y <span id="val_offy" class="value-display"></span></label>
                    <input type="range" id="aim_offy" min="-50" max="50" step="1" oninput="updateConfig()">
                </div>
            </div>
        </div>

        <script>
            // Initial values from server
            const config = {
                PID_KP: %s,
                PID_KI: %s,
                PID_KD: %s,
                MOUSE_SENSITIVITY: %s,
                AIM_DEADZONE: %s,
                AIM_HEIGHT_RATIO: %s,
                AIM_OFFSET_X: %s,
                AIM_OFFSET_Y: %s,
                HID_SMOOTH_FACTOR: %s,
                HID_UPDATE_INTERVAL: %s,
                ENABLE_HID: %s,
                RECOIL_STRENGTH: %s,
                SHOOT_COOLDOWN: %s,
                SHOOT_PREDICTION: %s,
                SHOOT_THRESHOLD: %s
            };

            function init() {
                document.getElementById('pid_kp').value = config.PID_KP;
                document.getElementById('pid_ki').value = config.PID_KI;
                document.getElementById('pid_kd').value = config.PID_KD;
                document.getElementById('mouse_sens').value = config.MOUSE_SENSITIVITY;
                document.getElementById('aim_dead').value = config.AIM_DEADZONE;
                document.getElementById('aim_ratio').value = config.AIM_HEIGHT_RATIO;
                document.getElementById('aim_offx').value = config.AIM_OFFSET_X;
                document.getElementById('aim_offy').value = config.AIM_OFFSET_Y;
                document.getElementById('hid_smooth').value = config.HID_SMOOTH_FACTOR;
                document.getElementById('hid_interval').value = config.HID_UPDATE_INTERVAL;
                document.getElementById('enable_hid').checked = config.ENABLE_HID;
                document.getElementById('recoil_strength').value = config.RECOIL_STRENGTH;
                document.getElementById('shoot_cooldown').value = config.SHOOT_COOLDOWN;
                document.getElementById('shoot_pred').value = config.SHOOT_PREDICTION;
                document.getElementById('shoot_thres').value = config.SHOOT_THRESHOLD;
                updateDisplays();
            }

            function updateDisplays() {
                document.getElementById('val_kp').innerText = document.getElementById('pid_kp').value;
                document.getElementById('val_ki').innerText = document.getElementById('pid_ki').value;
                document.getElementById('val_kd').innerText = document.getElementById('pid_kd').value;
                document.getElementById('val_sens').innerText = document.getElementById('mouse_sens').value;
                document.getElementById('val_dead').innerText = document.getElementById('aim_dead').value;
                document.getElementById('val_ratio').innerText = document.getElementById('aim_ratio').value;
                document.getElementById('val_offx').innerText = document.getElementById('aim_offx').value;
                document.getElementById('val_offy').innerText = document.getElementById('aim_offy').value;
                document.getElementById('val_smooth').innerText = document.getElementById('hid_smooth').value;
                document.getElementById('val_interval').innerText = document.getElementById('hid_interval').value;
                document.getElementById('val_recoil').innerText = document.getElementById('recoil_strength').value;
                document.getElementById('val_cooldown').innerText = document.getElementById('shoot_cooldown').value;
                document.getElementById('val_pred').innerText = document.getElementById('shoot_pred').value;
                document.getElementById('val_thres').innerText = document.getElementById('shoot_thres').value;
            }

            function updateConfig() {
                updateDisplays();
                const data = {
                    PID_KP: parseFloat(document.getElementById('pid_kp').value),
                    PID_KI: parseFloat(document.getElementById('pid_ki').value),
                    PID_KD: parseFloat(document.getElementById('pid_kd').value),
                    MOUSE_SENSITIVITY: parseFloat(document.getElementById('mouse_sens').value),
                    AIM_DEADZONE: parseInt(document.getElementById('aim_dead').value),
                    AIM_HEIGHT_RATIO: parseFloat(document.getElementById('aim_ratio').value),
                    AIM_OFFSET_X: parseInt(document.getElementById('aim_offx').value),
                    AIM_OFFSET_Y: parseInt(document.getElementById('aim_offy').value),
                    HID_SMOOTH_FACTOR: parseFloat(document.getElementById('hid_smooth').value),
                    HID_UPDATE_INTERVAL: parseFloat(document.getElementById('hid_interval').value),
                    ENABLE_HID: document.getElementById('enable_hid').checked,
                    RECOIL_STRENGTH: parseFloat(document.getElementById('recoil_strength').value),
                    SHOOT_COOLDOWN: parseFloat(document.getElementById('shoot_cooldown').value),
                    SHOOT_PREDICTION: parseFloat(document.getElementById('shoot_pred').value),
                    SHOOT_THRESHOLD: parseInt(document.getElementById('shoot_thres').value)
                };

                fetch('/update_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
            }

            init();
        </script>
      </body>
    </html>
    """ % (PID_KP, PID_KI, PID_KD, MOUSE_SENSITIVITY, AIM_DEADZONE, AIM_HEIGHT_RATIO, AIM_OFFSET_X, AIM_OFFSET_Y, HID_SMOOTH_FACTOR, HID_UPDATE_INTERVAL, "true" if ENABLE_HID else "false", RECOIL_STRENGTH, SHOOT_COOLDOWN, SHOOT_PREDICTION, SHOOT_THRESHOLD)

@app.route('/update_config', methods=['POST'])
def update_config():
    global PID_KP, PID_KI, PID_KD, MOUSE_SENSITIVITY, AIM_DEADZONE, AIM_HEIGHT_RATIO, AIM_OFFSET_X, AIM_OFFSET_Y, HID_SMOOTH_FACTOR, HID_UPDATE_INTERVAL, ENABLE_HID, RECOIL_STRENGTH, SHOOT_COOLDOWN, SHOOT_PREDICTION, SHOOT_THRESHOLD
    data = request.json
    if data:
        PID_KP = data.get('PID_KP', PID_KP)
        PID_KI = data.get('PID_KI', PID_KI)
        PID_KD = data.get('PID_KD', PID_KD)
        MOUSE_SENSITIVITY = data.get('MOUSE_SENSITIVITY', MOUSE_SENSITIVITY)
        AIM_DEADZONE = data.get('AIM_DEADZONE', AIM_DEADZONE)
        AIM_HEIGHT_RATIO = data.get('AIM_HEIGHT_RATIO', AIM_HEIGHT_RATIO)
        AIM_OFFSET_X = data.get('AIM_OFFSET_X', AIM_OFFSET_X)
        AIM_OFFSET_Y = data.get('AIM_OFFSET_Y', AIM_OFFSET_Y)
        HID_SMOOTH_FACTOR = data.get('HID_SMOOTH_FACTOR', HID_SMOOTH_FACTOR)
        HID_UPDATE_INTERVAL = data.get('HID_UPDATE_INTERVAL', HID_UPDATE_INTERVAL)
        ENABLE_HID = data.get('ENABLE_HID', ENABLE_HID)
        RECOIL_STRENGTH = data.get('RECOIL_STRENGTH', RECOIL_STRENGTH)
        SHOOT_COOLDOWN = data.get('SHOOT_COOLDOWN', SHOOT_COOLDOWN)
        SHOOT_PREDICTION = data.get('SHOOT_PREDICTION', SHOOT_PREDICTION)
        SHOOT_THRESHOLD = data.get('SHOOT_THRESHOLD', SHOOT_THRESHOLD)
        print(f"Config Updated: KP={PID_KP}, Recoil={RECOIL_STRENGTH}, Pred={SHOOT_PREDICTION}, Thres={SHOOT_THRESHOLD}")
    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 初始化模型
    if MODEL_PATH.endswith('.rknn'):
        # Use Multi-NPU wrapper for RKNN models
        # Try increasing threads to saturate NPU
        model_wrapper = MultiYOLO_RKNN_Wrapper(MODEL_PATH, num_threads=6)
    elif MODEL_PATH.endswith('.pt'):
        model_wrapper = YOLO_CPU_Wrapper(MODEL_PATH)
    else:
        print(f"Error: Unsupported model format: {MODEL_PATH}")
        sys.exit(1)

    # 启动检测线程
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    # 启动HID平滑移动线程
    t_hid = threading.Thread(target=hid_worker)
    t_hid.daemon = True
    t_hid.start()

    # 启动自动射击线程
    t_shoot = threading.Thread(target=shoot_worker)
    t_shoot.daemon = True
    t_shoot.start()

    # 启动可视化线程
    t_vis = threading.Thread(target=vis_worker)
    t_vis.daemon = True
    t_vis.start()

    # 监听所有 IP
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
