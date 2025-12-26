import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import time
import subprocess
import threading
import os
import sys
import ctypes
import atexit
from flask import Flask, Response, request, jsonify
from queue import Queue

# --- CPU 亲和性设置 (RK3588 A76 Big Cores) ---
try:
    # RK3588: 0-3=A55, 4-7=A76. Bind to 4,5,6,7
    os.sched_setaffinity(0, {4, 5, 6, 7})
    print("[System] Process bound to A76 Big Cores (4-7)")
except Exception as e:
    print(f"[System] Failed to set CPU affinity: {e}")

# --- 配置参数 ---
MODEL_PATH = './Models/cf-11n-rk3588-int8.rknn'
CONF_THRES = 0.45  # Lower threshold to catch weak detections
USE_RGB_INPUT = True 
CAMERA_INDEX = 20 # HDMI IN
PORT = 5000
TARGET_CLASS_ID = 1

# HID / 键鼠控制配置
ENABLE_HID = True
HID_DEVICE = '/dev/hidg1'
HID_MOVE_COOLDOWN = 0.0005
MOUSE_SENSITIVITY = 0.4
HID_SMOOTH_FACTOR = 0.3
HID_UPDATE_INTERVAL = 0.003

# 自动射击配置
AUTO_SHOOT = True
SHOOT_COOLDOWN = 0.03
SHOOT_THRESHOLD = 20
SHOOT_DURATION = 0.02
SHOOT_PREDICTION = 6
SHOOT_SUSTAIN_TIME = 0
RECOIL_STRENGTH = 3.0

# 瞄准偏移与死区
AIM_OFFSET_X = 0
AIM_OFFSET_Y = 0
AIM_DEADZONE = 0
AIM_HEIGHT_RATIO = 0.10

# PID 参数
PID_KP = 0.35
PID_KI = 0.001
PID_KD = 0.15
PID_MAX_INTEGRAL = 0

app = Flask(__name__)

# --- 全局变量 ---
model_wrapper = None
output_frame = None
hid_file = None
hid_buffer_x = 0.0
hid_buffer_y = 0.0
hid_buttons = 0
hid_lock = threading.Lock()
hid_cond = threading.Condition(hid_lock)
should_shoot = False
last_shoot_signal_time = 0

pid_state = {
    'last_error_x': 0, 'last_error_y': 0,
    'integral_x': 0, 'integral_y': 0,
    'tracking_frames': 0
}
frame_id = 0
lock = threading.Lock()
vis_queue = Queue(maxsize=2)

COCO_CLASSES = [
    'defenders','Infiltrators'
]

# --- 资源清理 ---
def cleanup():
    global hid_file
    if hid_file:
        try:
            hid_file.close()
        except:
            pass

atexit.register(cleanup)

# --- 初始化 HID ---
try:
    subprocess.run(['sudo', 'chmod', '666', HID_DEVICE], check=False)
    hid_file = open(HID_DEVICE, 'wb', buffering=0)
    print(f"[HID] Ready: {HID_DEVICE}")
except Exception as e:
    print(f"[HID] Error: {e}")
    ENABLE_HID = False

# --- 图像处理辅助函数 ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    h0, w0 = shape
    h, w = new_shape[1], new_shape[0]
    if h0 == 0 or w0 == 0: return img, 1.0, (0, 0)
    r = min(w / w0, h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    new_unpad = (max(1, new_unpad[0]), max(1, new_unpad[1]))
    dw, dh = w - new_unpad[0], h - new_unpad[1]
    dw /= 2; dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def draw_detections(img, dets):
    h, w = img.shape[:2]
    for x1, y1, x2, y2, score, cls in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
        # 如果是 defenders 显示红色，否则显示绿色
        color = (0, 0, 255) if label == 'defenders' else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f'{label} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# --- C 模型封装 ---
class CDetection(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float), ("y1", ctypes.c_float), ("x2", ctypes.c_float), ("y2", ctypes.c_float), ("score", ctypes.c_float), ("class_id", ctypes.c_int)]

class YOLO_RKNN_C_Wrapper:
    def __init__(self, model_path):
        self.lib = ctypes.CDLL('./c_src/librknn_yolo.so')
        self.lib.init_model.argtypes = [ctypes.c_char_p]
        self.lib.init_model.restype = ctypes.c_void_p
        self.lib.detect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_float, ctypes.c_float, ctypes.POINTER(CDetection), ctypes.c_int]
        self.lib.detect.restype = ctypes.c_int
        self.lib.release_model.argtypes = [ctypes.c_void_p]
        
        self.ctx = self.lib.init_model(model_path.encode('utf-8'))
        if not self.ctx: raise RuntimeError("Failed to init C RKNN model")
        self.model_wh = (640, 640)
        self.results_buffer = (CDetection * 100)()

    def detect(self, frame):
        if USE_RGB_INPUT: input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else: input_img = frame
        lb_img, ratio, (dw, dh) = letterbox(input_img, self.model_wh)
        if not lb_img.flags['C_CONTIGUOUS']: lb_img = np.ascontiguousarray(lb_img)
        
        # IOU Threshold hardcoded to 0.45
        count = self.lib.detect(self.ctx, lb_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_float(CONF_THRES), ctypes.c_float(0.45), self.results_buffer, 100)
        
        if count == 0: return np.zeros((0, 6), dtype=np.float32)
        dets = np.zeros((count, 6), dtype=np.float32)
        for i in range(count):
            d = self.results_buffer[i]
            dets[i] = [d.x1, d.y1, d.x2, d.y2, d.score, float(d.class_id)]
        dets[:, [0, 2]] -= dw; dets[:, [1, 3]] -= dh; dets[:, :4] /= ratio
        return dets

class MultiYOLO_RKNN_Wrapper:
    def __init__(self, model_path, num_threads=3):
        self.model_path = model_path
        self.num_threads = num_threads
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
            time.sleep(0.1)
            
    def _worker(self, thread_id):
        try:
            with self.init_lock:
                wrapper = YOLO_RKNN_C_Wrapper(self.model_path)
        except Exception as e:
            print(f"Thread {thread_id} init failed: {e}")
            return

        while self.running:
            try:
                frame_id, frame = self.input_queue.get(timeout=0.1)
                t0 = time.time()
                dets = wrapper.detect(frame)
                t1 = time.time()
                self.output_queue.put((frame_id, frame, dets, (t1 - t0) * 1000))
            except:
                continue
            
    def detect_async(self, frame, frame_id):
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except: pass
        self.input_queue.put((frame_id, frame))
            
    def get_result(self):
        if not self.output_queue.empty(): return self.output_queue.get()
        return None, None, None, 0

# --- 业务逻辑 ---
def shoot_worker():
    global should_shoot, hid_buffer_y, hid_buttons
    while True:
        if not ENABLE_HID or not hid_file:
            time.sleep(0.1); continue
        if should_shoot:
            try:
                with hid_lock:
                    hid_buttons = 0x01
                    hid_file.write(bytearray([hid_buttons, 0, 0, 0]))
                    if RECOIL_STRENGTH > 0: hid_buffer_y += RECOIL_STRENGTH
                time.sleep(SHOOT_DURATION)
                with hid_lock:
                    hid_buttons = 0
                    hid_file.write(bytearray([hid_buttons, 0, 0, 0]))
                time.sleep(SHOOT_COOLDOWN)
            except: pass
        else: time.sleep(0.01)

def hid_worker():
    global hid_buffer_x, hid_buffer_y, hid_buttons
    while True:
        if not ENABLE_HID or not hid_file:
            time.sleep(0.1); continue
        with hid_lock:
            if abs(hid_buffer_x) < 0.1 and abs(hid_buffer_y) < 0.1: hid_cond.wait(timeout=0.01)
            step_x = hid_buffer_x * HID_SMOOTH_FACTOR
            step_y = hid_buffer_y * HID_SMOOTH_FACTOR
            if abs(hid_buffer_x) > 0.5 and abs(step_x) < 1: step_x = 1 if hid_buffer_x > 0 else -1
            elif abs(hid_buffer_x) <= 0.5: step_x = 0
            if abs(hid_buffer_y) > 0.5 and abs(step_y) < 1: step_y = 1 if hid_buffer_y > 0 else -1
            elif abs(hid_buffer_y) <= 0.5: step_y = 0
            step_x = max(-127, min(127, step_x))
            step_y = max(-127, min(127, step_y))
            dx, dy = int(step_x), int(step_y)
            hid_buffer_x -= dx; hid_buffer_y -= dy
            current_buttons = hid_buttons
        if dx != 0 or dy != 0:
            try: hid_file.write(bytearray([current_buttons, dx & 0xFF, dy & 0xFF, 0]))
            except: pass
        time.sleep(HID_UPDATE_INTERVAL)

def process_hid_logic(dets, center_x, center_y):
    global pid_state, should_shoot, hid_buffer_x, hid_buffer_y, last_shoot_signal_time
    target = None
    min_dist = float('inf')
    for det in dets:
        if int(det[5]) == TARGET_CLASS_ID:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * AIM_HEIGHT_RATIO
            dist = (cx - center_x)**2 + (cy - center_y)**2
            if dist < min_dist: min_dist = dist; target = (cx, cy)
    
    if target:
        cx, cy = target
        raw_dx, raw_dy = cx - center_x, cy - center_y
        if pid_state['tracking_frames'] > 0:
            cd_x = raw_dx - pid_state['last_error_x']
            cd_y = raw_dy - pid_state['last_error_y']
        else: cd_x = cd_y = 0
        pid_state['tracking_frames'] += 1

        if AUTO_SHOOT and ENABLE_HID:
            pred_dx = raw_dx + (cd_x * SHOOT_PREDICTION)
            pred_dy = raw_dy + (cd_y * SHOOT_PREDICTION)
            if (abs(raw_dx) < SHOOT_THRESHOLD and abs(raw_dy) < SHOOT_THRESHOLD) or \
               (abs(pred_dx) < SHOOT_THRESHOLD and abs(pred_dy) < SHOOT_THRESHOLD):
                last_shoot_signal_time = time.time()
                should_shoot = True
            else:
                should_shoot = (time.time() - last_shoot_signal_time < SHOOT_SUSTAIN_TIME)
        else: should_shoot = False

        if abs(raw_dx) < AIM_DEADZONE: raw_dx = 0
        if abs(raw_dy) < AIM_DEADZONE: raw_dy = 0
        move_x = (PID_KP * raw_dx) + (PID_KD * cd_x)
        move_y = (PID_KP * raw_dy) + (PID_KD * cd_y)
        pid_state['last_error_x'] = raw_dx
        pid_state['last_error_y'] = raw_dy
        
        dx, dy = int(move_x), int(move_y)
        if ENABLE_HID and hid_file and (abs(dx) >= 1 or abs(dy) >= 1):
            with hid_lock:
                hid_buffer_x += dx * MOUSE_SENSITIVITY
                hid_buffer_y += dy * MOUSE_SENSITIVITY
                hid_cond.notify()
    else:
        pid_state = {'last_error_x': 0, 'last_error_y': 0, 'integral_x': 0, 'integral_y': 0, 'tracking_frames': 0}
        should_shoot = (AUTO_SHOOT and ENABLE_HID and (time.time() - last_shoot_signal_time < SHOOT_SUSTAIN_TIME))
        with hid_lock: hid_buffer_x = 0.0; hid_buffer_y = 0.0

def vis_worker():
    global output_frame
    while True:
        try:
            frame, dets, cx, cy, fps = vis_queue.get()
            vis = draw_detections(frame, dets)
            cv2.line(vis, (int(cx)-10, int(cy)), (int(cx)+10, int(cy)), (0,255,255), 1)
            cv2.line(vis, (int(cx), int(cy)-10), (int(cx), int(cy)+10), (0,255,255), 1)
            if AUTO_SHOOT: cv2.circle(vis, (int(cx), int(cy)), int(SHOOT_THRESHOLD), (0, 255, 255), 1)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            with lock: output_frame = vis
        except: pass

def detection_loop():
    global output_frame, model_wrapper
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera not found")
        return
    
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B', 'G', 'R', '3'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx, cy = (w / 2) + AIM_OFFSET_X, (h / 2) + AIM_OFFSET_Y
    print(f"[Camera] {w}x{h}")

    if hasattr(model_wrapper, 'start'): model_wrapper.start()

    fps_start = time.time()
    fps_cnt = 0
    fps = 0
    fid_cnt = 0
    last_fid = -1
    
    cam_fps_cnt = 0
    cam_fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        
        cam_fps_cnt += 1
        
        # 修复 BGR3 格式可能返回扁平数组的问题
        if frame.ndim == 2 and frame.shape[0] == 1:
            if frame.shape[1] == w * h * 3:
                frame = frame.reshape((h, w, 3))
            else:
                continue

        fid_cnt += 1
        model_wrapper.detect_async(frame, fid_cnt)
        
        best_res = None
        while True:
            fid, res_frame, dets, inf_time = model_wrapper.get_result()
            if res_frame is None: break
            if fid > last_fid:
                if best_res is None or fid > best_res[0]: best_res = (fid, res_frame, dets, inf_time)
        
        if best_res:
            fid, res_frame, dets, inf_time = best_res
            process_hid_logic(dets, cx, cy)
            last_fid = fid
            fps_cnt += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_cnt / (time.time() - fps_start)
                cam_fps = cam_fps_cnt / (time.time() - cam_fps_start)
                
                sys.stdout.write(f"\rFPS: {fps:.2f} | CamFPS: {cam_fps:.2f} | Inf: {inf_time:.2f}ms | Dets: {len(dets)}   ")
                sys.stdout.flush()
                
                fps_cnt = 0
                fps_start = time.time()
                cam_fps_cnt = 0
                cam_fps_start = time.time()
            
            if not vis_queue.full(): vis_queue.put((res_frame, dets, cx, cy, fps))

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None: time.sleep(0.01); continue
            # 检查图像尺寸，防止非法尺寸导致 imencode 崩溃
            if output_frame.shape[1] > 4000 or output_frame.shape[0] > 4000: continue
            try:
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except: pass
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html><head><title>YOLO Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>body{font-family:sans-serif;background:#333;color:#fff;padding:20px}.container{display:flex;flex-wrap:wrap;gap:20px}.video{flex:1;min-width:300px}.ctrl{flex:1;min-width:300px;background:#444;padding:20px;border-radius:8px}input[type=range]{width:100%%}</style>
    </head><body>
    <div class="container"><div class="video"><img src="/video_feed" width="100%%"></div>
    <div class="ctrl"><h2>Config</h2>
    <label>HID <input type="checkbox" id="enable_hid" onchange="up()"></label><br>
    <label>Smooth <input type="range" id="hid_smooth" min="0.05" max="1.0" step="0.05" oninput="up()"></label><br>
    <label>Recoil <input type="range" id="recoil" min="0" max="10" step="0.5" oninput="up()"></label><br>
    <label>Thres <input type="range" id="thres" min="10" max="200" step="5" oninput="up()"></label><br>
    <label>KP <input type="range" id="kp" min="0" max="2.0" step="0.01" oninput="up()"></label><br>
    <label>Sens <input type="range" id="sens" min="0.1" max="2.0" step="0.1" oninput="up()"></label>
    </div></div>
    <script>
    const cfg={PID_KP:%s,MOUSE_SENSITIVITY:%s,HID_SMOOTH_FACTOR:%s,ENABLE_HID:%s,RECOIL_STRENGTH:%s,SHOOT_THRESHOLD:%s};
    function init(){
        document.getElementById('kp').value=cfg.PID_KP;
        document.getElementById('sens').value=cfg.MOUSE_SENSITIVITY;
        document.getElementById('hid_smooth').value=cfg.HID_SMOOTH_FACTOR;
        document.getElementById('enable_hid').checked=cfg.ENABLE_HID;
        document.getElementById('recoil').value=cfg.RECOIL_STRENGTH;
        document.getElementById('thres').value=cfg.SHOOT_THRESHOLD;
    }
    function up(){
        fetch('/update_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
            PID_KP:parseFloat(document.getElementById('kp').value),
            MOUSE_SENSITIVITY:parseFloat(document.getElementById('sens').value),
            HID_SMOOTH_FACTOR:parseFloat(document.getElementById('hid_smooth').value),
            ENABLE_HID:document.getElementById('enable_hid').checked,
            RECOIL_STRENGTH:parseFloat(document.getElementById('recoil').value),
            SHOOT_THRESHOLD:parseInt(document.getElementById('thres').value)
        })});
    }
    init();
    </script></body></html>
    """ % (PID_KP, MOUSE_SENSITIVITY, HID_SMOOTH_FACTOR, "true" if ENABLE_HID else "false", RECOIL_STRENGTH, SHOOT_THRESHOLD)

@app.route('/update_config', methods=['POST'])
def update_config():
    global PID_KP, MOUSE_SENSITIVITY, HID_SMOOTH_FACTOR, ENABLE_HID, RECOIL_STRENGTH, SHOOT_THRESHOLD
    d = request.json
    if d:
        PID_KP = d.get('PID_KP', PID_KP)
        MOUSE_SENSITIVITY = d.get('MOUSE_SENSITIVITY', MOUSE_SENSITIVITY)
        HID_SMOOTH_FACTOR = d.get('HID_SMOOTH_FACTOR', HID_SMOOTH_FACTOR)
        ENABLE_HID = d.get('ENABLE_HID', ENABLE_HID)
        RECOIL_STRENGTH = d.get('RECOIL_STRENGTH', RECOIL_STRENGTH)
        SHOOT_THRESHOLD = d.get('SHOOT_THRESHOLD', SHOOT_THRESHOLD)
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    model_wrapper = MultiYOLO_RKNN_Wrapper(MODEL_PATH, num_threads=6)
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=hid_worker, daemon=True).start()
    threading.Thread(target=shoot_worker, daemon=True).start()
    threading.Thread(target=vis_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
