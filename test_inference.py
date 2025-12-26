import ctypes
import numpy as np
import cv2
import os

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
        if frame is None:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Use random noise to simulate input
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
        if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame)
        
        print("Calling detect...")
        count = self.lib.detect(self.ctx, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_float(0.25), ctypes.c_float(0.45), self.results_buffer, 100)
        print(f"Count: {count}")

if __name__ == "__main__":
    model_path = './Models/cf-11n-rk3588-int8.rknn'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        exit(1)
        
    wrapper = YOLO_RKNN_C_Wrapper(model_path)
    wrapper.detect(None)
