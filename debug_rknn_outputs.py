import sys
import os
import numpy as np
from rknnlite.api import RKNNLite

def check_model_structure(model_path):
    print(f"Checking model: {model_path}")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Load failed")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime failed")
        return

    # Create a dummy input
    # Assuming 640x640 RGB
    img = np.zeros((1, 640, 640, 3), dtype=np.uint8)
    
    try:
        outputs = rknn.inference(inputs=[img])
        print(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}, dtype: {out.dtype}")
            # Print first few values to see range
            flat = out.flatten()
            print(f"  First 5 values: {flat[:5]}")
            print(f"  Min: {np.min(out)}, Max: {np.max(out)}")
            
    except Exception as e:
        print(f"Inference failed: {e}")
    
    rknn.release()

if __name__ == "__main__":
    model_path = './Models/cf-11n-rk3588-int8.rknn'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    check_model_structure(model_path)
