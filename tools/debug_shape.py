import numpy as np
from rknnlite.api import RKNNLite
import sys

def inspect_model(model_path):
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Load failed")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init failed")
        return

    # Create dummy input 640x640 RGB
    img = np.zeros((1, 640, 640, 3), dtype=np.uint8)
    
    # Run
    outputs = rknn.inference(inputs=[img])
    
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        
    rknn.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_shape.py <model_path>")
    else:
        inspect_model(sys.argv[1])
