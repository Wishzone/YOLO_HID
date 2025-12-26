import os
import sys
import numpy as np
from rknnlite.api import RKNNLite

def inspect_model(model_path):
    print(f"Inspecting: {model_path}")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Load failed")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime failed")
        return

    # Access internal attributes if possible, or infer from inference
    # RKNNLite doesn't expose input_attrs directly like the C API does easily.
    # But we can try to run inference and see if it complains or check logs if verbose.
    
    # However, we can't easily get ZP/Scale from Python RKNNLite API directly without parsing logs or using C API.
    # But wait, the user has the C wrapper working!
    # We can use the C wrapper to print the attributes.
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_model(sys.argv[1])
