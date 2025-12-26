import os
import sys
import numpy as np
import argparse
from rknn.api import RKNN

def generate_dataset_list(dataset_path, output_file, num_images=100):
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        sys.exit(1)
        
    images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and ' ' not in f]
    if not images:
        print(f"No images found in {dataset_path}")
        sys.exit(1)
        
    # Select a subset of images
    images = images[:num_images]
    
    with open(output_file, 'w') as f:
        for img in images:
            f.write(os.path.join(dataset_path, img) + '\n')
    print(f"Generated dataset list with {len(images)} images at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize ONNX model to RKNN int8')
    parser.add_argument('onnx_model', help='Path to ONNX model')
    parser.add_argument('output_rknn', help='Path to output RKNN model')
    parser.add_argument('--dataset', default='DataSet/images/val', help='Path to calibration dataset images')
    parser.add_argument('--platform', default='rk3588', help='Target platform (e.g., rk3588)')
    args = parser.parse_args()

    ONNX_MODEL = args.onnx_model
    RKNN_MODEL = args.output_rknn
    DATASET_PATH = args.dataset
    DATASET_LIST = 'dataset_list.txt'
    QUANTIZE_ON = True

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    # IMPORTANT: Ultralytics YOLOv8/11 ONNX export typically includes the normalization (div by 255) inside the model.
    # Therefore, we should NOT normalize again in RKNN config.
    # Use mean=0, std=255 to normalize 0-255 input to 0-1.
    # This is required if the ONNX model does NOT have a Div(255) layer.
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform, 
                quantized_dtype='asymmetric_quantized-8', quantized_algorithm='kl_divergence')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    if not os.path.exists(ONNX_MODEL):
        print(f"ONNX model not found at {ONNX_MODEL}")
        sys.exit(1)
        
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)
    print('done')

    # Generate dataset list for quantization
    generate_dataset_list(DATASET_PATH, DATASET_LIST)

    # Build model
    print('--> Building model')
    # do_quantization=True enables int8 quantization
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET_LIST)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)
    print('done')
    
    print(f"Quantized model exported to {RKNN_MODEL}")
