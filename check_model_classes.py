import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def check_pt_model(model_path):
    try:
        import torch
        print(f"Loading .pt model from {model_path}...")
        # 使用 map_location 确保在没有 GPU 的设备上也能加载
        # 设置 weights_only=False 以消除 FutureWarning，因为我们需要加载完整的模型结构来获取 names
        # 如果只是加载权重，可以使用 weights_only=True，但这里我们需要元数据
        try:
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # 兼容旧版本 torch 不支持 weights_only 参数的情况
            model_data = torch.load(model_path, map_location='cpu')
        
        names = None
        # 尝试不同的结构来寻找 names
        if isinstance(model_data, dict):
            if 'model' in model_data:
                if hasattr(model_data['model'], 'names'):
                    names = model_data['model'].names
            if names is None and 'names' in model_data:
                names = model_data['names']
        elif hasattr(model_data, 'names'):
            names = model_data.names
            
        # 尝试获取输入尺寸
        imgsz = None
        if isinstance(model_data, dict):
            if 'train_args' in model_data:
                imgsz = model_data['train_args'].get('imgsz')
            elif 'args' in model_data:
                imgsz = model_data['args'].get('imgsz')
        
        if imgsz is None:
            # 尝试从模型对象中获取
            model_obj = model_data.get('model') if isinstance(model_data, dict) else model_data
            if hasattr(model_obj, 'args'):
                args = model_obj.args
                if isinstance(args, dict):
                    imgsz = args.get('imgsz')
                elif hasattr(args, 'imgsz'):
                    imgsz = args.imgsz

        if imgsz:
            print(f"Model Input Size: {imgsz}")

        if names:
            print(f"\nFound {len(names)} classes:")
            print(names)
            return names
        else:
            print("Could not find 'names' in the .pt file.")
            return None
            
    except Exception as e:
        print(f"Error loading .pt model: {e}")
        return None

def check_rknn_model(model_path):
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("Error: rknnlite not found. This script needs to run on an RKNN supported device (e.g. RPi) for .rknn files.")
        return

    rknn = RKNNLite()
    
    print(f"Loading RKNN model from {model_path}...")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model failed')
        return

    print('Init runtime environment...')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        return
        
    print('\n' + '='*30)
    print('Model Information:')
    print('='*30)
    # print(f'SDK Version: {rknn.get_sdk_version()}')
    # print(f'RKNN Object: {rknn}')
    
    # 测试不同的输入尺寸
    test_sizes = [(640, 640), (1920, 1080), (1280, 720), (320, 320)]
    valid_outputs = None
    valid_size = None

    # print("\nProbing input size...")
    for width, height in test_sizes:
        # print(f'Testing input size: {width}x{height} ... ', end='')
        try:
            # 创建虚拟输入 (NHWC format usually for RKNN inputs via Python API, but sometimes NCHW)
            # Most RKNN examples use NHWC for image input
            img = np.zeros((1, height, width, 3), dtype=np.uint8)
            
            outputs = rknn.inference(inputs=[img])
            # print(f"SUCCESS")
            valid_outputs = outputs
            valid_size = (width, height)
            break
        except Exception as e:
            pass
            # print(f"Failed")
            # print(f"Debug: {e}") 

    if valid_outputs is None:
        print("\nError: Could not determine valid input size. Inference failed for all tested sizes.")
        return

    print(f"Input Size: {valid_size[0]}x{valid_size[1]}")

    # print("\nOutput shapes:")
    # for i, out in enumerate(valid_outputs):
    #     print(f"  Output {i}: {out.shape}")

    # 分析形状以推测类别数量
    guessed_classes = []
    model_type = "Unknown"
    
    for out in valid_outputs:
        shape = out.shape
        # Check for YOLOv8/v11 pattern: (1, C, N) where C = 4 + classes
        # 例如 (1, 84, 8400) -> 80 classes
        if len(shape) == 3:
            # (1, 84, 8400) or (1, 8400, 84)
            c1 = shape[1]
            c2 = shape[2]
            
            # Case 1: (1, 84, 8400)
            if c2 > 100 and c1 < 200:
                nc = c1 - 4
                if nc > 0:
                    guessed_classes.append(nc)
                    model_type = "YOLOv8/v11 (Anchor-free)"
                    # print(f"  -> Possible YOLOv8/v11 format (1, C, N). If so, classes = {c1} - 4 = {nc}")
            
            # Case 2: (1, 8400, 84)
            if c1 > 100 and c2 < 200:
                nc = c2 - 4
                if nc > 0:
                    guessed_classes.append(nc)
                    model_type = "YOLOv8/v11 (Anchor-free)"
                    # print(f"  -> Possible YOLOv8/v11 format (1, N, C). If so, classes = {c2} - 4 = {nc}")

        # Check for YOLOv5 pattern: (1, 3, H, W, C) where C = 5 + classes
        # Or (1, 255, H, W) where 255 = 3 * (5 + classes)
        if len(shape) == 4:
            # (1, 255, H, W)
            c = shape[1]
            if c % 3 == 0:
                nc = (c // 3) - 5
                if nc > 0:
                    guessed_classes.append(nc)
                    model_type = "YOLOv5 (Anchor-based)"
                    # print(f"  -> Possible YOLOv5 format (NCHW). If so, classes = ({c}/3) - 5 = {nc}")
        
        if len(shape) == 5:
             # (1, 3, H, W, 85)
             nc = shape[4] - 5
             if nc > 0:
                 guessed_classes.append(nc)
                 model_type = "YOLOv5 (Anchor-based)"
                 # print(f"  -> Possible YOLOv5 format (NHWC). If so, classes = {shape[4]} - 5 = {nc}")

    if guessed_classes:
        # 找到最可能的类别数
        likely_class_count = max(set(guessed_classes), key=guessed_classes.count)
        print(f"Model Type: {model_type}")
        print(f"Class Count: {likely_class_count}")
        # print("Note: RKNN models do not store class names, only the number of classes can be inferred.")
        # print("To see the actual class names, please check the source .pt file using this script.")
    else:
        print("\nCould not infer class count from output shapes.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_classes.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)
        
    if model_path.endswith('.pt'):
        check_pt_model(model_path)
    elif model_path.endswith('.rknn'):
        check_rknn_model(model_path)
    else:
        print("Unsupported file extension. Please use .pt or .rknn")
