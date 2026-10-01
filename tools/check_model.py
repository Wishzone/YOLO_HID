import sys
import os
import numpy as np
import argparse

# Context manager to suppress C-level stdout/stderr but allow printing to real stdout
class SilenceAndPrint:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.real_stdout = None
        self.capture_path = "/tmp/rknn_capture.log"
        self.capture_file = None
        
    def __enter__(self):
        if self.verbose: 
            return sys.stdout
            
        # Flush python buffers
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Save original stdout for our use
        real_stdout_fd = os.dup(1)
        self.real_stdout = os.fdopen(real_stdout_fd, 'w')
        
        # Redirect 1 and 2 to temp file instead of devnull to capture logs
        self.capture_file = open(self.capture_path, 'w+')
        os.dup2(self.capture_file.fileno(), 1)
        os.dup2(self.capture_file.fileno(), 2)
        
        return self.real_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose: return
        # Do NOT restore stdout/stderr to avoid late logs from C libraries
        # Just close our private handle
        if self.real_stdout:
            self.real_stdout.close()
        if self.capture_file:
            self.capture_file.close()

    def get_captured_content(self):
        if self.verbose: return ""
        try:
            if self.capture_file:
                self.capture_file.flush()
                os.fsync(self.capture_file.fileno())
                self.capture_file.seek(0)
                return self.capture_file.read()
        except Exception as e:
            return ""
        return ""

def check_pt_model(model_path, verbose=False):
    # Use SilenceAndPrint to suppress logs
    with SilenceAndPrint(verbose) as out:
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Gather info
            info = {}
            info['model'] = os.path.basename(model_path)
            
            if hasattr(model, 'task'):
                info['task'] = model.task
                
            if hasattr(model, 'names'):
                info['names'] = model.names
                
            if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
                info['imgsz'] = model.overrides['imgsz']
            
            # Capture model info to get type
            import io
            import logging
            from ultralytics.utils import LOGGER
            
            capture_buf = io.StringIO()
            handler = logging.StreamHandler(capture_buf)
            LOGGER.addHandler(handler)
            
            model.info()
            
            LOGGER.removeHandler(handler)
            
            model_info_str = capture_buf.getvalue().strip()
            if model_info_str:
                # Example: "YOLO26n summary: ..."
                first_line = model_info_str.split('\n')[0]
                if "summary" in first_line:
                    info['type'] = first_line.split("summary")[0].strip()
                else:
                    info['type'] = first_line

            # Print to real stdout
            if out:
                out.write(f"Model: {info['model']}\n")
                if 'type' in info:
                    out.write(f"Type: {info['type']}\n")
                if 'task' in info:
                    out.write(f"Task: {info['task']}\n")
                if 'imgsz' in info:
                    out.write(f"Input Size: {info['imgsz']}\n")
                if 'names' in info:
                    names = info['names']
                    out.write(f"Class Count: {len(names)}\n")
                    # Format classes nicely
                    class_list = list(names.values())
                    out.write(f"Classes: {class_list}\n")

        except ImportError:
            if out: out.write("Error: ultralytics package not found.\n")
        except Exception as e:
            if out: out.write(f"Error checking .pt model: {e}\n")

def check_rknn_model(model_path, verbose=False):
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("Error: rknnlite package not found.")
        return

    result_info = {}
    result_info['model'] = os.path.basename(model_path)
    
    # Use SilenceAndPrint to suppress C-level logs but keep a handle to real stdout
    silencer = SilenceAndPrint(verbose)
    with silencer as out:
        # Force verbose=True to ensure we capture the version info logs
        # The output is silenced anyway, so the user won't see the spam
        rknn = RKNNLite(verbose=True) 
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            if out: out.write("Error: Load RKNN model failed\n")
            return
        ret = rknn.init_runtime()
        if ret != 0:
            if out: out.write("Error: Init runtime failed\n")
            return
            
        # Get SDK Version
        if hasattr(rknn, 'get_sdk_version'):
            sdk_ver = rknn.get_sdk_version()
            if sdk_ver:
                # Clean up SDK version string
                lines = sdk_ver.strip().split('\n')
                clean_ver = []
                for line in lines:
                    if "API:" in line or "DRV:" in line:
                        clean_ver.append(line.strip())
                if clean_ver:
                    result_info['sdk_version'] = ", ".join(clean_ver)
                else:
                    result_info['sdk_version'] = sdk_ver.strip()
                
        # Parse captured logs for model info
        logs = silencer.get_captured_content()
        
        if logs:
            import re
            # Look for "RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 ...)"
            match = re.search(r"RKNN Model Information, (.*)", logs)
            if match:
                result_info['model_info'] = match.group(1).strip()
                # Extract toolkit version specifically
                tv_match = re.search(r"toolkit version: ([^\(]+)", result_info['model_info'])
                if tv_match:
                    result_info['toolkit_version'] = tv_match.group(1).strip()
            else:
                # Try looser match
                match = re.search(r"toolkit version: ([^\(]+)", logs)
                if match:
                    result_info['toolkit_version'] = match.group(1).strip()
    
        # Inference Test
        test_sizes = [320, 416, 512, 640, 736, 768, 960, 1280]
        input_size = None
        outputs = None
        
        for size in test_sizes:
            try:
                img = np.zeros((1, size, size, 3), dtype=np.uint8)
                outputs = rknn.inference(inputs=[img])
                if outputs is not None:
                    input_size = size
                    break
            except:
                continue
                
        if outputs is None:
            result_info['status'] = "Failed to determine input size"
        else:
            result_info['input_size'] = f"{input_size}x{input_size}"
            
            if len(outputs) == 1 and outputs[0].ndim in (2, 3) and 6 in outputs[0].shape[-2:]:
                result_info['type'] = "YOLO26 (end-to-end XYXY detections)"
                result_info['note'] = "Class indices are returned directly; class count is not encoded in this shape"
            elif len(outputs) == 1:
                out_tensor = outputs[0]
                if out_tensor.ndim == 3:
                    channels = out_tensor.shape[1]
                    anchors = out_tensor.shape[2]
                    nc = channels - 4
                    result_info['type'] = "YOLO26 (single-output detection)"
                    result_info['class_count'] = nc
                    
                    inferred_size = int(np.sqrt(anchors * 1024 / 21))
                    if input_size != inferred_size:
                        result_info['note'] = f"Model structure suggests {inferred_size}x{inferred_size}"

            elif len(outputs) == 3 and all(t.ndim == 4 for t in outputs):
                result_info['type'] = "Raw multi-scale detection heads (verify model export format)"
                out0 = outputs[0]
                if out0.ndim == 4:
                    c = out0.shape[1]
                    if c in (6, 8):
                        result_info['type'] = "YOLO26-compatible raw heads (4 box channels + class scores)"
                        result_info['class_count'] = c - 4
                    result_info['note'] = f"Output shapes: {[list(t.shape) for t in outputs]}"
            
            else:
                result_info['type'] = f"Unknown ({len(outputs)} outputs)"
        
        rknn.release()

        # Print results using the captured stdout handle
        if out:
            out.write(f"Model: {result_info.get('model')}\n")
            if 'status' in result_info:
                out.write(f"Status: {result_info['status']}\n")
            else:
                if 'sdk_version' in result_info:
                    out.write(f"SDK Version: {result_info['sdk_version']}\n")
                if 'toolkit_version' in result_info:
                    out.write(f"Toolkit Version: {result_info['toolkit_version']}\n")
                # if 'model_info' in result_info:
                #     out.write(f"Model Info: {result_info['model_info']}\n")
                    
                out.write(f"Input Size: {result_info.get('input_size')}\n")
                out.write(f"Type: {result_info.get('type')}\n")
                if 'class_count' in result_info:
                    out.write(f"Class Count: {result_info['class_count']}\n")
                if 'note' in result_info:
                    out.write(f"Note: {result_info['note']}\n")

def main():
    parser = argparse.ArgumentParser(description='Check .pt or .rknn model parameters.')
    parser.add_argument('model_path', help='Path to the model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output (RKNN logs)')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found")
        sys.exit(1)
        
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == '.pt':
        check_pt_model(model_path, verbose=args.verbose)
    elif ext == '.rknn':
        check_rknn_model(model_path, verbose=args.verbose)
    else:
        print(f"Unsupported file extension: {ext}. Please use .pt or .rknn")

if __name__ == "__main__":
    main()
