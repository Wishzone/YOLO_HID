# YOLO11n RKNN Web Inference

这是一个基于 RKNN (Rockchip NPU) 的 YOLOv11/YOLOv5 目标检测项目，提供了一个简单的 Web 界面来查看实时检测结果。

## 功能特点
- 支持 RKNN 模型推理 (YOLOv11, YOLOv5 等)
- 基于 Flask 的 Web 视频流
- 包含模型检查工具
- 支持 USB HID 模拟 (键盘/鼠标)

## 环境要求
- Python 3.x
- rknnlite2
- opencv-python
- flask
- numpy

## 文件说明
- `simple_yolo_web.py`: 主程序，启动 Web 服务器并进行目标检测。
- `check_model_classes.py`: 检查 `.pt` 模型的类别名称或推断 `.rknn` 模型的类别数量。
- `check_rknn_model.py`: 简单的 RKNN 模型加载测试脚本。
- `hid_keyboard_mouse.sh`: 配置 USB Gadget 模式以模拟 HID 设备。
- `*.rknn`: 转换好的 RKNN 模型文件。
- `*.pt`: PyTorch 原始模型文件。

## 使用方法

### 1. 启动 Web 检测服务
```bash
/home/pi/anaconda3/envs/rknn/bin/python simple_yolo_web.py
```
默认端口为 5000，可以通过浏览器访问 `http://<设备IP>:5000` 查看视频流。

### 2. 检查模型类别
```bash
# 检查 .pt 文件 (显示具体类别名)
python check_model_classes.py yolo11n.pt

# 检查 .rknn 文件 (推断类别数量)
python check_model_classes.py yolov11n-cf-640-640.rknn
```

### 3. 配置 HID 模式
如果需要使用 HID 功能：
```bash
sudo ./hid_keyboard_mouse.sh
```

## 注意事项
- 确保使用的 `.rknn` 模型与你的硬件平台 (如 RK3588) 兼容。
- 如果更换模型，请在 `simple_yolo_web.py` 中修改 `MODEL_PATH` 和 `COCO_CLASSES`。
