# YOLO Web Streamer for RK3588

这是一个基于 Flask 和 OpenCV 的 YOLO 目标检测 Web 推流项目，专为 Rockchip RK3588 平台（如 NanoPC-T6）优化。它支持使用 RKNN NPU 加速推理，并包含 HID 模拟功能（用于自动瞄准/控制）。

## 功能特性

*   **Web 视频流**: 通过浏览器实时查看检测结果（访问 `http://<IP>:5000`）。
*   **多 NPU 加速**: 针对 RK3588 的 3 个 NPU 核心进行并行推理，显著提升 FPS。
*   **多模型支持**: 
    *   支持 `.rknn` 模型（使用 `rknnlite2` 进行 NPU 加速）。
    *   支持 `.pt` 模型（使用 `ultralytics` 进行 CPU 推理，仅用于调试）。
*   **HID 控制**: 检测到目标后，通过 USB HID 接口模拟鼠标移动（需要硬件支持及 `hidtx` 工具）。
*   **自动模型尺寸探测**: 自动识别 RKNN 模型的输入分辨率。

## 环境要求

*   **硬件**: Rockchip RK3588 开发板 (例如 NanoPC-T6)。
*   **操作系统**: Linux (Ubuntu/Debian)。
*   **Python 环境**: 建议使用 Conda 环境 (例如 `rknn`)。

### 依赖库

请确保安装了以下 Python 库：

```bash
pip install flask opencv-python numpy
# 如果使用 RKNN 模型
pip install rknn-toolkit-lite2
# 如果使用 PT 模型
pip install ultralytics
```

## 文件结构

*   `simple_yolo_web.py`: 主程序脚本。
*   `check_model.py`: 模型检查工具。
*   `hidtx`: HID 发送工具（二进制文件）。
*   `Models/`: 存放模型文件的目录（需自行创建或修改路径）。

## 快速开始

1.  **准备模型**:
    确保你的 `.rknn` 模型文件路径正确。默认路径在脚本中配置为 `./Models/yolo11n-rk3588.rknn`。

2.  **运行程序**:
    使用配置好 RKNN 环境的 Python 解释器运行脚本：

    ```bash
    # 假设你的环境名为 rknn
    /home/pi/anaconda3/envs/rknn/bin/python simple_yolo_web.py
    ```
    或者直接：
    ```bash
    python3 simple_yolo_web.py
    ```

3.  **访问 Web 界面**:
    在浏览器中输入开发板的 IP 地址和端口 5000，例如：
    `http://192.168.x.x:5000`

## 配置说明

可以在 `simple_yolo_web.py` 开头部分修改配置参数：

```python
MODEL_PATH = './Models/yolo11n-rk3588.rknn'  # 模型路径
CONF_THRES = 0.45                            # 置信度阈值
IOU_THRES = 0.2                              # NMS IOU 阈值
CAMERA_INDEXES = [20, 21, 11, 0]             # 摄像头索引尝试列表
TARGET_CLASS_ID = 0                          # 目标类别 ID (0 通常是 person)
HID_DEVICE = '/dev/hidg1'                    # HID 设备节点
```

## 注意事项

*   **HID 权限**: 使用 HID 功能通常需要 `sudo` 权限，或者配置相应的 udev 规则。脚本中尝试使用 `sudo` 调用 `hidtx`。
*   **摄像头**: 脚本会按顺序尝试 `CAMERA_INDEXES` 中的索引直到找到可用摄像头。
*   **性能**: 使用 `.rknn` 模型时，脚本会自动启动 3 个线程利用 RK3588 的 3 个 NPU 核心。

## 常见问题

*   **报错 `rknnlite package not found`**: 请检查是否在正确的 Conda/Python 环境中运行。
*   **画面卡顿**: 检查网络连接，或者尝试降低摄像头分辨率。
