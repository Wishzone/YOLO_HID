# YOLO11n RK3588 AI 助手

这是一个专为 Rockchip RK3588 平台（如 NanoPC-T6）打造的高性能 YOLO 目标检测与自动辅助系统。项目利用 RK3588 强大的 NPU 进行 3 核心并行推理，实现高帧率的目标检测，并通过 USB HID 接口模拟鼠标进行智能辅助操作。

## ✨ 主要功能

*   **🚀 极致性能**: 利用 RK3588 的 3 个 NPU 核心进行多线程并行推理，大幅提升 FPS。
*   **👁️ 实时监控**: 内置 Flask Web 服务器，通过浏览器实时查看低延迟的检测画面 (MJPEG 流)。
*   **🎯 智能瞄准**: 
    *   集成 **PID 控制算法**，实现平滑、精准的鼠标移动，拒绝机械式卡顿。
    *   支持 **动态死区** 和 **灵敏度调节**。
*   **🔥 自动射击**: 当准星锁定目标时，自动触发鼠标左键点击（支持冷却时间配置）。
*   **⚡ 高效传输**: 
    *   直接操作 `/dev/hidg1` 设备文件，无缓冲写入，延迟极低。
    *   优化了 Web 视频流编码，降低 CPU 占用。
*   **🛠️ 广泛兼容**: 支持 YOLOv5, YOLOv8, YOLOv10, YOLOv11 的 RKNN 模型。

## 🛠️ 硬件与环境要求

*   **开发板**: Rockchip RK3588 系列 (推荐 NanoPC-T6, Orange Pi 5 等)。
*   **摄像头**: USB 摄像头 (支持 MJPEG 格式更佳)。
*   **连接**: 开发板 OTG 接口需连接到目标主机（PC/游戏机）以模拟鼠标。
*   **系统**: Linux (Ubuntu/Debian/Armbian)。
*   **Python**: Python 3.8+ (建议使用 Conda 环境)。

## 📦 安装依赖

建议在 Conda 环境中运行：

```bash
# 激活环境
conda activate rknn

# 安装基础依赖
pip install flask opencv-python numpy

# 安装 RKNN Lite2 (用于 NPU 推理)
# 请从 Rockchip 官方仓库下载对应 Python 版本的 whl 包安装
pip install rknn_toolkit_lite2-*.whl

# (可选) 如果需要调试 .pt 模型
pip install ultralytics
```

## ⚙️ USB HID 配置 (关键)

本项目依赖 Linux USB Gadget API 来模拟鼠标。在运行程序前，必须确保 `/dev/hidg1` 设备存在。

如果你的系统中没有该设备，请创建一个启动脚本 `hid_setup.sh` 并以 root 权限运行：

```bash
#!/bin/bash
# 配置 USB Gadget 为鼠标设备

CONFIGFS_HOME=/sys/kernel/config/usb_gadget
GADGET_NAME=rknn_mouse
LANG=0x409

modprobe libcomposite

mkdir -p ${CONFIGFS_HOME}/${GADGET_NAME}
cd ${CONFIGFS_HOME}/${GADGET_NAME}

echo 0x1d6b > idVendor  # Linux Foundation
echo 0x0104 > idProduct # Multifunction Composite Gadget
echo 0x0100 > bcdDevice
echo 0x0200 > bcdUSB

mkdir -p strings/${LANG}
echo "RKNN-AI" > strings/${LANG}/manufacturer
echo "AI-Mouse" > strings/${LANG}/product
echo "12345678" > strings/${LANG}/serialnumber

# 配置 HID 功能
mkdir -p functions/hid.usb0
echo 1 > functions/hid.usb0/protocol
echo 1 > functions/hid.usb0/subclass
echo 8 > functions/hid.usb0/report_length
# 写入鼠标报告描述符
echo -ne \\x05\\x01\\x09\\x02\\xa1\\x01\\x09\\x01\\xa1\\x00\\x05\\x09\\x19\\x01\\x29\\x03\\x15\\x00\\x25\\x01\\x95\\x03\\x75\\x01\\x81\\x02\\x95\\x01\\x75\\x05\\x81\\x03\\x05\\x01\\x09\\x30\\x09\\x31\\x09\\x38\\x15\\x81\\x25\\x7f\\x75\\x08\\x95\\x03\\x81\\x06\\xc0\\xc0 > functions/hid.usb0/report_desc

mkdir -p configs/c.1/strings/${LANG}
echo "Config 1" > configs/c.1/strings/${LANG}/configuration
echo 250 > configs/c.1/MaxPower

# 关联功能
ln -s functions/hid.usb0 configs/c.1/

# 启用 Gadget (请根据实际 UDC 名称修改，通常是 fc000000.usb 或类似)
ls /sys/class/udc > UDC
chmod 777 /dev/hidg0 2>/dev/null || true
chmod 777 /dev/hidg1 2>/dev/null || true
```

## 🚀 运行项目

1.  **准备模型**:
    将转换好的 `.rknn` 模型放入 `Models/` 目录。
    修改 `yolo_web_hid.py` 中的 `MODEL_PATH` 变量指向你的模型。

2.  **启动脚本**:

    ```bash
    # 建议使用 sudo 以确保有权限访问 /dev/hidg1 和 摄像头
    sudo /home/pi/anaconda3/envs/rknn/bin/python yolo_web_hid.py
    ```

3.  **访问 Web 界面**:
    在浏览器中访问: `http://<开发板IP>:5000`

## 🔧 参数调优

在 `yolo_web_hid.py` 顶部可以调整核心参数：

```python
# 核心配置
CONF_THRES = 0.7         # 置信度阈值
TARGET_CLASS_ID = 0      # 目标类别 ID (0 通常是人)
ENABLE_HID = True        # 总开关

# 瞄准参数
AIM_OFFSET_X = 5         # 准星横向偏移校准
AIM_HEIGHT_RATIO = 0.10  # 瞄准高度 (0.0=头顶, 0.5=中心)
MOUSE_SENSITIVITY = 0.8  # 鼠标移动灵敏度

# 自动射击
AUTO_SHOOT = True        # 启用自动射击
SHOOT_THRESHOLD = 15     # 触发范围 (像素)
SHOOT_COOLDOWN = 0.2     # 射击冷却 (秒)

# PID 控制 (平滑移动)
PID_KP = 0.65            # 比例系数 (响应速度)
PID_KD = 0.40            # 微分系数 (阻尼/防抖)
```

## ⚠️ 免责声明

本项目仅供计算机视觉与嵌入式系统学习研究使用。请勿用于任何违反游戏公平性或法律法规的用途。作者不对使用本项目造成的任何后果负责。
