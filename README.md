# YOLO11n RK3588 AI 助手

这是一个专为 Rockchip RK3588 平台（如 NanoPC-T6, Orange Pi 5）打造的高性能 YOLO 目标检测与自动辅助系统。项目利用 RK3588 强大的 NPU 进行硬件加速推理，结合 C 语言底层优化，实现高帧率目标检测，并通过 USB HID 接口模拟鼠标进行智能辅助操作。

## ✨ 主要功能

*   **🚀 极致性能**: 
    *   利用 RK3588 NPU 进行 INT8 量化推理。
    *   底层后处理算法采用 C 语言编写 (`c_src`), 大幅降低 Python 开销。
    *   多线程异步处理视频流与推理。
*   **👁️ 实时监控**: 内置 Flask Web 服务器，通过浏览器 (端口 5000) 实时查看低延迟检测画面。
*   **🎮 硬件级模拟**: 
    *   通过 Linux USB Gadget API (`/dev/hidg1`) 模拟物理鼠标，无需软件驱动，难以被检测。
    *   支持动态 PID 控制算法，实现平滑、拟人的瞄准轨迹。
*   **🔥 智能辅助**: 
    *   **自动瞄准**: 自动吸附目标，支持动态死区和灵敏度调节。
    *   **自动射击**: 锁定目标后自动触发点击，支持冷却时间和后坐力补偿。
*   **🛠️ 广泛兼容**: 支持 YOLOv5, YOLOv8, YOLOv10, YOLOv11 等多种架构的 RKNN 模型。

## 🛠️ 硬件与环境要求

*   **开发板**: Rockchip RK3588 系列 (推荐 NanoPC-T6, Orange Pi 5 等)。
*   **连接**: 开发板 OTG 接口需连接到目标主机（PC/游戏机）以模拟鼠标。
*   **摄像头**: HDMI 输入或 USB 摄像头 (支持 MJPEG 格式更佳)。
*   **系统**: Linux (Ubuntu/Debian/Armbian) with Rockchip kernel 5.10+。

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd YOLO11n
```

### 2. 编译 C 代码
本项目使用 C 语言处理 NPU 输出以提升性能：
```bash
cd c_src
make
cd ..
# 编译成功后，应在 src 目录下看到 librknn_yolo.so
```

## 🚀 运行指南

### 1. 系统初始化 (每次重启后需运行)
运行 `setup.sh` 脚本以配置 USB HID 设备 (`/dev/hidg1`) 并开启 CPU/NPU 高性能模式。
```bash
sudo ./setup.sh
```
> **注意**: 此脚本需要 root 权限，且必须在连接 OTG 线缆的情况下运行，否则主机无法识别模拟鼠标。

### 2. 运行主程序
```bash
sudo python3 yolo_web_hid.py
```
*   程序默认监听 HDMI IN (Camera Index 20)，如需更改请编辑 `yolo_web_hid.py` 中的 `CAMERA_INDEX`。
*   默认加载模型: `./Models/cf-11n-rk3588-int8.rknn`。

## 📂 文件结构说明

*   `yolo_web_hid.py`: 主程序，包含 Web 服务、推理循环和 HID 控制逻辑。
*   `setup.sh`: 系统初始化脚本，配置 USB Gadget 和性能模式。
*   `check_model.py`: 用于检查 RKNN 模型详细信息的工具。
*   `c_src/`: C 语言源码目录，包含 NPU 后处理加速代码。
    *   `yolo_rknn.c`: 后处理实现。
    *   `Makefile`: 编译脚本。
*   `Models/`: 存放转换好的 `.rknn` 模型文件。

## 🔧 常见问题

**Q: 运行提示 `OSError: ./c_src/librknn_yolo.so: cannot open shared object file`**
A: 请确保你已经进入 `c_src` 目录并执行了 `make` 命令。

**Q: 提示找不到 `/dev/hidg1`**
A: 请确保已使用 `sudo` 运行了 `setup.sh`，并且开发板的 OTG 接口已正确连接到电脑。

**Q: 帧率很低**
A: 
1. 确保运行了 `setup.sh` 开启高性能模式。
2. 检查摄像头输入分辨率，推荐 1080p 或 720p。
3. 确保使用的是 INT8 量化的 RKNN 模型。
