# YOLO HID for NanoPC-T6 / RK3588

基于 RKNN Runtime 的 YOLO26 检测工程，包含 HDMI/V4L2 采集、RGA 图像预处理、USB HID 输出和 HTTP/MJPEG 预览。保留现有检测和控制参数。

## 目录

```text
src/core/       公共状态和检测数据类型
src/models/     YOLO26 RKNN 模型加载、推理及输出解析
src/modules/    V4L2、图像采集、USB HID 和 Web 预览
src/utils/      图像预处理及调度辅助函数
tools/         模型检查和 EDID 生成工具
scripts/       发布包生成脚本
tests/         后处理坐标、布局和阈值测试
Models/        本机模型及 EDID（不纳入 Git 和发布包）
build/obj/     编译中间文件及头文件依赖（自动生成）
dist/          发布包与 SHA-256 校验文件（自动生成）
```

## 构建

在 RK3588 的 AArch64 Linux 上准备 GCC/G++、GNU Make、pkg-config、OpenCV 4 开发包，以及与板卡驱动配套的 RKNN Runtime 和 RGA 库。编译使用系统环境，无需激活 Anaconda。

Ubuntu 可通过以下命令安装通用依赖；RKNN/RGA 的头文件和库使用板卡 SDK 提供的版本。

```bash
sudo apt install build-essential pkg-config libopencv-dev
make check-deps
make -j4
make test
```

也兼容原来的 `make -C src`。可执行文件为项目根目录下的 `yolo_app_26`。头文件修改会通过 `.d` 依赖自动触发重新编译；中间文件集中在 `build/obj/`。

```bash
make clean
make -j4
```

## 模型验证和运行

默认模型路径为 `Models/cs2-26n_rk3588_int8.rknn`，也可以通过 `--model` 指定其他模型。输入要求 RGB 640×640；支持两种输出结构：

- 1–3 个 YOLO26 原始特征图：NCHW 或 NHWC，通道为四个 LTRB 回归距离加类别分数。解码包含网格/步长还原。板卡上的 CS2、CF 模型分别为三尺度、4 类和 2 类。
- 单个端到端输出：`[N,6]`、`[6,N]`、`[1,N,6]` 或 `[1,6,N]`，每条检测为 `x1,y1,x2,y2,score,class`。

保留原工程不使用 NMS 的设置。原始特征图的类别通道与端到端输出的类别索引不同，不能用同一种解析方式。模型检查验证加载和一次 NPU 推理，检测精度仍需要实际输入图像验证。

类别分数模式默认 `auto`：输出出现 `[0,1]` 以外的类别值时按 logits 应用 sigmoid，否则按概率读取。已知模型建议显式指定，避免自动判断的歧义：本板 CS2 模型使用 `--scores logits`，CF 模型使用 `--scores probabilities`，避免对已有概率重复应用 sigmoid。

先检查 NPU 推理。这一步不会打开视频设备、启动 Web 服务或发送 HID 指令。

```bash
./yolo_app_26 --help
./yolo_app_26 --check-model
./yolo_app_26 --model Models/cf-26n-rk3588-int8.rknn --scores probabilities --check-model
```

完整运行需要 HDMI 输入信号、`/dev/video20` 采集设备和 USB OTG HID 接口。`setup.sh start` 会设置性能策略、视频 EDID 和 USB Gadget；按需执行。

```bash
cd /path/to/YOLO
sudo bash ./setup.sh start
sudo ./yolo_app_26 --model Models/cs2-26n_rk3588_int8.rknn --scores logits
```

浏览器访问 `http://<板卡IP>:8080/` 可查看所有类别的检测框、类别编号和置信度；页面顶部显示 USB 连接状态、检测数量、HID 成功发送和失败次数。可用页面按钮启用／暂停 HID，暂停时释放按钮并丢弃待发送移动。MJPEG 地址为 `/stream`，带框截图为 `/snapshot.jpg`，状态 JSON 为 `/status`。

Web 显示所有类别，HID 默认仍控制类别 0。可按训练标签指定目标类别；类别编号必须来自自己的模型，不能照搬其他模型的头部／身体顺序：

```bash
sudo ./yolo_app_26 --scores logits --target-classes 0,2
sudo ./yolo_app_26 --scores logits --target-classes all
sudo ./yolo_app_26 --scores logits --target-classes 2,3 --start-paused # 从 Web 启用 HID
sudo ./yolo_app_26 --scores logits --no-hid    # 仅检测和 Web 预览
```

HID 设备可通过 `--hid-device /dev/hidg1` 指定。USB OTG 数据接口必须连接被控制电脑，`USB configured` 表示已枚举；`not attached` 表示尚未建立连接。HID 写入失败会重试，程序正常退出时发送释放按钮的报表。`make test` 使用模拟接口验证 HID，不会操作实际鼠标。

使用 Ctrl+C 退出。停止 USB Gadget：

```bash
sudo bash ./setup.sh stop
```

EDID 文件保持为 `Models/1080p_multi_hz.edid`。需要重新生成时：

```bash
mkdir -p Models
(cd Models && python3 ../tools/gen_edid.py)
```

## 发布包

```bash
make package
cd dist
sha256sum -c yolo-hid-rk3588-*.tar.gz.sha256
```

发布包包含 AArch64 可执行文件、源码、构建和运行脚本，不包含本机模型、RKNN/RGA/OpenCV 动态库。新板卡需要准备相同的运行库，并将自己的模型放入 `Models/`。包内的 `BUILD_COMMIT` 标明构建所用提交；如果工作区有未提交修改，则标注 `-dirty`。
