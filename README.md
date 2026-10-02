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
tests/         后处理、目标跟踪、HID、Web 预览及 EDID 回归测试
Models/        本机模型及 EDID（不纳入 Git 和发布包）
build/obj/     编译中间文件及头文件依赖（自动生成）
dist/          发布包与 SHA-256 校验文件（自动生成）
```

## 构建

在 RK3588 的 AArch64 Linux 上准备 GCC/G++、GNU Make、pkg-config、OpenCV 4 开发包，以及与板卡驱动配套的 RKNN Runtime 和 RGA 库。编译使用系统环境，无需激活 Anaconda。

Ubuntu 可通过以下命令安装通用依赖；RKNN/RGA 的头文件和库使用板卡 SDK 提供的版本。

```bash
sudo apt install build-essential pkg-config libopencv-dev python3 v4l-utils edid-decode
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

- 1–3 个 YOLO26 原始特征图：NCHW 或 NHWC，通道为四个 LTRB 回归距离加类别分数。解码包含网格/步长还原，并按类别执行 NMS 去重。板卡上的 CS2、CF 模型分别为三尺度、4 类和 2 类。
- 单个端到端输出：`[N,6]`、`[6,N]`、`[1,N,6]` 或 `[1,6,N]`，每条检测为 `x1,y1,x2,y2,score,class`。

原始特征图会对同一个目标输出多个网格候选框，不能直接把所有候选当成独立目标。程序先收集全部候选，按置信度排序，再以默认 IoU 0.5 抑制同类别的重复框，最后限制输出数量；可通过 `--nms-iou 0.5` 调整阈值（大于 0、小于 1）。不同类别分别保留，同一个人的头部和身体可能各有一个框，因此页面的“检测框”数量不等于人数。端到端 `[N,6]` 模型沿用自身输出，不额外施加 NMS。

原始特征图的类别通道与端到端输出的类别索引不同，不能用同一种解析方式。模型检查验证加载和一次 NPU 推理，检测精度仍需要实际输入图像验证。

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

## 目标跟踪与选择

参考 `mydet.py` 的跨帧关联、目标锁定、头部选择和运动预测思路，新增适配当前 C++/RKNN 工程的几何跟踪器。检测结果仍先执行类别 NMS，再逐一匹配同类别框；每个旧目标在一帧中最多匹配一次，避免多个候选覆盖同一个目标。锁定选择带切换冷却和距离差阈值，减少相近目标之间的反复跳转。短暂漏检保留编号 120ms，但停止 HID 移动和点击；重新出现或切换目标时清空旧移动缓存并重置 PID 差分。

默认仍选择靠近画面中心的目标。头部类别需通过 `--head-classes` 显式指定，程序不假设奇数类别就是头部。配置后可在网页顶部切换“中心最近”“头部优先”“仅头部”“仅身体”；头部瞄准框中心，身体保持框高度 20% 的原有位置。画面黄色 `LOCK #编号` 十字表示当前选择，暂停 HID 时仍可预览。

本板 CS2 实测类别 2 为身体、类别 3 为头部，可使用：

```bash
sudo ./yolo_app_26 --scores logits --target-classes 2,3 --head-classes 3 --aim-mode head-priority --start-paused
```

其他模型应按自己的训练标签修改类别列表。`body-only` 表示目标类别列表中不属于 `--head-classes` 的类别。头部优先会对头部给予距离评分加成，仍允许显著靠近中心的身体目标被选中；不是把所有头部强行排在身体前。

可选参数：`--aim-radius 200` 将选择限制在原始 HDMI 画面中心 200 像素半径内，并画出范围圈（默认 0，即全画面）；这是选择范围，不裁剪模型输入。`--prediction-ms 20` 开启速度平滑后的提前量，允许 0–40ms，位移限制为框对角线的 25% 且最多 20 像素，漏检、方向反转或长时间停顿会重置／减弱预测；默认关闭。`--hid-smoothing 0.5` 可调节现有 HID 输出平滑（0.01–1，默认 1；数值较小更平缓、延迟也更高）。坐标始终使用真实输入尺寸和 letterbox 逆变换，不复制参考脚本的固定 416 尺寸、Windows 截屏或 CUDA 推理路径。

跟踪仅依据位置、框重叠和类别，不具备身份识别能力，目标重叠、快速穿越或长时间遮挡时可能重新分配编号。`/status` 增加 `aim_mode`、`target_id`、`target_visible`、`head_modes_available`、`aim_radius`；`POST /aim/nearest`、`/aim/head-priority`、`/aim/head-only`、`/aim/body-only` 用于切换模式，缺少头部类别配置时后三项返回 400。

使用 Ctrl+C 退出。停止 USB Gadget：

```bash
sudo bash ./setup.sh stop
```

## HDMI 输入刷新率

EDID 文件为 `Models/1080p_multi_hz.edid`，向信号源声明 1920×1080 的 60、120、144、165、180、240Hz 模式，首选保持 60Hz。高刷新率是否稳定仍取决于信号源、HDMI 线和板卡驱动。生成器修正了 CTA 数据块长度和 HDMI 扩展块排列，并检查详细时序字段范围、结构与校验和。

2026-10-01 在本机 Windows 独立 HDMI 直连 NanoPC-T6（内核 6.1.118）时，120/144/165/180/240Hz 均读到对应输入时序；240Hz 读回为 239.99Hz、558.88MHz，V4L2 以 BGR24 连续采集 480 帧成功，工具报告约 240fps。这是该连接条件下的短时实测；模型推理和 Web 预览帧率另受计算及编码速度影响。

已有旧版 EDID 时先重新生成；更新脚本会拒绝结构或校验和错误，安装 `edid-decode` 后还会执行完整合规检查，不会只修复校验和后继续加载：

```bash
python3 tools/gen_edid.py --output Models/1080p_multi_hz.edid
edid-decode --check Models/1080p_multi_hz.edid
# 停止正在采集视频的程序后加载，HDMI 会短暂重新连接。
sudo bash ./setup.sh edid
sudo bash ./setup.sh edid-status
```

`setup.sh edid` 仅处理 HDMI EDID，不改 USB HID；写入后读回并逐字节比较，失败会返回非零状态。重复加载相同文件会验证读回后直接退出，避免反复断开输入。缺少文件时 `setup.sh start` 或 `edid` 会自动生成；可通过 `VIDEO_DEV` 和 `HDMI_EDID_FILE` 环境变量指定设备及文件。

在 Windows 的“设置 → 系统 → 屏幕 → 高级显示”中选择 `RK3588-Multi`，分辨率设为 1920×1080，先尝试 120 或 144Hz。若仍显示旧模式，重新插拔 HDMI 后再查看。更改刷新率前退出采集程序，确认新信号锁定后再启动，避免旧采集队列在信号切换后停止出帧。`edid-status` 的像素时钟和总行列数反映实际输入信号；应用的 FPS 参数或 Web 预览速度不会改变信号源刷新率。EDID 是运行时配置，板卡重启后应在启动采集前重新执行 `setup.sh start`。

## 发布包

```bash
make package
cd dist
sha256sum -c yolo-hid-rk3588-*.tar.gz.sha256
```

发布包包含 AArch64 可执行文件、源码、构建和运行脚本，不包含本机模型、RKNN/RGA/OpenCV 动态库。新板卡需要准备相同的运行库，并将自己的模型放入 `Models/`。包内的 `BUILD_COMMIT` 标明构建所用提交；如果工作区有未提交修改，则标注 `-dirty`。
