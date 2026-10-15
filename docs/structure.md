# 项目结构与模块职责

项目按源码、工具、测试、文档和本机数据组织。根目录 `Makefile` 是构建入口，`setup.sh` 是板卡配置入口。新增功能放入对应源码模块，运行参数与操作步骤分别维护在 [runtime.md](runtime.md) 和 [hdmi.md](hdmi.md)。

## 源码模块

| 路径 | 职责 |
| --- | --- |
| `src/main_yolo26.cpp` | 解析命令行参数，初始化设备和工作线程，调度检测、跟踪及 HID 控制 |
| `src/core/types.h` | C/C++ 共用的检测结果结构 |
| `src/core/globals.*` | 跨线程运行状态、HID 缓冲区与控制设置 |
| `src/models/yolo26_rknn.*` | RKNN 模型加载、输入输出管理、推理和资源释放 |
| `src/models/yolo26_postprocess.*` | 输出布局解析、置信度转换、候选解码和类别 NMS |
| `src/modules/v4l2_driver.*` | V4L2 设备、缓冲区、采集队列和 HDMI fence 管理 |
| `src/modules/camera_capture.*` | 视频采集线程与最新帧交接 |
| `src/modules/target_tracker.*` | 跨帧关联、目标锁定、瞄准模式及有限运动预测 |
| `src/modules/aim_controller.*` | 实机与仿真共用的逐帧 PD、死区、倍率及点击判定 |
| `src/modules/aim_simulation.*` | 固定准星闭环仿真、延迟和 HID 时序模型及入框统计 |
| `src/modules/hid_motion.h` | 实机与仿真共用的 HID 整数拆分、平滑和限幅 |
| `src/modules/hid_mouse.*` | USB HID 报表、移动拆分、失败重试及按键释放 |
| `src/modules/web_server.*` | 画面拷贝、检测与目标叠加、MJPEG、状态和控制接口 |
| `src/utils/sys_utils.*` | 图像预处理、CPU 亲和性、调度与内存设置 |

应用从 V4L2 采集 HDMI 图像，经预处理送入 RKNN；后处理结果交给跟踪器选择目标，再由 HID 工作线程发送控制报表。Web 线程使用画面和检测结果生成预览，并向主循环提供控制设置。

## 配套文件

| 路径 | 内容 |
| --- | --- |
| `Makefile` | 编译、依赖检查、回归测试、清理与发布目标 |
| `src/Makefile` | 兼容原来的 `make -C src`，转发到根目录规则 |
| `setup.sh` | 性能设置、USB Gadget、EDID 加载与读回检查 |
| `tools/check_model.py` | 可选 Python 模型检查工具，所需依赖取决于模型格式 |
| `tools/gen_edid.py` | EDID 生成与结构、校验和检查 |
| `tools/aim_simulator.cpp` | 不依赖设备库的离线仿真命令行及 JSON／CSV 导出 |
| `tools/aim_simulator.py` / `tools/aim_simulator.html` | 本地仿真服务与可视化调参、回放界面 |
| `tests/test_postprocess.c` | 输出解析与 NMS 回归测试 |
| `tests/test_tracker.cpp` | 目标关联、模式切换、坐标还原与预测测试 |
| `tests/test_hid.cpp` | 模拟 HID 写入、拆分、重试、暂停及退出测试 |
| `tests/test_web.cpp` | 检测叠加、坐标、图像拷贝及帧所有权测试 |
| `tests/test_edid.py` | EDID 时序、布局和校验测试 |
| `tests/test_simulation.cpp` / `tests/test_simulator_api.py` | PD 与 HID 反馈、仿真指标、参数校验及 HTTP 导出测试 |
| `scripts/package.sh` | 生成含源码、程序和文档的发布包与 SHA-256 校验文件 |
| `docs/` | 项目结构、构建发布、运行和 HDMI 配置说明 |

## 本机数据和生成文件

| 路径 | 生成方式及 Git 状态 |
| --- | --- |
| `Models/*.rknn` | 自行提供的模型，Git 忽略 |
| `Models/1080p_multi_hz.edid` | EDID 工具或配置脚本生成，Git 忽略 |
| `build/obj/` | Make 生成的对象文件及 `.d` 头文件依赖，Git 忽略 |
| `build/test_*` | Make 生成的测试程序，Git 忽略 |
| `yolo_app_26` | Make 生成的应用程序，Git 忽略 |
| `dist/` | `make package` 生成的发布包及校验文件，Git 忽略 |

`Models/` 保存本机运行数据，与 `src/models/` 中的推理源码分开。发布包创建空的 `Models/` 目录；模型需在目标板卡上另行准备。构建命令与依赖要求见[开发文档](development.md)。
