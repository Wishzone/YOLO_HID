# YOLO HID for NanoPC-T6 / RK3588

基于 RKNN Runtime 的 YOLO26 检测工程，包含 HDMI/V4L2 采集、RGA 图像预处理、USB HID 输出和 HTTP/MJPEG 预览。

## 文档导航

- [项目结构与模块职责](docs/structure.md)
- [构建、测试与发布](docs/development.md)
- [模型验证、运行参数与目标跟踪](docs/runtime.md)
- [HDMI 输入刷新率与 EDID](docs/hdmi.md)

## 文件结构

```text
YOLO/
├── Makefile           统一构建、测试和发布入口
├── setup.sh           板卡性能、HDMI EDID 和 USB Gadget 配置
├── docs/              项目结构、开发与运行文档
├── src/
│   ├── Makefile       make -C src 的兼容入口
│   ├── main_yolo26.cpp 应用入口及检测、控制调度
│   ├── core/          公共状态和检测数据类型
│   ├── models/        RKNN 模型加载、推理与后处理
│   ├── modules/       视频采集、目标跟踪、HID 和 Web
│   └── utils/         图像预处理及系统辅助函数
├── tests/             C/C++ 与 Python 回归测试
├── tools/             模型检查和 EDID 生成工具
├── scripts/           发布包生成脚本
├── Models/            本机模型及 EDID，Git 忽略
├── build/             对象文件、依赖和测试程序，自动生成
├── dist/              发布包及 SHA-256 文件，自动生成
└── yolo_app_26        编译生成的应用程序，Git 忽略
```

## 快速开始

在 RK3588 AArch64 Linux 上准备 GCC/G++、Make、pkg-config、OpenCV 4，以及板卡配套的 RKNN Runtime、RGA 头文件和库。详见[依赖准备](docs/development.md)。

```bash
make check-deps
make -j4
make test PYTHON=/usr/bin/python3
./yolo_app_26 --help
./yolo_app_26 --check-model
```

默认模型为 `Models/cs2-26n_rk3588_int8.rknn`，模型文件需自行放入本机目录。本板 CS2 类别备注为 0、1 为警，2、3 为匪；其他模型按自己的训练标签配置。

连接 HDMI 输入及 USB OTG 后运行：

```bash
sudo bash ./setup.sh start
sudo ./yolo_app_26 --scores logits
```

浏览器访问 `http://<板卡IP>:8080/`。类别选择、HID 暂停、目标跟踪和模型格式说明见[运行文档](docs/runtime.md)；输入刷新率配置见 [HDMI 文档](docs/hdmi.md)。

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `make` / `make -j4` | 编译应用 |
| `make test` | 运行回归测试 |
| `make clean` | 清理编译产物 |
| `make rebuild` | 清理后重新编译 |
| `make package` | 生成发布包与校验文件 |
| `make -C src` | 兼容原构建入口 |

源码与文档纳入 Git；本机模型、EDID、编译产物和发布包由 `.gitignore` 排除。
