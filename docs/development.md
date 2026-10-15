# 构建、测试与发布

所有命令均在项目根目录执行。构建规则统一位于根目录 `Makefile`，`src/Makefile` 保留为兼容入口。

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

## 测试

`make test` 运行后处理、HID、Web 预览、目标跟踪、闭环仿真五组 C/C++ 测试，以及 EDID 和仿真接口的 Python 测试。HID 使用模拟写入，Web 使用测试图像，仿真接口使用本地临时端口；不会启动采集或发送实际鼠标报表。测试可执行文件位于 `build/`。

如果当前 Python 来自 Anaconda，可以显式使用系统 Python：

```bash
make test PYTHON=/usr/bin/python3
```

`make clean` 清理对象文件、依赖文件、测试程序和应用程序；本机模型与 `dist/` 发布包保留。

离线仿真只需 C++11 编译器、Make 和 Python 3 标准库：`make simulate` 启动调参界面，`make build/aim_simulator` 仅编译命令行工具，不需要 OpenCV、RKNN 或 RGA。参数与标定步骤见[仿真文档](simulation.md)。

## 发布包

```bash
make package
cd dist
sha256sum -c yolo-hid-rk3588-*.tar.gz.sha256
```

发布包包含 AArch64 可执行文件、源码、构建和运行脚本，不包含本机模型、RKNN/RGA/OpenCV 动态库。新板卡需要准备相同的运行库，并将自己的模型放入 `Models/`。包内的 `BUILD_COMMIT` 标明构建所用提交；如果工作区有未提交修改，则标注 `-dirty`。

发布包同时包含 `docs/` 使用文档；解压后仍可从根目录构建，或通过 `make -C src` 使用兼容入口。
