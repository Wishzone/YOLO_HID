# 模型验证与运行

以下命令均在项目根目录执行。首次部署请先完成[构建与依赖准备](development.md)。

## 模型与控制

本板 CS2 模型的类别备注：0、1 为警，2、3 为匪。其他模型以各自的训练标签为准。

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
