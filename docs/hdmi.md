# HDMI 输入与 EDID

以下命令均在项目根目录执行。EDID 生成工具位于 `tools/gen_edid.py`，本机生成文件位于 `Models/`。

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
