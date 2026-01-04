#!/bin/bash

# YOLO11n RK3588 Setup & Optimization Script
# Run as root

# 检查是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

gadget=g1

# --- 1. Performance Settings ---
setup_performance() {
    echo "--- [1/3] Setting Performance Mode ---"
    
    echo "Setting CPU to performance mode..."
    # 使用循环设置所有 CPU 核心
    for governor in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        if [ -e "$governor" ]; then
            echo performance > "$governor"
        fi
    done

    echo "Setting NPU to performance mode..."
    if [ -e /sys/class/devfreq/fdab0000.npu/governor ]; then
        echo performance > /sys/class/devfreq/fdab0000.npu/governor
    fi

    echo "Setting GPU to performance mode..."
    if [ -e /sys/class/devfreq/fb000000.gpu/governor ]; then
        echo performance > /sys/class/devfreq/fb000000.gpu/governor
    fi

    echo "Setting DMC (Memory) to performance mode..."
    if [ -e /sys/class/devfreq/dmc/governor ]; then
        echo performance > /sys/class/devfreq/dmc/governor
    fi

    echo "Current Frequencies:"
    if [ -e /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq ]; then
        echo "CPU0: $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)"
    fi
    if [ -e /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq ]; then
        echo "CPU4: $(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq)"
    fi
    if [ -e /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq ]; then
        echo "CPU6: $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)"
    fi
    if [ -e /sys/class/devfreq/fdab0000.npu/cur_freq ]; then
        echo "NPU: $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
    fi
    echo "--------------------------------"
}

# --- 2. Latency Optimization (IRQ, Network, V4L2) ---
setup_latency() {
    echo "--- [2/3] Optimizing System Latency ---"

    # 2.1 IRQ Affinity (Bind USB/HDMI interrupts to Big Cores 4-7)
    echo "Optimizing IRQ Affinity..."
    # Find USB controller IRQs (dwc3 is common for USB3 on RK3588)
    for irq in $(grep "dwc3" /proc/interrupts | awk -F: '{print $1}'); do
        # f0 = 11110000 (Cores 4-7)
        if [ -d "/proc/irq/$irq" ]; then
            echo f0 > /proc/irq/$irq/smp_affinity
        fi
    done
    # Find HDMI RX IRQs (hdmi-in or similar)
    for irq in $(grep "hdmi" /proc/interrupts | awk -F: '{print $1}'); do
        if [ -d "/proc/irq/$irq" ]; then
            echo f0 > /proc/irq/$irq/smp_affinity
        fi
    done

    # 2.2 Network Latency (Skipped)
    # echo "Tuning Network Stack..."
    # sysctl -w net.core.rmem_max=16777216 > /dev/null
    # sysctl -w net.core.wmem_max=16777216 > /dev/null
    # sysctl -w net.ipv4.tcp_low_latency=1 > /dev/null

    # 2.3 V4L2 Settings (HDMI IN)
    VIDEO_DEV=/dev/video20
    if [ -e $VIDEO_DEV ]; then
        echo "Tuning V4L2 Device ($VIDEO_DEV)..."
        # Disable auto-exposure/focus if they exist
        v4l2-ctl -d $VIDEO_DEV -c exposure_auto=1 2>/dev/null
        v4l2-ctl -d $VIDEO_DEV -c focus_auto=0 2>/dev/null
        
        # Load Custom EDID for 1080p 165Hz
        EDID_FILE="$(dirname $0)/Models/1080p_165hz.edid"
        if [ -f "$EDID_FILE" ]; then
            echo "Loading Custom EDID from $EDID_FILE..."
            v4l2-ctl -d $VIDEO_DEV --set-edid=file="$EDID_FILE",format=raw --fix-edid-checksums
        fi
        
        # Note: HDMI Input framerate is determined by source, skipping set-parm
    else
        echo "$VIDEO_DEV not found, skipping V4L2 tuning."
    fi

    # 2.4 Kernel Parameters (Scheduler & VM)
    echo "Tuning Kernel Parameters..."
    # 减少 Swap 使用，防止内存换页造成的卡顿
    sysctl -w vm.swappiness=1 > /dev/null
    # 调整脏页回写，避免 I/O 突发阻塞
    sysctl -w vm.dirty_background_ratio=5 > /dev/null
    sysctl -w vm.dirty_ratio=10 > /dev/null
    # 禁用透明大页 (THP) 以减少内存分配延迟抖动
    if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
        echo never > /sys/kernel/mm/transparent_hugepage/enabled
    fi

    # --- Advanced Video Latency Tuning (User Requested) ---
    echo "Applying Advanced Video Latency Settings..."
    
    # 1. Videobuf2 Core Parameters
    # 尝试减少最小缓冲区数量 (如果模块参数暴露)
    if [ -e /sys/module/videobuf2_core/parameters/min_buffers_needed ]; then
        echo 2 > /sys/module/videobuf2_core/parameters/min_buffers_needed
        echo "Set videobuf2 min_buffers_needed to 2"
    fi

    # 2. Frame Skip (Vendor Specific)
    if [ -e /proc/sys/video/frame_skip ]; then
        echo 0 > /proc/sys/video/frame_skip
        echo "Disabled frame skipping"
    fi

    echo "--------------------------------"
}

# --- 3. HID Gadget Settings ---
start_hid_gadget(){ 
    echo "--- [3/3] Starting HID Gadget ---"
    has_mount=$(mount -l | grep /sys/kernel/config)
    if [[ -z  $has_mount ]];then
        mount -t configfs none /sys/kernel/config
    fi
    
    # 确保 usb_gadget 目录存在
    if [[ ! -d /sys/kernel/config/usb_gadget ]]; then
        echo "Error: /sys/kernel/config/usb_gadget does not exist. ConfigFS might not be enabled."
        return 1
    fi

    cd /sys/kernel/config/usb_gadget

    if [[ ! -d ${gadget} ]]; then
mkdir ${gadget}
fi
    cd ${gadget}

    # 加载模块
    if [ -d "/lib/modules/$(uname -r)" ]; then
        cd "/lib/modules/$(uname -r)"
    elif [ -d "/lib/modules/6.1.118" ]; then
        cd /lib/modules/6.1.118
    fi
    
    depmod -a
    modprobe libcomposite
    modprobe usb_f_hid
    
    cd /sys/kernel/config/usb_gadget/${gadget}

    #设置USB协议版本USB2.0
    echo 0x0200 > bcdUSB

    #定义产品的VendorID和ProductID
    echo "0x0525"  > idVendor
    echo "0xa4ac" > idProduct

    #实例化"英语"ID：
    if [[ ! -d strings/0x409 ]]; then
        mkdir strings/0x409
    fi

    #将开发商、产品和序列号字符串写入内核
    echo "76543210" > strings/0x409/serialnumber
    echo "mkelehk"  > strings/0x409/manufacturer
    echo "keyboard_mouse"  > strings/0x409/product

    #创建一个USB配置实例
    if [[ ! -d configs/c.1 ]]; then
        mkdir configs/c.1
    fi

    #定义配置描述符使用的字符串
if [[ ! -d configs/c.1/strings/0x409 ]]; then
mkdir configs/c.1/strings/0x409
fi   

    echo "hid" > configs/c.1/strings/0x409/configuration

    #创建两个接口
    if [[ ! -d functions/hid.0 ]]; then
        mkdir functions/hid.0   #键盘
    fi
    if [[ ! -d functions/hid.1 ]]; then
        mkdir functions/hid.1   #鼠标
    fi

    #接口0，模拟键盘  
    echo 1 > functions/hid.0/subclass   #启动设备符
    echo 1 > functions/hid.0/protocol   #键盘协议
    echo 8 > functions/hid.0/report_length  #标识该hid设备每次发送的报表长度为8字节
    echo -ne \\x05\\x01\\x09\\x06\\xa1\\x01\\x05\\x07\\x19\\xe0\\x29\\xe7\\x15\\x00\\x25\\x01\\x75\\x01\\x95\\x08\\x81\\x02\\x95\\x01\\x75\\x08\\x81\\x03\\x95\\x05\\x75\\x01\\x05\\x08\\x19\\x01\\x29\\x05\\x91\\x02\\x95\\x01\\x75\\x03\\x91\\x03\\x95\\x06\\x75\\x08\\x15\\x00\\x25\\x65\\x05\\x07\\x19\\x00\\x29\\x65\\x81\\x00\\xc0 > functions/hid.0/report_desc  #配置hid描述符

#接口1，模拟键盘鼠标
    echo 1 > functions/hid.1/subclass   #启动设备符
    echo 2 > functions/hid.1/protocol   #鼠标协议
    echo 4 > functions/hid.1/report_length  # 相对值是4
    echo -ne \\x05\\x01\\x09\\x02\\xa1\\x01\\x09\\x01\\xa1\\x00\\x05\\x09\\x19\\x01\\x29\\x05\\x15\\x00\\x25\\x01\\x95\\x05\\x75\\x01\\x81\\x02\\x95\\x01\\x75\\x03\\x81\\x01\\x05\\x01\\x09\\x30\\x09\\x31\\x09\\x38\\x15\\x81\\x25\\x7F\\x75\\x08\\x95\\x03\\x81\\x06\\xc0\\xc0 > functions/hid.1/report_desc

    #捆绑接口到配置config.1
    ln -sf functions/hid.0 configs/c.1
    ln -sf functions/hid.1 configs/c.1

#配置USB3.0 OTG0的工作模式为Device（设备）：
    if [ -e /sys/kernel/debug/usb/fc000000.usb/mode ]; then
    echo device > /sys/kernel/debug/usb/fc000000.usb/mode
    fi

    echo "sleep 3s"
    sleep 3s

    #将gadget驱动注册到UDC上
    echo fc000000.usb > UDC
    echo "---------------------------"
}

stop_hid_gadget() {
    echo "--- Stopping HID Gadget ---"
    cd /sys/kernel/config/usb_gadget/${gadget}
    echo "" > UDC
    rmmod usb_f_hid
    rmmod libcomposite
    echo "---------------------------"
}

# --- Main Execution ---

case $1 in
    start)
        setup_performance
        setup_latency
        start_hid_gadget
        echo "=== Setup Complete ==="
        ;;
    stop)
        stop_hid_gadget
        ;;
    *)
        echo "Usage: $0 (start | stop)"
        echo "  start: Set performance mode, optimize latency, and start HID gadget"
        echo "  stop : Stop HID gadget"
        ;;
esac
