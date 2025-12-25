#!/bin/bash

# 检查是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

gadget=g1

# --- Performance Settings ---
setup_performance() {
    echo "--- Setting Performance Mode ---"
    
    echo "Setting CPU to performance mode..."
    if [ -d /sys/devices/system/cpu/cpufreq/policy0 ]; then
        echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
    fi
    if [ -d /sys/devices/system/cpu/cpufreq/policy4 ]; then
        echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
    fi
    if [ -d /sys/devices/system/cpu/cpufreq/policy6 ]; then
        echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
    fi

    echo "Setting NPU to performance mode..."
    if [ -e /sys/class/devfreq/fdab0000.npu/governor ]; then
        echo performance > /sys/class/devfreq/fdab0000.npu/governor
    fi

    echo "Setting GPU to performance mode..."
    if [ -e /sys/class/devfreq/fb000000.gpu/governor ]; then
        echo performance > /sys/class/devfreq/fb000000.gpu/governor
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
    if [ -e /sys/class/devfreq/fb000000.gpu/cur_freq ]; then
        echo "GPU: $(cat /sys/class/devfreq/fb000000.gpu/cur_freq)"
    fi
    echo "--------------------------------"
}

# --- HID Gadget Settings ---
start_hid_gadget(){ 
    echo "--- Starting HID Gadget ---"
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

    # 当我们创建完这个文件夹之后，系统自动的在这个文件夹中创建usb相关的内容 ，这些内容需要由创建者自己填写
    if [[ ! -d ${gadget} ]]; then
		mkdir ${gadget}
	fi
    cd ${gadget}

    # 加载模块
    # 尝试进入当前内核版本的模块目录，如果失败则尝试原脚本的硬编码路径
    if [ -d "/lib/modules/$(uname -r)" ]; then
        cd "/lib/modules/$(uname -r)"
    elif [ -d "/lib/modules/6.1.118" ]; then
        cd /lib/modules/6.1.118
    fi
    
    depmod -a
    modprobe libcomposite
    modprobe usb_f_hid             #需要先将usb_f_hid.ko驱动拷贝过来
    
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

    #将gadget驱动注册到UDC上，插上USB线到电脑上，电脑就会枚举USB设备。
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
        start_hid_gadget
        ;;
    stop)
        stop_hid_gadget
        ;;
    *)
        echo "Usage: $0 (start | stop)"
        echo "  start: Set performance mode and start HID gadget"
        echo "  stop : Stop HID gadget"
        ;;
esac
