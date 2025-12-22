#!/bin/bash

gadget=g1

do_start(){ 
    has_mount=$(mount -l | grep /sys/kernel/config)
    if [[ -z  $has_mount ]];then
        mount -t configfs none /sys/kernel/config
    fi
    cd /sys/kernel/config/usb_gadget

    # 当我们创建完这个文件夹之后，系统自动的在这个文件夹中创建usb相关的内容 ，这些内容需要由创建者自己填写
    if [[ ! -d ${gadget} ]]; then
		mkdir ${gadget}
	fi
    cd ${gadget}

    cd /lib/modules/6.1.118
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
    mkdir strings/0x409

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
    mkdir functions/hid.0   #键盘
    mkdir functions/hid.1   #鼠标

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
    ln -s functions/hid.0 configs/c.1
    ln -s functions/hid.1 configs/c.1
	
	#配置USB3.0 OTG0的工作模式为Device（设备）：
	echo device > /sys/kernel/debug/usb/fc000000.usb/mode

    echo "sleep 3s"
    sleep 3s

    #将gadget驱动注册到UDC上，插上USB线到电脑上，电脑就会枚举USB设备。
    echo fc000000.usb > UDC
}

do_stop() {
    cd /sys/kernel/config/usb_gadget/${gadaget}
    echo "" > UDC
    rmmod usb_f_hid.ko
    rmmod libcomposite.ko
}

case $1 in
    start)
        echo "Start hid gadget "
        do_start 
        ;;
    stop)
        echo "Stop hid gadget"
        do_stop
        ;;
    *)
        echo "Usage: $0 (stop | start)"
        ;;
esac
