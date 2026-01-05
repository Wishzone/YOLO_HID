import cv2
import time
import numpy as np
import sys
import gc
import random
try:
    from pynput import mouse
except ImportError:
    print("请先安装 pynput 库: pip install pynput")
    sys.exit(1)

# --- 配置 ---
IMAGE_PATH = './Timingtest/test.png'
TIMEOUT = 1.0
MOVEMENT_THRESHOLD = 1
TEST_ROUNDS = 50  # 测试轮数
WINDOW_X = 0      # 窗口显示的起始X坐标 (用于多屏设置，例如副屏在右侧可设为1920)
WINDOW_Y = 0      # 窗口显示的起始Y坐标

# 全局变量
start_time = 0
end_time = 0
is_moved = False
start_mouse_pos = None
waiting_for_response = False

def on_move(x, y):
    global end_time, is_moved, start_time, start_mouse_pos, waiting_for_response
    
    # 如果不在等待响应状态，直接忽略
    if not waiting_for_response:
        return

    # 如果已经检测到移动，忽略后续事件
    if is_moved:
        return

    # 计算相对于起始位置的移动距离平方 (避免开方运算以提高速度)
    dist_sq = (x - start_mouse_pos[0])**2 + (y - start_mouse_pos[1])**2
    
    if dist_sq > MOVEMENT_THRESHOLD**2:
        end_time = time.perf_counter() # 使用高精度计时器
        is_moved = True
        waiting_for_response = False # 停止等待

def main():
    global start_time, is_moved, start_mouse_pos, waiting_for_response, end_time

    # 读取图片
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"警告: 无法找到 {IMAGE_PATH}，将使用生成图片代替。")
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, "TEST IMAGE", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # 准备窗口
    window_name = 'Latency Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, WINDOW_X, WINDOW_Y) # 移动窗口到指定屏幕位置
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- 预热阶段 ---
    # 首次调用 OpenCV 函数会有加载开销，先运行一次不计入统计
    print("正在预热...")
    black_img = np.zeros_like(img)
    black_img[:] = (128, 128, 128) # 改为灰色背景，防止过曝
    cv2.imshow(window_name, black_img)
    cv2.waitKey(1)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)
    
    print("=== 自动化延迟测试 (高精度版) ===")
    print(f"计划进行 {TEST_ROUNDS} 轮测试，请保持鼠标静止...")
    print("提示: 测试过程中按 ESC 或 q 键可随时停止测试")

    # 启动监听 (只启动一次，避免反复创建线程的开销)
    listener = mouse.Listener(on_move=on_move)
    listener.start()
    mouse_controller = mouse.Controller()
    
    latencies = []

    try:
        for i in range(TEST_ROUNDS):
            print(f"\n--- 第 {i+1}/{TEST_ROUNDS} 轮 ---")
            
            # 1. 重置状态
            is_moved = False
            waiting_for_response = False
            start_time = 0
            end_time = 0
            
            # 2. 显示黑屏准备
            cv2.imshow(window_name, black_img)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                raise KeyboardInterrupt
            
            # 3. 随机等待 (0.4 ~ 0.6秒)，防止预测
            wait_time = random.uniform(0.4, 0.6)
            start_wait = time.perf_counter()
            while time.perf_counter() - start_wait < wait_time:
                if cv2.waitKey(10) & 0xFF in [27, ord('q')]: # 保持窗口响应，但不要太频繁
                    raise KeyboardInterrupt
            
            # 4. 记录基准位置
            start_mouse_pos = mouse_controller.position
            
            # 5. 关键路径开始：禁用 GC 以减少抖动
            gc.disable()
            
            # 6. 显示图片并计时
            # 注意：imshow 后必须接 waitKey 才能刷新窗口，但 waitKey 本身有延迟
            # 我们尽量让计时紧贴着刷新操作
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: # 强制刷新事件循环
                raise KeyboardInterrupt
            start_time = time.perf_counter() # 立即计时
            
            waiting_for_response = True # 开启监听判定
            
            # 7. 等待响应循环
            while not is_moved:
                # 检查超时
                if time.perf_counter() - start_time > TIMEOUT:
                    break
                # 极短的 sleep 让出 CPU，避免死循环占用 100% CPU 导致系统卡顿
                # 但在极高精度要求下，可以去掉 sleep，但这通常会影响系统其他进程
                # cv2.waitKey(1) 既能处理窗口事件，又能充当 sleep
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    raise KeyboardInterrupt
            
            # 8. 恢复 GC
            gc.enable()
            
            if is_moved:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                print(f"延迟: {latency_ms:.2f} ms")
            else:
                print("超时")
                
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        listener.stop()
        cv2.destroyAllWindows()

    # --- 结果统计 ---
    if latencies:
        avg_latency = np.mean(latencies)
        std_dev = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print("\n" + "="*30)
        print("       测试结果统计       ")
        print("="*30)
        print(f"样本数量 : {len(latencies)}")
        print(f"平均延迟 : {avg_latency:.2f} ms")
        print(f"标准差   : {std_dev:.2f} ms (越小越稳定)")
        print(f"最小延迟 : {min_latency:.2f} ms")
        print(f"最大延迟 : {max_latency:.2f} ms")
        print("="*30)
    else:
        print("\n没有收集到有效的测试数据")

if __name__ == "__main__":
    main()
