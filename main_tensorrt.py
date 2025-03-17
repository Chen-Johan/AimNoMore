import torch
import numpy as np
import cv2
import time
import win32api
import win32con
from numba import prange, njit
import cupy as cp
from ultralytics import YOLO
import gameSelection
from config import (
    aaMovementAmp, aim_power, aim_distance,
    headshot_mode, headshot_offset, body_offset,
    useMask, maskHeight, maskWidth, aaQuitKey, aaQuitKeyCode,
    confidence, max_det, cpsDisplay, visuals,
    centerOfScreen, MODEL_SIZE,
    model_path, target_class,
    target_default_class, target_alternate_class,
    target_key_codes, mouse_aim_key,
    device_param, device_type, ONNXDevice
)

#------------------------ 优化函数 - 增强JIT编译优化 --------------------------
@njit(fastmath=True, cache=True)
def calculate_mouse_move_optimized(x_mid, y_mid, c_width, c_height, box_height, headshot_offset_val, body_offset_val, is_headshot, aim_power_val, aa_movement_amp):
    """优化的鼠标移动计算 - 参数直接传入以避免全局查找开销"""
    # 计算与目标的偏移
    dx = x_mid - c_width
    dy = (y_mid - (box_height * (headshot_offset_val if is_headshot else body_offset_val))) - c_height
    
    # 使用非线性映射计算移动距离 - 内联计算提高速度
    dx_abs = abs(dx)
    dy_abs = abs(dy)
    # 使用快速近似计算替代精确幂计算
    dx_powered = dx_abs * dx_abs if aim_power_val >= 0.9 else dx_abs ** aim_power_val
    dy_powered = dy_abs * dy_abs if aim_power_val >= 0.9 else dy_abs ** aim_power_val
    
    move_x = int((1 if dx > 0 else -1 if dx < 0 else 0) * dx_powered * aa_movement_amp)
    move_y = int((1 if dy > 0 else -1 if dy < 0 else 0) * dy_powered * aa_movement_amp)
    
    return move_x, move_y

@njit(fastmath=True, cache=True, parallel=True)
def fast_process_targets(targets, model_size, aim_dist_sq):
    """高度优化的目标处理函数 - 合并距离计算和目标筛选"""
    n = len(targets)
    center_x = model_size / 2
    center_y = model_size / 2
    
    # 预分配数组以避免内存分配开销
    screen_x = np.empty(n, dtype=np.float32)
    screen_y = np.empty(n, dtype=np.float32)
    distances_squared = np.empty(n, dtype=np.float32)
    
    # 并行计算，使用距离平方避免开平方运算
    for i in prange(n):
        # 直接计算屏幕坐标
        screen_x[i] = targets[i, 0] * model_size
        screen_y[i] = targets[i, 1] * model_size
        
        # 计算距离平方（避免开平方）
        dx = screen_x[i] - center_x
        dy = screen_y[i] - center_y
        distances_squared[i] = dx*dx + dy*dy
    
    # 找到距离最近的目标索引
    nearest_idx = np.argmin(distances_squared)
    
    # 检查是否在瞄准范围内（距离平方与阈值平方比较）
    is_in_range = distances_squared[nearest_idx] <= aim_dist_sq
    
    # 仅返回必要信息
    if is_in_range:
        return True, nearest_idx, screen_x[nearest_idx], screen_y[nearest_idx]
    else:
        return False, -1, 0.0, 0.0

#------------------------ 绘图函数 --------------------------
def draw_detection_results(frame, boxes, confs):
    """绘制检测结果：边界框和置信度，不显示标签"""
    img = frame.copy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        conf_text = f"{conf:.2f}"
        cv2.putText(img, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

#------------------------ 目标类别切换 --------------------------
class TargetClassSwitcher:
    """目标类别切换器 - 高效地管理目标类别切换"""
    def __init__(self, default_class=None):
        self.default_class = default_class if default_class is not None else target_default_class  # 从配置获取默认类别
        self.alternate_class = target_alternate_class  # 从配置获取替代类别
        self.current_class = self.default_class  # 当前使用的类别
        self.multi_class_mode = False  # 是否处于多类别模式
        
        # 按键状态跟踪（防止持续触发）
        self.f1_pressed = False
        self.f2_pressed = False
        
        # 从配置获取按键码
        self.multi_key = target_key_codes["multi"]  # 多目标模式键码
        self.switch_key = target_key_codes["switch"]  # 单目标切换键码
    
    def update(self):
        """更新目标类别，根据按键状态"""
        # 检查多目标模式键状态
        key_multi_current = bool(win32api.GetAsyncKeyState(self.multi_key) & 0x8000)
        if key_multi_current and not self.f1_pressed:
            self.multi_class_mode = True
            if cpsDisplay:
                print(f"切换到多目标模式: 同时检测类别 {self.default_class} 和 {self.alternate_class}")
        self.f1_pressed = key_multi_current
        
        # 检查单目标切换键状态
        key_switch_current = bool(win32api.GetAsyncKeyState(self.switch_key) & 0x8000)
        if key_switch_current and not self.f2_pressed:
            self.multi_class_mode = False  # 关闭多类别模式
            self.current_class = self.alternate_class if self.current_class == self.default_class else self.default_class
            if cpsDisplay:
                print(f"切换到单目标模式: 类别 {self.current_class}")
        self.f2_pressed = key_switch_current
        
        # 返回当前类别设置
        if self.multi_class_mode:
            return [self.default_class, self.alternate_class]  # 多类别模式返回类别列表
        else:
            return self.current_class  # 单类别模式返回单一类别

#------------------------ 主函数 --------------------------
def main():
    """主程序入口"""
    # 初始化
    camera, cWidth, cHeight = gameSelection.gameSelection()
    count, sTime = 0, time.time()
    model_size = MODEL_SIZE
    
    # 创建目标类别切换器 - 使用配置值
    target_switcher = TargetClassSwitcher()
    current_target = target_class  # 初始值从配置加载
    
    # 预计算一些常量以避免重复计算
    aim_distance_squared = aim_distance * aim_distance  # 预先计算距离平方阈值
    
    # 使用固定的CPU亲和性提高性能
    try:
        import psutil
        # 将进程限制在物理核心(如果有多于一个核心)上，避免在逻辑核心间切换
        p = psutil.Process()
        # 使用系统可用核心的一半来运行程序(通常这些是物理核心)
        available_cores = list(range(psutil.cpu_count(logical=False)))
        if available_cores:
            p.cpu_affinity(available_cores)
    except (ImportError, AttributeError):
        pass
    
    # 启用cuDNN自动调优并设置高性能模式
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    
    # 设置PyTorch性能选项
    torch.set_float32_matmul_precision('high')  # 使用高精度浮点运算
    
    # 预先分配缓冲区和变量，使用固定内存减少CPU-GPU传输时间
    frame_buffer = cp.empty((1, 3, model_size, model_size), dtype=cp.float16)
    targets_buffer = np.zeros((max_det, 4), dtype=np.float32)
    boxes_buffer = np.zeros((max_det, 4), dtype=np.float32)
    
    # 加载模型 - 使用配置中的模型路径
    model = YOLO(model_path, task='detect')
    model.imgsz = (model_size, model_size)
    
    # 根据设备类型进行相应配置
    if device_type == ONNXDevice.CPU:
        # CPU模式 - 使用cpu设备
        print("使用CPU模式运行ONNX模型")
        model.device = device_param  # 设置为'cpu'
    elif device_type == ONNXDevice.AMD:
        # AMD模式 - 使用0号GPU设备
        print("使用AMD/NVIDIA GPU模式运行ONNX模型")
        model.device = device_param  # 设置为'0'
    else:
        # NVIDIA模式 - 默认使用CUDA，无需特别设置
        print("使用NVIDIA GPU模式运行TensorRT引擎")
    
    # 主循环 - 使用最简单直接的方式，减少不必要的同步
    with torch.no_grad():
        # 使用配置中导出的键码，不再需要在这里计算
        while not win32api.GetAsyncKeyState(aaQuitKeyCode):
            loop_start = time.time()
            
            # 检查是否需要切换目标类别 - 高效实现
            current_target = target_switcher.update()
            
            # 1. 图像获取与预处理 - 直接处理，避免多余复制
            frame = camera.get_latest_frame()
            
            # 直接在CuPy上处理图像数据，减少CPU运算
            npImg = cp.asarray(frame, dtype=cp.float16)[:, :, :3]
            cp.multiply(npImg, cp.float16(1.0/255.0), out=npImg)
            frame_buffer[0] = cp.transpose(npImg, (2, 0, 1))
            im = torch.from_numpy(cp.asnumpy(frame_buffer)).cuda(non_blocking=True)
            
            # 2. 模型推理 - 使用当前选择的目标类别
            results = model.predict(
                source=im,
                imgsz=model_size,
                conf=confidence,
                iou=confidence,
                max_det=max_det,
                classes=current_target,  # 使用当前选择的目标类别
                half=True,
                verbose=False,
                stream=False,
                save=False,
                show=False,
            )
            result = results[0]
            boxes_count = len(result.boxes)
            
            # 3. 处理检测结果
            if boxes_count > 0 and centerOfScreen:
                # 处理边界框 - 直接使用预分配的缓冲区
                boxes = result.boxes.xyxy.cpu().numpy()
                np.copyto(boxes_buffer[:boxes_count], boxes)  # 使用copyto避免新分配
                
                # 批量计算目标数据 - 用向量化操作替代循环
                x_centers = (boxes_buffer[:boxes_count, 0] + boxes_buffer[:boxes_count, 2]) / 2 / model_size
                y_centers = (boxes_buffer[:boxes_count, 1] + boxes_buffer[:boxes_count, 3]) / 2 / model_size
                widths = (boxes_buffer[:boxes_count, 2] - boxes_buffer[:boxes_count, 0]) / model_size
                heights = (boxes_buffer[:boxes_count, 3] - boxes_buffer[:boxes_count, 1]) / model_size
                
                # 填充目标缓冲区
                curr_targets = targets_buffer[:boxes_count]
                curr_targets[:, 0] = x_centers
                curr_targets[:, 1] = y_centers
                curr_targets[:, 2] = widths
                curr_targets[:, 3] = heights
                
                # 4. 高效处理并找出最近目标
                in_range, nearest_idx, xMid, yMid = fast_process_targets(
                    curr_targets, model_size, aim_distance_squared
                )
                
                # 5. 如果找到目标且按下配置的鼠标键，执行瞄准
                if in_range and win32api.GetAsyncKeyState(mouse_aim_key) & 0x8000:
                    box_height = curr_targets[nearest_idx, 3] * model_size
                    
                    # 直接调用优化的鼠标移动函数，传递所有常量
                    move_x, move_y = calculate_mouse_move_optimized(
                        xMid, yMid, cWidth, cHeight, box_height,
                        headshot_offset, body_offset, headshot_mode,
                        aim_power, aaMovementAmp
                    )
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
            
            # 6. 可视化处理 - 只在必要时执行
            if visuals:
                if boxes_count > 0:
                    confs = result.boxes.conf.cpu().numpy()
                    annotated_frame = draw_detection_results(frame, boxes, confs)
                    cv2.imshow('Live Feed', annotated_frame)
                else:
                    cv2.imshow('Live Feed', frame)
                
                if cv2.waitKey(1) & 0xFF == ord(aaQuitKey):
                    break
            
            # 7. 计算和显示FPS
            count += 1
            if time.time() - sTime > 1:
                fps = count
                if cpsDisplay:
                    # 显示当前模式信息
                    if isinstance(current_target, list):
                        mode_info = f"多目标模式: 类别 {current_target}"
                    else:
                        mode_info = f"单目标模式: 类别 {current_target}"
                    print(f"CPS: {fps} | {mode_info}")
                count, sTime = 0, time.time()
    
    # 8. 清理资源
    camera.stop()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("发生错误:")
        traceback.print_exc()
        print(f"错误信息: {str(e)}")
        print("程序异常退出。如有问题请查看上方错误信息或联系开发者。")
        input("按回车键退出...")