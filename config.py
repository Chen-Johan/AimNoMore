"""
配置文件系统
这个模块负责管理和验证所有的配置参数，包括：
- 屏幕捕获设置
- 瞄准参数与目标检测配置
- 按键与控制设置
- 显示与界面选项
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os

#------------------------ 枚举类和键码映射 --------------------------
class MaskSide(Enum):
    LEFT = "left"    # 左侧遮罩
    RIGHT = "right"  # 右侧遮罩

class ONNXDevice(Enum):
    CPU = 1      # CPU处理器，使用 'cpu' 作为参数，文件后缀 .onnx
    AMD = 2      # AMD显卡，使用 device '0' 作为参数，文件后缀 .onnx  
    NVIDIA = 3   # NVIDIA显卡，使用 device '0' 作为参数，使用 TensorRT 引擎，文件后缀 .engine

# 鼠标按键常量定义
MOUSE_BUTTONS = {
    "LEFT": 0x01,      # 鼠标左键
    "RIGHT": 0x02,     # 鼠标右键
    "MIDDLE": 0x04,    # 鼠标中键
    "SIDE_1": 0x05,    # 鼠标侧键1
    "SIDE_2": 0x06     # 鼠标侧键2
}

# 键盘按键虚拟键码映射
KEY_CODES = {
    # 功能键
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73,
    "F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77,
    "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    # 修饰键
    "SHIFT": 0x10, "CTRL": 0x11, "ALT": 0x12,
    # 其他常用键
    "ESC": 0x1B, "TAB": 0x09, "CAPS": 0x14,
    "SPACE": 0x20, "ENTER": 0x0D, "BACKSPACE": 0x08
}

def get_key_code(key_name):
    """获取按键的虚拟键码"""
    if key_name in KEY_CODES:
        return KEY_CODES[key_name]
    elif key_name in MOUSE_BUTTONS:
        return MOUSE_BUTTONS[key_name]
    elif len(key_name) == 1:
        return ord(key_name)
    return KEY_CODES.get(key_name.upper(), ord('P'))

#------------------------ 配置类定义 --------------------------
@dataclass
class ScreenConfig:
    """屏幕捕获配置"""
    height: int = 320        # 捕获区域的高度
    width: int = 320         # 捕获区域的宽度
    target_fps: int = 100    # 目标帧率
    
    def validate(self):
        assert self.height > 0, "屏幕高度必须大于0"
        assert self.width > 0, "屏幕宽度必须大于0"
        assert self.target_fps > 0, "目标FPS必须大于0"

@dataclass
class MaskConfig:
    """屏幕遮罩配置"""
    enabled: bool = False            # 是否启用遮罩
    side: MaskSide = MaskSide.LEFT   # 遮罩位置：左侧或右侧
    width: int = 200                 # 遮罩宽度
    height: int = 200                # 遮罩高度
    
    def validate(self):
        if self.enabled:
            assert self.width > 0, "遮罩宽度必须大于0"
            assert self.height > 0, "遮罩高度必须大于0"
    
    def to_dict(self):
        """自定义转换为字典方法，处理枚举值"""
        result = asdict(self)
        result['side'] = self.side.value  # 将枚举转换为值
        return result

@dataclass
class AimConfig:
    """瞄准参数配置"""
    movement_amp: float = 1.1    # 移动倍率：值越小移动越平滑，值越大移动越快
    aim_power: float = 0.72      # 非线性映射指数(0.7-0.9)，影响瞄准曲线
    aim_distance: int = 200      # 开始瞄准的最大距离
    
    def validate(self):
        assert 0.7 <= self.aim_power <= 0.9, "瞄准曲线指数必须在0.7-0.9之间"
        assert self.aim_distance > 0, "瞄准距离必须大于0"

@dataclass
class DetectionConfig:
    """目标检测配置"""
    confidence: float = 0.5    # 置信度阈值：低于此值的检测结果将被忽略
    max_det: int = 3           # 最大检测目标数
    
    def validate(self):
        assert 0 <= self.confidence <= 1, "置信度必须在0-1之间"
        assert self.max_det > 0, "最大检测目标数必须大于0"

@dataclass
class AimModeConfig:
    """瞄准模式配置"""
    headshot_mode: bool = True     # True为瞄准头部，False为瞄准身体
    headshot_offset: float = 0.36  # 爆头模式下的垂直偏移
    body_offset: float = 0.2       # 身体模式下的垂直偏移
    
    def validate(self):
        assert 0 <= self.headshot_offset <= 1, "爆头偏移必须在0-1之间"
        assert 0 <= self.body_offset <= 1, "身体偏移必须在0-1之间"

@dataclass
class DisplayConfig:
    """显示配置"""
    cps_display: bool = True         # 是否在终端显示每秒校正次数
    visuals: bool = False            # 是否显示视觉效果（边界框等）
    center_of_screen: bool = True    # 是否优先选择屏幕中心的目标

@dataclass
class ModelConfig:
    """模型配置"""
    # 基本模型路径（不含扩展名）
    base_path: str = "models/CS2/prune_25/yolo11n320CS2_prune_25"  # 模型基础路径，不含扩展名
    
    def get_model_path(self, device_type: ONNXDevice) -> str:
        """根据设备类型获取正确的模型路径"""
        if device_type == ONNXDevice.NVIDIA:
            return f"{self.base_path}.engine"  # NVIDIA 使用 TensorRT 引擎
        else:
            return f"{self.base_path}.onnx"    # CPU 和 AMD 使用 ONNX 格式
    
    def get_device_param(self, device_type: ONNXDevice) -> str:
        """根据设备类型获取设备参数"""
        if device_type == ONNXDevice.CPU:
            return "cpu"   # CPU 使用 'cpu'
        elif device_type == ONNXDevice.AMD:
            return "0"     # AMD 使用 '0'
        else:
            return None    # NVIDIA TensorRT 不需要单独的设备参数
    
    def validate(self):
        # 检查基本路径的有效性
        base_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), self.base_path))
        assert os.path.exists(base_dir), f"模型目录不存在: {base_dir}"
        
        # 注意：这里不检查具体文件，因为扩展名会根据设备类型变化

@dataclass
class ControlConfig:
    """控制和按键配置"""
    # 目标类别设置
    default_class: int = 0        # 默认目标类别 (CT阵营)
    alternate_class: int = 2      # 替代目标类别 (T阵营)
    
    # 按键设置
    quit_key: str = "F11"         # 退出程序的按键
    multi_key: str = "F1"         # 多目标模式切换键 (同时检测两个类别)
    switch_key: str = "F2"        # 单目标切换键 (在两个单一类别间切换)
    
    # 鼠标设置 (可选值: LEFT, RIGHT, MIDDLE, SIDE_1, SIDE_2)
    aim_button: str = "SIDE_1"    # 瞄准按键，默认为鼠标侧键1
    
    def get_key_codes(self):
        """获取按键的虚拟键码"""
        return {
            "quit": get_key_code(self.quit_key),
            "multi": get_key_code(self.multi_key),
            "switch": get_key_code(self.switch_key),
            "aim": get_key_code(self.aim_button)
        }

#------------------------ 主配置类 --------------------------
@dataclass
class ConfigManager:
    """主配置类，管理所有配置并负责加载/保存"""
    # 核心配置组
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    aim_mode: AimModeConfig = field(default_factory=AimModeConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    onnx_device: ONNXDevice = ONNXDevice.NVIDIA
    
    def __post_init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.load_config()
        self.validate()
        self.save_config()
    
    def validate(self):
        """验证所有配置项"""
        self.screen.validate()
        self.mask.validate()
        self.aim.validate()
        self.detection.validate()
        self.aim_mode.validate()
        self.model.validate()
    
    def to_dict(self):
        """将配置转换为可序列化的字典"""
        return {
            # 屏幕和捕获
            "screen": asdict(self.screen),
            "mask": self.mask.to_dict(),
            
            # 瞄准和检测设置
            "aim": asdict(self.aim),
            "detection": asdict(self.detection),
            "aim_mode": asdict(self.aim_mode),
            
            # 模型和设备
            "model": asdict(self.model),
            "onnx_device": self.onnx_device.value,
            
            # 控制与显示
            "control": asdict(self.control),
            "display": asdict(self.display)
        }
    
    def save_config(self, filepath=None):
        """保存配置到文件"""
        if filepath is None:
            filepath = self.config_path
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
            print(f"配置已保存到: {filepath}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def load_config(self, filepath=None):
        """从文件加载配置"""
        if filepath is None:
            filepath = self.config_path
            
        if not os.path.exists(filepath):
            print(f"配置文件不存在: {filepath}，使用默认配置")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 简化的配置加载逻辑，移除向后兼容处理
            self._update_from_dict(config_dict)
            print(f"配置已加载: {filepath}")
            
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
    
    def _update_from_dict(self, config_dict):
        """从字典更新配置"""
        # 屏幕配置
        if "screen" in config_dict:
            self.screen = ScreenConfig(**config_dict["screen"])
        
        # 遮罩配置 
        if "mask" in config_dict:
            mask_dict = config_dict["mask"].copy()
            if "side" in mask_dict:
                mask_dict["side"] = MaskSide(mask_dict["side"])
            self.mask = MaskConfig(**mask_dict)
        
        # 瞄准和检测配置
        if "aim" in config_dict:
            self.aim = AimConfig(**config_dict["aim"])
            
        if "detection" in config_dict:
            self.detection = DetectionConfig(**config_dict["detection"])
            
        if "aim_mode" in config_dict:
            self.aim_mode = AimModeConfig(**config_dict["aim_mode"])
            
        if "display" in config_dict:
            self.display = DisplayConfig(**config_dict["display"])
            
        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        
        # 控制配置
        if "control" in config_dict:
            self.control = ControlConfig(**config_dict["control"])
            
        # 设备配置
        if "onnx_device" in config_dict:
            self.onnx_device = ONNXDevice(config_dict["onnx_device"])

#------------------------ 全局实例和变量导出 --------------------------
# 创建单一配置实例
config = ConfigManager()

# 从配置导出所有变量，简化且更易于维护
# 屏幕和捕获
screenShotHeight = config.screen.height
screenShotWidth = config.screen.width
MODEL_SIZE = config.screen.width  # 使用屏幕宽度作为模型尺寸
target_fps = config.screen.target_fps

# 遮罩设置
useMask = config.mask.enabled
maskSide = config.mask.side.value
maskWidth = config.mask.width
maskHeight = config.mask.height

# 瞄准参数
aaMovementAmp = config.aim.movement_amp
aim_power = config.aim.aim_power
aim_distance = config.aim.aim_distance

# 检测参数
confidence = config.detection.confidence
max_det = config.detection.max_det

# 瞄准模式
headshot_mode = config.aim_mode.headshot_mode
headshot_offset = config.aim_mode.headshot_offset
body_offset = config.aim_mode.body_offset

# 显示设置
cpsDisplay = config.display.cps_display
visuals = config.display.visuals
centerOfScreen = config.display.center_of_screen

# 模型设置 - 根据设备类型获取正确路径和参数
model_base_path = config.model.base_path
device_type = config.onnx_device
model_path = config.model.get_model_path(device_type)
device_param = config.model.get_device_param(device_type)

# 控制设置
target_class = config.control.default_class  # 默认类别
target_default_class = config.control.default_class 
target_alternate_class = config.control.alternate_class
aaQuitKey = config.control.quit_key

# 键码映射
key_codes = config.control.get_key_codes()
aaQuitKeyCode = key_codes["quit"]
target_key_codes = {
    "multi": key_codes["multi"],
    "switch": key_codes["switch"]
}
mouse_aim_key = key_codes["aim"]

