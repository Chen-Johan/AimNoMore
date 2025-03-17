import pygetwindow
import time
import bettercam
from typing import Union

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import screenShotHeight, screenShotWidth, target_fps

def gameSelection() -> (bettercam.BetterCam, int, Union[int, None]):
    # 缓存窗口列表
    videoGameWindows = pygetwindow.getAllWindows()
    valid_windows = [(i, w) for i, w in enumerate(videoGameWindows) if w.title.strip()]
    
    if not valid_windows:
        print("未找到有效窗口")
        return None
        
    print("=== 可用窗口 ===")
    for index, window in valid_windows:
        print(f"[{index}]: {window.title}")
        
    # 使用异常处理优化输入逻辑
    try:
        userInput = int(input("请输入要选择的窗口编号: "))
        videoGameWindow = videoGameWindows[userInput]
    except (ValueError, IndexError):
        print("输入无效，请重试")
        return None

    # Activate that Window
    activationRetries = 30
    activationSuccess = False
    while (activationRetries > 0):
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as e:
            print("Failed to activate game window: {}".format(str(e)))
            print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            activationRetries = 0
            break
        # wait a little bit before the next try
        time.sleep(3.0)
        activationRetries = activationRetries - 1
    # if we failed to activate the window then we'll be unable to send input to it
    # so just exit the script now
    if activationSuccess == False:
        return None
    print("Successfully activated the game window...")

    # Starting screenshoting engine
    left = ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2)
    top = videoGameWindow.top + \
        (videoGameWindow.height - screenShotHeight) // 2
    right, bottom = left + screenShotWidth, top + screenShotHeight

    region: tuple = (left, top, right, bottom)

    # Calculating the center Autoaim box
    cWidth: int = screenShotWidth // 2
    cHeight: int = screenShotHeight // 2

    print(region)

    # 优化相机初始化
    camera_config = {
        "region": region,
        "output_color": "BGR", # 原本BGRA
        "max_buffer_len": 512
    }
    
    camera = bettercam.create(**camera_config)
    if not camera:
        print("Better Camera 启动失败")
        return None
        
    # 使用配置中的 target_fps
    camera.start(target_fps=target_fps, video_mode=True)

    return camera, cWidth, cHeight