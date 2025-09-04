import pyautogui
import time
import os

# 截图保存文件夹
save_dir = "screenshots"
os.makedirs(save_dir, exist_ok=True)

# 截图数量和间隔
num_screenshots = 100
interval = 0.1  # 秒

# 设置pyautogui的安全设置
pyautogui.FAILSAFE = True

print(f"Starting to take {num_screenshots} full screen screenshots...")
print("Move mouse to top-left corner to stop if needed")
time.sleep(5)

for i in range(num_screenshots):
    filename = os.path.join(save_dir, f"screenshot_{i+1:03d}.png")
    
    try:
        # 截取全屏截图
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        
        print(f"Saved {filename}")
        
        # 等待指定间隔
        time.sleep(interval)
        
    except Exception as e:
        print(f"Error taking screenshot {i+1}: {e}")
        break

print(f"Done! {num_screenshots} full screen screenshots saved to {save_dir} folder.")