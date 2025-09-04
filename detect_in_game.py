import pygame
import random
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# ---- External capture mode: capture game.py window by title and auto-click apples ----
EXTERNAL_MODE = True  # set True to enable external window capture and auto-click

if EXTERNAL_MODE:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import mss
    import pyautogui
    try:
        import pygetwindow as gw
    except Exception:
        gw = None

    WINDOW_TITLE = "YOLO Auto Click Game"
    MODEL_PATH = 'runs/train/yolo_game_model24/weights/best.pt'
    IMG_SIZE = 1440
    CONF_THRES = 0.25
    AUTO_CLICK_MIN_CONF = 0.35
    AUTO_CLICK_COOLDOWN = 0.2

    def find_window_rect(title: str):
        if gw is None:
            raise RuntimeError("pygetwindow not installed: pip install pygetwindow")
        wins = gw.getWindowsWithTitle(title)
        if not wins:
            return None
        win = wins[0]
        if win.isMinimized:
            try:
                win.restore()
            except Exception:
                pass
        try:
            win.activate()
        except Exception:
            pass
        # left, top, width, height
        return win.left, win.top, win.width, win.height

    def main_ext():
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found: {MODEL_PATH}")
            sys.exit(1)

        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            model.to('cuda')
            try:
                model.model.half()
            except Exception:
                pass
            print("Detect device:", next(model.model.parameters()).device)
        else:
            print("CUDA not available, using CPU")

        torch.backends.cudnn.benchmark = True

        last_click_ts = 0.0
        with mss.mss() as sct:
            while True:
                rect = find_window_rect(WINDOW_TITLE)
                if rect is None:
                    print(f"Window not found: {WINDOW_TITLE}")
                    time.sleep(0.5)
                    continue
                left, top, width, height = rect
                mon = {"left": left, "top": top, "width": width, "height": height}
                shot = sct.grab(mon)
                frame = np.array(shot)  # BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Optional upscale to training height
                if height > 0 and IMG_SIZE and IMG_SIZE != height:
                    scale = IMG_SIZE / float(height)
                    frame = cv2.resize(frame, (int(width * scale), IMG_SIZE), interpolation=cv2.INTER_LINEAR)

                with torch.inference_mode():
                    res = model(frame, conf=CONF_THRES, imgsz=IMG_SIZE, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)

                # Parse detections
                apples = []
                for r in res:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    for b in r.boxes:
                        conf = float(b.conf[0].detach().cpu().numpy())
                        cls = int(b.cls[0].detach().cpu().numpy())
                        name = model.names.get(cls, str(cls))
                        if name != 'apple' or conf < AUTO_CLICK_MIN_CONF:
                            continue
                        x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
                        apples.append((conf, (x1, y1, x2, y2)))

                # Auto click highest-confidence apple
                now_t = time.time()
                if apples and now_t - last_click_ts >= AUTO_CLICK_COOLDOWN:
                    conf, (x1, y1, x2, y2) = max(apples, key=lambda t: t[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    # If we upscaled, cx,cy are in resized coords; map back to screen
                    if height > 0 and IMG_SIZE and IMG_SIZE != height:
                        inv_scale = height / float(IMG_SIZE)
                        cx = int(cx * inv_scale)
                        cy = int(cy * inv_scale)
                    screen_x = left + cx
                    screen_y = top + cy
                    pyautogui.click(x=screen_x, y=screen_y)
                    last_click_ts = now_t

                # Aim ~60Hz loop
                time.sleep(1/60)

    if __name__ == '__main__':
        try:
            main_ext()
        except KeyboardInterrupt:
            pass
        sys.exit(0)

# 初始化
pygame.init()

# 窗口大小
WIDTH, HEIGHT = 2560, 1440
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("YOLO Auto Click Game with AI Detection")

# 加载背景
background = pygame.image.load("background.png")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# 加载图片
images = {
    "apple": pygame.image.load("apple.png"),
    "banana": pygame.image.load("banana.png"),
    "cherry": pygame.image.load("cherry.png")
}

# 缩小图片大小
for k in images:
    images[k] = pygame.transform.scale(images[k], (80, 80))

# 修复模型路径 - 使用正确的路径
model_path = 'annotated_ds/yolo_game_model.pt'
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Available model files:")
    if os.path.exists('runs/train'):
        for root, dirs, files in os.walk('runs/train'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
    sys.exit(1)

# 加载训练好的YOLO模型
print("Loading YOLO model...")
model = YOLO(model_path)
print("Model loaded successfully!")

# 游戏参数
target = "apple"
score = 0
clock = pygame.time.Clock()
current_image = None
rect = None
detection_results = []
last_detection_time = 0
detection_interval = 0.25  # 增加到1秒检测一次，减少性能压力

# 自动点击配置
AUTO_CLICK_ENABLED = True
AUTO_CLICK_MIN_CONF = 0.35
AUTO_CLICK_COOLDOWN = 0.2  # seconds
last_auto_click_time = 0.0

# 刷新计时器
refresh_time = 500  # 毫秒
last_refresh = pygame.time.get_ticks()

force_detect = False

def capture_screen():
    """捕获游戏窗口的截图 - 修复版本"""
    try:
        # 获取游戏窗口的截图
        screen_surface = pygame.display.get_surface()
        
        # 使用更可靠的方法
        width, height = screen_surface.get_size()
        screen_array = pygame.surfarray.pixels3d(screen_surface)
        screen_array = np.array(screen_array)
        screen_array = screen_array.transpose([1, 0, 2])
        
        # 转换为OpenCV格式
        screen_cv = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
        
        # 保存调试截图
        cv2.imwrite('debug_screen.jpg', screen_cv)
        print(f"Screenshot saved: {screen_cv.shape}")
        
        return screen_cv
    except Exception as e:
        print(f"Screenshot error: {e}")
        return None

def detect_objects(screen_cv):
    """使用YOLO模型检测物体 - 改进版本"""
    if screen_cv is None:
        return []
    
    try:
        print("Running detection...")
        # 降低置信度阈值，增加检测灵敏度
        results = model(screen_cv, conf=0.05, imgsz=1440,verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"Found {len(boxes)} detections")
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # 获取置信度和类别
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    print(f"Detected {class_name} at ({x1:.1f}, {y1:.1f}) with confidence {conf:.2f}")
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            else:
                print("No detections found")
        
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []

def draw_detections(surface, detections):
    """在游戏界面上绘制检测结果 - 改进版本"""
    font = pygame.font.SysFont(None, 24)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        conf = det['confidence']
        
        # 根据类别选择不同颜色
        if class_name == 'apple':
            color = (0, 255, 0)      # 绿色
        elif class_name == 'banana':
            color = (0, 255, 255)    # 黄色
        elif class_name == 'cherry':
            color = (0, 0, 255)      # 红色
        else:
            color = (255, 255, 0)    # 青色
        
        # 绘制边界框
        pygame.draw.rect(surface, color, (int(x1), int(y1), int(x2-x1), int(y2-y1)), 3)
        
        # 绘制标签背景
        label = f"{class_name}: {conf:.2f}"
        text = font.render(label, True, color)
        text_rect = text.get_rect()
        text_rect.topleft = (int(x1), int(y1) - 25)
        
        # 绘制黑色背景
        pygame.draw.rect(surface, (0, 0, 0), text_rect.inflate(10, 5))
        surface.blit(text, text_rect)

# 游戏循环
while True:
    # 绘制背景
    win.blit(background, (0, 0))

    now = pygame.time.get_ticks()
    # 刷新水果（保持你原有的判断）
    if now - last_refresh > refresh_time or current_image is None:
        current_image = random.choice(list(images.keys()))
        x, y = random.randint(0, WIDTH-80), random.randint(0, HEIGHT-80)
        rect = pygame.Rect(x, y, 80, 80)
        last_refresh = now
        # 关键：刷新时清空旧检测并强制下一步立即检测
        detection_results = []
        force_detect = True
        print(f"\n--- New image: {current_image} ---")


    # 先绘制背景和水果
    win.blit(background, (0, 0))
    if current_image:
        win.blit(images[current_image], (rect.x, rect.y))

    # 检测时机：若刚刷新则立即检测；否则按节流检测
    current_time = time.time()
    need_detect = force_detect or (current_time - last_detection_time > detection_interval)
    if need_detect:
        screen_cv = capture_screen()
        if screen_cv is not None:
            results = model(screen_cv, conf=0.25, imgsz=1440, device=0, verbose=False)
            parsed = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                    parsed.append({'class': model.names[cls], 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
            detection_results = parsed

            # 新增：自动点击苹果
            if AUTO_CLICK_ENABLED and detection_results:
                # 找到置信度最高的苹果
                apples = [d for d in detection_results if d['class'] == 'apple' and d['confidence'] >= AUTO_CLICK_MIN_CONF]
                if apples:
                    best = max(apples, key=lambda d: d['confidence'])
                    x1, y1, x2, y2 = best['bbox']
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    now_t = time.time()
                    if now_t - last_auto_click_time >= AUTO_CLICK_COOLDOWN:
                        # 合成一次左键点击 (DOWN+UP)
                        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': (cx, cy), 'button': 1}))
                        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {'pos': (cx, cy), 'button': 1}))
                        last_auto_click_time = now_t

        last_detection_time = current_time
        force_detect = False  # 本次强制检测完成


    # 绘制AI检测结果
    draw_detections(win, detection_results)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # 新增：按下 ESC 键退出
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

        # 鼠标点击
        if event.type == pygame.MOUSEBUTTONDOWN:
            if rect and rect.collidepoint(event.pos):
                if current_image == target:
                    score += 1

    # 绘制分数
    font = pygame.font.SysFont(None, 40)
    text = font.render(f"Score: {score}", True, (0, 0, 0))
    win.blit(text, (10, 10))

    # 绘制目标提示
    target_text = font.render(f"Target: {target}", True, (0, 0, 0))
    win.blit(target_text, (10, 50))

    # 绘制检测状态
    status_text = font.render(f"Detections: {len(detection_results)}", True, (0, 0, 0))
    win.blit(status_text, (10, 90))

    pygame.display.update()
    clock.tick(120)  # 用 60 FPS 保证刷新稳定