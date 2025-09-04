import os
import sys
import time
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

HAS_WIN32 = False
try:
    import win32gui, win32con, win32api
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

# ---------------- Config ----------------
WINDOW_TITLE = "YOLO Auto Click Game"  # must match your game window title
MODEL_PATH = 'annotated_ds/yolo_game_model.pt'
IMG_SIZE = 1440            # match training size
CONF_THRES = 0.25          # detection confidence
AUTO_CLICK_MIN_CONF = 0.25 # lower threshold a bit to trigger sooner
AUTO_CLICK_COOLDOWN = 0.1 # seconds, shorter cooldown
LOOP_FPS = 120             # higher loop rate for quicker reaction
BURST_CLICKS = 1           # send a short burst of clicks
BURST_INTERVAL = 0.03      # seconds between burst clicks

# -------------- Helpers -----------------
def find_window_rect(title: str):
    if gw is None:
        raise RuntimeError("pygetwindow not installed. Install with: pip install pygetwindow")
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
    return win.left, win.top, win.width, win.height  # screen coords

# --------------- Main -------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        # List alternatives to help user
        base = 'runs/train'
        if os.path.exists(base):
            print("Available model files under runs/train:")
            for root, _, files in os.walk(base):
                for f in files:
                    if f.endswith('.pt'):
                        print(os.path.join(root, f))
        sys.exit(1)

    # PyAutoGUI tuning
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    model = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        model.to('cuda')
        try:
            model.model.half()  # FP16 for speed if supported
        except Exception:
            pass
        print("Detect device:", next(model.model.parameters()).device)
    else:
        print("CUDA not available, using CPU")

    # Persist model buffers between calls for speed and enable cudnn autotune
    try:
        model.predict(persist=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    # Warmup to stabilize latency
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    with torch.inference_mode():
        try:
            _ = model(dummy, conf=0.1, imgsz=IMG_SIZE, device=0, verbose=False)
        except Exception:
            pass

    last_click_ts = 0.0
    frame_time = 1.0 / LOOP_FPS

    with mss.mss() as sct:
        while True:
            start_t = time.time()

            rect = find_window_rect(WINDOW_TITLE)
            if rect is None:
                # Window not found; small delay then retry
                time.sleep(0.2)
                continue
            left, top, width, height = rect
            mon = {"left": left, "top": top, "width": width, "height": height}

            shot = sct.grab(mon)
            frame = np.array(shot)  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Avoid resize if window height already equals training size
            resized = frame
            scale = 1.0
            if height > 0 and IMG_SIZE and IMG_SIZE != height:
                scale = IMG_SIZE / float(height)
                # Use fast linear resize; smaller compute if downscaling
                resized = cv2.resize(frame, (int(width * scale), IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            with torch.inference_mode():
                res = model(resized, conf=CONF_THRES, imgsz=IMG_SIZE, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)

            # Parse detections (apples only)
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

            # Auto-click highest-confidence apple quickly
            now_t = time.time()
            if apples and now_t - last_click_ts >= AUTO_CLICK_COOLDOWN:
                conf, (x1, y1, x2, y2) = max(apples, key=lambda t: t[0])
                # Slightly inset to avoid border clicks
                inset = 3
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if scale != 1.0:
                    inv = 1.0 / scale
                    cx = int(cx * inv)
                    cy = int(cy * inv)
                # Clamp to client area
                cx = max(inset, min(width - inset - 1, cx))
                cy = max(inset, min(height - inset - 1, cy))

                # Prefer direct window messages for reliability
                hwnd = win._hWnd if hasattr(win, '_hWnd') else None
                sent = send_click_to_window(hwnd, cx, cy, burst=BURST_CLICKS)
                if not sent:
                    screen_x = left + cx
                    screen_y = top + cy
                    for _ in range(BURST_CLICKS):
                        try:
                            clicker.mouseDown(x=screen_x, y=screen_y, button='left')
                            clicker.mouseUp(x=screen_x, y=screen_y, button='left')
                        except TypeError:
                            clicker.click(x=screen_x, y=screen_y)
                        if BURST_INTERVAL > 0:
                            time.sleep(BURST_INTERVAL)
                last_click_ts = now_t

            # keep loop rate
            elapsed = time.time() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
