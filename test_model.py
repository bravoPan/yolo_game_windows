# file: test_model.py
import os
import glob
import cv2
from ultralytics import YOLO

MODEL_PATH = 'annotated_ds/yolo_game_model.pt'
IMAGES_DIR = 'dataset/images'  # 你的截图目录
CONF_THRES = 0.1

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    if not os.path.isdir(IMAGES_DIR):
        print(f"Images dir not found: {IMAGES_DIR}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    # 只匹配 screenshot_*.png
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, 'screenshot_*.png')))
    if not image_paths:
        print(f"No images matched: {os.path.join(IMAGES_DIR, 'screenshot_*.png')}")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot read: {img_path}")
            continue

        results = model(img, conf=CONF_THRES, verbose=False)
        print(f"\nImage: {os.path.basename(img_path)}")
        total = 0
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                name = model.names.get(cls, str(cls))
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                print(f" - {name}: conf={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                total += 1
        if total == 0:
            print(" - No detections")

if __name__ == "__main__":
    main()