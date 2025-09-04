import pygame
import random
import sys

# 初始化
pygame.init()

# 窗口大小
WIDTH, HEIGHT = 2560, 1440
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("YOLO Auto Click Game")

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

# 游戏参数
target = "apple"
score = 0
clock = pygame.time.Clock()
current_image = None
rect = None
apples_count = 0

# 刷新计时器
refresh_time = 600  # 毫秒
last_refresh = pygame.time.get_ticks()

# 游戏循环
while True:
    # 绘制背景
    win.blit(background, (0, 0))

    now = pygame.time.get_ticks()
    # 每 0.5 秒刷新图片
    if now - last_refresh > refresh_time or current_image is None:
        current_image = random.choice(list(images.keys()))
        x, y = random.randint(0, WIDTH-80), random.randint(0, HEIGHT-80)
        rect = pygame.Rect(x, y, 80, 80)
        last_refresh = now
        
        # 新增：累计出现过的苹果总数
        if current_image == 'apple':
            apples_count += 1

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

    # 绘制图片
    if current_image:
        win.blit(images[current_image], (rect.x, rect.y))

    # 绘制分数
    font = pygame.font.SysFont(None, 40)
    text = font.render(f"Score: {score}", True, (0, 0, 0))
    win.blit(text, (10, 10))

    # 绘制目标
    target_text = font.render(f"Target: {target}", True, (0, 0, 0))
    win.blit(target_text, (10, 50))

    # 显示累计苹果总数
    apples_text = font.render(f"Apples: {apples_count}", True, (0, 0, 0))
    win.blit(apples_text, (10, 90))

    pygame.display.update()
    clock.tick(60)  # 用 60 FPS 保证刷新稳定
