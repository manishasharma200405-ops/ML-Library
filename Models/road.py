"""Creates road.mp4 - a fake road where the car slowly drifts left and right.

"""
import os
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
W, H, FRAMES, FPS = 960, 540, 300, 30
out = cv2.VideoWriter("road.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

for i in range(FRAMES):
    shift = int(110 * np.sin(2 * np.pi * i / 150))      # car drifts left/right over time
    img = np.full((H, W, 3), 60, np.uint8)               # grey road
    cv2.rectangle(img, (0, 0), (W, 200), (200, 160, 120), -1)   # sky
    cv2.line(img, (150 + shift, H), (450 + shift // 4, 330), (255, 255, 255), 10)
    cv2.line(img, (810 + shift, H), (510 + shift // 4, 330), (255, 255, 255), 10)
    out.write(img)

out.release()
print("road.mp4 created")