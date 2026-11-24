import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from manga_translator.utils import (
    detect_bubbles_with_mangalens,
    build_bubble_mask_from_mangalens_result,
)

INPUT_DIR = "manga_det_dataset/test/images"
OUT_BUBBLE_DIR = "output/generated_masks/bubble_masks"   # 生成类似图2的黑白气泡掩码
OUT_TEXT_DIR = "output/generated_masks/text_color_masks" # 生成类似图1的黑白+洋红双色文字掩码
os.makedirs(OUT_BUBBLE_DIR, exist_ok=True)
os.makedirs(OUT_TEXT_DIR, exist_ok=True)

yolo_model = YOLO("models/detection/yolo_custom.pt")

for img_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")))[:20]:
    name = os.path.splitext(os.path.basename(img_path))[0]
    bgr = cv2.imread(img_path)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 1. 生成【图2：黑底白字气泡二值掩码】
    try:
        res_lens = detect_bubbles_with_mangalens(rgb, return_annotated=False, verbose=False)
        bubble_mask = build_bubble_mask_from_mangalens_result(res_lens, (h, w))
    except Exception:
        bubble_mask = np.zeros((h, w), dtype=np.uint8)

    # 用微调的 YOLOv8 检测框补充/强化气泡区域
    yolo_res = yolo_model.predict(rgb, imgsz=800, conf=0.45, verbose=False)[0]
    yolo_box_mask = np.zeros((h, w), dtype=np.uint8)
    for box in yolo_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(yolo_box_mask, (x1, y1), (x2, y2), 255, -1)
        if np.count_nonzero(bubble_mask[y1:y2, x1:x2]) == 0:
            cv2.ellipse(
                bubble_mask,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                0, 0, 360, 255, -1,
            )

    cv2.imwrite(os.path.join(OUT_BUBBLE_DIR, f"{name}_bubble_mask.png"), bubble_mask)

    # 2. 生成【图1：白底 + 气泡内黑字 + 气泡外洋红文字掩码】
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 提取深色笔迹像素
    _, ink_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    # 过滤掉大面积黑色背景块，保留文字笔画
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    clean_text_mask = np.zeros_like(ink_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if 8 <= area <= 2500 and bw < w * 0.25 and bh < h * 0.25:
            clean_text_mask[labels == i] = 255

    # 白底画布 (BGR)
    vis_text = np.full((h, w, 3), 255, dtype=np.uint8)
    in_bubble = (clean_text_mask > 0) & ((bubble_mask > 0) | (yolo_box_mask > 0))
    out_bubble = (clean_text_mask > 0) & (~in_bubble)

    vis_text[in_bubble] = (0, 0, 0)       # 气泡内文字：黑色
    vis_text[out_bubble] = (255, 0, 255)  # 气泡外文字/标题/拟声词：洋红色 (BGR: 255,0,255)

    cv2.imwrite(os.path.join(OUT_TEXT_DIR, f"{name}_text_mask.png"), vis_text)

print("批量掩码图生成完成！请查看 output/generated_masks/ 目录。")
