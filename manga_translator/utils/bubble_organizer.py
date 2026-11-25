import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_cjk_font(size: int = 18):
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def sort_reading_order(bubbles: list, direction: str = "rtl", row_tolerance: int = 60, img_h: int = None) -> list:
    """
    Stage 3: 按阅读方向排序气泡
    bubbles: [{"bbox": [x1, y1, x2, y2], "text": "...", "conf": 0.9}, ...]
    """
    if not bubbles:
        return []

    tol = max(30, int(img_h * 0.04)) if img_h else row_tolerance

    for b in bubbles:
        x1, y1, x2, y2 = b["bbox"]
        b["cx"] = (x1 + x2) / 2.0
        b["cy"] = (y1 + y2) / 2.0

    # 先按 Y 坐标聚类分行，再在行内按 X 排序
    bubbles.sort(key=lambda item: item["cy"])
    rows = []
    current_row = [bubbles[0]]

    for b in bubbles[1:]:
        if abs(b["cy"] - current_row[0]["cy"]) <= tol:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)

    ordered = []
    order_idx = 1
    for row in rows:
        row.sort(key=lambda item: -item["cx"] if direction == "rtl" else item["cx"])
        for b in row:
            item_dict = {
                "reading_order": order_idx,
                "bbox": b["bbox"],
                "confidence": round(float(b.get("conf", 1.0)), 4),
                "text": b["text"],
            }
            if "translation" in b:
                item_dict["translation"] = b["translation"]
            ordered.append(item_dict)
            order_idx += 1
    return ordered


def render_stage1_detection_image(image_rgb: np.ndarray, raw_boxes: list) -> np.ndarray:
    """生成 Stage 1 独立检测结果图（绿框 + 置信度）"""
    vis = image_rgb.copy()
    for idx, box in enumerate(raw_boxes):
        x1, y1, x2, y2 = box["bbox"]
        conf = box.get("conf", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.putText(
            vis,
            f"Det:{conf:.2f}",
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
    return vis


def save_and_visualize_stage3(
    image_rgb: np.ndarray,
    ordered_bubbles: list,
    json_path: str,
    mode_name: str = "proposed",
    show_translation: bool = True,
) -> np.ndarray:
    """导出结构化 JSON（含原文与译文）并在原图绘制阅读顺序编号与文本"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_mode": mode_name,
                "total_bubbles": len(ordered_bubbles),
                "bubbles": ordered_bubbles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    vis = image_rgb.copy()
    for item in ordered_bubbles:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis,
            f"#{item['reading_order']}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 0, 0),
            2,
        )

    pil_img = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_img)
    font = _load_cjk_font(18)

    for item in ordered_bubbles:
        x1, y1, x2, y2 = item["bbox"]
        # 优先显示翻译结果；若未翻译或关闭显示则显示 OCR 原文
        raw_display = item.get("translation") if (show_translation and item.get("translation")) else item.get("text", "")
        text_preview = (raw_display or "").replace("\n", " ")[:22]
        tx, ty = x1 + 38, max(y1 - 22, 4)

        bbox_txt = draw.textbbox((tx, ty), text_preview, font=font)
        draw.rectangle(bbox_txt, fill=(255, 255, 255))
        draw.text((tx, ty), text_preview, fill=(210, 20, 40), font=font)

    return np.array(pil_img)
