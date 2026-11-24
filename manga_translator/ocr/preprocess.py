import cv2
import numpy as np


def preprocess_bubble(crop_img: np.ndarray, enable: bool = True, target_height: int = 96) -> np.ndarray:
    """
    Stage 2: 对裁剪后的气泡区域进行图像预处理：缩放、双边滤波去噪、CLAHE对比度增强
    """
    if not enable or crop_img is None or crop_img.size == 0:
        return crop_img

    # 1. Resizing: 保持宽高比将低分辨率气泡放大至合适高度，提升小字识别率
    h, w = crop_img.shape[:2]
    if h < target_height and h > 0:
        scale = target_height / float(h)
        crop_img = cv2.resize(crop_img, (max(1, int(w * scale)), target_height), interpolation=cv2.INTER_CUBIC)

    # 2. Denoising: 双边滤波去噪（保留文字边缘锐度同时平滑网点噪点）
    denoised = cv2.bilateralFilter(crop_img, d=5, sigmaColor=50, sigmaSpace=50)

    # 3. Contrast Enhancement: 转至 LAB 空间对亮度通道做 CLAHE 自适应直方图均衡
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return result


def build_stage2_comparison_grid(crop_pairs: list, max_items: int = 12) -> np.ndarray:
    """
    将 [原裁剪切片 | 预处理后切片] 拼成对比网格图，用于报告展示与阶段验证
    crop_pairs: list of (raw_crop_rgb, proc_crop_rgb)
    """
    if not crop_pairs:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    rows = []
    target_h = 110
    for idx, (raw_c, proc_c) in enumerate(crop_pairs[:max_items]):
        if raw_c is None or raw_c.size == 0:
            continue
        rh, rw = raw_c.shape[:2]
        new_w = max(1, int(rw * (target_h / float(rh))))
        r_resized = cv2.resize(raw_c, (new_w, target_h), interpolation=cv2.INTER_AREA)
        p_resized = cv2.resize(proc_c, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        sep = np.full((target_h, 6, 3), (255, 0, 0), dtype=np.uint8)  # 红色分隔线
        pair_row = np.hstack([r_resized, sep, p_resized])
        rows.append(pair_row)

    if not rows:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((target_h, max_w - r.shape[1], 3), 240, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
        padded_rows.append(np.full((4, max_w, 3), 200, dtype=np.uint8))

    return np.vstack(padded_rows)
