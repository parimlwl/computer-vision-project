import os
import sys
import cv2
import numpy as np


def restore_noisy_blurry_image(
    img_bgr: np.ndarray,
    denoise_strength: float = 10.0,
    sharpen_amount: float = 1.2,
    use_clahe: bool = True,
) -> np.ndarray:
    """
    针对 [高斯噪声 + 模糊] 的两阶段图像恢复管线（先去噪、后去模糊锐化）：
    1. Non-Local Means (NLM) 去噪：剥离高斯随机噪点同时保护边缘结构
    2. Unsharp Masking (反锐化掩模)：增强模糊笔画的边缘梯度
    3. (可选) LAB 亮度通道 CLAHE：提升低对比度清晰度
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("输入图像为空")

    # 1. 去高斯噪声 (Fast Non-Local Means Denoising)
    # h: 亮度分量滤波强度 (建议 6~15，噪声越大值越大)
    # hColor: 色彩分量滤波强度 (黑白漫画可设为与 h 相同)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=denoise_strength,
        hColor=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2. 去模糊 / 边缘锐化 (Unsharp Masking: sharp = orig + amount * (orig - blurred))
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(
        denoised, 1.0 + sharpen_amount,
        gaussian_blur, -sharpen_amount,
        0
    )

    # 3. 可选：LAB 空间 L 通道 CLAHE 局部对比度增强（防止模糊导致字迹发灰）
    if use_clahe:
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_ch)
        sharpened = cv2.cvtColor(cv2.merge((l_enhanced, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    return sharpened


def process_file(input_path: str, output_path: str = None, denoise_strength: float = 10.0, sharpen_amount: float = 1.2):
    # 兼容中文路径读取
    raw_bytes = np.fromfile(input_path, dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    restored = restore_noisy_blurry_image(
        img,
        denoise_strength=denoise_strength,
        sharpen_amount=sharpen_amount,
        use_clahe=True,
    )

    if not output_path:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_restored{ext or '.png'}"

    # 拼一张 [原图 | 恢复后] 左右对比图方便展示
    sep = np.full((img.shape[0], 6, 3), (0, 0, 255), dtype=np.uint8)
    compare = np.hstack([img, sep, restored])

    # 兼容中文路径保存
    ext = os.path.splitext(output_path)[1] or ".png"
    cv2.imencode(ext, restored)[1].tofile(output_path)

    compare_path = f"{os.path.splitext(output_path)[0]}_compare{ext}"
    cv2.imencode(ext, compare)[1].tofile(compare_path)

    print(f"[OK] 去噪锐化结果已保存至: {output_path}")
    print(f"[OK] 左右对比图已保存至:   {compare_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: uv run --no-sync python denoise_deblur.py <图片路径> [去噪强度默认10] [锐化强度默认1.2]")
        sys.exit(1)

    in_img = sys.argv[1]
    d_str = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    s_amt = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    process_file(in_img, denoise_strength=d_str, sharpen_amount=s_amt)
