import os
import glob
import time
import json
import asyncio
import numpy as np
import pandas as pd
from PIL import Image
from manga_translator.manga_translator import MangaTranslator
from manga_translator.config import Config, Detector, Ocr

TEST_IMG_DIR = "manga_det_dataset/test/images"
# 如果你有准备人工校对的文本真值 JSON（格式: {"图片名.jpg": "整页按顺序拼接的参考文本"}），填入路径即可自动算真实 CER；
# 若暂无文本真值，脚本会自动以完整版 Proposed 的高质量识别结果为参考基准计算相对字符差异率，并统计真实检测数与耗时。
GT_TEXT_JSON = "manga_det_dataset/test/gt_texts.json"
OUT_CSV = "output/cds540_experiment_results.csv"


def compute_cer(pred_text: str, gt_text: str) -> float:
    """计算字符错误率 CER = Levenshtein编辑距离 / 参考文本长度"""
    pred = "".join(str(pred_text).split())
    gt = "".join(str(gt_text).split())
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = [[0] * (len(gt) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(gt) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(gt) + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(pred)][len(gt)] / float(len(gt))


async def run_all_experiments():
    os.environ["CDS540_ENABLED"] = "1"
    img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.*")))[:15]  # 取前15张测试图快速跑出对比数据
    if not img_paths:
        print(f"未在 {TEST_IMG_DIR} 找到测试图片！")
        return

    gt_texts = {}
    if os.path.exists(GT_TEXT_JSON):
        with open(GT_TEXT_JSON, "r", encoding="utf-8") as f:
            gt_texts = json.load(f)

    # 加载项目配置文件
    cfg_path = "config/config.json" if os.path.exists("config/config.json") else "config/config.example.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = Config.parse_obj(cfg_dict) if hasattr(Config, "parse_obj") else Config(**cfg_dict)
    config.detector.detector = Detector.yolo_custom
    config.detector.detection_size = 800

    translator = MangaTranslator({"use_gpu": False, "verbose": False})

    modes = [
        ("baseline", "Mode 1: Baseline (Whole-Page OCR, No Detection)"),
        ("proposed_no_prep", "Mode 2: Proposed (YOLOv8 Crop, No Preprocess)"),
        ("proposed", "Mode 3: Proposed (YOLOv8 Crop + Resize/Denoise/CLAHE)"),
    ]

    all_mode_outputs = {}
    summary_rows = []

    for mode_key, mode_label in modes:
        os.environ["CDS540_MODE"] = mode_key
        print(f"\n>>> 正在运行实验: {mode_label} ...")
        latencies = []
        bubble_counts = []
        page_texts = {}

        for p in img_paths:
            img_name = os.path.basename(p)
            pil_img = Image.open(p).convert("RGB")
            pil_img.name = p

            ctx = await translator._run_cds540_staged_pipeline(pil_img, config)
            metrics = getattr(ctx, "cds540_metrics", {})
            latencies.append(metrics.get("elapsed_ms", 0.0))
            bubbles = metrics.get("bubbles", [])
            bubble_counts.append(len(bubbles))
            merged_str = "".join([b.get("text", "") for b in bubbles])
            page_texts[img_name] = merged_str

        all_mode_outputs[mode_key] = page_texts
        summary_rows.append({
            "Experiment_Mode": mode_label,
            "Avg_Detected_Bubbles": round(float(np.mean(bubble_counts)), 2),
            "Avg_Processing_Time_ms": round(float(np.mean(latencies)), 1),
            "_raw_texts": page_texts,
        })

    # 计算 CER（优先使用外部标注的 gt_texts.json，否则以 Mode 3 完整版输出作为伪真值计算相对差异率）
    ref_texts = gt_texts if gt_texts else all_mode_outputs["proposed"]
    cer_col_name = "Mean_CER_vs_GT" if gt_texts else "Relative_CER_vs_Proposed"

    for row in summary_rows:
        raw = row.pop("_raw_texts")
        cers = [compute_cer(raw[k], ref_texts.get(k, "")) for k in raw]
        row[cer_col_name] = round(float(np.mean(cers)), 4)

    df = pd.DataFrame(summary_rows)
    os.makedirs("output", exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n================ CDS540 对照实验汇总表 ================")
    print(df.to_string(index=False))
    print(f"\n实验数据已保存至: {OUT_CSV}")


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
