from ultralytics import YOLO

def main():
    # 加载轻量级预训练模型（首次运行会自动下载约 6MB 的 yolov8n.pt）
    model = YOLO("yolov8n.pt")

    # 启动 CPU 微调训练
    model.train(
        data="manga_data.yaml",
        epochs=40,               # 471 张图微调 40 轮足够收敛
        imgsz=800,               # 兼顾漫画小字清晰度与 CPU 训练速度
        batch=8,                 # CPU 内存批大小
        device="cpu",            # 明确指定 CPU 训练
        workers=4,               # 多线程数据加载
        optimizer="AdamW",
        lr0=0.001,

        # --- 漫画专属数据增强（严禁水平/垂直翻转文字） ---
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        mosaic=0.5,
        close_mosaic=10,

        project="runs/manga_det",
        name="yolov8n_manga_v1"
    )

if __name__ == "__main__":
    main()
