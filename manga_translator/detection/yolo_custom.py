import os
import cv2
import numpy as np
from ultralytics import YOLO
from .common import CommonDetector
from ..utils import Quadrilateral


class CustomYOLODetector(CommonDetector):
    def __init__(self):
        super().__init__()
        self.model = None

    async def _load(self, device: str):
        weight_path = os.path.join("models", "detection", "../../models/detection/yolo_custom.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"未找到自定义 YOLO 权重文件: {weight_path}")
        self.model = YOLO(weight_path)
        self.device = "cpu" if device in ("mps", "rocm") else device

    async def _unload(self):
        del self.model
        self.model = None

    async def _detect(
            self,
            image: np.ndarray,
            detect_size: int,
            text_threshold: float,
            box_threshold: float,
            unclip_ratio: float,
            verbose: bool = False,
            *args,
            **kwargs,
    ):
        if self.model is None:
            await self._load(getattr(self, "device", "cpu"))

        img_h, img_w = image.shape[:2]

        # 保持与训练一致的 imgsz=800
        results = self.model.predict(
            source=image,
            imgsz=800,
            conf=max(box_threshold, 0.25),
            iou=0.45,
            device=self.device,
            verbose=False,
        )[0]

        textlines = []
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            # 0: text-black-bubble, 1: text-black-free, 2: text-white-bubble, 3: text-white-free
            if cls_id not in (0, 1, 2, 3):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])

            # 动态安全扩边 (Padding 6%)，避免裁剪切片时切断偏旁或日文浊音符号
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = bw * 0.06, bh * 0.06
            px1 = max(0, int(x1 - pad_x))
            py1 = max(0, int(y1 - pad_y))
            px2 = min(img_w, int(x2 + pad_x))
            py2 = min(img_h, int(y2 + pad_y))

            pts = np.array([[px1, py1], [px2, py1], [px2, py2], [px1, py2]], dtype=np.int32)
            textlines.append(Quadrilateral(pts, "", score))

            # 根据识别出的黑白字类别自适应生成精准文字擦除掩码
            roi = gray[py1:py2, px1:px2]
            if roi.size > 0:
                thresh_type = cv2.THRESH_BINARY if cls_id in (2, 3) else cv2.THRESH_BINARY_INV
                _, bin_text = cv2.threshold(roi, 0, 255, thresh_type + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                bin_dilated = cv2.dilate(bin_text, kernel, iterations=2)
                mask[py1:py2, px1:px2] = np.maximum(mask[py1:py2, px1:px2], bin_dilated)

        return textlines, mask, None
