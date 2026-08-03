import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_id: int


class Detector:
    def __init__(
        self,
        model_type: str = "yolov11",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_type = model_type.lower()
        self.model_path = Path(model_path) if model_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model()
        self.names = getattr(self.model, "names", {}) if self.model is not None else {}

    def _load_model(self):
        try:
            if self.model_type in {"yolov11", "yolov5", "yolo"}:
                if self.model_path is not None and self.model_path.exists():
                    logger.info("Loading custom YOLO model from %s", self.model_path)
                    model = torch.hub.load(
                        "ultralytics/yolov5",
                        "custom",
                        path=str(self.model_path),
                        force_reload=False,
                        trust_repo=True,
                    )
                else:
                    logger.info("Loading default YOLOv5s model for fallback inference")
                    model = torch.hub.load(
                        "ultralytics/yolov5",
                        "yolov5s",
                        pretrained=True,
                        trust_repo=True,
                    )
                model.to(self.device)
                model.conf = self.conf_threshold
                model.iou = self.iou_threshold
                return model

            if self.model_type == "rt-detr":
                if self.model_path is not None and self.model_path.exists():
                    logger.info("Loading RT-DETR model from %s", self.model_path)
                    model = torch.load(str(self.model_path), map_location=self.device)
                    model.eval()
                    return model
                logger.warning("RT-DETR model path not found; using YOLO fallback")
                return self._load_model_override("yolov5")

        except Exception as exc:
            logger.warning("Model load failed: %s", exc)
            logger.warning("Falling back to CPU and default YOLO model")
            try:
                    model = torch.hub.load(
                        "ultralytics/yolov5",
                        "yolov5s",
                        pretrained=True,
                        trust_repo=True,
                    )
            except Exception as fallback_exc:
                logger.error("Fallback model load failed: %s", fallback_exc)
                return None

    def _load_model_override(self, model_type: str):
        self.model_type = model_type
        return self._load_model()

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            return []

        if self.model_type in {"yolov11", "yolov5", "yolo"}:
            results = self.model(frame)
            detections = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                label = self.names.get(class_id, f"class_{class_id}")
                detections.append(Detection(label=label, confidence=float(conf), bbox=(x1, y1, x2, y2), class_id=class_id))
            return detections

        logger.debug("Falling back to YOLO inference for unsupported model type %s", self.model_type)
        return self.predict(frame)
