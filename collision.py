import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tracking import Track

@dataclass
class CollisionAlert:
    track_id: int
    label: str
    ttc: float
    bbox: Tuple[int, int, int, int]


class CollisionPredictor:
    def __init__(self, safe_ttc: float = 3.5):
        self.safe_ttc = safe_ttc

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def predict(
        self,
        tracks: List[Track],
        image_shape: Optional[Tuple[int, int, int]] = None,
    ) -> List[CollisionAlert]:
        alerts: List[CollisionAlert] = []
        if not tracks or image_shape is None:
            return alerts

        height = image_shape[0]
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            _, cy = self._bbox_center(track.bbox)
            dy = track.velocity[1]
            if dy >= -0.5:
                continue
            distance_metric = height - y2
            speed = max(abs(dy), 1.0)
            ttc = float(distance_metric / speed)
            if ttc <= self.safe_ttc:
                alerts.append(CollisionAlert(track_id=track.track_id, label=track.label, ttc=ttc, bbox=track.bbox))

        return alerts

    def draw(self, frame: np.ndarray, alerts: List[CollisionAlert]) -> np.ndarray:
        overlay = frame.copy()
        for alert in alerts:
            x1, y1, x2, y2 = alert.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"ALERT {alert.track_id} {alert.label} TTC {alert.ttc:.1f}s",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
