import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TrafficSign:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class TrafficSignRecognizer:
    def __init__(self):
        self.sign_labels = {
            "red_circle": "Stop/Prohibitory",
            "blue_circle": "Mandatory",
            "yellow_triangle": "Warning",
            "speed_limit": "Speed Limit",
        }

    @staticmethod
    def _color_mask(frame: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

    def detect(self, frame: np.ndarray) -> List[TrafficSign]:
        results: List[TrafficSign] = []
        height, width = frame.shape[:2]

        red_mask = cv2.bitwise_or(
            self._color_mask(frame, (0, 70, 50), (10, 255, 255)),
            self._color_mask(frame, (160, 70, 50), (180, 255, 255)),
        )
        blue_mask = self._color_mask(frame, (100, 100, 50), (140, 255, 255))
        yellow_mask = self._color_mask(frame, (15, 100, 100), (35, 255, 255))

        for mask, label_key in [(red_mask, "red_circle"), (blue_mask, "blue_circle"), (yellow_mask, "yellow_triangle")]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 20 or h < 20 or w > width * 0.5 or h > height * 0.5:
                    continue
                label = self.sign_labels[label_key]
                results.append(TrafficSign(label=label, confidence=0.65, bbox=(x, y, x + w, y + h)))

        return results

    def draw(self, frame: np.ndarray, signs: List[TrafficSign]) -> np.ndarray:
        overlay = frame.copy()
        for sign in signs:
            x1, y1, x2, y2 = sign.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 204, 0), 2)
            cv2.putText(overlay, f"{sign.label} {sign.confidence:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
