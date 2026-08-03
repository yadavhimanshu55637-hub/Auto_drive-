import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class LaneLine:
    points: np.ndarray
    slope: float
    intercept: float


class LaneDetector:
    def __init__(
        self,
        blur_ksize: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        rho: int = 1,
        theta: float = np.pi / 180,
        hough_threshold: int = 50,
        min_line_length: int = 80,
        max_line_gap: int = 40,
    ):
        self.blur_ksize = blur_ksize
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.rho = rho
        self.theta = theta
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([
            [0, height],
            [int(width * 0.45), int(height * 0.6)],
            [int(width * 0.55), int(height * 0.6)],
            [width, height],
        ])
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(edges, mask)

    def detect(self, frame: np.ndarray) -> List[LaneLine]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        region = self._region_of_interest(edges)
        lines = cv2.HoughLinesP(
            region,
            self.rho,
            self.theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        lane_lines: List[LaneLine] = []
        if lines is None:
            return lane_lines

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.3:
                continue
            lane_lines.append(LaneLine(points=np.array([x1, y1, x2, y2]), slope=slope, intercept=intercept))

        return self._consolidate_lane_lines(lane_lines)

    def _consolidate_lane_lines(self, lanes: List[LaneLine]) -> List[LaneLine]:
        if not lanes:
            return []
        left_lines = [lane for lane in lanes if lane.slope < 0]
        right_lines = [lane for lane in lanes if lane.slope > 0]
        consolidated = []
        for group in (left_lines, right_lines):
            if not group:
                continue
            avg_slope = float(np.mean([lane.slope for lane in group]))
            avg_intercept = float(np.mean([lane.intercept for lane in group]))
            consolidated.append(LaneLine(points=np.array([0, int(avg_intercept), 1, int(avg_slope + avg_intercept)]), slope=avg_slope, intercept=avg_intercept))
        return consolidated

    def draw(self, frame: np.ndarray, lanes: List[LaneLine]) -> np.ndarray:
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for lane in lanes:
            x1 = 0
            y1 = int(lane.intercept)
            x2 = width
            y2 = int(lane.slope * width + lane.intercept)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
