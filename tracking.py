import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _HAS_DEEPSORT = True
except ImportError:
    DeepSort = None
    _HAS_DEEPSORT = False

logger = logging.getLogger(__name__)

@dataclass
class Track:
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)


class MultiObjectTracker:
    def __init__(self, enable_deepsort: bool = True, max_iou_distance: float = 0.3):
        self.enable_deepsort = enable_deepsort and _HAS_DEEPSORT
        self.max_iou_distance = max_iou_distance
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.deepsort: Optional[DeepSort] = None

        if self.enable_deepsort:
            logger.info("Initializing DeepSORT tracker")
            try:
                self.deepsort = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
            except Exception as exc:
                logger.warning("DeepSORT initialization failed: %s", exc)
                logger.warning("Falling back to simple IOU tracker")
                self.deepsort = None
                self.enable_deepsort = False

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections: List["Detection"], frame: Optional[np.ndarray] = None) -> List[Track]:
        if self.deepsort is not None and frame is not None:
            detections_for_ds = []
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                detections_for_ds.append(([x1, y1, x2 - x1, y2 - y1], detection.confidence, detection.label))
            tracks = self.deepsort.update_tracks(detections_for_ds, frame=frame)
            output_tracks: List[Track] = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tlbr = track.to_tlbr()
                bbox = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                output_tracks.append(
                    Track(
                        track_id=track.track_id,
                        label=track.det_confidences[0][0] if track.det_confidences else "object",
                        bbox=bbox,
                        confidence=track.det_confidences[0][1] if track.det_confidences else 0.0,
                        class_id=-1,
                    )
                )
            return output_tracks

        return self._simple_iou_tracking(detections)

    def _simple_iou_tracking(self, detections: List["Detection"]) -> List[Track]:
        current_tracks: Dict[int, Track] = {}
        unmatched_tracks = set(self.tracks.keys())

        for det in detections:
            best_match = None
            best_iou = 0.0
            for track_id in list(unmatched_tracks):
                track = self.tracks[track_id]
                score = self._iou(track.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_match = track_id
            if best_match is not None and best_iou >= self.max_iou_distance:
                track = self.tracks[best_match]
                vx, vy = self._compute_velocity(track.bbox, det.bbox)
                updated_history = track.history[-4:] + [det.bbox]
                current_tracks[best_match] = Track(
                    track_id=best_match,
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    velocity=(vx, vy),
                    history=updated_history,
                )
                unmatched_tracks.remove(best_match)
            else:
                current_tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    velocity=(0.0, 0.0),
                    history=[det.bbox],
                )
                self.next_id += 1

        self.tracks = current_tracks
        return list(self.tracks.values())

    @staticmethod
    def _compute_velocity(previous_bbox: Tuple[int, int, int, int], current_bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x_prev, y_prev = MultiObjectTracker._center(previous_bbox)
        x_curr, y_curr = MultiObjectTracker._center(current_bbox)
        return float(x_curr - x_prev), float(y_curr - y_prev)
