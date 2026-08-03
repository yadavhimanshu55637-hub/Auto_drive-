import logging
from pathlib import Path

import cv2

from collision import CollisionPredictor
from detection import Detector
from lane_detection import LaneDetector
from traffic_sign import TrafficSignRecognizer
from tracking import MultiObjectTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main(video_source=0):
    detector = Detector(model_type="yolov11", conf_threshold=0.3)
    lane_detector = LaneDetector()
    sign_recognizer = TrafficSignRecognizer()
    tracker = MultiObjectTracker()
    collision_predictor = CollisionPredictor(safe_ttc=3.0)

    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        logging.error("Unable to open video source: %s", video_source)
        return

    window_name = "Autonomous Driving Perception"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        detections = detector.predict(frame)
        tracks = tracker.update(detections, frame=frame)
        lanes = lane_detector.detect(frame)
        signs = sign_recognizer.detect(frame)
        alerts = collision_predictor.predict(tracks, image_shape=frame.shape)

        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"{det.label} {det.confidence:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"ID {track.track_id} {track.label}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        annotated = lane_detector.draw(annotated, lanes)
        annotated = sign_recognizer.draw(annotated, signs)
        annotated = collision_predictor.draw(annotated, alerts)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
