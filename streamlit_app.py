import io
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

from collision import CollisionPredictor
from lane_detection import LaneDetector
from traffic_sign import TrafficSignRecognizer
from tracking import MultiObjectTracker

Detector = None
_detector_import_error = None
try:
    from detection import Detector
except ImportError as exc:
    _detector_import_error = exc


@dataclass
class FrameMetrics:
    frame_index: int
    detected_classes: Counter
    object_count: int
    collision_count: int
    average_speed: float


@st.cache_resource
def load_detector(model_type: str, conf_threshold: float, iou_threshold: float):
    if _detector_import_error is not None:
        raise RuntimeError(
            "Failed to import Detector. Install the required dependencies and try again. "
            f"Original error: {_detector_import_error}"
        )
    return Detector(model_type=model_type, conf_threshold=conf_threshold, iou_threshold=iou_threshold)


@st.cache_resource
def load_lane_detector() -> LaneDetector:
    return LaneDetector()


@st.cache_resource
def load_sign_recognizer() -> TrafficSignRecognizer:
    return TrafficSignRecognizer()


@st.cache_resource
def load_tracker() -> MultiObjectTracker:
    return MultiObjectTracker()


@st.cache_resource
def load_collision_predictor() -> CollisionPredictor:
    return CollisionPredictor(safe_ttc=3.5)


def annotate_frame(
    frame: np.ndarray,
    detections: List,
    tracks: List,
    lanes,
    signs,
    alerts,
    show_detection: bool,
    show_tracks: bool,
    show_lanes: bool,
    show_signs: bool,
    show_alerts: bool,
) -> np.ndarray:
    annotated = frame.copy()
    if show_detection:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 150, 255), 2)
            cv2.putText(
                annotated,
                f"{det.label} {det.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    if show_tracks:
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
    if show_lanes:
        annotated = load_lane_detector().draw(annotated, lanes)
    if show_signs:
        annotated = load_sign_recognizer().draw(annotated, signs)
    if show_alerts:
        annotated = load_collision_predictor().draw(annotated, alerts)
    return annotated


def compute_speed_estimate(tracks: List) -> float:
    if not tracks:
        return 0.0
    speeds = [np.linalg.norm(track.velocity) for track in tracks]
    return float(np.mean(speeds) * 8.0)


def frame_analysis(frame: np.ndarray, detector: Detector, lane_detector: LaneDetector, sign_recognizer: TrafficSignRecognizer, tracker: MultiObjectTracker, collision_predictor: CollisionPredictor) -> Tuple[List, List, List, List, List[FrameMetrics], np.ndarray]:
    detections = detector.predict(frame)
    tracks = tracker.update(detections, frame=frame)
    lanes = lane_detector.detect(frame)
    signs = sign_recognizer.detect(frame)
    alerts = collision_predictor.predict(tracks, image_shape=frame.shape)
    speed = compute_speed_estimate(tracks)
    metrics = FrameMetrics(
        frame_index=0,
        detected_classes=Counter([det.label for det in detections]),
        object_count=len(detections),
        collision_count=len(alerts),
        average_speed=speed,
    )
    return detections, tracks, lanes, signs, alerts, metrics


def run_video_analysis(
    video_path: str,
    max_frames: int,
    detector: Detector,
    lane_detector: LaneDetector,
    sign_recognizer: TrafficSignRecognizer,
    tracker: MultiObjectTracker,
    collision_predictor: CollisionPredictor,
) -> Tuple[List[FrameMetrics], np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    metrics: List[FrameMetrics] = []
    annotated_frame = None
    frame_index = 0

    while frame_index < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.predict(frame)
        tracks = tracker.update(detections, frame=frame)
        lanes = lane_detector.detect(frame)
        signs = sign_recognizer.detect(frame)
        alerts = collision_predictor.predict(tracks, image_shape=frame.shape)
        speed = compute_speed_estimate(tracks)

        metrics.append(
            FrameMetrics(
                frame_index=frame_index,
                detected_classes=Counter([det.label for det in detections]),
                object_count=len(detections),
                collision_count=len(alerts),
                average_speed=speed,
            )
        )

        if frame_index == 0:
            annotated_frame = annotate_frame(
                frame,
                detections,
                tracks,
                lanes,
                signs,
                alerts,
                show_detection=True,
                show_tracks=True,
                show_lanes=True,
                show_signs=True,
                show_alerts=True,
            )

        frame_index += 1

    cap.release()
    return metrics, annotated_frame if annotated_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)


def build_speedometer(speed_kmh: float):
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=speed_kmh,
            delta={"reference": 80, "position": "bottom"},
            gauge={
                "axis": {"range": [0, 180]},
                "bar": {"color": "#ff4b4b"},
                "steps": [
                    {"range": [0, 60], "color": "#31a354"},
                    {"range": [60, 120], "color": "#f1c232"},
                    {"range": [120, 180], "color": "#d62728"},
                ],
            },
            title={"text": "Estimated System Speed (km/h)"},
        )
    )
    fig.update_layout(height=330, margin=dict(t=20, b=20, l=20, r=20))
    return fig


def build_object_count_chart(frame_metrics: List[FrameMetrics]):
    if not PLOTLY_AVAILABLE or not frame_metrics:
        return None
    x = [m.frame_index for m in frame_metrics]
    y = [m.object_count for m in frame_metrics]
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#1f77b4")))
    fig.update_layout(title="Objects Detected per Frame", xaxis_title="Frame", yaxis_title="Object Count", height=320, margin=dict(t=40, b=40, l=40, r=20))
    return fig


def build_collision_timeline(frame_metrics: List[FrameMetrics]):
    if not PLOTLY_AVAILABLE or not frame_metrics:
        return None
    x = [m.frame_index for m in frame_metrics]
    y = [m.collision_count for m in frame_metrics]
    fig = go.Figure(data=go.Bar(x=x, y=y, marker_color="#d62728"))
    fig.update_layout(title="Collision Warnings per Frame", xaxis_title="Frame", yaxis_title="Warnings", height=320, margin=dict(t=40, b=40, l=40, r=20))
    return fig


def build_class_distribution(frame_metrics: List[FrameMetrics]):
    if not PLOTLY_AVAILABLE:
        return None
    counter = Counter()
    for metric in frame_metrics:
        counter.update(metric.detected_classes)
    if not counter:
        return None
    labels = list(counter.keys())
    values = list(counter.values())
    fig = go.Figure(data=go.Pie(labels=labels, values=values, hole=0.3))
    fig.update_layout(title="Detected Object Class Distribution", height=340, margin=dict(t=40, b=40, l=40, r=20))
    return fig


def main():
    st.set_page_config(page_title="Autonomous Driving Dashboard", layout="wide", page_icon="🚘")
    st.title("Autonomous Driving Perception Dashboard")
    st.markdown(
        "Use the dashboard to analyze video snapshots, uploaded footage, or camera captures with object detection, lane detection, sign recognition, tracking, and collision prediction."
    )

    if _detector_import_error is not None:
        st.error(
            "The object detection backend could not be loaded because required dependencies are missing. "
            "Install the dependencies from requirements.txt and restart the app."
        )
        st.code(str(_detector_import_error))
        return

    with st.sidebar:
        st.header("Controls")
        source = st.radio("Data source", ["Image", "Upload Video", "Camera Snapshot"])
        model_type = st.selectbox("Detection model", ["yolov11", "rt-detr"])
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.8, 0.35, 0.05)
        iou_threshold = st.slider("NMS IoU threshold", 0.1, 0.7, 0.45, 0.05)
        max_frames = st.slider("Frames to analyze", 1, 40, 12)
        show_detection = st.checkbox("Show detections", True)
        show_tracks = st.checkbox("Show tracks", True)
        show_lanes = st.checkbox("Show lane overlay", True)
        show_signs = st.checkbox("Show traffic sign overlay", True)
        show_alerts = st.checkbox("Show collision alerts", True)
        analyze_button = st.button("Analyze now")

    detector = load_detector(model_type, conf_threshold, iou_threshold)
    lane_detector = load_lane_detector()
    sign_recognizer = load_sign_recognizer()
    tracker = load_tracker()
    collision_predictor = load_collision_predictor()

    analysis_context = st.empty()
    final_metrics: List[FrameMetrics] = []
    annotated_frame: Optional[np.ndarray] = None
    source_preview = None

    if source == "Image":
        image_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
        if image_file is not None and analyze_button:
            image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.predict(frame)
            tracks = tracker.update(detections, frame=frame)
            lanes = lane_detector.detect(frame)
            signs = sign_recognizer.detect(frame)
            alerts = collision_predictor.predict(tracks, image_shape=frame.shape)
            annotated_frame = annotate_frame(
                frame,
                detections,
                tracks,
                lanes,
                signs,
                alerts,
                show_detection,
                show_tracks,
                show_lanes,
                show_signs,
                show_alerts,
            )
            final_metrics = [FrameMetrics(frame_index=0, detected_classes=Counter([det.label for det in detections]), object_count=len(detections), collision_count=len(alerts), average_speed=compute_speed_estimate(tracks))]
            source_preview = st.image(annotated_frame, caption="Annotated image", use_column_width=True)

    elif source == "Upload Video":
        video_file = st.file_uploader("Upload a driving video", type=["mp4", "mov", "avi", "mkv"])
        if video_file is not None and analyze_button:
            temp_path = os.path.join(".", "temp_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            with st.spinner("Processing video frames..."):
                final_metrics, annotated_frame = run_video_analysis(
                    temp_path,
                    max_frames,
                    detector,
                    lane_detector,
                    sign_recognizer,
                    tracker,
                    collision_predictor,
                )
            source_preview = st.image(annotated_frame, caption="Annotated video key frame", use_column_width=True)
            os.remove(temp_path)

    else:
        camera_image = st.camera_input("Take a snapshot of the driving scene")
        if camera_image is not None and analyze_button:
            image_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.predict(frame)
            tracks = tracker.update(detections, frame=frame)
            lanes = lane_detector.detect(frame)
            signs = sign_recognizer.detect(frame)
            alerts = collision_predictor.predict(tracks, image_shape=frame.shape)
            annotated_frame = annotate_frame(
                frame,
                detections,
                tracks,
                lanes,
                signs,
                alerts,
                show_detection,
                show_tracks,
                show_lanes,
                show_signs,
                show_alerts,
            )
            final_metrics = [FrameMetrics(frame_index=0, detected_classes=Counter([det.label for det in detections]), object_count=len(detections), collision_count=len(alerts), average_speed=compute_speed_estimate(tracks))]
            source_preview = st.image(annotated_frame, caption="Annotated camera snapshot", use_column_width=True)

    if final_metrics:
        totals = {"objects": sum(m.object_count for m in final_metrics), "collisions": sum(m.collision_count for m in final_metrics), "avg_speed": np.mean([m.average_speed for m in final_metrics])}
        st.markdown("---")
        st.subheader("Vehicle Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames analyzed", len(final_metrics))
        col2.metric("Total objects", totals["objects"])
        col3.metric("Collision warnings", totals["collisions"])
        col4.metric("Avg speed estimate", f"{totals['avg_speed']:.1f} km/h")

        left_col, right_col = st.columns([2, 1])
        with left_col:
            object_chart = build_object_count_chart(final_metrics)
            if object_chart is not None:
                st.plotly_chart(object_chart, use_container_width=True)
            else:
                st.subheader("Objects Detected per Frame")
                st.line_chart([m.object_count for m in final_metrics])

            collision_chart = build_collision_timeline(final_metrics)
            if collision_chart is not None:
                st.plotly_chart(collision_chart, use_container_width=True)
            else:
                st.subheader("Collision Warnings per Frame")
                st.bar_chart([m.collision_count for m in final_metrics])

        with right_col:
            class_chart = build_class_distribution(final_metrics)
            if class_chart is not None:
                st.plotly_chart(class_chart, use_container_width=True)
            else:
                st.subheader("Detected Object Class Distribution")
                class_counter = Counter()
                for metric in final_metrics:
                    class_counter.update(metric.detected_classes)
                if class_counter:
                    st.dataframe({"Class": list(class_counter.keys()), "Count": list(class_counter.values())})
                else:
                    st.write("No detected object classes to display.")

            speed_chart = build_speedometer(totals["avg_speed"])
            if speed_chart is not None:
                st.plotly_chart(speed_chart, use_container_width=True)
            else:
                st.metric("Estimated System Speed", f"{totals['avg_speed']:.1f} km/h", delta=f"{totals['avg_speed'] - 80:.1f} km/h")

        st.markdown("### Breakdown Table")
        breakdown = []
        for metric in final_metrics:
            breakdown.append({
                "Frame": metric.frame_index,
                "Objects": metric.object_count,
                "Collisions": metric.collision_count,
                "Speed": f"{metric.average_speed:.1f}",
                "Class summary": ", ".join([f"{label}:{count}" for label, count in metric.detected_classes.items()]),
            })
        st.dataframe(breakdown)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Use the button above to analyze the chosen source. For the best result, upload a short video clip or a clear road image."
    )


if __name__ == "__main__":
    main()
