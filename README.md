# Autonomous Driving Perception System

A starter perception pipeline for object detection, lane detection, traffic sign recognition, collision prediction, and multi-object tracking.

## Files

- `app.py` - main execution script
- `detection.py` - YOLO-style object detector wrapper
- `lane_detection.py` - lane line detection using edge and Hough analysis
- `tracking.py` - multi-object tracker with DeepSORT fallback
- `traffic_sign.py` - simple traffic sign color-shape detection
- `collision.py` - time-to-collision alerts based on tracked object motion
- `streamlit_app.py` - interactive Streamlit dashboard for analysis and visualization
- `requirements.txt` - dependency list

## Setup

1. Create or activate the Python 3.11 virtual environment bundled with this workspace:

```powershell
cd "c:\Users\laksh\OneDrive\Desktop\Auto_drive"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell script execution is disabled, use the virtual environment interpreter directly instead of activating:

```powershell
cd "c:\Users\laksh\OneDrive\Desktop\Auto_drive"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you do not want DeepSORT tracking, the pipeline still runs without `deep_sort_realtime`.

## Run

Run the desktop perception pipeline:

```powershell
cd "c:\Users\laksh\OneDrive\Desktop\Auto_drive"
.\.venv\Scripts\python.exe app.py
```

Run the Streamlit dashboard:

```powershell
cd "c:\Users\laksh\OneDrive\Desktop\Auto_drive"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Press `Esc` to exit the live video window in `app.py`.

## Notes

- `app.py` defaults to webcam input; pass a filename or RTSP URL to use another video source.
- YOLO is loaded through `torch.hub`. Model weights are downloaded automatically when needed.
- The system is scaffolded for research and prototyping, not production-grade deployment.
