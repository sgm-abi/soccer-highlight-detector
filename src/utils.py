import cv2
import numpy as np
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    """Get basic info about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


def extract_frame(video_path: str, timestamp_sec: float) -> np.ndarray:
    """Extract a single frame at a given timestamp."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame at {timestamp_sec}s")
    return frame


def timestamp_to_sec(timestamp: str) -> float:
    """Convert MM:SS or HH:MM:SS string to seconds."""
    parts = timestamp.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Invalid timestamp format: {timestamp}")


def sec_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "output/video.mp4"
    info = get_video_info(path)
    print(f"Resolution: {info['width']}x{info['height']}")
    print(f"FPS: {info['fps']}")
    print(f"Duration: {sec_to_timestamp(info['duration_sec'])}")
