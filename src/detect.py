import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
from sklearn.cluster import KMeans


def _best_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_path: str = "yolov8x.pt", confidence: float = 0.3):
    """Load YOLO model for standard detection."""
    model = YOLO(model_path)
    model.to(_best_device())
    return model


def load_sahi_model(model_path: str = "yolov8x.pt", confidence: float = 0.3):
    """Load YOLO model wrapped in SAHI for small object detection."""
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence,
        device=_best_device(),
    )


def detect_frame(model, frame: np.ndarray) -> list:
    """Run standard YOLO detection on a single frame."""
    results = model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": model.names[cls],
                    "box_color": (0, 255, 0),
                    "team": None,
                }
            )
    return detections


def detect_frame_sahi(
    sahi_model, frame: np.ndarray, slice_size: int = 640, overlap: float = 0.2
) -> list:
    """Run SAHI sliced detection on a single frame (better for small objects)."""
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    detections = []
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        detections.append(
            {
                "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                "confidence": obj.score.value,
                "class_id": obj.category.id,
                "class_name": obj.category.name,
                "box_color": (0, 255, 0),
                "team": None,
            }
        )
    return detections


def filter_by_class(detections: list, class_names: list) -> list:
    """Filter detections by class name."""
    return [d for d in detections if d["class_name"] in class_names]


def is_on_pitch(
    bbox: list,
    frame_shape: tuple,
    x_margin: float = 0.05,
    y_top: float = 0.1,
    y_bottom: float = 0.95,
) -> bool:
    """Check if a detection is within the pitch area (excludes sideline persons).

    Args:
        bbox: [x1, y1, x2, y2]
        frame_shape: (height, width, channels)
        x_margin: fraction to exclude on left/right edges
        y_top: fraction to exclude at top
        y_bottom: fraction to exclude at bottom
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return x_margin * w < cx < (1 - x_margin) * w and y_top * h < cy < y_bottom * h


def get_dominant_color(frame: np.ndarray, bbox: list) -> tuple:
    """Extract dominant color from upper half of bounding box (jersey area)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mid_y = y1 + (y2 - y1) // 2
    crop = frame[y1:mid_y, x1:x2]

    if crop.size == 0:
        return (128, 128, 128)

    pixels = crop.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        pixels, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    return tuple(int(c) for c in centers[0])


def assign_teams(detections: list, frame: np.ndarray) -> list:
    """Assign team colors to player detections using K-Means clustering."""
    players = [d for d in detections if d["class_name"] == "person"]
    if len(players) < 2:
        return detections

    colors = []
    for p in players:
        color = get_dominant_color(frame, p["bbox"])
        colors.append(color)

    km = KMeans(n_clusters=2, n_init=10)
    labels = km.fit_predict(colors)

    team_colors = [(0, 0, 255), (255, 0, 0)]  # rot, blau (BGR)
    for i, p in enumerate(players):
        p["team"] = labels[i]
        p["box_color"] = team_colors[labels[i]]

    return detections


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import extract_frame

    video_path = sys.argv[1] if len(sys.argv) > 1 else "output/video.mp4"
    timestamp = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0

    print("Loading SAHI model...")
    sahi_model = load_sahi_model()

    print(f"Extracting frame at {timestamp}s...")
    frame = extract_frame(video_path, timestamp)

    print("Running SAHI detection...")
    detections = detect_frame_sahi(sahi_model, frame)
    relevant = filter_by_class(detections, ["person", "sports ball"])

    print("Assigning teams...")
    relevant = assign_teams(relevant, frame)

    # Statistik
    teams = {}
    for d in relevant:
        t = d.get("team")
        teams[t] = teams.get(t, 0) + 1
    print(f"Team distribution: {teams}")

    # Frame annotieren
    for d in relevant:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = d["box_color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        team_label = {None: "ball", -1: "side", 99: "ref"}.get(
            d["team"], f"T{d['team']}"
        )
        label = f"{team_label} {d['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out = f"output/detection_test_team_enhanced_{timestamp}.jpg"
    cv2.imwrite(out, frame)
    print(f"Saved annotated frame to: {out}")
