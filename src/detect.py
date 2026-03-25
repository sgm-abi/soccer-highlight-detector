import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path


def load_model(model_path: str = "yolov8x.pt", confidence: float = 0.3):
    """Load YOLO model for standard detection."""
    return YOLO(model_path)


def load_sahi_model(model_path: str = "yolov8x.pt", confidence: float = 0.3):
    """Load YOLO model wrapped in SAHI for small object detection."""
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence,
    )


def detect_frame(model, frame: np.ndarray) -> list:
    """Run standard YOLO detection on a single frame.

    Returns list of detections: [x1, y1, x2, y2, confidence, class_id]
    """
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
            }
        )
    return detections


def filter_by_class(detections: list, class_names: list) -> list:
    """Filter detections by class name, e.g. ['person', 'sports ball']."""
    return [d for d in detections if d["class_name"] in class_names]


if __name__ == "__main__":
    import sys
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
    print(f"Found {len(relevant)} relevant objects:")
    for d in relevant:
        print(f"  {d['class_name']}: {d['confidence']:.2f}")

    for d in relevant:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = (0, 255, 0) if d["class_name"] == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{d['class_name']} {d['confidence']:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    out = "output/detection_test_sahi.jpg"
    cv2.imwrite(out, frame)
    print(f"Saved annotated frame to: {out}")
