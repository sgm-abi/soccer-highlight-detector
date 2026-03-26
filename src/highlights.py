import cv2
import numpy as np
import librosa
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HighlightEvent:
    """A detected highlight moment."""

    start_sec: float
    end_sec: float
    score: float
    reasons: list[str] = field(default_factory=list)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_audio_spikes(
    video_path: str, threshold: float = 2.0, min_gap_sec: float = 5.0
) -> list[float]:
    """Detect crowd noise spikes (goals, cheering) via audio energy.

    Args:
        video_path: Path to video file
        threshold: Std deviations above mean to count as spike
        min_gap_sec: Minimum seconds between spikes

    Returns:
        List of timestamps in seconds where spikes occur
    """
    import tempfile, subprocess

    print("Analyzing audio...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "22050", "-vn", tmp_wav],
        check=True,
        capture_output=True,
    )
    y, sr = librosa.load(tmp_wav, mono=True)
    __import__("os").unlink(tmp_wav)

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    mean, std = rms.mean(), rms.std()
    spike_mask = rms > (mean + threshold * std)
    spike_times = times[spike_mask]

    filtered = []
    last = -min_gap_sec
    for t in spike_times:
        if t - last >= min_gap_sec:
            filtered.append(float(t))
            last = t

    return filtered


def detect_player_cluster(
    detections: list,
    frame_shape: tuple,
    min_players: int = 4,
    cluster_radius: float = 0.15,
) -> Optional[tuple]:
    """Detect if many players are clustered together.

    Args:
        detections: List of detection dicts from detect.py
        frame_shape: (height, width, channels)
        min_players: Minimum players in cluster to count
        cluster_radius: Fraction of frame width for cluster radius

    Returns:
        (cx, cy) center of cluster in pixels, or None
    """
    players = [d for d in detections if d["class_name"] == "person"]
    if len(players) < min_players:
        return None

    h, w = frame_shape[:2]
    radius_px = cluster_radius * w

    centers = np.array(
        [
            [(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]
            for d in players
        ]
    )

    for i, c in enumerate(centers):
        dists = np.linalg.norm(centers - c, axis=1)
        nearby = np.sum(dists < radius_px)
        if nearby >= min_players:
            cluster_players = centers[dists < radius_px]
            cx, cy = cluster_players.mean(axis=0)
            return (float(cx), float(cy))

    return None


def detect_ball_near_goal(
    detections: list, frame_shape: tuple, goal_margin: float = 0.15
) -> bool:
    """Check if ball is near either goal (left or right edge of frame).

    Args:
        detections: List of detection dicts
        frame_shape: (height, width, channels)
        goal_margin: Fraction of frame width to consider as goal zone

    Returns:
        True if ball is in goal zone
    """
    h, w = frame_shape[:2]
    balls = [d for d in detections if d["class_name"] == "sports ball"]

    for ball in balls:
        cx = (ball["bbox"][0] + ball["bbox"][2]) / 2
        if cx < goal_margin * w or cx > (1 - goal_margin) * w:
            return True
    return False


def detect_fast_motion(
    prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 5.0
) -> bool:
    """Detect fast motion between two frames using optical flow magnitude.

    Args:
        prev_frame: Previous video frame
        curr_frame: Current video frame
        threshold: Mean optical flow magnitude threshold

    Returns:
        True if fast motion detected
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(magnitude.mean()) > threshold


def score_frame(
    detections: list,
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    audio_spike: bool,
    config: dict,
) -> tuple[float, list[str]]:
    """Compute highlight score for a single frame using enabled detectors.

    Returns:
        (score, reasons) where score is 0.0-1.0
    """
    score = 0.0
    reasons = []
    detectors = config["detectors"]

    if detectors["ball_near_goal"]["enabled"]:
        cfg = detectors["ball_near_goal"]
        if detect_ball_near_goal(
            detections, frame.shape, goal_margin=cfg["goal_margin"]
        ):
            score += cfg["weight"]
            reasons.append("ball_near_goal")

    if detectors["player_cluster"]["enabled"]:
        cfg = detectors["player_cluster"]
        cluster = detect_player_cluster(
            detections,
            frame.shape,
            min_players=cfg["min_players"],
            cluster_radius=cfg["cluster_radius"],
        )
        if cluster:
            score += cfg["weight"]
            reasons.append("player_cluster")

    if detectors["audio_spike"]["enabled"] and audio_spike:
        score += detectors["audio_spike"]["weight"]
        reasons.append("audio_spike")

    if detectors["fast_motion"]["enabled"] and prev_frame is not None:
        cfg = detectors["fast_motion"]
        if detect_fast_motion(prev_frame, frame, threshold=cfg["threshold"]):
            score += cfg["weight"]
            reasons.append("fast_motion")

    return score, reasons


def merge_highlight_frames(
    highlight_frames: list, padding_sec: float = 3.0, merge_gap_sec: float = 4.0
) -> list[HighlightEvent]:
    """Merge nearby highlight frames into HighlightEvent objects.

    Args:
        highlight_frames: List of (timestamp, score, reasons)
        padding_sec: Seconds to pad each event
        merge_gap_sec: Merge events closer than this many seconds

    Returns:
        List of merged HighlightEvent objects
    """
    if not highlight_frames:
        return []

    events = []
    start, best_score, all_reasons = highlight_frames[0]
    end = start

    for timestamp, score, reasons in highlight_frames[1:]:
        if timestamp - end <= merge_gap_sec:
            end = timestamp
            best_score = max(best_score, score)
            all_reasons.extend(r for r in reasons if r not in all_reasons)
        else:
            events.append(
                HighlightEvent(
                    start_sec=max(0, start - padding_sec),
                    end_sec=end + padding_sec,
                    score=best_score,
                    reasons=all_reasons,
                )
            )
            start, best_score, all_reasons = timestamp, score, list(reasons)
            end = start

    events.append(
        HighlightEvent(
            start_sec=max(0, start - padding_sec),
            end_sec=end + padding_sec,
            score=best_score,
            reasons=all_reasons,
        )
    )

    return events


def extract_highlights(
    video_path: str, sahi_model, config_path: str = "config.yaml"
) -> list[HighlightEvent]:
    """Run full highlight detection pipeline on a video.

    Args:
        video_path: Path to video file
        sahi_model: Loaded SAHI model from detect.py
        config_path: Path to config YAML file

    Returns:
        List of HighlightEvent objects
    """
    from src.detect import detect_frame_sahi, filter_by_class

    config = load_config(config_path)
    vid_cfg = config["video"]
    det_cfg = config["detectors"]

    # Audio nur wenn aktiviert
    audio_spikes = []
    if det_cfg["audio_spike"]["enabled"]:
        audio_spikes = detect_audio_spikes(
            video_path,
            threshold=det_cfg["audio_spike"]["threshold"],
            min_gap_sec=det_cfg["audio_spike"]["min_gap_sec"],
        )
        print(
            f"Found {len(audio_spikes)} audio spikes: "
            f"{[f'{t:.1f}s' for t in audio_spikes]}"
        )

    # Wenn keine visuellen Detektoren aktiv: direkt aus Audio-Spikes Events bauen
    visual_detectors = ["ball_near_goal", "player_cluster", "fast_motion"]
    need_visual = any(det_cfg[d]["enabled"] for d in visual_detectors)

    if not need_visual:
        print("Visual detectors disabled — using audio spikes directly.")
        highlight_frames = [(t, 1.0, ["audio_spike"]) for t in audio_spikes]
        return merge_highlight_frames(
            highlight_frames,
            padding_sec=vid_cfg["padding_sec"],
            merge_gap_sec=vid_cfg["merge_gap_sec"],
        )

    from src.detect import detect_frame_sahi, filter_by_class

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    highlight_frames = []
    prev_frame = None
    frame_idx = 0

    print(
        f"Scanning {total_frames} frames "
        f"(every {vid_cfg['sample_every_n_frames']})..."
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % vid_cfg["sample_every_n_frames"] == 0:
            timestamp = frame_idx / fps
            audio_spike = any(abs(t - timestamp) < 2.0 for t in audio_spikes)

            detections = detect_frame_sahi(sahi_model, frame)
            relevant = filter_by_class(detections, ["person", "sports ball"])

            score, reasons = score_frame(
                relevant, frame, prev_frame, audio_spike, config
            )

            if score >= vid_cfg["score_threshold"]:
                highlight_frames.append((timestamp, score, reasons))

            prev_frame = frame.copy()

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Progress: {frame_idx}/{total_frames}")

    cap.release()

    return merge_highlight_frames(
        highlight_frames,
        padding_sec=vid_cfg["padding_sec"],
        merge_gap_sec=vid_cfg["merge_gap_sec"],
    )


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.detect import load_sahi_model
    from src.utils import sec_to_timestamp
    from src.export import export_events

    video_path = sys.argv[1] if len(sys.argv) > 1 else "output/video.mp4"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    print(f"Video:  {video_path}")
    print(f"Config: {config_path}")

    config = load_config(config_path)
    visual_detectors = ["ball_near_goal", "player_cluster", "fast_motion"]
    need_visual = any(config["detectors"][d]["enabled"] for d in visual_detectors)

    sahi_model = None
    if need_visual:
        print("Loading model...")
        sahi_model = load_sahi_model()

    print("Extracting highlights...")
    events = extract_highlights(video_path, sahi_model, config_path)

    print("\n=== HIGHLIGHTS ===")
    for i, e in enumerate(events):
        print(
            f"  [{i+1}] {sec_to_timestamp(e.start_sec)} → "
            f"{sec_to_timestamp(e.end_sec)} "
            f"(score: {e.score:.2f}, reasons: {e.reasons})"
        )

    exp_cfg = config.get("export", {})
    crossfade_sec = exp_cfg.get("crossfade_sec", 0.5)

    # Dateiname aus aktiven Detektoren ableiten, außer explizit angegeben
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        detector_labels = {
            "audio_spike": "audio",
            "ball_near_goal": "ball",
            "player_cluster": "cluster",
            "fast_motion": "motion",
        }
        active = [
            detector_labels[d]
            for d in detector_labels
            if config["detectors"].get(d, {}).get("enabled", False)
        ]
        suffix = "_".join(active) if active else "highlights"
        default_out = exp_cfg.get("output", "output/highlights.mp4")
        base_dir = os.path.dirname(default_out)
        output_path = os.path.join(base_dir, f"highlights_{suffix}.mp4")

    preview = exp_cfg.get("preview", False)
    print(
        f"\nExporting {len(events)} clips {'[preview]' if preview else ''} → {output_path}"
    )
    export_events(
        video_path, events, output_path, crossfade_sec=crossfade_sec, preview=preview
    )
    print(f"Done: {output_path}")
