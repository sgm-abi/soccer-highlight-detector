import cv2
import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HighlightEvent:
    """A detected highlight moment."""

    start_sec: float
    end_sec: float
    score: float
    reasons: list[str] = field(default_factory=list)


def detect_audio_spikes(
    video_path: str, threshold: float = 4.0, min_gap_sec: float = 15.0
) -> list[float]:
    """Detect crowd noise spikes (goals, cheering) via audio energy.

    Args:
        video_path: Path to video file
        threshold: Std deviations above mean to count as spike
        min_gap_sec: Minimum seconds between spikes

    Returns:
        List of timestamps in seconds where spikes occur
    """
    print("Analyzing audio...")
    y, sr = librosa.load(video_path, mono=True)

    # RMS energy per frame
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Spikes = deutlich über Durchschnitt
    mean, std = rms.mean(), rms.std()
    spike_mask = rms > (mean + threshold * std)
    spike_times = times[spike_mask]

    # Zu dichte Spikes zusammenfassen
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
    """Detect if many players are clustered together (e.g. corner, goal).

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

    # Für jeden Spieler: wie viele andere sind in der Nähe?
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
) -> tuple[float, list[str]]:
    """Compute highlight score for a single frame.

    Returns:
        (score, reasons) where score is 0.0-1.0
    """
    score = 0.0
    reasons = []

    if detect_ball_near_goal(detections, frame.shape):
        score += 0.4
        reasons.append("ball_near_goal")

    cluster = detect_player_cluster(detections, frame.shape)
    if cluster:
        score += 0.3
        reasons.append("player_cluster")

    if audio_spike:
        score += 0.2
        reasons.append("audio_spike")

    if prev_frame is not None and detect_fast_motion(prev_frame, frame):
        score += 0.1
        reasons.append("fast_motion")

    return score, reasons


def extract_highlights(
    video_path: str,
    sahi_model,
    score_threshold: float = 0.4,
    padding_sec: float = 3.0,
    sample_every_n_frames: int = 15,
) -> list[HighlightEvent]:
    """Run full highlight detection pipeline on a video.

    Args:
        video_path: Path to video file
        sahi_model: Loaded SAHI model from detect.py
        score_threshold: Minimum score to count as highlight
        padding_sec: Seconds to add before/after highlight
        sample_every_n_frames: Only analyze every Nth frame (speed vs accuracy)

    Returns:
        List of HighlightEvent objects
    """
    from src.detect import detect_frame_sahi, filter_by_class

    # Audio-Spikes vorab analysieren
    audio_spikes = detect_audio_spikes(video_path)
    print(
        f"Found {len(audio_spikes)} audio spikes: {[f'{t:.1f}s' for t in audio_spikes]}"
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    highlight_frames = []
    prev_frame = None
    frame_idx = 0

    print(f"Scanning {total_frames} frames (every {sample_every_n_frames})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            timestamp = frame_idx / fps

            # Audio-Spike in ±2s Fenster?
            audio_spike = any(abs(t - timestamp) < 2.0 for t in audio_spikes)

            # Detektionen
            detections = detect_frame_sahi(sahi_model, frame)
            relevant = filter_by_class(detections, ["person", "sports ball"])

            # Score berechnen
            score, reasons = score_frame(relevant, frame, prev_frame, audio_spike)

            if score >= score_threshold:
                highlight_frames.append((timestamp, score, reasons))

            prev_frame = frame.copy()

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames")

    cap.release()

    # Zusammenhängende Highlight-Frames zu Events zusammenfassen
    events = merge_highlight_frames(highlight_frames, padding_sec)
    print(f"Found {len(events)} highlight events")
    return events


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
            start, best_score, all_reasons = timestamp, score, reasons
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


if __name__ == "__main__":
    import sys
    import os
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import get_video_info, sec_to_timestamp
    from src.export import export_clips

    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="output/video.mp4")
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip YOLO: detect audio spikes and export clips directly",
    )
    parser.add_argument(
        "--pre", type=float, default=6.0, help="Seconds before spike (default 6s)"
    )
    parser.add_argument(
        "--post", type=float, default=3.0, help="Seconds after spike (default 3s)"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Audio spike threshold (std devs)"
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=0.5,
        help="Crossfade duration in seconds (0 = cut)",
    )
    parser.add_argument("--output", default="output/highlights.mp4")
    args = parser.parse_args()

    video_path = args.video
    info = get_video_info(video_path)
    fps = info["fps"]

    if args.audio_only:
        print("=== Audio-only mode ===")
        spikes = detect_audio_spikes(video_path, threshold=args.threshold)
        print(f"Found {len(spikes)} audio spikes")
        for i, t in enumerate(spikes):
            print(f"  [{i+1}] {sec_to_timestamp(t)}")

        print(
            f"\nExporting {len(spikes)} clips (-{args.pre}s/+{args.post}s) → {args.output}"
        )
        export_clips(
            video_path,
            spikes,
            args.output,
            pre_sec=args.pre,
            post_sec=args.post,
            crossfade_sec=args.crossfade,
        )
        print(f"Done: {args.output}")

    else:
        from src.detect import load_sahi_model

        print("Loading model...")
        sahi_model = load_sahi_model()

        print("Extracting highlights...")
        events = extract_highlights(video_path, sahi_model)

        print("\n=== HIGHLIGHTS ===")
        for i, e in enumerate(events):
            print(
                f"  [{i+1}] {sec_to_timestamp(e.start_sec)} → "
                f"{sec_to_timestamp(e.end_sec)} "
                f"(score: {e.score:.2f}, reasons: {e.reasons})"
            )
