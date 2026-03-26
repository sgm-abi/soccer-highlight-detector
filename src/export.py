import subprocess
import tempfile
import os
import shutil
from pathlib import Path

XFADE_BATCH_SIZE = 10  # Max clips per xfade pass (prevents OOM with many clips)


def export_events(
    video_path: str,
    events: list,
    output_path: str,
    crossfade_sec: float = 0.5,
    preview: bool = False,
) -> str:
    """Export HighlightEvents as a highlight video with optional crossfade.

    Args:
        video_path: Source video path
        events: List of HighlightEvent objects (with start_sec/end_sec)
        output_path: Output video path
        crossfade_sec: Crossfade duration between clips in seconds (0 = hard cut)
        preview: Insert title cards between clips for easy review (ignores crossfade)

    Returns:
        Path to output video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Video-Metadaten für Titelkarten
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    w, h, fps_frac = probe.stdout.strip().split(",")
    w, h = int(w), int(h)
    fps_num, fps_den = fps_frac.split("/")
    fps = float(fps_num) / float(fps_den)

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        clip_durations = []
        for i, e in enumerate(events):
            clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            duration = e.end_sec - e.start_sec
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{e.start_sec:.3f}",
                    "-i", video_path,
                    "-t", f"{duration:.3f}",
                    "-c", "copy",
                    clip_path,
                ],
                check=True,
                capture_output=True,
            )
            clip_paths.append(clip_path)
            clip_durations.append(duration)
            print(f"  Clip {i+1}/{len(events)}: {e.start_sec:.1f}s – {e.end_sec:.1f}s")

        if preview:
            _concat_with_title_cards(clip_paths, events, output_path, w, h, fps, tmpdir)
        else:
            _concat_with_crossfade(clip_paths, clip_durations, output_path, crossfade_sec, tmpdir)

    return output_path


def _make_title_card(text: str, output_path: str, w: int, h: int, fps: float, duration: float = 1.0):
    """Generate a black separator frame (1s). Text is printed to console only."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:size={w}x{h}:rate={fps}",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"  → {text}")


def _concat_with_title_cards(
    clip_paths: list, events: list, output_path: str,
    w: int, h: int, fps: float, tmpdir: str,
):
    """Re-encode clips to match title card format, then concat with separators."""
    from src.utils import sec_to_timestamp

    # Clips auf gleiche Codec-Parameter bringen (nötig für Concat mit Titelkarten)
    reenc_paths = []
    for i, p in enumerate(clip_paths):
        reenc = os.path.join(tmpdir, f"reenc_{i:03d}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", p,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                reenc,
            ],
            check=True, capture_output=True,
        )
        reenc_paths.append(reenc)

    all_paths = []
    for i, (clip, e) in enumerate(zip(reenc_paths, events)):
        ts = sec_to_timestamp(e.start_sec)
        card_path = os.path.join(tmpdir, f"card_{i:03d}.mp4")
        _make_title_card(f"Clip {i+1} / {len(events)}  ({ts})", card_path, w, h, fps)
        all_paths.append(card_path)
        all_paths.append(clip)

    _simple_concat(all_paths, output_path)


def _concat_with_crossfade(
    clip_paths: list,
    clip_durations: list,
    output_path: str,
    crossfade_sec: float,
    tmpdir: str,
):
    """Concatenate clips with crossfade, batching to avoid ffmpeg OOM."""
    n = len(clip_paths)
    if n == 0:
        raise ValueError("No clips to concatenate")

    if n == 1 or crossfade_sec <= 0:
        _simple_concat(clip_paths, output_path)
        return

    if n > XFADE_BATCH_SIZE:
        # Process in batches, then concat batches with hard cut
        batch_outputs = []
        for batch_idx in range(0, n, XFADE_BATCH_SIZE):
            batch_clips = clip_paths[batch_idx:batch_idx + XFADE_BATCH_SIZE]
            batch_durs = clip_durations[batch_idx:batch_idx + XFADE_BATCH_SIZE]
            batch_out = os.path.join(tmpdir, f"batch_{batch_idx // XFADE_BATCH_SIZE:03d}.mp4")
            if len(batch_clips) == 1:
                shutil.copy(batch_clips[0], batch_out)
            else:
                _xfade_clips(batch_clips, batch_durs, batch_out, crossfade_sec)
            batch_outputs.append(batch_out)
            print(f"  Batch {batch_idx // XFADE_BATCH_SIZE + 1}/{-(-n // XFADE_BATCH_SIZE)} done")
        _simple_concat(batch_outputs, output_path)
    else:
        _xfade_clips(clip_paths, clip_durations, output_path, crossfade_sec)


def _simple_concat(clip_paths: list, output_path: str):
    """Concatenate clips without transitions using ffmpeg concat demuxer."""
    list_path = os.path.join(os.path.dirname(clip_paths[0]), "concat.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path],
        check=True,
        capture_output=True,
    )


def _xfade_clips(
    clip_paths: list, clip_durations: list, output_path: str, crossfade_sec: float
):
    """Apply xfade transitions between a small batch of clips."""
    n = len(clip_paths)
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    offsets = []
    cumulative = 0.0
    for i in range(n - 1):
        cumulative += clip_durations[i] - crossfade_sec
        offsets.append(cumulative)

    v_parts = [f"[0:v][1:v]xfade=transition=fade:duration={crossfade_sec}:offset={offsets[0]:.3f}[v1]"]
    a_parts = [f"[0:a][1:a]acrossfade=d={crossfade_sec}[a1]"]

    for i in range(2, n):
        cur_v = "vout" if i == n - 1 else f"v{i}"
        cur_a = "aout" if i == n - 1 else f"a{i}"
        v_parts.append(f"[v{i-1}][{i}:v]xfade=transition=fade:duration={crossfade_sec}:offset={offsets[i-1]:.3f}[{cur_v}]")
        a_parts.append(f"[a{i-1}][{i}:a]acrossfade=d={crossfade_sec}[{cur_a}]")

    if n == 2:
        v_parts[0] = v_parts[0].replace("[v1]", "[vout]")
        a_parts[0] = a_parts[0].replace("[a1]", "[aout]")

    subprocess.run(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", "; ".join(v_parts + a_parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
