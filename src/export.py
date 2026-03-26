import subprocess
import tempfile
import os
from pathlib import Path


def export_clips(
    video_path: str,
    timestamps: list[float],
    output_path: str,
    pre_sec: float = 6.0,
    post_sec: float = 3.0,
    crossfade_sec: float = 0.5,
) -> str:
    """Cut clips around timestamps and concatenate into one highlight video.

    Crossfade transitions between clips via ffmpeg xfade/acrossfade filters.
    Asymmetric padding: more before the spike (action) than after (reaction).

    Args:
        video_path: Source video path
        timestamps: Spike timestamps in seconds
        output_path: Output video path
        pre_sec: Seconds before each spike (default 6s — catches buildup/shot)
        post_sec: Seconds after each spike (default 3s — short celebration)
        crossfade_sec: Crossfade duration between clips in seconds (0 = cut)

    Returns:
        Path to output video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Clip-Dauern berechnen (erster Clip kann kürzer sein wenn nahe Videostart)
    clip_durations = []
    clip_specs = []
    for t in timestamps:
        start = max(0.0, t - pre_sec)
        duration = (t - start) + post_sec
        clip_durations.append(duration)
        clip_specs.append((start, duration))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Clips schneiden (stream copy, schnell)
        clip_paths = []
        for i, (start, duration) in enumerate(clip_specs):
            clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{start:.3f}",
                    "-i", video_path,
                    "-t", f"{duration:.3f}",
                    "-c", "copy",
                    clip_path,
                ],
                check=True,
                capture_output=True,
            )
            clip_paths.append(clip_path)
            print(f"  Clip {i+1}/{len(timestamps)}: {start:.1f}s – {start+duration:.1f}s")

        if len(clip_paths) == 1 or crossfade_sec <= 0:
            # Kein Überblenden: einfacher Concat
            list_path = os.path.join(tmpdir, "clips.txt")
            with open(list_path, "w") as f:
                for p in clip_paths:
                    f.write(f"file '{p}'\n")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    "-c", "copy",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
        else:
            # Überblenden via xfade + acrossfade
            inputs = []
            for p in clip_paths:
                inputs += ["-i", p]

            # xfade-Offsets: kumulierte Dauer minus Überblendzeit pro Übergang
            offsets = []
            cumulative = 0.0
            for i in range(len(clip_paths) - 1):
                cumulative += clip_durations[i] - crossfade_sec
                offsets.append(cumulative)

            n = len(clip_paths)
            v_parts = []
            a_parts = []

            # Ersten Übergang
            v_parts.append(
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_sec}:offset={offsets[0]:.3f}[v1]"
            )
            a_parts.append(f"[0:a][1:a]acrossfade=d={crossfade_sec}[a1]")

            # Weitere Übergänge
            for i in range(2, n):
                prev_v = f"v{i-1}"
                prev_a = f"a{i-1}"
                cur_v = f"v{i}" if i < n - 1 else "vout"
                cur_a = f"a{i}" if i < n - 1 else "aout"
                v_parts.append(
                    f"[{prev_v}][{i}:v]xfade=transition=fade:duration={crossfade_sec}:offset={offsets[i-1]:.3f}[{cur_v}]"
                )
                a_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={crossfade_sec}[{cur_a}]")

            # Letztes Label anpassen wenn nur 2 Clips
            if n == 2:
                v_parts[0] = v_parts[0].replace("[v1]", "[vout]")
                a_parts[0] = a_parts[0].replace("[a1]", "[aout]")

            filter_complex = "; ".join(v_parts + a_parts)

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    *inputs,
                    "-filter_complex", filter_complex,
                    "-map", "[vout]",
                    "-map", "[aout]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p",  # Instagram-kompatibel
                    output_path,
                ],
                check=True,
                capture_output=True,
            )

    return output_path
