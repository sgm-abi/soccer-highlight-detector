import subprocess
from pathlib import Path


def download_video(url: str, output_path: str = "output/video.mp4") -> str:
    """Download a video from YouTube using yt-dlp."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            url,
            "-o",
            output_path,
            "--no-playlist",
        ],
        check=True,
    )

    return output_path


def get_video_path(source: str, output_path: str = "output/video.mp4") -> str:
    """Return local path – either use existing file or download from URL.

    Args:
        source: Local file path or YouTube URL
        output_path: Where to save if downloading

    Returns:
        Path to the video file
    """
    if Path(source).exists():
        print(f"Using local file: {source}")
        return source
    else:
        print(f"Downloading from: {source}")
        return download_video(source, output_path)


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/whp_ir4SE9Q"
    path = get_video_path(source)
    print(f"Video ready at: {path}")
