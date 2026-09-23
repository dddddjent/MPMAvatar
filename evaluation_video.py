"""Encode prediction, reference and side-by-side evaluation videos."""

from pathlib import Path
import subprocess


def encode_comparison(directory: Path, frames: list[int], fps: int = 25) -> None:
    """Keep prediction on the left and reference on the right at source FPS."""
    assert frames == list(range(frames[0], frames[0] + len(frames)))
    for name in ("pred", "gt"):
        for frame in frames:
            assert (directory / name / f"{frame:04d}.png").is_file()
        subprocess.run([
            "ffmpeg", "-nostdin", "-n", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-start_number", str(frames[0]),
            "-i", str(directory / name / "%04d.png"), "-frames:v", str(len(frames)),
            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(directory / f"{name}.mp4"),
        ], check=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-n", "-hide_banner", "-loglevel", "error",
        "-i", str(directory / "pred.mp4"), "-i", str(directory / "gt.mp4"),
        "-filter_complex", "hstack", "-an", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "comparison.mp4"),
    ], check=True)
