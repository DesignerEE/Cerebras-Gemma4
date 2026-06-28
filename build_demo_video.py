#!/usr/bin/env python3
"""Build the VisionOps hackathon demo video from screenshots.

Audio is optional. Run with --audio to mix in background + sound effects
from demo_assets/. Default output is a silent video with burned-in captions.
"""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Shot:
    file: str                    # image filename in demo_shots/
    duration: float              # seconds
    caption: str                 # caption burned into the video
    sfx: Optional[str] = None    # audio filename in demo_assets/ (played at shot start)
    sfx_volume: float = 1.0      # sound effect volume


SHOTS = [
    Shot("01_boot.png", 4.0, "VisionOps — See the problem. Fix the problem."),
    Shot("02_vision_upload.png", 5.0, "Point any camera at the failing system."),
    Shot("03_vision_diagnosis.png", 8.0, "Gemma 4 31B sees the issue and prescribes a fix."),
    Shot("04_news_scouts.png", 6.0, "While it diagnoses, autonomous agents gather live context."),
    Shot("05_news_report.png", 7.0, "A commander agent synthesizes everything into an action plan."),
    Shot("06_race_start.png", 4.0, "When speed matters, cfire saturates Cerebras inference."),
    Shot("07_race_peak.png", 7.0, "Thousands of tokens per second, live."),
    Shot("08_headroom.png", 5.0, "Real-time telemetry from the edge."),
    Shot("09_benchmark_memory.png", 5.0, "It remembers the fastest setup for next time."),
    Shot("10_physical_alert.png", 5.0, "The dashboard acts in the physical world."),
    Shot("11_end_card.png", 4.0, "Multi-agent. Multimodal. Physically aware."),
]

OUTPUT_FILE = "demo_video.mp4"
RESOLUTION = (1920, 1080)
FPS = 30
BACKGROUND_AUDIO = "demo_assets/engine_idle.wav"
BACKGROUND_VOLUME = 0.25
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux default
# Fallback fonts to try if the above is missing
FONT_FALLBACKS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "C:/Windows/Fonts/arialbd.ttf",           # Windows
]
FADE_DURATION = 0.5  # crossfade between shots

# ─── Helpers ─────────────────────────────────────────────────────────────────


def run(cmd: list[str]) -> None:
    """Run an ffmpeg command and print it."""
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_font() -> Optional[str]:
    """Return the first available caption font."""
    candidates = [CAPTION_FONT] + FONT_FALLBACKS
    for f in candidates:
        if f and Path(f).exists():
            return f
    return None


def build_shot_segment(
    image_path: Path,
    duration: float,
    caption: str,
    output_path: Path,
    font: Optional[str],
) -> None:
    """Turn a single screenshot into an MP4 segment with optional caption."""
    width, height = RESOLUTION

    # Scale image to fit inside RESOLUTION with black padding (contain)
    video_filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        "pad={}:{}:(ow-iw)/2:(oh-ih)/2:black".format(width, height),
    ]

    if font and caption:
        # Burn caption at bottom center with semi-transparent background strip
        safe_caption = caption.replace("'", "'\\''")  # escape for shell filter string
        video_filters.append(
            f"drawtext=fontfile={font}:text='{safe_caption}':"
            f"fontcolor=white:fontsize=48:box=1:boxcolor=black@0.6:"
            f"boxborderw=20:x=(w-text_w)/2:y=h-text_h-80:line_spacing=8"
        )

    vf = ",".join(video_filters)
    # Loop image for duration; format yuv420p for compatibility.
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(image_path),
        "-vf", vf,
        "-c:v", "libx264", "-t", str(duration),
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-an", str(output_path),
    ])


def build_video_only(shots_dir: Path, tmpdir: Path, font: Optional[str]) -> Path:
    """Concatenate shot segments into one silent video."""
    segments: list[Path] = []

    for idx, shot in enumerate(SHOTS, 1):
        img = shots_dir / shot.file
        if not img.exists():
            raise FileNotFoundError(f"Missing screenshot: {img}")

        seg = Path(tmpdir) / f"seg_{idx:02d}.mp4"
        build_shot_segment(img, shot.duration, shot.caption, seg, font)
        segments.append(seg)

    # Build crossfaded concat via xfade filter
    total_shots = len(segments)
    inputs = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    filter_parts = []
    last = "[0:v]"
    for i in range(1, total_shots):
        # offset for xfade = previous output end minus fade duration
        offset = sum(SHOTS[j].duration for j in range(i)) - FADE_DURATION * i
        out_label = f"[vf{i}]" if i < total_shots - 1 else "[outv]"
        filter_parts.append(
            f"{last}[{i}:v]xfade=transition=fade:duration={FADE_DURATION}:offset={offset}{out_label}"
        )
        last = out_label

    video_output = Path(tmpdir) / "video_silent.mp4"
    run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(video_output),
    ])
    return video_output


def build_audio_mix(assets_dir: Path, tmpdir: Path) -> Path:
    """Create a mixed audio track: background loop + per-shot sound effects."""
    total_duration = sum(s.duration for s in SHOTS)

    bg = assets_dir / BACKGROUND_AUDIO
    if not bg.exists():
        print(f"Warning: background audio not found: {bg}; final video will have no audio.")
        # Return silent track
        silent = Path(tmpdir) / "audio.m4a"
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", str(total_duration), "-c:a", "aac", str(silent)])
        return silent

    # Loop/pad background to total duration, fade in/out
    bg_looped = Path(tmpdir) / "bg_looped.m4a"
    run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(bg),
        "-af", f"volume={BACKGROUND_VOLUME},afade=t=in:ss=0:d=1,afade=t=out:st={total_duration-2}:d=2",
        "-t", str(total_duration), "-c:a", "aac", str(bg_looped),
    ])

    # Build per-shot SFX delayed to start times
    sfx_inputs = ["-i", str(bg_looped)]
    delay_filters = ["[0:a]volume=1[bg]"]
    sfx_files = []
    current_time = 0.0
    for shot in SHOTS:
        if shot.sfx:
            sfx_path = assets_dir / shot.sfx
            if sfx_path.exists():
                sfx_files.append((current_time, shot.sfx, shot.sfx_volume))
            else:
                print(f"Warning: missing SFX: {sfx_path}")
        current_time += shot.duration

    if not sfx_files:
        return bg_looped

    # Add each SFX as input and delay/advolume
    for j, (start, filename, volume) in enumerate(sfx_files, start=1):
        sfx_path = assets_dir / filename
        sfx_inputs += ["-i", str(sfx_path)]
        delay_ms = int(start * 1000)
        delay_filters.append(
            f"[{j}:a]adelay={delay_ms}|{delay_ms},volume={volume}[sfx{j}]"
        )

    # Mix all: background + every delayed SFX
    sfx_labels = "".join(f"[sfx{j}]" for j in range(1, len(sfx_files) + 1))
    num_inputs = 1 + len(sfx_files)
    delay_filters.append(
        f"[bg]{sfx_labels}amix=inputs={num_inputs}:duration=first:dropout_transition=0[aout]"
    )

    audio_output = Path(tmpdir) / "audio_mixed.m4a"
    run([
        "ffmpeg", "-y", *sfx_inputs,
        "-filter_complex", ";".join(delay_filters),
        "-map", "[aout]", "-t", str(total_duration),
        "-c:a", "aac", str(audio_output),
    ])
    return audio_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VisionOps demo video from screenshots.")
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Mix in background audio and sound effects from demo_assets/",
    )
    args = parser.parse_args()

    shots_dir = Path("demo_shots")
    assets_dir = Path("demo_assets")

    if not shots_dir.exists():
        shots_dir.mkdir(parents=True)
        print(f"Created {shots_dir}/ — place screenshots there.")
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True)
        print(f"Created {assets_dir}/ — place audio files there (only used with --audio).")

    # Quick validation
    missing_images = [s.file for s in SHOTS if not (shots_dir / s.file).exists()]
    if missing_images:
        print("ERROR: missing screenshots in demo_shots/:")
        for f in missing_images:
            print(f"  - {f}")
        print("\nCapture those shots, then re-run this script.")
        return 1

    font = find_font()
    if font:
        print(f"Using caption font: {font}")
    else:
        print("WARNING: no caption font found; captions will be skipped.")

    with tempfile.TemporaryDirectory(prefix="visionops_demo_") as tmpdir:
        print("\n[1/2] Building silent video from screenshots...")
        video = build_video_only(shots_dir, Path(tmpdir), font)

        if args.audio:
            print("\n[2/2] Mixing audio and combining...")
            audio = build_audio_mix(assets_dir, Path(tmpdir))
            run([
                "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                OUTPUT_FILE,
            ])
        else:
            print("\n[2/2] Skipping audio (use --audio to add sound).")
            run(["ffmpeg", "-y", "-i", str(video), "-c:v", "copy", "-an", OUTPUT_FILE])

    print(f"\nDone: {OUTPUT_FILE}")
    print(f"Duration: {sum(s.duration for s in SHOTS)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
