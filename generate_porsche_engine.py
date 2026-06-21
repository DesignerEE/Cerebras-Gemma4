#!/usr/bin/env python3
"""
Generate more realistic Porsche F1 engine sound loops using MiniMax Music.

MiniMax has no dedicated sound-effects endpoint, so we ask the music model for
instrumental "pure engine sound" audio, then trim and crossfade it into
seamless loops that the browser can pitch-shift for idle / rev / high RPM.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import subprocess
import requests

ENV_FILE = Path.home() / ".kimi" / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("MINIMAX_API_KEY="):
            os.environ.setdefault("MINIMAX_API_KEY", line.split("=", 1)[1].strip())

API_KEY = os.environ.get("MINIMAX_API_KEY")
API_URL = "https://api.minimax.io/v1/music_generation"
OUT_DIR = Path(__file__).parent / "static" / "sounds"

# name -> (prompt, loop_duration_seconds)
SOUNDS: dict[str, tuple[str, float]] = {
    "porsche_engine_idle": (
        "Porsche Formula 1 V6 turbo hybrid engine idling in the pit lane, "
        "close microphone raw exhaust note, deep low-end mechanical rumble, "
        "steady high-performance racing RPM, no music, no melody, no rhythm, "
        "pure engine sound, motorsport sound design, instrumental",
        6.0,
    ),
    "porsche_engine_rev": (
        "Porsche Formula 1 V6 turbo hybrid engine accelerating hard from idle "
        "to redline, close microphone raw exhaust scream, high pitched racing "
        "engine roaring, mechanical power build up, exhaust pop, no music, "
        "no melody, no rhythm, pure engine sound, motorsport sound design, instrumental",
        6.0,
    ),
    "porsche_engine_high": (
        "Porsche Formula 1 V6 turbo hybrid engine at full throttle down the "
        "main straight, close microphone high RPM engine scream, intense "
        "racing exhaust note, constant high pitched wail, no music, no melody, "
        "no rhythm, pure engine sound, motorsport sound design, instrumental",
        6.0,
    ),
}


def generate(name: str, prompt: str, target: float) -> Path:
    if not API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not found")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "music-2.6-free",
        "prompt": prompt,
        "is_instrumental": True,
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "mp3",
        },
    }

    print(f"Generating {name}...", flush=True)
    start = time.time()
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=600)
    elapsed = time.time() - start

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    base = data.get("base_resp", {})
    if base.get("status_code") != 0 or not data.get("data", {}).get("audio"):
        raise RuntimeError(f"API error: {base}")

    raw_bytes = bytes.fromhex(data["data"]["audio"])
    extra = data.get("extra_info", {})
    duration_s = extra.get("music_duration", 0) / 1000
    print(f"  raw: {len(raw_bytes)/1024:.0f} KB, {duration_s:.1f}s in {elapsed:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{name}_raw.mp3"
    raw_path.write_bytes(raw_bytes)
    out_path = OUT_DIR / f"{name}.mp3"

    # Pick the middle `target` seconds of the raw clip (usually the most stable),
    # then apply high-pass to remove mud, low-pass to tame music artifacts,
    # fade in/out, and loudness-normalize.
    start_sec = max(0.0, (duration_s - target) / 2)
    fade_in = 0.15
    fade_out = 0.25
    fade_out_start = max(0.0, target - fade_out)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-ss", str(start_sec), "-t", str(target),
            "-af",
            f"highpass=f=60,lowpass=f=10000,"
            f"afade=t=in:ss=0:d={fade_in},"
            f"afade=t=out:st={fade_out_start}:d={fade_out},"
            f"loudnorm=I=-14:TP=-1.5:LRA=4",
            "-b:a", "192k",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw_path.unlink(missing_ok=True)
    print(f"  loop: {out_path.name} ({out_path.stat().st_size/1024:.0f} KB, {target:.1f}s)", flush=True)
    return out_path


def main():
    for name, (prompt, duration) in SOUNDS.items():
        try:
            generate(name, prompt, duration)
        except Exception as e:
            print(f"  FAILED {name}: {e}", file=sys.stderr, flush=True)
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
