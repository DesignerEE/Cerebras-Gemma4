#!/usr/bin/env python3
"""
Generate realistic F1 sound effects using MiniMax Music Generation.

Learns from /home/ai/Work/claude/minimax_music_gen.py:
  - Use /v1/music_generation endpoint
  - Use is_instrumental=True (no lyrics needed)
  - Prompt the model with a sound-effect-as-music description
  - Decode hex audio response to MP3
  - Trim the generated music to a short sound-effect clip with pydub

Requires MINIMAX_API_KEY. Reads from ~/.kimi/.env if available.

Run:
    .venv/bin/python generate_f1_sounds_minimax.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import subprocess

import requests

# Load key from ~/.kimi/.env if present
ENV_FILE = Path.home() / ".kimi" / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("MINIMAX_API_KEY=") and "=" in line:
            os.environ.setdefault("MINIMAX_API_KEY", line.split("=", 1)[1].strip())

API_KEY = os.environ.get("MINIMAX_API_KEY")
API_URL = "https://api.minimax.io/v1/music_generation"
OUT_DIR = Path(__file__).parent / "static" / "sounds"

# name -> (prompt, target_duration_seconds)
SOUNDS: dict[str, tuple[str, float]] = {
    "engine_idle": (
        "Formula 1 racing car engine idling in the pit lane, low mechanical rumble, high performance V6 hybrid engine, steady RPM, motorsport ambience, instrumental sound effect",
        4.0,
    ),
    "engine_rev": (
        "Formula 1 engine accelerating from idle to high RPM, roaring racing car engine revving, mechanical power build up, motorsport sound effect, instrumental",
        4.0,
    ),
    "gear_click": (
        "Formula 1 steering wheel paddle shift click, metallic mechanical click, precise motorsport gear change, short sharp sound effect, instrumental",
        1.5,
    ),
    "light_beep": (
        "Electronic race start light beep, Formula 1 starting grid signal tone, short high pitched electronic beep, instrumental sound effect",
        1.5,
    ),
    "go_beep": (
        "Formula 1 lights out start tone, racing start signal, rising electronic tone, urgent motorsport sound effect, instrumental",
        2.0,
    ),
    "chequered": (
        "Victory fanfare, Formula 1 chequered flag celebration, triumphant brass and strings, short instrumental music cue",
        3.0,
    ),
    "radio_static": (
        "Team radio static burst, walkie talkie static noise, motorsport radio communication, short crackly sound effect, instrumental",
        1.5,
    ),
    "scout_deploy": (
        "Formula 1 car leaving pit lane, accelerating swoosh, racing car launch, fast motorsport sound effect, instrumental",
        2.5,
    ),
}


def generate_sound(name: str, prompt: str, target_duration: float) -> bytes:
    if not API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not found. Set it in ~/.kimi/.env or environment.")

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

    audio_bytes = bytes.fromhex(data["data"]["audio"])
    extra = data.get("extra_info", {})
    duration_s = extra.get("music_duration", 0) / 1000
    print(f"  raw: {len(audio_bytes)/1024:.0f} KB, {duration_s:.1f}s in {elapsed:.1f}s", flush=True)

    # Trim to target duration using ffmpeg
    tmp_path = OUT_DIR / f"{name}_raw.mp3"
    tmp_path.write_bytes(audio_bytes)
    out_path = OUT_DIR / f"{name}.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp_path),
            "-t", str(target_duration),
            "-af", "afade=t=out:st=" + str(max(0, target_duration - 0.2)) + ":d=0.2,loudnorm",
            "-b:a", "192k",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_path.unlink(missing_ok=True)
    print(f"  trimmed: {out_path.name} ({out_path.stat().st_size/1024:.0f} KB, {target_duration:.1f}s)", flush=True)
    return out_path.read_bytes()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    for name, (prompt, duration) in SOUNDS.items():
        try:
            generate_sound(name, prompt, duration)
        except Exception as e:
            print(f"  FAILED {name}: {e}", file=sys.stderr, flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
