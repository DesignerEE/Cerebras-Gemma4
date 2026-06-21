#!/usr/bin/env python3
"""
Generate realistic F1-style sound effects as WAV files.

Minimax public API is text-to-speech / music only — it does not have a
dedicated sound-effects endpoint. This script procedurally generates
high-quality racing sounds using numpy + scipy, no API key required.

Run:
    .venv/bin/python generate_f1_sounds.py

Outputs to static/sounds/:
    engine_idle.wav   - low idling V6 hybrid rumble
    engine_rev.wav    - accelerating engine sweep
    gear_click.wav    - mechanical paddle-shift click
    light_beep.wav    - start-light beep
    go_beep.wav       - lights-out go tone
    chequered.wav     - victory fanfare
    radio_static.wav  - team radio static + click
    scout_deploy.wav  - scout "leaving pits" swoosh
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
OUT_DIR = Path(__file__).parent / "static" / "sounds"


def save_wav(name: str, data: np.ndarray):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Normalize to 16-bit
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(OUT_DIR / name, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"  {name}")


def silence(duration: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration))


def fade(data: np.ndarray, attack: float = 0.01, release: float = 0.01) -> np.ndarray:
    n = len(data)
    atk = int(SAMPLE_RATE * attack)
    rel = int(SAMPLE_RATE * release)
    env = np.ones(n)
    if atk > 0:
        env[:atk] = np.linspace(0, 1, atk)
    if rel > 0:
        env[-rel:] = np.linspace(1, 0, rel)
    return data * env


def tone(freq: float, duration: float, vol: float = 0.5, wave: str = "sine") -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    if wave == "sine":
        return vol * np.sin(2 * np.pi * freq * t)
    if wave == "square":
        return vol * np.sign(np.sin(2 * np.pi * freq * t))
    if wave == "saw":
        return vol * (2 * (t * freq - np.floor(t * freq + 0.5)))
    if wave == "triangle":
        return vol * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    return vol * np.sin(2 * np.pi * freq * t)


def noise(duration: float, vol: float = 0.3, color: str = "white") -> np.ndarray:
    n = int(SAMPLE_RATE * duration)
    if color == "white":
        return vol * (np.random.random(n) * 2 - 1)
    if color == "pink":
        # Simple pink noise approximation
        white = np.random.randn(n)
        pink = np.cumsum(white)
        pink = pink / np.max(np.abs(pink))
        return vol * pink
    return vol * (np.random.random(n) * 2 - 1)


def engine_idle() -> np.ndarray:
    """Low idling F1 engine rumble with harmonics and subtle modulation."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    base = 80
    signal = np.zeros_like(t)
    # Harmonics
    for h, amp in [(1, 0.35), (2, 0.25), (3, 0.15), (4, 0.08), (6, 0.05)]:
        signal += amp * np.sin(2 * np.pi * base * h * t)
    # Modulation (engine lope)
    lfo = 1 + 0.04 * np.sin(2 * np.pi * 12 * t)
    signal *= lfo
    # Add mechanical grit
    grit = 0.05 * noise(duration, color="pink")
    signal = fade(signal + grit, 0.05, 0.2)
    return signal


def engine_rev() -> np.ndarray:
    """Engine RPM sweep from idle to redline and back."""
    duration = 2.5
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # RPM curve: idle -> redline -> idle
    rpm = 80 + 700 * np.sin(np.pi * t / duration) ** 2
    signal = np.zeros(n)
    for h in [1, 2, 3, 4, 6]:
        signal += 0.2 * np.sin(2 * np.pi * np.cumsum(rpm * h) / SAMPLE_RATE)
    # Add exhaust pops
    pops = noise(duration, vol=0.03) * (1 + np.sin(2 * np.pi * 30 * t))
    signal = fade(signal + pops, 0.05, 0.3)
    return signal


def gear_click() -> np.ndarray:
    """Mechanical paddle shift click."""
    click = tone(1200, 0.04, vol=0.5, wave="square") * np.linspace(1, 0, int(SAMPLE_RATE * 0.04))
    click2 = tone(2200, 0.03, vol=0.3, wave="square") * np.linspace(1, 0, int(SAMPLE_RATE * 0.03))
    out = np.concatenate([click, silence(0.02), click2])
    return fade(out, 0.0, 0.01)


def light_beep() -> np.ndarray:
    """Short start-light beep."""
    return fade(tone(880, 0.12, vol=0.4, wave="sine"), 0.01, 0.05)


def go_beep() -> np.ndarray:
    """Rising 'GO' tone."""
    duration = 0.5
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 440 * np.exp(3 * t / duration)
    signal = 0.4 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    return fade(signal, 0.01, 0.1)


def chequered() -> np.ndarray:
    """Victory fanfare arpeggio."""
    notes = [523.25, 659.25, 783.99, 1046.50]  # C major
    parts = []
    for freq in notes:
        parts.append(fade(tone(freq, 0.18, vol=0.35, wave="square"), 0.01, 0.05))
        parts.append(silence(0.04))
    return np.concatenate(parts)


def radio_static() -> np.ndarray:
    """Radio static burst with a click."""
    static = noise(0.12, vol=0.15, color="pink")
    click = tone(1800, 0.03, vol=0.3, wave="square") * np.linspace(1, 0, int(SAMPLE_RATE * 0.03))
    return fade(np.concatenate([static[: len(static) // 2], click, static[len(static) // 2 :]]), 0.0, 0.05)


def scout_deploy() -> np.ndarray:
    """Rising swoosh for scout deployment."""
    duration = 0.35
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 300 * np.exp(4 * t / duration)
    signal = 0.35 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    swoosh = noise(duration, vol=0.08) * np.linspace(0, 1, n)
    return fade(signal + swoosh, 0.01, 0.08)


def main():
    print(f"Generating F1 sounds to {OUT_DIR}...")
    sounds = {
        "engine_idle.wav": engine_idle(),
        "engine_rev.wav": engine_rev(),
        "gear_click.wav": gear_click(),
        "light_beep.wav": light_beep(),
        "go_beep.wav": go_beep(),
        "chequered.wav": chequered(),
        "radio_static.wav": radio_static(),
        "scout_deploy.wav": scout_deploy(),
    }
    for name, data in sounds.items():
        save_wav(name, data)
    print("Done.")


if __name__ == "__main__":
    main()
