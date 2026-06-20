#!/usr/bin/env python3
"""
generate_realistic_sounds.py - Realistic F1 sound effects via physical modeling.

Pure numpy/scipy synthesis (no API, no key, no cost). Replaces the music-model
approach with first-principles engine physics:

  ENGINE (1.6L V6 turbo hybrid, 90 deg bank angle, even-firing split-pin crank):
    * Firing train: 3 firings per crank revolution at 120 deg spacing
    * Exhaust acoustic resonance: 5-mode damped-sinusoid impulse response
    * Turbocharger compressor whine (1.5-2.5 kHz, throttle-modulated)
    * MGU-K electric motor whine (6-10 kHz, ERS-modulated)
    * Intake plenum Helmholtz rumble (55-180 Hz band-limited noise)
    * Valvetrain ticks (cam-frequency impulses)
    * Pit-lane convolution reverb (synthetic IR with early reflections)

  SFX:
    * gear_click   - paddle-shift: broadband transient + mesh whine + low thunk
    * light_beep   - pure 880 Hz sine, clean ADSR
    * go_beep      - rising exponential sweep (lights-out)
    * chequered    - brass-style C major arpeggio via filtered sawtooth
    * radio_static - telephone-band noise + PL tone + bookend clicks
    * scout_deploy - doppler pitch swoosh with band-limited air rush

Output: static/sounds/*.wav -> ffmpeg -> *.mp3 (192 kbps).
Run:    .venv/bin/python generate_realistic_sounds.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import iirfilter, lfilter, convolve

SR = 44100
OUT_DIR = Path(__file__).parent / "static" / "sounds"


# ---------------------------------------------------------------------------
# Output + envelope helpers
# ---------------------------------------------------------------------------

def save(name: str, data: np.ndarray) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.clip(data, -1.0, 1.0)
    wav_path = OUT_DIR / f"{name}.wav"
    wavfile.write(wav_path, SR, (data * 32767).astype(np.int16))
    mp3_path = OUT_DIR / f"{name}.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path),
         "-b:a", "192k", "-codec:a", "libmp3lame", str(mp3_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav_path.unlink(missing_ok=True)
    print(f"  {mp3_path.name:<22} {mp3_path.stat().st_size // 1024:>4} KB  "
          f"{len(data) / SR:.2f}s", flush=True)
    return mp3_path


def seamless_loop(signal: np.ndarray, crossfade_ms: float = 60.0) -> np.ndarray:
    """Return a signal of length len(signal) - cf whose wrap point lands on
    two ADJACENT samples of the original (so it is genuinely click-free).
    The first cf samples blend the original tail into the head."""
    cf = int(SR * crossfade_ms / 1000)
    L = len(signal)
    if cf >= L // 4:
        return signal
    ang = np.linspace(0, np.pi / 2, cf)
    fade_in = np.sin(ang) ** 2
    fade_out = np.cos(ang) ** 2
    out = np.empty(L - cf, dtype=np.float64)
    out[:cf] = signal[:cf] * fade_in + signal[-cf:] * fade_out
    out[cf:] = signal[cf:L - cf]
    return out


def normalize(signal: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(signal)) + 1e-9
    return signal * (target_peak / peak)


# ---------------------------------------------------------------------------
# Engine physical model
# ---------------------------------------------------------------------------

def exhaust_tone(rpm_curve: np.ndarray) -> np.ndarray:
    """Exhaust note via direct harmonic synthesis of the firing fundamental.

    For an even-firing V6, 3 cylinders fire per crank revolution, so the
    exhaust fundamental f0 = RPM/20. Synthesizing the harmonic stack directly
    (instead of convolving a broadband impulse train with an IR) produces a
    clean, predictable spectrum where the firing fundamental always dominates,
    matching real F1 recordings where the fundamental peak is unambiguous.

    Harmonic amplitudes are tuned against a Ferrari 2024 reference recording
    (spectral centroid 1006 Hz, fundamental at 700 Hz when RPM ~ 14000)."""
    # Per-sample firing fundamental frequency
    f0 = rpm_curve / 20.0
    # Integrate to phase (allows smooth RPM sweeps without discontinuities)
    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    # Harmonic stack: amplitudes fall off as 1/k with mild empirical shaping
    harmonics = [
        (1, 1.00),   # fundamental (firing rate)
        (2, 0.55),   # 2nd harmonic
        (3, 0.32),   # 3rd harmonic
        (4, 0.18),   # 4th harmonic
        (5, 0.10),   # 5th harmonic (rasp)
        (6, 0.05),   # 6th harmonic (crackle)
    ]
    signal = np.zeros_like(rpm_curve)
    for k, amp in harmonics:
        signal += amp * np.sin(k * phase)
    # Add subtle combustion jitter (per-firing amplitude variation) for organic
    # texture without destabilizing the spectrum. Modulate at the firing rate.
    jitter = 1.0 + 0.06 * np.sin(2.0 * np.pi * f0 * 0.5)
    return signal * jitter


def turbo_whine(duration: float, throttle: np.ndarray) -> np.ndarray:
    """Turbocharger compressor Blade Pass Frequency whine.
    Turbo spins up to ~125,000 RPM with 7-9 compressor blades → BPF = 11-15 kHz.
    This is the dominant high-frequency characteristic of the modern V6 hybrid era
    and was absent from naturally-aspirated V8/V10 engines."""
    # BPF tracks throttle (boost pressure), not crank RPM
    freq = 11000.0 + 4000.0 * throttle
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    sig = (np.sin(phase)
           + 0.45 * np.sin(2.0 * phase)
           + 0.22 * np.sin(3.0 * phase)
           + 0.10 * np.sin(4.0 * phase))
    # Compressor spools with lag (turbo inertia).
    # Real-world F1 recordings (mic + YouTube compression) show ~0% energy in
    # the 11-16 kHz band - the BPF exists physically but is attenuated by air
    # absorption at distance and rolled off by phone mics + lossy encoding.
    # Match the audible reality, not just the physics.
    lag = lfilter([0.04], [1.0, -0.96], throttle)
    return sig * (0.05 + 0.28 * lag) * 0.05


def mgu_whine(duration: float, ers_load: np.ndarray) -> np.ndarray:
    """MGU-K electric motor whine. Lower frequency than turbo due to higher pole
    count and lower max rotational speed (~50,000 RPM). Range 2-6 kHz."""
    freq = 2000.0 + 4000.0 * ers_load
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    sig = np.sin(phase) + 0.35 * np.sin(2.0 * phase)
    return sig * ers_load * 0.12


def wastegate_chatter(duration: float, throttle: np.ndarray) -> np.ndarray:
    """Wastegate chatter - signature modern-F1 crackle when throttle lifts and
    excess boost pressure is vented. Absent from NA-era engines. Emits short
    broadband noise bursts when the throttle derivative goes negative."""
    n = int(SR * duration)
    signal = np.zeros(n)
    diff = np.diff(throttle, prepend=throttle[0])
    lift_events = np.where(diff < -0.05)[0]
    for ev in lift_events:
        # A flutter of 4-7 crackle bursts over ~100 ms
        for k in range(np.random.randint(4, 8)):
            pos = ev + k * int(SR * 0.014) + np.random.randint(0, 120)
            if pos >= n:
                continue
            burst_len = int(SR * 0.009)
            burst = np.random.randn(burst_len) * np.exp(-np.arange(burst_len) / (SR * 0.0018))
            end = min(pos + burst_len, n)
            signal[pos:end] += burst[:end - pos] * 0.20
    return signal


def intake_rumble(duration: float, throttle: np.ndarray) -> np.ndarray:
    """Intake plenum Helmholtz-style band-limited noise (55-180 Hz)."""
    n = int(SR * duration)
    noise = np.random.randn(n)
    b, a = iirfilter(3, [55.0 / (SR / 2), 180.0 / (SR / 2)], btype='band', ftype='butter')
    filtered = lfilter(b, a, noise)
    return filtered * (0.15 + 0.50 * throttle) * 0.40


def valvetrain(duration: float, rpm_curve: np.ndarray) -> np.ndarray:
    """Mechanical valvetrain ticks at cam frequency (half crank speed)."""
    n = int(SR * duration)
    ticks = np.zeros(n)
    angle = 0.0
    threshold = 90.0  # valve event every 90 deg cam = 180 deg crank
    for i in range(1, n):
        angle += rpm_curve[i] / 60.0 / 2.0 * 360.0 / SR
        if angle >= threshold:
            angle -= threshold
            ticks[i] = np.random.uniform(0.4, 1.0)
    tick_ir = np.exp(-np.arange(int(SR * 0.002)) / (SR * 0.0005))
    return np.convolve(ticks, tick_ir, mode='same') * 0.10


def pit_reverb_ir(duration_ms: float = 150.0) -> np.ndarray:
    """Synthetic small-room IR for pit-lane ambience."""
    n = int(SR * duration_ms / 1000)
    t = np.arange(n) / SR
    ir = np.random.randn(n) * np.exp(-t / (duration_ms / 1000 / 4))
    # Early reflections from concrete walls
    for delay_ms, gain in [(12, 0.35), (28, 0.22), (45, 0.15), (70, 0.08)]:
        d = int(SR * delay_ms / 1000)
        if d < n:
            ir[d] += gain * (1.0 + 0.1 * np.random.randn())
    return ir / (np.max(np.abs(ir)) + 1e-9)


def apply_reverb(signal: np.ndarray, ir: np.ndarray, wet: float = 0.18) -> np.ndarray:
    wet_sig = convolve(signal, ir, mode='same') * wet
    return signal + wet_sig


# RPM + throttle curves -----------------------------------------------------

def rpm_idle(duration: float, base: float = 3500.0) -> np.ndarray:
    n = int(SR * duration)
    # Smooth brownian walk (small drift around base)
    walk = np.cumsum(np.random.randn(n))
    walk = lfilter([0.00005], [1.0, -0.99995], walk)
    walk = walk / (np.max(np.abs(walk)) + 1e-9) * 80.0
    t = np.arange(n) / SR
    lope = 30.0 * np.sin(2 * np.pi * 3 * t) + 15.0 * np.sin(2 * np.pi * 12 * t)
    return base + walk + lope


def rpm_rev_sweep(duration: float) -> np.ndarray:
    n = int(SR * duration)
    t = np.arange(n) / SR
    x = t / duration
    # Asymmetric: ramp up, hold near redline (~11,000 RPM fuel-flow limited), slow fall
    base = 3500.0 + 7500.0 * np.sin(np.pi * x) ** 2
    ripple = 200.0 * np.sin(2 * np.pi * 4 * x) * np.sin(np.pi * x)
    return base + ripple


def rpm_high(duration: float, base: float = 13500.0) -> np.ndarray:
    return rpm_idle(duration, base=base)


def throttle_idle(duration: float) -> np.ndarray:
    n = int(SR * duration)
    return np.full(n, 0.12) + 0.05 * np.random.rand(n)


def throttle_rev(duration: float) -> np.ndarray:
    n = int(SR * duration)
    t = np.arange(n) / SR
    x = t / duration
    th = np.where(x < 0.30, x / 0.30,
         np.where(x < 0.60, 1.0, 1.0 - (x - 0.60) / 0.40))
    th = np.clip(th, 0.10, 1.0)
    return th + 0.02 * np.random.rand(n)


def throttle_high(duration: float) -> np.ndarray:
    n = int(SR * duration)
    return np.full(n, 0.92) + 0.03 * np.random.rand(n)


def build_engine(rpm_curve: np.ndarray, throttle: np.ndarray, duration: float) -> np.ndarray:
    exhaust = exhaust_tone(rpm_curve)
    signal = (
        exhaust * 0.75
        + turbo_whine(duration, throttle)
        + mgu_whine(duration, throttle * 0.6) * 0.7
        + intake_rumble(duration, throttle)
        + valvetrain(duration, rpm_curve)
        + wastegate_chatter(duration, throttle)
    )
    # High-pass at 50 Hz: real recordings show ~0% sub-bass (mic + air cannot
    # capture <50 Hz engine block vibration, and YouTube HPFs it further).
    b_hp, a_hp = iirfilter(2, 50.0 / (SR / 2), btype='high', ftype='butter')
    signal = lfilter(b_hp, a_hp, signal)
    signal = apply_reverb(signal, pit_reverb_ir(), wet=0.18)
    signal = seamless_loop(signal, crossfade_ms=80.0)
    # Modern V6 hybrid is ~4-6 dB quieter than NA V8/V10 era because the turbo
    # and MGU-H absorb acoustic energy that previously escaped as exhaust note.
    return normalize(signal, target_peak=0.55)


# ---------------------------------------------------------------------------
# SFX
# ---------------------------------------------------------------------------

def sfx_gear_click() -> np.ndarray:
    """F1 paddle-shift: broadband metallic transient + mesh whine + low thunk."""
    click_n = int(SR * 0.015)
    t1 = np.arange(click_n) / SR
    click = np.random.randn(click_n) * np.exp(-t1 / 0.001)
    click *= 1.0 + 0.5 * np.sin(2 * np.pi * 1800 * t1)
    whine_n = int(SR * 0.040)
    t2 = np.arange(whine_n) / SR
    whine_freq = 1500.0 * np.exp(-3.0 * t2 / 0.04)
    whine_phase = 2 * np.pi * np.cumsum(whine_freq) / SR
    whine = 0.18 * np.sin(whine_phase) * np.exp(-t2 / 0.012)
    thunk_n = int(SR * 0.030)
    t3 = np.arange(thunk_n) / SR
    thunk = 0.35 * np.sin(2 * np.pi * 90 * t3) * np.exp(-t3 / 0.010)
    out = np.concatenate([
        click,
        np.zeros(int(SR * 0.005)),
        whine,
        np.zeros(int(SR * 0.003)),
        thunk,
        np.zeros(int(SR * 0.020)),
    ])
    return normalize(out, target_peak=0.85)


def sfx_light_beep() -> np.ndarray:
    """Pure F1 start-light beep - 880 Hz sine with clean ADSR + 3rd harmonic."""
    duration = 0.18
    n = int(SR * duration)
    t = np.arange(n) / SR
    signal = np.sin(2 * np.pi * 880 * t) + 0.05 * np.sin(2 * np.pi * 2640 * t)
    attack = int(SR * 0.003)
    decay = int(SR * 0.020)
    release = int(SR * 0.050)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    if attack + decay < n:
        env[attack:attack + decay] = np.linspace(1, 0.85, decay)
    env[-release:] *= np.linspace(1, 0, release)
    return normalize(signal * env * 0.5, target_peak=0.75)


def sfx_go_beep() -> np.ndarray:
    """F1 lights-out: rising exponential sweep 440 -> 1320 Hz."""
    duration = 0.45
    n = int(SR * duration)
    t = np.arange(n) / SR
    freq = 440 * np.exp(1.1 * t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    signal = 0.5 * (np.sin(phase) + 0.15 * np.sin(2 * phase))
    env = np.ones(n)
    env[:int(SR * 0.010)] = np.linspace(0, 1, int(SR * 0.010))
    env[-int(SR * 0.080):] = np.linspace(1, 0, int(SR * 0.080))
    return normalize(signal * env, target_peak=0.80)


def sfx_chequered() -> np.ndarray:
    """Victory fanfare: brass-style C major arpeggio via low-passed sawtooth."""
    notes = [(523.25, 0.20), (659.25, 0.20), (783.99, 0.20), (1046.50, 0.45)]
    parts = []
    for freq, dur in notes:
        n = int(SR * dur)
        t = np.arange(n) / SR
        vibrato = 1.0 + 0.005 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * np.cumsum(freq * vibrato) / SR
        saw = 2.0 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        # Brass-like lowpass cutoff tracks pitch
        b, a = iirfilter(2, (freq * 6) / (SR / 2), btype='low', ftype='butter')
        brass = lfilter(b, a, saw)
        env = np.ones(n)
        a_atk = int(SR * 0.030)
        a_rel = int(SR * 0.080)
        env[:a_atk] = np.linspace(0, 1, a_atk)
        env[-a_rel:] = np.linspace(1, 0, a_rel)
        parts.append(brass * env * 0.35)
        parts.append(np.zeros(int(SR * 0.04)))
    out = np.concatenate(parts)
    out = apply_reverb(out, pit_reverb_ir(200.0), wet=0.22)
    return normalize(out, target_peak=0.85)


def sfx_radio_static() -> np.ndarray:
    """Team radio burst: bookend clicks + telephone-band static + PL tone."""
    duration = 0.45
    n = int(SR * duration)
    t = np.arange(n) / SR
    click = np.zeros(n)
    click_n = int(SR * 0.005)
    click[:click_n] = np.random.randn(click_n) * np.exp(-np.arange(click_n) / (SR * 0.0008))
    click[-click_n:] = np.random.randn(click_n) * np.exp(-np.arange(click_n)[::-1] / (SR * 0.0008))
    noise = np.random.randn(n)
    b, a = iirfilter(3, [300.0 / (SR / 2), 3400.0 / (SR / 2)], btype='band', ftype='butter')
    static = lfilter(b, a, noise)
    ramp = int(SR * 0.04)
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    static *= env * 0.35
    pl = 0.05 * np.sin(2 * np.pi * 100 * t)
    bf, af = iirfilter(2, [800.0 / (SR / 2), 1200.0 / (SR / 2)], btype='band', ftype='butter')
    formant = lfilter(bf, af, np.random.randn(n)) * 0.15 * env
    return normalize(click + static + pl + formant, target_peak=0.75)


def sfx_scout_deploy() -> np.ndarray:
    """Pit-exit acceleration swoosh: doppler pitch + band-limited air rush."""
    duration = 0.60
    n = int(SR * duration)
    t = np.arange(n) / SR
    x = t / duration
    freq = 250.0 + 800.0 * np.sin(np.pi * x) ** 1.5
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = 0.30 * np.sin(phase)
    noise = np.random.randn(n)
    b, a = iirfilter(2, [800.0 / (SR / 2), 6000.0 / (SR / 2)], btype='band', ftype='butter')
    swoosh = lfilter(b, a, noise)
    swoosh *= np.sin(np.pi * x) ** 2 * 0.30
    env = np.ones(n)
    env[:int(SR * 0.02)] = np.linspace(0, 1, int(SR * 0.02))
    env[-int(SR * 0.10):] = np.linspace(1, 0, int(SR * 0.10))
    return normalize((tone + swoosh) * env, target_peak=0.80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating realistic F1 sounds to {OUT_DIR} ...\n", flush=True)
    duration = 6.5  # generates longer, seamless-loop trims to ~6.4 s

    print("Engines (V6 turbo hybrid physical model):", flush=True)
    save("engine_idle", build_engine(rpm_idle(duration),       throttle_idle(duration), duration))
    save("engine_rev",  build_engine(rpm_rev_sweep(duration),  throttle_rev(duration),  duration))
    save("engine_high", build_engine(rpm_high(duration),       throttle_high(duration), duration))

    print("\nSFX (improved procedural):", flush=True)
    save("gear_click",   sfx_gear_click())
    save("light_beep",   sfx_light_beep())
    save("go_beep",      sfx_go_beep())
    save("chequered",    sfx_chequered())
    save("radio_static", sfx_radio_static())
    save("scout_deploy", sfx_scout_deploy())

    print("\nDone.")


if __name__ == "__main__":
    main()
