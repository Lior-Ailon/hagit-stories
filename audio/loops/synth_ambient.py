"""Original, synthetic ambient loops for the story scenes (no third-party audio, no license needed).

Usage:  python -I synth_ambient.py <output_dir>
Needs:  numpy, scipy, imageio-ffmpeg   (pip install numpy scipy imageio-ffmpeg)

Every loop is made seamless: a longer take is generated and its tail is cross-faded
into its head, so the loop point has no click and no level jump.

  crickets_loop  evening / night: a soft chorus of crickets
  field_loop     open country: gentle breeze + grass rustle (much quieter than wind_loop)
  dream_loop     Jacob's ladder: airy shimmering pad with sparse high chimes
  stream_loop    a well / small brook: soft water babble
  tent_loop      inside the tent / home: warm room tone + faint cloth rustles
  hearth_loop    cooking fire: low flame rumble + crackles (longer than fire_loop)
"""
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

import imageio_ffmpeg

SR = 44100
LOOP_SEC = 24.0
XFADE_SEC = 3.0
FF = imageio_ffmpeg.get_ffmpeg_exe()
rng = np.random.default_rng(2026)


# ---------- helpers ----------
def noise(n):
    return rng.standard_normal(n)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x)


def smooth(n, rate_hz):
    """Slow random curve in [0, 1] (about `rate_hz` new targets per second)."""
    k = int(n / SR * rate_hz) + 4
    pts = rng.random(k)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)


def reverb(x, secs=2.2, wet=0.35):
    m = int(secs * SR)
    ir = noise(m) * np.exp(-np.arange(m) / (SR * secs / 4.5))
    ir = lp(ir, 5500)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    y = fftconvolve(x, ir)[: len(x)]
    return (1 - wet) * x + wet * y * np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))


def make_loop(x):
    """Cross-fade the tail into the head so the loop point is seamless."""
    L = int(LOOP_SEC * SR)
    F = int(XFADE_SEC * SR)
    out = x[:L].copy()
    w = np.linspace(0, np.pi / 2, F)
    out[:F] = x[:F] * np.sin(w) + x[L:L + F] * np.cos(w)
    return out


def normalise(x, rms_db, peak_db=-7.0):
    x = x - np.mean(x)
    x = x * (10 ** (rms_db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12))
    ceiling = 10 ** (peak_db / 20)
    return ceiling * np.tanh(x / ceiling)


def total_len():
    return int((LOOP_SEC + XFADE_SEC) * SR)


# ---------- the sounds ----------
def crickets():
    n = total_len()
    secs = n / SR
    out = np.zeros(n)
    for _ in range(6):
        f = rng.uniform(4300, 5300)
        amp = rng.uniform(0.4, 1.0)
        pos = rng.uniform(0, 0.6)
        while pos < secs - 0.3:
            pulses = int(rng.integers(3, 5))
            for p in range(pulses):
                i = int((pos + p * 0.032) * SR)
                m = int(0.024 * SR)
                if i + m >= n:
                    break
                tt = np.arange(m) / SR
                env = np.sin(np.pi * tt / 0.024) ** 2
                out[i:i + m] += amp * env * np.sin(2 * np.pi * f * tt) * (0.85 + 0.15 * p / pulses)
            pos += rng.uniform(0.35, 0.7)
    out *= 0.55 + 0.45 * smooth(n, 0.35)
    out += 0.08 * lp(noise(n), 420) * (0.5 + 0.5 * smooth(n, 0.1))
    return normalise(make_loop(out), -27.0)


def field():
    n = total_len()
    breeze = lp(noise(n), 650) * (0.30 + 0.70 * smooth(n, 0.12) ** 2)
    rustle = bp(noise(n), 2200, 6500) * (smooth(n, 0.22) ** 3) * 0.22
    body = lp(noise(n), 110) * 0.35
    return normalise(make_loop(breeze + rustle + body), -29.0)


def dream():
    n = total_len()
    t = np.arange(n) / SR
    pad = np.zeros(n)
    for f in (146.83, 220.00, 293.66, 369.99, 440.00, 659.25):           # D major add9
        for d in (-0.0016, 0.0, 0.0017):
            phase = rng.uniform(0, 2 * np.pi)
            pad += (0.35 + 0.65 * smooth(n, 0.14)) * np.sin(2 * np.pi * f * (1 + d) * t + phase) / (1 + f / 400)
    sparkle = np.zeros(n)
    for f in (1174.66, 1479.98, 1760.00, 1975.53, 2349.32):              # high D pentatonic
        pos = rng.uniform(0, 1.5)
        while pos < n / SR - 2.5:
            i = int(pos * SR)
            m = int(2.2 * SR)
            tt = np.arange(m) / SR
            sparkle[i:i + m] += rng.uniform(0.08, 0.22) * np.exp(-tt / 0.55) * np.sin(2 * np.pi * f * tt)
            pos += rng.uniform(1.6, 4.5)
    air = bp(noise(n), 1500, 5000) * 0.05 * (0.4 + 0.6 * smooth(n, 0.2))
    mix = reverb(pad * 0.5 + sparkle + air, 2.6, 0.45)
    return normalise(make_loop(mix), -28.0)


def stream():
    n = total_len()
    secs = n / SR
    bubbles = np.zeros(n)
    for _ in range(int(secs * 170)):
        i = int(rng.uniform(0, secs - 0.12) * SR)
        dur = rng.uniform(0.02, 0.07)
        m = int(dur * SR)
        tt = np.arange(m) / SR
        f = rng.uniform(250, 1400) * (1 + rng.uniform(2, 6) * tt)       # rising "bloop"
        bubbles[i:i + m] += rng.uniform(0.2, 1.0) * np.exp(-tt / (dur * 0.35)) * np.sin(2 * np.pi * np.cumsum(f) / SR)
    body = bp(noise(n), 400, 3500) * 0.35
    out = (bubbles * 0.45 + body) * (0.7 + 0.3 * smooth(n, 0.4))
    return normalise(make_loop(out), -28.0)


def tent():
    n = total_len()
    room = lp(noise(n), 260) * (0.55 + 0.45 * smooth(n, 0.08))
    air = bp(noise(n), 900, 2400) * 0.03 * (0.3 + 0.7 * smooth(n, 0.12))
    out = room + air
    pos = rng.uniform(1, 3)
    while pos < n / SR - 1:
        i = int(pos * SR)
        m = int(rng.uniform(0.15, 0.35) * SR)
        burst = bp(noise(m), 1500, 4500) * np.hanning(m) * rng.uniform(0.05, 0.14)
        out[i:i + m] += burst[: n - i]
        pos += rng.uniform(3.0, 8.0)
    return normalise(make_loop(out), -30.0)


def hearth():
    n = total_len()
    secs = n / SR
    rumble = lp(noise(n), 160) * (0.5 + 0.5 * smooth(n, 0.7))
    imp = np.zeros(n)
    t = 0.0
    activity = smooth(n, 0.25)
    while t < secs:
        i = min(n - 1, int(t * SR))
        rate = 14 + 30 * activity[i]
        t += rng.exponential(1.0 / rate)
        j = min(n - 1, int(t * SR))
        imp[j] += min(1.0, rng.pareto(2.2) * 0.25 + 0.05) * rng.choice([-1.0, 1.0])
    crackle = bp(imp, 1200, 6500) * 2.2 + bp(imp, 220, 900) * 0.9
    return normalise(make_loop(rumble * 0.35 + crackle), -27.0)


SOUNDS = {
    "crickets_loop": crickets,
    "field_loop": field,
    "dream_loop": dream,
    "stream_loop": stream,
    "tent_loop": tent,
    "hearth_loop": hearth,
}


def export(name, x, out_dir):
    wav = os.path.join(out_dir, name + ".tmp.wav")
    mp3 = os.path.join(out_dir, name + ".mp3")
    wavfile.write(wav, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run([FF, "-y", "-v", "error", "-i", wav, "-ac", "1", "-ar", str(SR), "-b:a", "96k", mp3], check=True)
    os.remove(wav)
    return mp3


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in SOUNDS.items():
        print("making", name, flush=True)
        export(name, fn(), out_dir)
    print("done")
