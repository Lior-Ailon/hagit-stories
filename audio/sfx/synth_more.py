"""More original, synthetic story sounds. No third-party audio, no license needed.
Usage: python synth_more.py <output_dir>      (needs numpy, scipy, imageio-ffmpeg)

  bell.mp3             a sheep's bell: irregular soft clanks
  flute.mp3            a calm, slow flute melody (for Jacob, the "quiet tent-dweller")
  caravan.mp3          a slow caravan: soft hooves and camel bells
  heavy_steps.mp3      heavy, close footsteps (Esau walks in)
  steps_away.mp3       the same footsteps, fading as he walks away
  drama.mp3            a short rising tension cue with a drum roll and a final hit
  heartbeat_fast.mp3   a strong, fast heartbeat (a frightened child's)
  tok.mp3              four wood/drum hits, one second apart (to accompany "ויאכל וישת ויקום וילך")
  growl.mp3            a low, rough animal growl
"""
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
rng = np.random.default_rng(5779)


def lp(x, f, o=2):
    return sosfilt(butter(o, f, "lowpass", fs=SR, output="sos"), x)


def hp(x, f, o=2):
    return sosfilt(butter(o, f, "highpass", fs=SR, output="sos"), x)


def bp(x, lo, hi, o=2):
    return sosfilt(butter(o, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def place(dst, src, at):
    i = int(at * SR)
    if i >= len(dst):
        return
    m = min(len(src), len(dst) - i)
    dst[i:i + m] += src[:m]


def reverb(x, secs=1.8, wet=0.3):
    m = int(secs * SR)
    ir = rng.standard_normal(m) * np.exp(-np.arange(m) / (SR * secs / 4.5))
    ir = lp(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    y = fftconvolve(x, ir)[: len(x)]
    return (1 - wet) * x + wet * y * np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))


def finish(name, x, out_dir, rms_db=-21.0, peak=0.5, fade=0.05):
    x = x - np.mean(x)
    x *= 10 ** (rms_db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    x = peak * np.tanh(x / peak)
    f = int(fade * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    wav, mp3 = os.path.join(out_dir, name + ".tmp.wav"), os.path.join(out_dir, name + ".mp3")
    wavfile.write(wav, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", wav, "-ac", "1", "-b:a", "96k", mp3], check=True)
    os.remove(wav)
    print("wrote", mp3, flush=True)


# ---------- sounds ----------
def bell(out):
    y = np.zeros(int(SR * 12))
    t = 0.3
    while t < 11:
        base = rng.uniform(1050, 1500)
        n = int(0.7 * SR)
        tt = np.arange(n) / SR
        s = sum(a * np.exp(-tt * d) * np.sin(2 * np.pi * base * r * tt)
                for a, r, d in ((1.0, 1.0, 9), (0.6, 2.76, 14), (0.35, 5.4, 20)))
        s += 0.15 * hp(rng.standard_normal(n), 3000) * np.exp(-tt * 60)       # the little "clank"
        place(y, s * rng.uniform(0.5, 1.0), t)
        t += rng.uniform(0.35, 1.2)
    finish("bell", y, out, -23)


def flute(out):
    notes = [(587.33, 2.0), (740.0, 1.4), (880.0, 2.4), (784.0, 1.6), (659.25, 2.2), (587.33, 1.4), (493.88, 1.8), (440.0, 3.0),
             (493.88, 1.4), (587.33, 2.4), (659.25, 1.8), (587.33, 3.4)]
    total = sum(d for _, d in notes) + 1.0
    y = np.zeros(int(SR * total))
    t = 0.2
    for f, d in notes:
        n = int((d + 0.6) * SR)
        tt = np.arange(n) / SR
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.0 * tt) * np.minimum(1, tt / 0.6)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        tone = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
        env = np.minimum(1, tt / 0.15) * np.exp(-np.maximum(0, tt - d) * 6)
        breath = bp(rng.standard_normal(n), 1800, 4500) * 0.05 * env
        place(y, (tone * 0.5 + breath) * env, t)
        t += d - 0.12
    y = reverb(y, 2.2, 0.35)
    finish("flute", y, out, -24, fade=0.4)


def caravan(out):
    y = np.zeros(int(SR * 16))
    t = 0.1
    while t < 15.3:
        for k in range(2):                                    # two soft hoof falls
            n = int(0.16 * SR)
            tt = np.arange(n) / SR
            thud = lp(rng.standard_normal(n), 380) * np.exp(-tt * 28) * 1.0 + hp(rng.standard_normal(n), 2500) * np.exp(-tt * 70) * 0.08
            place(y, thud * rng.uniform(0.6, 1.0), t + k * 0.28)
        n = int(0.9 * SR)                                     # a camel bell on every stride
        tt = np.arange(n) / SR
        f = rng.uniform(760, 980)
        b = (np.exp(-tt * 5.5) * np.sin(2 * np.pi * f * tt) + 0.45 * np.exp(-tt * 9) * np.sin(2 * np.pi * f * 2.43 * tt))
        place(y, b * 0.28, t + 0.06)
        t += 0.9 + rng.uniform(-0.03, 0.03)
    finish("caravan", reverb(y, 1.2, 0.18), out, -24, fade=0.3)


def footsteps(seconds, period, fade_out=False):
    y = np.zeros(int(SR * seconds))
    t, i = 0.15, 0
    while t < seconds - 0.7:
        n = int(0.4 * SR)
        tt = np.arange(n) / SR
        thump = np.sin(2 * np.pi * (52 + 6 * (i % 2)) * tt) * np.exp(-tt * 13)
        crunch = lp(rng.standard_normal(n), 1400) * np.exp(-tt * 30) * 0.7 + hp(rng.standard_normal(n), 3000) * np.exp(-tt * 55) * 0.12
        place(y, (thump * 1.0 + crunch) * rng.uniform(0.85, 1.0), t)
        t += period + rng.uniform(-0.03, 0.03)
        i += 1
    if fade_out:
        y *= np.linspace(1, 0.04, len(y)) ** 1.6
    return y


def heavy_steps(out):
    finish("heavy_steps", footsteps(8, 0.62), out, -19, fade=0.2)


def steps_away(out):
    finish("steps_away", lp(footsteps(10, 0.6, True), 3500), out, -19, fade=0.1)


def drama(out):
    secs = 12.0
    n = int(SR * secs)
    t = np.arange(n) / SR
    drone = np.zeros(n)
    for f, a in ((55.0, 1.0), (82.41, 0.7), (110.0, 0.4)):
        saw = 2 * ((f * t) % 1.0) - 1
        drone += a * lp(saw, 700)
    drone *= (0.35 + 0.65 * (t / secs) ** 1.5) * (1 + 0.25 * np.sin(2 * np.pi * (5 + 6 * t / secs) * t))
    swell = lp(rng.standard_normal(n), 3000) * (t / secs) ** 3 * 0.35
    roll = np.zeros(n)
    tt = 0.2
    while tt < secs - 0.9:
        hit = np.sin(2 * np.pi * 55 * np.arange(int(0.5 * SR)) / SR) * np.exp(-np.arange(int(0.5 * SR)) / SR * 9)
        place(roll, hit * (0.25 + 0.75 * tt / secs), tt)
        tt += max(0.07, 0.34 * (1 - tt / secs) ** 1.4 + 0.07)
    final = np.zeros(n)
    m = int(1.6 * SR)
    k = np.arange(m) / SR
    place(final, (np.sin(2 * np.pi * 48 * k) * np.exp(-k * 3.2) + 0.6 * lp(rng.standard_normal(m), 1200) * np.exp(-k * 6)) * 1.8, secs - 1.5)
    finish("drama", reverb(drone * 0.7 + swell + roll * 0.9 + final, 2.0, 0.25), out, -19, fade=0.05)


def heartbeat_fast(out):
    seconds, period = 6.0, 0.42
    y = np.zeros(int(SR * seconds))

    def thump(f0, amp):
        n = int(0.3 * SR)
        tt = np.arange(n) / SR
        fr = f0 * (1 + 1.2 * np.exp(-tt * 25))
        return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 20) * amp

    t = 0.25
    while t < seconds - 0.4:
        place(y, thump(64, 1.0), t)
        place(y, thump(78, 0.75), t + 0.17)
        t += period
    finish("heartbeat_fast", lp(y, 260), out, -19, fade=0.05)


def tok(out):
    y = np.zeros(int(SR * 4.2))
    for i, (f, bright) in enumerate(((95, 0.1), (130, 0.2), (170, 0.3), (230, 0.45))):    # four rising hits
        n = int(0.4 * SR)
        tt = np.arange(n) / SR
        fr = f * (1 + 0.6 * np.exp(-tt * 40))
        s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 16)
        s += bright * bp(rng.standard_normal(n), 800, 3000) * np.exp(-tt * 45)
        place(y, s, 0.1 + i * 1.0)
    finish("tok", y, out, -18, fade=0.01)


def growl(out):
    secs = 3.2
    n = int(SR * secs)
    t = np.arange(n) / SR
    f0 = 62 + 10 * np.sin(2 * np.pi * 1.3 * t) + 6 * np.sin(2 * np.pi * 9 * t)
    saw = 2 * ((np.cumsum(f0) / SR) % 1.0) - 1
    rough = saw * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 28 * t)) * 0.5 + 0.2 * rng.standard_normal(n))
    y = bp(rough, 70, 900, 3) * np.minimum(1, t / 0.25) * np.exp(-np.maximum(0, t - 2.0) * 2.2)
    y += 0.25 * bp(rng.standard_normal(n), 300, 1500) * np.minimum(1, t / 0.25) * np.exp(-np.maximum(0, t - 2.0) * 2.2)
    finish("growl", y, out, -19, fade=0.05)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for fn in (bell, flute, caravan, heavy_steps, steps_away, drama, heartbeat_fast, tok, growl):
        fn(out)
    print("done")
