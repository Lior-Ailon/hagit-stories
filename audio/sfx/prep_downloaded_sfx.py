"""Turn the public-domain / CC0 recordings (see CREDITS.md) into short, normalised story sounds.
Usage: python prep_downloaded_sfx.py <raw_dir> <output_dir>

  raw_dir holds: baby1.ogg  (CC0, "Crying newborn baby", the_yura, Wikimedia Commons)
                 wolf.ogg   (public domain, "Wolf howls", Wikimedia Commons)
                 lion.ogg   (public domain, "Lion raring-sound1TamilNadu178", Wikimedia Commons)   [optional]
  creates baby1.mp3, baby2.mp3 (the same baby, pitched up, from a different moment), wolf.mp3, lion.mp3
"""
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

SR = 44100
FF = imageio_ffmpeg.get_ffmpeg_exe()
raw_dir, out_dir = sys.argv[1], sys.argv[2]


def decode(path):
    p = subprocess.run([FF, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def loudest(y, secs, start_after=0.0):
    hop = int(0.05 * SR)
    n = len(y) // hop
    env = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    k = max(1, int(secs / 0.05))
    sm = np.convolve(env, np.ones(k) / k, mode="valid")
    lo = int(start_after / 0.05)
    i = lo + int(np.argmax(sm[lo:])) if lo < len(sm) else int(np.argmax(sm))
    a = i * hop
    return y[a: a + int(secs * SR)]


def save(name, y, rms_db=-20.0, peak=0.5, fade=0.15):
    y = y - np.mean(y)
    y = y * (10 ** (rms_db / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    y = peak * np.tanh(y / peak)
    f = int(fade * SR)
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    wav, mp3 = os.path.join(out_dir, name + ".tmp.wav"), os.path.join(out_dir, name + ".mp3")
    wavfile.write(wav, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    subprocess.run([FF, "-y", "-v", "error", "-i", wav, "-ac", "1", "-b:a", "96k", mp3], check=True)
    os.remove(wav)
    print("wrote", mp3, f"{len(y) / SR:.1f}s")


def resample(y, factor):
    """Pitch up by `factor` (and a little shorter)."""
    idx = np.arange(0, len(y) - 1, factor)
    return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


os.makedirs(out_dir, exist_ok=True)
baby = decode(os.path.join(raw_dir, "baby1.ogg"))
save("baby1", loudest(baby, 6.0))
b2 = loudest(baby, 7.0, start_after=1.0)
save("baby2", resample(b2, 1.22)[: int(6.0 * SR)])
wolf = decode(os.path.join(raw_dir, "wolf.ogg"))
save("wolf", loudest(wolf, 6.0))
lion_path = os.path.join(raw_dir, "lion.ogg")
if os.path.exists(lion_path):
    save("lion", loudest(decode(lion_path), 5.0))
