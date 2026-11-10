"""A soft, excited heartbeat ("lub-dub") for the moments the story says "בום בום בום".
Original, synthetic: no third-party audio, no license needed.
Usage: python synth_heartbeat.py <output_dir>      (needs numpy, scipy, imageio-ffmpeg)"""
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
SECONDS = 7.0
PERIOD = 0.62            # about 97 beats per minute: a nervous child's heart


def thump(f0, dur, amp):
    n = int(SR * dur)
    t = np.arange(n) / SR
    freq = f0 * (1 + 1.2 * np.exp(-t * 25))                 # quick pitch drop, like a soft drum
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 18) * amp


def main(out_dir):
    y = np.zeros(int(SR * SECONDS))
    t = 0.35
    while t < SECONDS - 0.5:
        for at, f0, amp in ((t, 62, 1.0), (t + 0.24, 74, 0.7)):      # lub, dub
            s = thump(f0, 0.35, amp)
            i = int(at * SR)
            y[i:i + len(s)] += s[: len(y) - i]
        t += PERIOD
    y = sosfilt(butter(2, 240, "lowpass", fs=SR, output="sos"), y)
    y *= 10 ** (-21 / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-12)       # about -21 dBFS RMS, like the other effects
    y = np.clip(y, -0.5, 0.5)
    os.makedirs(out_dir, exist_ok=True)
    wav, mp3 = os.path.join(out_dir, "heartbeat.tmp.wav"), os.path.join(out_dir, "heartbeat.mp3")
    wavfile.write(wav, SR, (y * 32767).astype(np.int16))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", wav, "-ac", "1", "-b:a", "96k", mp3], check=True)
    os.remove(wav)
    print("wrote", mp3)


if __name__ == "__main__":
    main(sys.argv[1])
