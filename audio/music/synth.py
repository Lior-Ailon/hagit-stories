"""Original gentle music themes for Hagit's kids' parasha podcast.
Karplus-Strong harp/oud plucks + warm pad + soft bells + synthetic reverb.
Usage: python3 synth.py OUTDIR
"""
import sys, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
rng = np.random.default_rng(7)
NOTE = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

def hz(n):  # 'A4'
    name, octv = n[:-1], int(n[-1])
    return 440.0 * 2 ** ((NOTE[name] + 12 * (octv + 1) - 69) / 12)

def pluck(f, dur, bright=0.5, decay=0.996):
    n = int(SR * dur); p = max(2, int(SR / f))
    buf = rng.uniform(-1, 1, p)
    # soften excitation
    buf = sosfilt(butter(1, min(0.99, 0.15 + bright * 0.6), output='sos'), buf)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = decay * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    env = np.minimum(1, np.arange(n) / (SR * 0.004))
    return out * env

def pad(freqs, dur, cutoff=1400, vol=0.18):
    n = int(SR * dur); t = np.arange(n) / SR; s = np.zeros(n)
    for f in freqs:
        for det in (-0.12, 0.0, 0.11):
            ff = f * 2 ** (det / 12)
            s += np.sin(2*np.pi*ff*t) + 0.35*np.sin(2*np.pi*2*ff*t + 0.3) + 0.12*np.sin(2*np.pi*3*ff*t)
    s = sosfilt(butter(2, cutoff, fs=SR, output='sos'), s)
    a = int(SR * min(1.2, dur/3)); r = int(SR * min(1.5, dur/2.5))
    env = np.ones(n); env[:a] = np.linspace(0, 1, a); env[-r:] *= np.linspace(1, 0, r)
    return s / (len(freqs) * 3) * env * vol

def bell(f, dur=3.0, vol=0.12):
    n = int(SR * dur); t = np.arange(n) / SR
    s = (np.sin(2*np.pi*f*t) * np.exp(-t*1.6) + 0.5*np.sin(2*np.pi*f*2.76*t) * np.exp(-t*3.2)
         + 0.25*np.sin(2*np.pi*f*5.4*t) * np.exp(-t*5))
    return s * vol

def tick(dur=0.08, vol=0.05):
    n = int(SR*dur); s = rng.uniform(-1,1,n) * np.exp(-np.arange(n)/(SR*0.012))
    return sosfilt(butter(2, 5000, 'hp', fs=SR, output='sos'), s) * vol

def drum(dur=0.35, f=95, vol=0.18):
    n=int(SR*dur); t=np.arange(n)/SR
    return np.sin(2*np.pi*f*t*(1-0.3*t))*np.exp(-t*12)*vol

def reverb(x, secs=2.6, wet=0.28):
    n = int(SR*secs); t = np.arange(n)/SR
    ir = rng.standard_normal(n) * np.exp(-t * 3.0 / secs * 2.2)
    ir = sosfilt(butter(1, 4500, fs=SR, output='sos'), ir); ir[0] = 0; ir /= np.sqrt((ir**2).sum())
    y = fftconvolve(x, ir)[:len(x)]
    return (1-wet)*x + wet*y*0.9

def place(buf, sig, at):
    i = int(at*SR); j = min(len(buf), i+len(sig))
    if i < len(buf): buf[i:j] += sig[:j-i]

def chord_notes(name, octv=3):
    # name like 'D', 'Bm', 'E' ; returns root, third, fifth names w/ octave
    minor = name.endswith('m'); root = name[:-1] if minor else name
    r = NOTE[root]; th = (r + (3 if minor else 4)) % 12; fi = (r + 7) % 12
    inv = {v:k for k,v in NOTE.items() if len(k)==1 or '#' in k}
    def nm(pc, o): return inv[pc] + str(o + (1 if pc < r else 0))
    return [nm(r, octv), nm(th, octv), nm(fi, octv)]

def theme(prog, bpm, bars_per_chord, beats, arp_pattern, melody, out, *, bright=0.5, bells=False,
          shaker=False, drums=False, pad_cut=1400, loops=2):
    beat = 60/bpm; bar = beat*beats
    total = len(prog)*bars_per_chord*bar*loops
    L = np.zeros(int(SR*(total+4))); R = np.zeros_like(L)
    t0 = 0.0
    for lp in range(loops):
        for ci, c in enumerate(prog):
            notes = chord_notes(c, 3); hi = chord_notes(c, 4)
            pool = [notes[0], notes[1], notes[2], hi[0], hi[1], hi[2]]
            seg = bars_per_chord*bar
            p = pad([hz(n)/2 if k==0 else hz(n) for k,n in enumerate(notes)], seg+0.6, pad_cut)
            place(L, p, t0); place(R, p, t0)
            steps = int(seg/(beat/2))
            for s in range(steps):
                idx = arp_pattern[s % len(arp_pattern)]
                if idx is None: continue
                f = hz(pool[idx]); pl = pluck(f, 2.2, bright) * (0.30 if s%4 else 0.38)
                pan = 0.35 + 0.3*((idx%3)/2)
                place(L, pl*(1-pan), t0+s*beat/2); place(R, pl*pan, t0+s*beat/2)
                if shaker and s%2==1:
                    tk = tick(); place(L, tk*0.8, t0+s*beat/2); place(R, tk, t0+s*beat/2)
            if drums:
                for b in range(int(seg/beat)):
                    if b % beats == 0:
                        d = drum(); place(L, d, t0+b*beat); place(R, d, t0+b*beat)
            if bells and (ci % 2 == 0):
                bl = bell(hz(hi[2])*2, vol=0.06); place(L, bl*0.4, t0+beat*0.5); place(R, bl*0.6, t0+beat*0.5)
            t0 += seg
    # melody (original): list of (beat_offset, note, length_beats); repeats each loop, second time an octave up softer
    loop_len = len(prog)*bars_per_chord*bar
    for lp in range(loops):
        for (bo, n, lb) in melody:
            f = hz(n) * (1 if lp == 0 else 1)
            m = pluck(f, max(1.5, lb*beat+1.2), min(1, bright+0.25), 0.9975) * 0.42
            vib = 1 + 0.0*np.arange(len(m))
            place(L, m*0.55, lp*loop_len + bo*beat); place(R, m*0.45, lp*loop_len + bo*beat)
    mix = np.stack([reverb(L), reverb(R)], 1)
    mix /= np.max(np.abs(mix)) + 1e-9; mix *= 0.85
    wavfile.write(out, SR, (mix*32767).astype(np.int16))
    return total

if __name__ == '__main__':
    o = sys.argv[1]
    # Lech Lecha — the journey: warm D major, flowing harp, 84bpm
    theme(['D','G','Bm','A','D','G','A','D'], 84, 1, 4, [0,2,4,2,3,2,4,5],
          [(0,'F#5',2),(2,'A5',2),(4,'B5',3),(7,'A5',1),(8,'F#5',2),(10,'E5',2),(12,'C#5',2),(14,'E5',2),
           (16,'F#5',2),(18,'D5',2),(20,'B4',3),(23,'D5',1),(24,'E5',2),(26,'C#5',2),(28,'D5',4)],
          f'{o}/theme_lech.wav', bright=0.45, drums=True)
    # Toldot — two brothers, playful G major, pizzicato + shaker, 100bpm
    theme(['G','C','D','G','Em','C','D','G'], 100, 1, 4, [0,None,2,3,1,None,4,2],
          [(0,'D5',1),(1,'G5',1),(2,'B5',1),(3,'A5',1),(4,'G5',2),(6,'E5',2),(8,'F#5',1),(9,'A5',1),(10,'D5',2),
           (12,'G5',3),(16,'B4',1),(17,'D5',1),(18,'E5',2),(20,'C5',2),(22,'E5',2),(24,'D5',1),(25,'F#5',1),(26,'A5',2),(28,'G5',4)],
          f'{o}/theme_toldot.wav', bright=0.65, shaker=True)
    # Vayetzei — the night and the ladder: A minor with the E-major (freygish) colour, bells, 66bpm
    theme(['Am','Dm','E','Am','F','Dm','E','Am'], 66, 1, 4, [0,2,4,5,4,2,3,1],
          [(0,'E5',2),(2,'F5',1),(3,'G#5',1),(4,'A5',3),(7,'G#5',1),(8,'F5',2),(10,'E5',2),(12,'D5',2),(14,'E5',2),
           (16,'C5',2),(18,'E5',2),(20,'A5',3),(23,'F5',1),(24,'E5',2),(26,'D5',1),(27,'C5',1),(28,'B4',1),(29,'A4',3)],
          f'{o}/theme_vayetzei.wav', bright=0.4, bells=True, pad_cut=1100)
    print('done')
