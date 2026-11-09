"""Happy, bouncy original themes for kids: major keys, 112-124bpm,
ukulele-like strums, glockenspiel melody, bouncing bass, soft claps + tambourine."""
import sys, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import synth as S

SR = S.SR; rng = np.random.default_rng(11)

def glock(f, dur=1.6, vol=0.16):
    n = int(SR*dur); t = np.arange(n)/SR
    s = (np.sin(2*np.pi*f*t)*np.exp(-t*3.2) + 0.35*np.sin(2*np.pi*f*2.0*t)*np.exp(-t*6)
         + 0.18*np.sin(2*np.pi*f*3.0*t)*np.exp(-t*9))
    s[:int(SR*0.002)] *= np.linspace(0, 1, int(SR*0.002))
    return s*vol

def bass(f, dur, vol=0.22):
    n = int(SR*dur); t = np.arange(n)/SR
    s = np.sin(2*np.pi*f*t) + 0.25*np.sin(2*np.pi*2*f*t)
    env = np.exp(-t*4.5); env[:int(SR*0.005)] *= np.linspace(0, 1, int(SR*0.005))
    return s*env*vol

def clap(vol=0.07):
    n = int(SR*0.18); s = np.zeros(n)
    for k, d in enumerate((0, 0.009, 0.018)):
        i = int(d*SR); m = n-i; burst = rng.uniform(-1, 1, m)*np.exp(-np.arange(m)/(SR*(0.01 if k < 2 else 0.05)))
        s[i:] += burst
    s = sosfilt(butter(2, [900, 5500], 'bandpass', fs=SR, output='sos'), s)
    return s*vol

def tamb(vol=0.035):
    n = int(SR*0.12); s = rng.uniform(-1, 1, n)*np.exp(-np.arange(n)/(SR*0.03))
    return sosfilt(butter(2, 7000, 'hp', fs=SR, output='sos'), s)*vol

def strum(chord, vol=0.22, bright=0.75, down=True):
    sig = np.zeros(int(SR*1.6))
    notes = chord if down else chord[::-1]
    for k, n in enumerate(notes):
        p = S.pluck(S.hz(n), 1.5, bright, 0.994)*vol
        i = int(k*0.012*SR); sig[i:i+len(p)] += p[:len(sig)-i]
    return sig

def song(prog, bpm, melody, out, loops=2, swing=0.0):
    beat = 60/bpm; bar = 4*beat; L = np.zeros(int(SR*(len(prog)*bar*loops+3))); R = np.zeros_like(L)
    t0 = 0.0
    for lp in range(loops):
        for c in prog:
            lo = S.chord_notes(c, 3); hi = S.chord_notes(c, 4)
            uke = [lo[1], lo[2], hi[0], hi[1]]
            # strum pattern: D . D U . U D U  (eighths)
            pat = [(0, True, 1.0), (1, True, .75), (1.5, False, .6), (2.5, False, .65), (3, True, .8), (3.5, False, .6)]
            for (b, d, v) in pat:
                at = t0 + b*beat + (swing*beat*0.5 if (b*2) % 2 == 1 else 0)
                s = strum(uke, 0.20*v, 0.8, d); S.place(L, s*0.6, at); S.place(R, s*0.4, at)
            root = S.hz(lo[0])/2; fifth = S.hz(lo[2])/2
            for b, f in ((0, root), (1.5, root), (2, fifth), (3, root*2 if root < 70 else root)):
                bs = bass(f, beat*1.2); S.place(L, bs, t0+b*beat); S.place(R, bs, t0+b*beat)
            for b in (1, 3):
                cp = clap(); S.place(L, cp*0.45, t0+b*beat); S.place(R, cp*0.55, t0+b*beat)
            for k in range(8):
                tb = tamb(0.03 if k % 2 else 0.018); S.place(L, tb*0.4, t0+k*beat/2); S.place(R, tb*0.6, t0+k*beat/2)
            t0 += bar
    loop_len = len(prog)*bar
    for lp in range(loops):
        for (bo, n, lb) in melody:
            dur = max(1.2, lb*beat+0.8)
            g = glock(S.hz(n)*2, dur, 0.15)   # glockenspiel an octave up
            p = S.pluck(S.hz(n), dur, 0.9, 0.996)*0.22  # doubled by a bright pluck
            m = min(len(g), len(p)); g, p = g[:m], p[:m]
            at = lp*loop_len + bo*beat
            S.place(L, g*0.45+p*0.55, at); S.place(R, g*0.55+p*0.45, at)
    mix = np.stack([S.reverb(L, 1.6, 0.18), S.reverb(R, 1.6, 0.18)], 1)
    mix /= np.max(np.abs(mix))+1e-9; mix *= 0.85
    wavfile.write(out, SR, (mix*32767).astype(np.int16))

if __name__ == '__main__':
    o = sys.argv[1]
    # Lech Lecha — happy journey, D major 116
    song(['D','A','Bm','G','D','A','G','D'], 116,
         [(0,'F#5',1),(1,'A5',1),(2,'D6',1.5),(3.5,'A5',.5),(4,'C#6',1),(5,'B5',1),(6,'A5',2),(8,'B5',1),(9,'A5',1),(10,'F#5',1),(11,'D5',1),
          (12,'G5',1),(13,'F#5',1),(14,'E5',2),(16,'F#5',1),(17,'A5',1),(18,'D6',1.5),(19.5,'C#6',.5),(20,'E6',1),(21,'D6',1),(22,'C#6',2),
          (24,'B5',1),(25,'A5',1),(26,'G5',1),(27,'E5',1),(28,'D5',2),(30,'A5',.5),(30.5,'D6',1.5)],
         f'{o}/theme_lech.wav', swing=0.12)
    # Toldot — two brothers, bouncy G major 124
    song(['G','C','G','D','G','C','D','G'], 124,
         [(0,'D5',.5),(.5,'G5',.5),(1,'B5',1),(2,'G5',1),(3,'B5',1),(4,'C6',1),(5,'E6',1),(6,'C6',2),(8,'B5',.5),(8.5,'D6',.5),(9,'G6',1),(10,'D6',1),(11,'B5',1),
          (12,'A5',1),(13,'F#5',1),(14,'D5',2),(16,'G5',1),(17,'B5',1),(18,'D6',2),(20,'E6',1),(21,'C6',1),(22,'G5',2),(24,'A5',1),(25,'C6',1),(26,'B5',1),(27,'A5',1),(28,'G5',3)],
         f'{o}/theme_toldot.wav', swing=0.18)
    # Vayetzei — the ladder: joyful rising melody, F major 112 (angels going up and down)
    song(['F','C','Dm','Bb','F','C','Bb','F'], 112,
         [(0,'C5',.5),(.5,'F5',.5),(1,'A5',.5),(1.5,'C6',1.5),(3,'A5',1),(4,'G5',.5),(4.5,'A5',.5),(5,'G5',.5),(5.5,'E5',.5),(6,'C5',2),
          (8,'D5',.5),(8.5,'F5',.5),(9,'A5',.5),(9.5,'D6',1.5),(11,'C6',1),(12,'Bb5',1),(13,'A5',1),(14,'G5',2),
          (16,'C5',.5),(16.5,'F5',.5),(17,'A5',.5),(17.5,'C6',1),(18.5,'F6',1.5),(20,'E6',1),(21,'C6',1),(22,'G5',2),
          (24,'Bb5',1),(25,'A5',1),(26,'G5',1),(27,'E5',1),(28,'F5',3)],
         f'{o}/theme_vayetzei.wav', swing=0.1)
    print('done')
