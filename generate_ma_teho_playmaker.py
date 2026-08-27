#!/usr/bin/env python3
"""Generate 'Ma Teho - Playmaker.mp3'.

A 1:1 melodic/structural re-production of the user's own reference track
'wanna play ? (Remix) (Cover) (Cover) (Cover).mp3' with a completely new
instrumentation and plugin palette, produced from the deep audio analysis
stored in ANALYSIS.md:

  - 143.55 BPM, 4/4, 112 bars, first downbeat at 0.070 s  (~187.32 s of music)
  - Key G# minor; loop family i-VI (G#m / E) with D#sus4 turnarounds
  - Syncopated 3-kick groove (16th slots 0 / 6 / 11), snare on beats 2 & 4,
    dense accented 16th hats
  - Section map (bars): 1-8 intro | 9-24 hook | 25-40 riff | 41-47 breakdown |
    48-79 main B (long "sky" notes 71-76) | 80 fill | 81-86 breakdown 2 |
    87 riser | 88-102 final drop | 103-112 outro fade
  - Target loudness -13.7 LUFS integrated, true peak ceiling -1.0 dBTP

New instrumentation (vs. reference): hybrid punchy acoustic-EDM drum kit,
round analog mono bass with sub octave, FM electric piano + analog pad for
harmony, hybrid 3-osc formant lead with duophonic fifth, Karplus-Strong
plucked electric guitar for the riff, glassy FM bells and arp plucks in the
breakdowns, reverse-crash / riser / sub-drop transition FX.

DSP: Pedalboard (Moog-style ladder filters, chorus, ping-pong delay, plate &
hall reverb sends, bus compression, brickwall limiting), pyloudnorm metering.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pedalboard as pb
import pyloudnorm as pyln
import scipy.signal
import soundfile as sf

# ---------------------------------------------------------------------------
# Project constants (all measured from the reference)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000
BPM = 143.55
BEAT = 60.0 / BPM                 # 0.417973 s
BAR = 4.0 * BEAT                  # 1.671892 s
SIX = BEAT / 4.0                  # one 16th note
T0 = 0.070                        # first downbeat offset in the reference
N_BARS = 112
MUSIC_END = T0 + N_BARS * BAR     # 187.318 s
TOTAL_DUR = 187.36                # match reference file duration
SEED = 20260827

OUT_TITLE = "Playmaker"
OUT_ARTIST = "ma teho"
OUT_ALBUM_ARTIST = "ma teho"
OUT_COMPOSER = "ma teho"
OUT_COPYRIGHT = "© 2026 ma teho. All rights reserved."
OUT_COMMENT = "Re-production of the artist's own composition; new instrumentation by ma teho."
TARGET_LUFS = -13.7
CEILING_DBTP = -1.0

ROOT = Path(__file__).resolve().parent
OUT_MP3 = ROOT / f"Ma Teho - {OUT_TITLE}.mp3"
COVER_PNG = ROOT / "cover_playmaker.png"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------
GMIN_PC = (8, 10, 11, 1, 3, 4, 6)          # G# natural minor pitch classes
CHORD_ROOTS = {"G#m": 32, "E": 28, "Emaj7": 28, "D#sus4": 39}   # bass-register roots (MIDI)
# chord tones (MIDI around C4) used for pads / EP / arps
CHORD_VOICINGS = {
    "G#m":    [56, 59, 63, 68],            # G#3 B3 D#4 A#4  (add9 color)
    "E":      [52, 59, 64, 66],            # E3  B3 E4  F#4  (add9 color)
    "Emaj7":  [52, 56, 63, 66],            # E3  G#3 D#4 F#4
    "D#sus4": [51, 58, 63, 68],            # D#3 A#3 D#4 G#4
}

def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)

def bar_start(bar: int) -> float:
    """1-based bar number -> absolute seconds."""
    return T0 + (bar - 1) * BAR

def slot_time(bar: int, slot: int) -> float:
    """1-based bar, 0-15 16th slot -> absolute seconds."""
    return bar_start(bar) + slot * SIX

# ---------------------------------------------------------------------------
# Wavetable oscillator bank (band-limited saw / square / sine)
# ---------------------------------------------------------------------------
WT = 4096
_phase_axis = np.arange(WT, dtype=np.float64) / WT

def _sum_harm(max_h: int, odd_only: bool = False) -> np.ndarray:
    hs = np.arange(1, max_h + 1, 2 if odd_only else 1, dtype=np.float64)
    ph = 2.0 * np.pi * np.outer(_phase_axis, hs)
    w = np.sum(np.sin(ph) / hs, axis=1)
    return (w / np.max(np.abs(w))).astype(np.float32)

WTABLES = {
    "saw_low": _sum_harm(48), "saw_mid": _sum_harm(24), "saw_high": _sum_harm(10),
    "sq_low": _sum_harm(47, True), "sq_mid": _sum_harm(23, True),
    "sine": np.sin(2.0 * np.pi * _phase_axis).astype(np.float32),
}

def osc(table: str, freq: float, n: int, phase0: float = 0.0, fm: np.ndarray | None = None) -> np.ndarray:
    """Vectorized band-limited wavetable osc. `fm` in normalized phase units."""
    idx = np.arange(n, dtype=np.float64)
    ph = phase0 + freq * idx / SAMPLE_RATE
    if fm is not None:
        ph = ph + fm
    ph = ph % 1.0
    pos = ph * WT
    i0 = pos.astype(np.int64)
    i1 = (i0 + 1) % WT
    frac = (pos - i0).astype(np.float32)
    t = WTABLES[table]
    return (1.0 - frac) * t[i0] + frac * t[i1]

def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n = min(n, max(1, int(a * SAMPLE_RATE)))
    d_n = min(max(0, n - a_n), int(d * SAMPLE_RATE))
    r_n = min(n, int(r * SAMPLE_RATE))
    env = np.full(n, s, dtype=np.float32)
    env[:a_n] = np.linspace(0, 1, a_n, dtype=np.float32)
    if d_n:
        env[a_n:a_n + d_n] = np.linspace(1, s, d_n, dtype=np.float32)
    if r_n and r_n < n:
        env[-r_n:] *= np.linspace(1, 0, r_n, dtype=np.float32)
    return env

def pan(sig: np.ndarray, p: float) -> np.ndarray:
    ang = (np.clip(p, -1, 1) + 1) * np.pi / 4
    gl, gr = np.cos(ang), np.sin(ang)
    if sig.ndim == 2:          # already stereo: apply balance, keep energy
        return sig * (np.sqrt(2.0) * np.array([gl, gr], dtype=np.float32))
    return np.stack([sig * gl, sig * gr], axis=1)

def place(buf: np.ndarray, sig: np.ndarray, t: float) -> None:
    i = int(t * SAMPLE_RATE)
    j = min(len(buf), i + len(sig))
    if i < len(buf) and j > i:
        buf[i:j] += sig[: j - i]

# ---------------------------------------------------------------------------
# Bus buffers
# ---------------------------------------------------------------------------
N_TOTAL = int(TOTAL_DUR * SAMPLE_RATE)
def new_bus() -> np.ndarray:
    return np.zeros((N_TOTAL, 2), dtype=np.float32)

# ===========================================================================
# 1) DRUM KIT  (new palette: hybrid acoustic-EDM)
# ===========================================================================
def kick(strong: bool = True) -> np.ndarray:
    dur = 0.42 if strong else 0.30
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f_env = 43.0 + (120.0 - 43.0) * np.exp(-22.0 * t)
    ph = 2 * np.pi * np.cumsum(f_env) / SAMPLE_RATE
    body = np.sin(ph) * np.exp(-5.6 * t)
    chest = np.sin(2 * ph) * np.exp(-16.0 * t) * 0.30
    beater = np.sin(2 * np.pi * 170.0 * t) * np.exp(-60.0 * t) * 0.50
    click = rng.standard_normal(n).astype(np.float32)
    sos = scipy.signal.butter(2, [1800, 6000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    click = scipy.signal.sosfilt(sos, click) * np.exp(-300.0 * t) * 0.5
    sig = np.tanh((body * 1.30 + chest + beater + click) * 1.6)
    fade = int(0.03 * SAMPLE_RATE)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return np.stack([sig, sig], axis=1) * (0.9 if strong else 0.62)

def snare() -> np.ndarray:
    dur = 0.55
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * (190 - 40 * np.minimum(t * 8, 1)) * t) * np.exp(-17.0 * t) * 0.7
    nz = rng.standard_normal(n).astype(np.float32)
    hi = scipy.signal.sosfilt(scipy.signal.butter(4, [1500, 7500], btype="bandpass", fs=SAMPLE_RATE, output="sos"), nz)
    lo = scipy.signal.sosfilt(scipy.signal.butter(4, [400, 1500], btype="bandpass", fs=SAMPLE_RATE, output="sos"), nz)
    crack = hi * np.exp(-22.0 * t) * 1.0 + lo * np.exp(-14.0 * t) * 0.6
    body = tone + crack
    # big arena tail via plate reverb applied to the whole snare bus later; add a short room here
    return np.stack([body, body], axis=1)

def clap() -> np.ndarray:
    dur = 0.35
    n = int(dur * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    for off, g in [(0, 1.0), (0.011, 0.75), (0.022, 0.8), (0.034, 0.6)]:
        i = int(off * SAMPLE_RATE)
        m = n - i
        t = np.arange(m) / SAMPLE_RATE
        burst = rng.standard_normal(m).astype(np.float32)
        burst = scipy.signal.sosfilt(scipy.signal.butter(4, [900, 7000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), burst)
        burst *= np.exp(-32.0 * t) * g
        out[i:] += burst
    return np.stack([out, out], axis=1)

def hat(open_: bool = False) -> np.ndarray:
    dur = 0.30 if open_ else 0.06
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    metallic = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) for f in (3170, 4450, 5330, 6760, 8110)) / 5
    nz = rng.standard_normal(n).astype(np.float32)
    nz = scipy.signal.sosfilt(scipy.signal.butter(4, [6500, 13000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), nz)
    env = np.exp(-(14.0 if open_ else 65.0) * t)
    sig = (nz * 0.7 + metallic * 0.35) * env
    sig *= np.minimum(t / 0.001, 1)
    return np.stack([sig, sig], axis=1)

def ride() -> np.ndarray:
    dur = 0.5
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    nz = rng.standard_normal(n).astype(np.float32)
    nz = scipy.signal.sosfilt(scipy.signal.butter(4, [5000, 12000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), nz)
    ping = np.sin(2 * np.pi * 830.0 * t) * np.exp(-40.0 * t) * 0.12
    sig = (nz * np.exp(-8.0 * t) * 0.5 + ping)
    return np.stack([sig * 0.9, sig * 1.1], axis=1)

def crash(reverse: bool = False) -> np.ndarray:
    dur = 2.6
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    l = rng.standard_normal(n).astype(np.float32)
    r = rng.standard_normal(n).astype(np.float32)
    bp = scipy.signal.butter(4, [2200, 11500], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    l = scipy.signal.sosfilt(bp, l); r = scipy.signal.sosfilt(bp, r)
    l *= np.exp(-1.9 * t); r *= np.exp(-1.9 * t)
    sig = np.stack([l, r], axis=1)
    if reverse:
        sig = sig[::-1].copy()
        sig *= np.linspace(0.2, 1.0, n)[:, None]
    rv = pb.Reverb(room_size=0.62, damping=0.5, wet_level=0.35, dry_level=0.8)
    return rv(sig, SAMPLE_RATE)

def snare_roll_hit() -> np.ndarray:
    s = snare() * 0.55
    return s

# ===========================================================================
# 2) FX  (risers, sub drop, downlifter, sweeps)
# ===========================================================================
def riser(bars: float = 2.0) -> np.ndarray:
    dur = bars * BAR
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    nz = rng.standard_normal(n).astype(np.float32)
    nz = np.stack([nz, np.roll(nz, 997)], axis=1)
    prog = pb.Pedalboard([
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=800, resonance=0.35),
        pb.Chorus(rate_hz=2.2, depth=0.7, mix=0.5),
    ])
    sig = prog(nz, SAMPLE_RATE)
    # manual upward filter sweep by crossfading three pre-filtered layers
    sos1 = scipy.signal.butter(4, [300, 1200], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos2 = scipy.signal.butter(4, [1200, 4200], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos3 = scipy.signal.butter(4, [4200, 13000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    la = scipy.signal.sosfilt(sos1, sig[:, 0]); lb = scipy.signal.sosfilt(sos2, sig[:, 0]); lc = scipy.signal.sosfilt(sos3, sig[:, 0])
    ra = scipy.signal.sosfilt(sos1, sig[:, 1]); rb = scipy.signal.sosfilt(sos2, sig[:, 1]); rc = scipy.signal.sosfilt(sos3, sig[:, 1])
    k = np.linspace(0, 1, n) ** 1.6
    w1 = np.clip(1.6 - 2.4 * k, 0, 1); w2 = np.clip(1.6 - np.abs(2.4 * k - 1.2), 0, 1); w3 = np.clip(2.4 * k - 1.2, 0, 1)
    L = la * w1 + lb * w2 + lc * w3
    R = ra * w1 + rb * w2 + rc * w3
    vol = (np.exp(np.linspace(-2.2, 0.35, n)) - 0.1)
    out = np.stack([L, R], axis=1) * vol[:, None]
    return out * 0.30

def sub_drop(freq: float = 41.2) -> np.ndarray:
    dur = 1.6
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f = freq * 0.45 + freq * 0.55 * np.exp(-3.2 * t)
    ph = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    sig = np.sin(ph) * np.exp(-2.1 * t) * 0.9
    return np.stack([sig, sig], axis=1)

def downlifter() -> np.ndarray:
    dur = 0.9
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f = 900 * np.exp(-2.4 * t) + 120
    ph = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    sig = osc("saw_mid", 1.0, n)  # placeholder, replaced below
    sig = WTABLES["saw_mid"][(ph % 1.0 * WT).astype(np.int64) % WT]
    sig *= np.exp(-3.5 * t) * 0.25
    lfo = np.sin(2 * np.pi * 7 * t)
    sig = sig * (0.7 + 0.3 * lfo)
    return np.stack([sig, sig[::-1] * 0.9], axis=1) * 0.8

def noise_sweep_down(bars: float = 2.0) -> np.ndarray:
    n = int(bars * BAR * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    nz = rng.standard_normal(n).astype(np.float32)
    nz = scipy.signal.sosfilt(scipy.signal.butter(4, [800, 3000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), nz)
    env = np.exp(np.linspace(0.0, -4.5, n))
    return np.stack([nz * env, np.roll(nz * env, 1319)], axis=1) * 0.22

# ===========================================================================
# 3) BASS  (round analog mono bass + sub octave, sidechained later)
# ===========================================================================
def bass_note(midi: float, dur: float, bright: float = 1.0) -> np.ndarray:
    n = max(8, int(dur * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    f0 = midi_hz(midi)
    vib = 0.0035 * np.sin(2 * np.pi * 4.7 * t)
    ph1 = rng.uniform(0, 1) + (f0 * (1.003 + vib) * t)
    ph2 = rng.uniform(0, 1) + (f0 * (0.997 + vib) * t)
    tbl = "saw_low" if f0 < 120 else "saw_mid"
    w1 = osc(tbl, f0 * 1.003, n, rng.uniform(0, 1))
    w2 = osc(tbl, f0 * 0.997, n, rng.uniform(0, 1))
    w3 = osc("sq_low", f0 * 0.5, n, rng.uniform(0, 1)) * 0.5
    env = adsr(n, 0.004, 0.10, 0.75, min(0.09, dur * 0.4))
    mid = (w1 + w2 * 0.9 + w3) * env
    # resonant low-pass with per-note brightness
    fc = float(np.clip(f0 * 6.5 * bright, 520.0, 2600.0))
    mid2 = pb.Pedalboard([pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=fc, resonance=0.22),
                          pb.Distortion(drive_db=5.0)])(np.stack([mid, mid], axis=1), SAMPLE_RATE)
    # sub layer at the fundamental (E1=41 Hz / G#1=52 Hz register like the reference)
    sub = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.3)) * adsr(n, 0.006, 0.06, 0.95, min(0.16, dur * 0.5))
    sub = np.tanh(sub * 2.0) * 1.55
    return (np.stack([sub, sub], axis=1) + mid2 * 0.42).astype(np.float32)

# ===========================================================================
# 4) HARMONY: FM electric piano + analog pad
# ===========================================================================
def ep_note(midi: float, dur: float) -> np.ndarray:
    """Glassy FM tine electric piano (2-op + tine overtone)."""
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = midi_hz(midi)
    idx_env = (2.1 * np.exp(-6.0 * t) + 0.25)
    fm = idx_env * np.sin(2 * np.pi * f0 * 3.0 * t) / (2 * np.pi)   # mod ratio 3
    car = np.sin(2 * np.pi * f0 * t + 2 * np.pi * fm)
    tine = np.sin(2 * np.pi * f0 * 3.01 * t) * np.exp(-14.0 * t) * 0.35
    thump = np.sin(2 * np.pi * f0 * 0.5 * t) * np.exp(-9.0 * t) * 0.18
    env = adsr(n, 0.004, 0.9, 0.35, min(0.4, dur * 0.5))
    sig = (car + tine + thump) * env
    trem = 1.0 - 0.18 * (1 - np.exp(-6 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.1 * t))
    return (sig * trem).astype(np.float32)

def pad_note(midi: float, dur: float) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    f0 = midi_hz(midi)
    out = np.zeros(n, dtype=np.float32)
    for det, panpos, tbl in [(-0.006, -0.7, "saw_low"), (0.0, 0.0, "saw_mid"), (0.006, 0.7, "saw_low")]:
        w = osc(tbl, f0 * (1 + det), n, rng.uniform(0, 1))
        out += w * 0.33
    env = adsr(n, 0.6, 0.5, 0.85, min(1.2, dur * 0.45))
    return out * env

# ===========================================================================
# 5) LEADS: hybrid formant lead + KS guitar pluck + arp + bell
# ===========================================================================
def lead_note(midi: float, dur: float, fifth: bool = True) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = midi_hz(midi)
    vib = 0.011 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - 0.13) / 0.3, 0, 1)
    drift = 0.002 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
    fm_v = (vib + drift)
    tbl = "saw_high" if f0 > 600 else "saw_mid"
    c = osc(tbl, f0, n, rng.uniform(0, 1), fm=fm_v)
    lo = osc(tbl, f0 * 1.006, n, rng.uniform(0, 1), fm=fm_v) * 0.6
    ro = osc(tbl, f0 * 0.994, n, rng.uniform(0, 1), fm=fm_v) * 0.6
    sq = osc("sq_mid", f0 / 2, n, rng.uniform(0, 1)) * 0.28
    core = c + lo + ro + sq
    env = adsr(n, 0.012, 0.20, 0.8, min(0.30, dur * 0.5))
    core *= env
    # formant-ish resonant filters (two peaks) -> vocal hybrid character
    sos_f1 = scipy.signal.butter(2, [700, 1300], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_f2 = scipy.signal.butter(2, [2200, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    f1 = scipy.signal.sosfilt(sos_f1, core) * 0.9
    f2 = scipy.signal.sosfilt(sos_f2, core) * 0.35
    body = pb.Pedalboard([pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=3900, resonance=0.18),
                          pb.Chorus(rate_hz=0.9, depth=0.3, mix=0.3)])(np.stack([core, core], axis=1), SAMPLE_RATE)
    sig = body + np.stack([f1, f1], axis=1) * 0.5 + np.stack([f2, f2], axis=1) * 0.5
    if fifth:
        f5 = midi_hz(midi - 7)
        w5 = osc(tbl, f5, n, rng.uniform(0, 1), fm=fm_v * 0.6) * env * 0.22
        sig = sig + np.stack([w5 * 0.8, w5], axis=1)
    return sig.astype(np.float32)

def ks_pluck(midi: float, dur: float) -> np.ndarray:
    """Karplus-Strong plucked electric guitar."""
    n = int(dur * SAMPLE_RATE)
    f0 = midi_hz(midi)
    period = max(2, int(round(SAMPLE_RATE / f0)))
    burst = rng.standard_normal(period).astype(np.float32)
    burst = scipy.signal.sosfilt(
        scipy.signal.butter(2, [max(40.0, f0 * 0.8), 9000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), burst)
    burst /= (np.max(np.abs(burst)) + 1e-9)
    damp = 0.497 + 0.002 * np.exp(-f0 / 900)   # loop gain per pass
    a = np.zeros(period + 2)
    a[0] = 1.0
    a[period] = -damp
    a[period + 1] = -damp
    x = np.zeros(n, dtype=np.float32)
    x[: period] = burst
    total = scipy.signal.lfilter([1.0], a, x).astype(np.float32)
    env = np.ones(n, dtype=np.float32)
    rel = min(n - 1, int(0.05 * SAMPLE_RATE))
    env[-rel:] *= np.linspace(1, 0, rel)
    pick = rng.standard_normal(int(0.004 * SAMPLE_RATE)).astype(np.float32)
    pick = scipy.signal.sosfilt(scipy.signal.butter(2, [2000, 8000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), pick)
    sig = total * env
    sig[: len(pick)] += pick * 0.25
    warm = pb.Pedalboard([pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF12, cutoff_hz=5200, resonance=0.12),
                          pb.Chorus(rate_hz=1.4, depth=0.35, mix=0.35)])(np.stack([sig, sig], axis=1), SAMPLE_RATE)
    return warm.astype(np.float32)

def arp_note(midi: float, dur: float) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    f0 = midi_hz(midi)
    w = osc("sq_mid", f0, n, rng.uniform(0, 1)) * 0.6 + osc("saw_high", f0, n, rng.uniform(0, 1)) * 0.4
    env = adsr(n, 0.002, 0.05, 0.18, min(0.05, dur * 0.6))
    sig = w * env
    filt = pb.Pedalboard([pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=2600, resonance=0.4)])
    return filt(np.stack([sig, sig], axis=1), SAMPLE_RATE)

def bell(midi: float, dur: float) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = midi_hz(midi)
    fm = (1.8 * np.exp(-8 * t) + 0.1) * np.sin(2 * np.pi * f0 * 3.5 * t) / (2 * np.pi)
    car = np.sin(2 * np.pi * f0 * t + 2 * np.pi * fm) * np.exp(-2.8 * t)
    shimmer = np.sin(2 * np.pi * f0 * 2.76 * t) * np.exp(-5.0 * t) * 0.2
    return ((car + shimmer) * 0.9).astype(np.float32)

# ===========================================================================
# ARRANGEMENT DATA
# ===========================================================================
def build_chord_map() -> list[str]:
    """112-bar chord map derived from the analysis."""
    m = []
    L1 = ["G#m", "E", "G#m", "D#sus4"]        # intro / breakdowns / outro
    L2 = ["G#m", "E", "E", "D#sus4"]          # main hook loop
    L3 = ["E", "G#m", "E", "D#sus4"]          # riff loop (leans E)
    LM = ["Emaj7", "G#m", "E", "G#m"]         # B-section variation (bars 57-64)
    m += L1 * 2                                                    # 1-8
    m += L2 * 4                                                    # 9-24
    m += L3 * 4                                                    # 25-40
    # 41-47 breakdown (7 bars)
    m += ["G#m", "E", "G#m", "D#sus4", "G#m", "E", "D#sus4"]       # 41-47
    m += L2 * 2                                                    # 48-55
    m += LM * 2                                                    # 56-63
    m += ["E", "G#m", "Emaj7", "G#m",                              # 64-67
          "E", "G#m", "Emaj7", "G#m",                              # 68-71
          "E", "G#m", "E", "D#sus4",                               # 72-75
          "E", "G#m", "E", "D#sus4"]                               # 76-79
    m += ["G#m", "E", "G#m", "D#sus4", "G#m", "E", "D#sus4"]       # 80-86 (80 = fill/transition bar)
    m += ["D#sus4"]                                                # 87 riser
    m += L2 * 3                                                    # 88-99
    m += ["E", "G#m", "E", "D#sus4"]                               # 100-103?? -> keep in C
    # fix: 88-99 is 12 bars (L2*3), then 100-102: E G#m E, 103 starts outro
    m = m[:100]
    m += ["E", "G#m", "E"]                                         # 101-103? adjust below
    m = m[:102]
    m += ["D#sus4"]                                                # 103 -> actually bar 103
    # outro 104-112
    m += ["G#m", "E", "G#m", "D#sus4", "G#m", "E", "D#sus4", "G#m", "E"][: 112 - len(m)]
    while len(m) < N_BARS:
        m.append("G#m")
    return m[:N_BARS]

CHORDS = build_chord_map()

# Lead melody events: (bar, slot, length_sixteenths, midi)
MELODY: list[tuple[int, int, int, int]] = [
    # pickup into the first drop
    (7, 15, 1, 80), (8, 2, 11, 83), (8, 14, 2, 82),
    # A1 hook phrase (bars 9-16)
    (9, 2, 6, 82),
    (10, 12, 4, 82),
    (11, 5, 2, 82), (11, 11, 5, 83),
    (12, 3, 1, 78), (12, 5, 3, 78),
    (13, 5, 2, 80),
    (14, 2, 2, 78), (14, 10, 6, 82),
    (15, 4, 2, 82), (15, 10, 6, 80),
    (16, 12, 2, 75),
    # call/response stabs (17-24)
    (17, 4, 1, 80), (18, 7, 1, 80), (18, 12, 2, 75), (20, 11, 1, 75),
    (21, 13, 1, 80), (22, 10, 2, 75), (23, 12, 1, 80), (24, 10, 2, 75),
    # A2 riff (25-39) - played by the guitar pluck
    (25, 2, 1, 80), (25, 15, 1, 83),
    (26, 5, 1, 80), (26, 10, 1, 75), (26, 13, 1, 80),
    (27, 1, 1, 80),
    (28, 9, 2, 75), (28, 12, 1, 80),
    (29, 0, 2, 80), (29, 4, 1, 80), (29, 12, 1, 80),
    (30, 0, 1, 80), (30, 3, 3, 80), (30, 8, 1, 75), (30, 12, 1, 80),
    (32, 3, 1, 80), (32, 5, 1, 75), (32, 10, 6, 80),
    (33, 10, 6, 80),
    (34, 2, 2, 80), (34, 10, 2, 80), (34, 15, 5, 80),
    (35, 14, 2, 80),
    (36, 1, 3, 80), (36, 10, 6, 80),
    (37, 14, 2, 80),
    (38, 1, 5, 80), (38, 6, 1, 75), (38, 9, 7, 80),
    (39, 9, 2, 80), (39, 11, 1, 83), (39, 14, 1, 80),
    # B phrases (48-56)
    (48, 7, 3, 80), (48, 11, 2, 83),
    (49, 1, 1, 78), (49, 5, 2, 80), (49, 9, 1, 80),
    (50, 3, 2, 75),
    (51, 2, 1, 80), (51, 4, 5, 80), (51, 12, 1, 82),
    (52, 10, 1, 83), (52, 11, 4, 80),
    (53, 2, 6, 80), (53, 13, 2, 80),
    (54, 0, 1, 78), (54, 2, 2, 75), (54, 13, 2, 80),
    (55, 1, 3, 80), (55, 7, 1, 80),
    (56, 4, 2, 80), (56, 11, 2, 80),
    # sparse anchors (57-64)
    (58, 1, 2, 75), (62, 0, 1, 75), (64, 0, 1, 75), (64, 9, 1, 75), (64, 15, 1, 80),
    # sky anthem (65-79): long sustained G#5 with swells
    (65, 10, 2, 80),
    (66, 0, 1, 82), (66, 2, 1, 80), (66, 8, 4, 80), (66, 14, 2, 80),
    (67, 2, 2, 80),
    (69, 11, 1, 75), (69, 14, 1, 75), (69, 15, 1, 82),
    (70, 5, 1, 80), (70, 8, 2, 80),
    (71, 12, 4, 80), (72, 0, 16, 80), (74, 10, 6, 80), (76, 10, 6, 80),
    (78, 2, 6, 80), (78, 10, 2, 80), (78, 14, 2, 80),
    (79, 1, 1, 83), (79, 3, 2, 80),
    # breakdown 2 + riser (81-87)
    (81, 2, 5, 80), (81, 8, 8, 82),
    (83, 4, 1, 85), (83, 8, 2, 80), (83, 11, 2, 78), (83, 15, 1, 82),
    (84, 1, 3, 83), (84, 5, 1, 80), (84, 7, 8, 80),
    (87, 5, 1, 85), (87, 6, 2, 78),
    # final drop (88-102)
    (88, 10, 2, 80), (89, 10, 3, 82), (90, 6, 5, 76),
    (92, 1, 2, 75), (94, 9, 2, 80), (95, 8, 2, 75),
    (96, 12, 4, 85), (97, 0, 1, 85), (97, 2, 1, 85), (97, 7, 1, 82),
    (98, 0, 2, 83), (98, 4, 1, 82), (98, 9, 1, 82),
    (100, 14, 2, 85), (101, 13, 3, 83),
    # outro fragments
    (104, 2, 1, 80), (104, 8, 1, 75), (105, 5, 4, 82), (106, 11, 4, 85),
    (107, 0, 1, 80), (107, 7, 1, 80), (107, 8, 8, 82),
    (111, 5, 3, 80), (111, 9, 3, 80), (111, 13, 3, 80),
    (112, 1, 2, 80), (112, 5, 4, 80),
]

# Bass rhythm patterns per 4-bar loop, as (slot, len_sixteenths, degree) where
# degree is one of: 0 root, 1 minor 3rd, 3 fifth, 4 octave, -1 passing 4th below octave
BASS_PAT_A = [(0, 3, 0), (3, 2, 3), (6, 2, 0), (8, 2, 4), (11, 3, 3), (14, 2, 4)]
BASS_PAT_B = [(0, 2, 0), (3, 1, 1), (6, 3, 0), (10, 2, 4), (12, 2, 3), (14, 2, 4)]
BASS_PAT_BREAK = [(0, 6, 0), (11, 5, 4)]
BASS_PASSING = 5  # semitone offset for passing 4th (perfect 4th above root)

# ===========================================================================
# SECTION DEFINITIONS
# ===========================================================================
def in_range(bar: int, a: int, b: int) -> bool:
    return a <= bar <= b

def groove_bars(bar: int) -> str | None:
    """Which drum arrangement applies at this bar."""
    if in_range(bar, 9, 24):  return "A"
    if in_range(bar, 25, 40): return "A2"
    if in_range(bar, 48, 79): return "B"
    if in_range(bar, 88, 102): return "C"
    return None

FILL_BARS = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 87, 88, 96, 102}

# ===========================================================================
# RENDER PIPELINE
# ===========================================================================
def render() -> np.ndarray:
    drums = new_bus()     # dry drums (reverb on bus later)
    bass = new_bus()
    keys = new_bus()      # EP + pad (sidechained)
    lead = new_bus()      # hybrid lead
    guitar = new_bus()    # KS pluck
    arp = new_bus()
    bells = new_bus()
    fx = new_bus()        # risers/drops/sweeps

    kick_times: list[float] = []

    # --- pre-computed one-shots ---
    KICK = kick(True); KICK_SOFT = kick(False)
    SNARE = snare(); CLAP = clap()
    HAT = hat(False); OHAT = hat(True); RIDE = ride()
    CRASH = crash(False); RCRASH = crash(True)

    # ------------------------------------------------------------------
    # DRUMS
    # ------------------------------------------------------------------
    for bar in range(1, N_BARS + 1):
        g = groove_bars(bar)
        if g:
            # kick slots per style (16th slots)
            if g == "A":
                kicks = [0, 6, 11]
            elif g == "A2":
                kicks = [0, 6, 11] + ([14] if bar % 2 == 0 else [])
            elif g == "B":
                kicks = [0, 6, 11] if bar % 4 else [0, 6, 9, 11]
            else:  # C
                kicks = [0, 6, 11] + ([3, 14] if bar % 2 else [])
            for s in kicks:
                place(drums, KICK, slot_time(bar, s))
                kick_times.append(slot_time(bar, s))
            # snare on beats 2 & 4 + ghosts
            for s in (4, 12):
                place(drums, SNARE, slot_time(bar, s))
            if bar % 2 == 1:
                place(drums, SNARE * 0.35, slot_time(bar, 7))
            else:
                place(drums, SNARE * 0.30, slot_time(bar, 15))
            if g in ("B", "C"):
                place(drums, CLAP * (0.7 if g == "B" else 0.9), slot_time(bar, 12))
            # hats: dense 16ths, accented offbeats
            for s in range(16):
                vel = 1.00 if s in (2, 6, 10, 14) else (0.66 if s % 4 == 0 else 0.48)
                if g == "C" and s % 2 == 0:
                    continue  # ride carries 8ths in C; closed hats on offbeat 16ths only
                if s == 10 and g in ("A", "B") and bar % 2 == 1:
                    place(drums, OHAT * vel, slot_time(bar, s))
                else:
                    place(drums, HAT * vel, slot_time(bar, s))
            if g == "C":
                for s in range(0, 16, 2):
                    place(drums, RIDE * (0.8 if s % 4 == 0 else 0.5), slot_time(bar, s))
        # fills
        if bar in FILL_BARS:
            for i, s in enumerate([12, 13, 14, 15]):
                place(drums, snare_roll_hit(), slot_time(bar, s) - 0.004 * i)
        # transition bars 40 / 80: hats only + roll
        if bar in (40, 80):
            for s_ in range(0, 12, 2):
                place(drums, HAT * 0.35, slot_time(bar, s_))
        # intro hats fading in
        if in_range(bar, 4, 6):
            for s in range(0, 16, 2):
                place(drums, HAT * (0.22 + 0.06 * (bar - 4)), slot_time(bar, s))
        if bar == 7:
            for s in range(0, 16, 2):
                place(drums, HAT * 0.4, slot_time(bar, s))
            place(drums, SNARE * 0.5, slot_time(bar, 12))
        # crashes / impacts on section heads
        if bar in (9, 25, 48, 88, 96):
            place(drums, CRASH * (1.0 if bar != 96 else 0.8), bar_start(bar))
            place(drums, KICK, bar_start(bar))
            kick_times.append(bar_start(bar))
        if bar in (10, 26, 49, 89):
            place(fx, RCRASH * 0.9, bar_start(bar) - BAR * 0.9)
        if bar in (41, 103):
            place(fx, noise_sweep_down(2.0), bar_start(bar))

    # breakdown fills into sections
    for b0, nhits in ((47, 8), (86, 6)):
        step = 16 // nhits
        for i in range(nhits):
            t = slot_time(b0, i * step)
            place(drums, snare_roll_hit() * (0.4 + 0.6 * i / max(1, nhits - 1)), t)

    # risers
    place(fx, riser(2.0), bar_start(7))
    place(fx, riser(2.0), bar_start(46))
    place(fx, riser(2.0), bar_start(85))
    place(fx, riser(1.0) * 1.1, bar_start(87))
    # sub drops at breakdown starts / final drop
    place(fx, sub_drop(midi_hz(39 - 12)), slot_time(40, 8))
    place(fx, sub_drop(midi_hz(39 - 12)), slot_time(87, 0))
    place(fx, downlifter(), bar_start(48))

    # ------------------------------------------------------------------
    # BASS
    # ------------------------------------------------------------------
    for bar in range(1, N_BARS + 1):
        g = groove_bars(bar)
        chord = CHORDS[bar - 1]
        root = CHORD_ROOTS[chord]
        if chord == "E":
            root = 28
        if g in ("A", "A2", "B", "C"):
            pat = BASS_PAT_B if bar % 2 == 0 else BASS_PAT_A
            if g == "C":
                pat = BASS_PAT_A + [(8, 2, 1)]
            bright = 1.0 + (0.25 if g == "C" else 0.0)
            # continuous sub bed keeps the low end ringing between hits
            place(bass, bass_note(root, BAR * 1.0, 0.75) * 0.42, bar_start(bar))
            for (s, ln, deg) in pat:
                note = root + {0: 0, 1: 3, 3: 7, 4: 12, -1: BASS_PASSING}[deg]
                if bar in FILL_BARS and s in (12, 14):
                    note = root + 12  # octave lift into fills
                place(bass, bass_note(note, ln * SIX * 1.35, bright), slot_time(bar, s))
        elif in_range(bar, 41, 47) or in_range(bar, 80, 87):
            for (s, ln, deg) in BASS_PAT_BREAK:
                note = root + (12 if deg == 4 else 0)
                place(bass, bass_note(note, ln * SIX, 0.8) * 0.35, slot_time(bar, s))
        elif bar <= 8 or bar == 40 or bar >= 80:
            # soft whole-bar sub
            place(bass, bass_note(root, BAR * 0.98, 0.7) * 0.30, bar_start(bar))

    # ------------------------------------------------------------------
    # HARMONY: pad (whole bar) + EP (rhythmic stabs)
    # ------------------------------------------------------------------
    for bar in range(1, N_BARS + 1):
        chord = CHORDS[bar - 1]
        voicing = CHORD_VOICINGS.get(chord, CHORD_VOICINGS["G#m"])
        g = groove_bars(bar)
        quiet = (bar <= 8 or bar == 40 or in_range(bar, 41, 47)
                 or in_range(bar, 80, 86) or bar >= 104)
        # pad
        for i, note in enumerate(voicing):
            p = pad_note(note, BAR * 1.05)
            pp = pan(p, [-0.6, -0.2, 0.2, 0.6][i % 4])
            place(keys, pp * (0.30 if quiet else 0.42), bar_start(bar))
        # EP stabs
        if g:
            slots = [0, 6, 11] if g in ("A", "A2") else ([0, 6, 11, 14] if g == "C" else [0, 6, 10])
            for k_i, s in enumerate(slots):
                for j, note in enumerate(voicing[1:]):
                    q = ep_note(note, SIX * 2.2)
                    qp = pan(q, [-0.5, 0.1, 0.55][j % 3])
                    place(keys, qp * 0.16, slot_time(bar, s) + 0.012 * (k_i % 2))
        elif quiet:
            for s, ln in ((0, 6), (11, 5)):
                for j, note in enumerate(voicing[1:]):
                    q = ep_note(note, ln * SIX * 1.4)
                    qp = pan(q, [-0.4, 0.0, 0.5][j % 3])
                    place(keys, qp * 0.10, slot_time(bar, s))

    # ------------------------------------------------------------------
    # LEAD (hook phrases) & GUITAR (riff) & ARP & BELLS
    # ------------------------------------------------------------------
    lead_skip_bars = set(range(25, 40))  # guitar takes over the riff in A2
    for (bar, slot, ln, midi) in MELODY:
        t = slot_time(bar, slot)
        dur = ln * SIX * 1.02
        quiet_zone = (bar <= 8 or bar == 40 or in_range(bar, 41, 47)
                      or in_range(bar, 80, 86) or bar >= 104)
        if bar in lead_skip_bars:
            pl = ks_pluck(midi, min(dur * 1.4, 1.2))
            place(guitar, pan(pl, 0.25) * 0.8, t)
        else:
            lv = lead_note(midi, dur)
            lvl = 0.40 if quiet_zone else 0.62
            place(lead, pan(lv, -0.08) * lvl, t)
            # guitar doubles an octave-driven unison in the final drop
            if in_range(bar, 88, 102):
                pl = ks_pluck(midi, min(dur * 1.3, 1.1))
                place(guitar, pan(pl, 0.3) * 0.55, t)
    # guitar also answers the hook stabs (17-24) with ghost plucks
    for (bar, slot, ln, midi) in MELODY:
        if in_range(bar, 17, 24):
            pl = ks_pluck(midi + 12, 0.35)
            place(guitar, pan(pl, 0.45) * 0.30, slot_time(bar, slot) + SIX * 0.5)

    # arp in breakdowns: 16ths over chord tones
    for zone in (range(41, 48), range(81, 88)):
        for bar in zone:
            chord = CHORDS[bar - 1]
            tones = CHORD_VOICINGS.get(chord, CHORD_VOICINGS["G#m"]) + \
                    [n + 12 for n in CHORD_VOICINGS.get(chord, CHORD_VOICINGS["G#m"])[:2]]
            seq = [tones[(i * 2) % len(tones)] + 12 for i in range(16)]
            for i, note in enumerate(seq):
                a = arp_note(note, SIX * 1.1)
                place(arp, pan(a, -0.8 + 1.6 * (i % 2)) * 0.22, slot_time(bar, i))

    # bells in final drop (counter accents)
    for bar, slot, midi in [(90, 0, 76), (92, 8, 80), (94, 0, 83), (96, 8, 88),
                            (98, 0, 85), (100, 8, 88), (102, 0, 90)]:
        b = bell(midi, 1.4)
        place(bells, pan(b, 0.5 if bar % 2 else -0.5) * 0.24, slot_time(bar, slot))

    # ------------------------------------------------------------------
    # SIDECHAIN (bass + keys pump keyed by kick)
    # ------------------------------------------------------------------
    duck = np.ones(N_TOTAL, dtype=np.float32)
    kt = np.array(kick_times)
    idx = ((kt * SAMPLE_RATE).astype(int))
    idx = idx[idx < N_TOTAL]
    sc_len = int(0.16 * SAMPLE_RATE)
    dec = np.exp(-np.linspace(0, 7.5, sc_len)).astype(np.float32)
    duck_env = np.zeros(N_TOTAL + sc_len, dtype=np.float32)
    for i in idx:
        duck_env[i:i + sc_len] += dec
    duck = 1.0 - 0.55 * np.clip(duck_env[:N_TOTAL], 0, 1)
    bass *= duck[:, None]
    keys *= (1.0 - 0.32 * (1 - duck))[:, None]

    # ------------------------------------------------------------------
    # BUS PROCESSING
    # ------------------------------------------------------------------
    SR = SAMPLE_RATE

    # drum bus: glue comp + plate room + presence
    drum_chain = pb.Pedalboard([
        pb.Compressor(threshold_db=-16, ratio=1.9, attack_ms=8.0, release_ms=110),
        pb.Reverb(room_size=0.28, damping=0.42, wet_level=0.14, dry_level=1.0),
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowShelfFilter(cutoff_frequency_hz=90, gain_db=1.5, q=0.7),
    ])
    drums = drum_chain(drums, SR)

    # snare needs a big tail -> dedicated plate send computed from dry snare hits
    # (approximated by sending the whole drum bus through a second reverb and
    #  mixing in only where snare hits occur is overkill; global small plate is fine)

    bass_chain = pb.Pedalboard([
        pb.Compressor(threshold_db=-16, ratio=3.0, attack_ms=8.0, release_ms=80),
        pb.HighpassFilter(cutoff_frequency_hz=24),
        pb.LowShelfFilter(cutoff_frequency_hz=70, gain_db=1.0, q=0.7),
    ])
    bass = bass_chain(bass, SR)

    keys_chain = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=120),
        pb.Chorus(rate_hz=0.6, depth=0.4, mix=0.3),
        pb.Reverb(room_size=0.55, damping=0.5, wet_level=0.30, dry_level=0.85),
    ])
    keys = keys_chain(keys, SR)

    lead_fx = pb.Pedalboard([
        pb.Delay(delay_seconds=BEAT * 0.75, feedback=0.34, mix=0.26),
        pb.Reverb(room_size=0.62, damping=0.4, wet_level=0.28, dry_level=0.9),
        pb.HighpassFilter(cutoff_frequency_hz=120),
    ])
    lead = lead_fx(lead, SR)

    guitar_fx = pb.Pedalboard([
        pb.Delay(delay_seconds=BEAT * 0.5, feedback=0.25, mix=0.22),
        pb.Reverb(room_size=0.45, damping=0.45, wet_level=0.24, dry_level=0.9),
        pb.HighpassFilter(cutoff_frequency_hz=180),
    ])
    guitar = guitar_fx(guitar, SR)

    arp_fx = pb.Pedalboard([
        pb.Delay(delay_seconds=BEAT * 0.75, feedback=0.45, mix=0.4),
        pb.Reverb(room_size=0.7, damping=0.35, wet_level=0.35, dry_level=0.8),
        pb.HighpassFilter(cutoff_frequency_hz=300),
    ])
    arp = arp_fx(arp, SR)

    bells_fx = pb.Pedalboard([
        pb.Reverb(room_size=0.85, damping=0.3, wet_level=0.45, dry_level=0.7),
        pb.HighpassFilter(cutoff_frequency_hz=400),
    ])
    bells = bells_fx(bells, SR)

    fx_fx = pb.Pedalboard([
        pb.Reverb(room_size=0.6, damping=0.4, wet_level=0.2, dry_level=1.0),
        pb.HighpassFilter(cutoff_frequency_hz=200),
    ])
    fx = fx_fx(fx, SR)

    # ------------------------------------------------------------------
    # MIX
    # ------------------------------------------------------------------
    mix = (drums * 1.00 + bass * 1.25 + keys * 1.15 + lead * 1.12 +
           guitar * 0.68 + arp * 0.52 + bells * 1.00 + fx * 0.85)

    # section zone gains (measured against reference bar-RMS profile)
    zone_db = np.zeros(N_BARS)
    zone_db[0:4] = -21.0       # intro (very quiet)
    zone_db[4:6] = -18.5
    zone_db[6] = -16.0         # bar 7 pickup
    zone_db[7] = -10.0         # bar 8 ramp
    zone_db[39] = -10.0        # bar 40 turnaround
    zone_db[40:43] = -21.0     # breakdown 1
    zone_db[43:47] = -14.0
    zone_db[79] = -17.0        # bar 80 turnaround
    zone_db[80:83] = -17.0     # breakdown 2
    zone_db[83:86] = -15.0
    zone_db[102] = -12.0       # bar 103 outro settle
    zone_db[103:108] = -21.0
    zone_db[108:110] = -20.0
    zone_db[110] = -26.0
    zone_db[111] = -44.0
    zone_gain = 10 ** (zone_db / 20.0)
    samp_gain = np.ones(N_TOTAL, dtype=np.float32)
    for b in range(N_BARS):
        i0 = int(bar_start(b + 1) * SAMPLE_RATE)
        i1 = int((bar_start(b + 1) + BAR) * SAMPLE_RATE) if b < N_BARS - 1 else N_TOTAL
        samp_gain[i0:i1] = zone_gain[b]
    mix = mix * samp_gain[:, None]

    # master chain
    master = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=22),
        pb.Compressor(threshold_db=-18, ratio=1.35, attack_ms=30, release_ms=220),
        pb.LowShelfFilter(cutoff_frequency_hz=52, gain_db=6.0, q=0.7),
        pb.PeakFilter(cutoff_frequency_hz=41, gain_db=5.0, q=1.2),
        pb.PeakFilter(cutoff_frequency_hz=170, gain_db=3.0, q=0.9),
        pb.PeakFilter(cutoff_frequency_hz=380, gain_db=4.5, q=0.9),
        pb.PeakFilter(cutoff_frequency_hz=2800, gain_db=4.0, q=0.8),
        pb.HighShelfFilter(cutoff_frequency_hz=9000, gain_db=-0.5, q=0.7),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-1.0, release_ms=60.0),
    ])
    mix = master(mix, SR)

    # iterative normalize -> limit (makeup gain loop) to hit target LUFS at
    # a dense, competitive true-peak level like the reference master
    meter = pyln.Meter(SR)
    limiter = pb.Limiter(threshold_db=CEILING_DBTP, release_ms=45.0)

    def limited_lufs(g_db: float) -> float:
        trial = mix * (10 ** (g_db / 20.0))
        trial = limiter(trial, SR)
        return meter.integrated_loudness(trial.astype(np.float64))

    # bisection on drive gain: limiter loudness response is monotonic-ish
    lo_db, hi_db = -12.0, 24.0
    for _ in range(12):
        mid_db = 0.5 * (lo_db + hi_db)
        v = limited_lufs(mid_db)
        if v < TARGET_LUFS:
            lo_db = mid_db
        else:
            hi_db = mid_db
        if hi_db - lo_db < 0.05:
            break
    drive_db = 0.5 * (lo_db + hi_db)
    mix = limiter(mix * (10 ** (drive_db / 20.0)), SR)
    print(f"master drive: {drive_db:+.2f} dB", flush=True)
    # final safety: only trims if overs exceed ceiling
    up = scipy.signal.resample_poly(mix, 2, 1, axis=0)
    pk = float(np.max(np.abs(up)))
    if pk > 10 ** (CEILING_DBTP / 20.0):
        mix *= (10 ** (CEILING_DBTP / 20.0) / pk) * 0.999

    # long musical fade at the tail (reference fades to silence ~186.9 s)
    fade_start = 183.9
    i_fade = int(fade_start * SAMPLE_RATE)
    fade = np.exp(np.linspace(0, -9.0, len(mix) - i_fade)).astype(np.float32)
    mix[i_fade:] *= fade[:, None]
    return mix.astype(np.float32)

# ---------------------------------------------------------------------------
# Encode + tag
# ---------------------------------------------------------------------------
def encode_and_tag(mix: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "master.wav")
        sf.write(wav, mix, SAMPLE_RATE, subtype="FLOAT")
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-i", wav]
        if COVER_PNG.exists():
            cmd += ["-i", str(COVER_PNG)]
        cmd += ["-map", "0:a"]
        if COVER_PNG.exists():
            cmd += ["-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic",
                    "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)"]
        cmd += ["-c:a", "libmp3lame", "-b:a", "320k", "-ar", "48000", "-ac", "2",
                "-id3v2_version", "3", "-write_xing", "1",
                "-metadata", f"title={OUT_TITLE}",
                "-metadata", f"artist={OUT_ARTIST}",
                "-metadata", f"album_artist={OUT_ALBUM_ARTIST}",
                "-metadata", f"composer={OUT_COMPOSER}",
                "-metadata", f"copyright={OUT_COPYRIGHT}",
                "-metadata", f"comment={OUT_COMMENT}",
                "-metadata", "genre=Electronic",
                "-metadata", "date=2026",
                str(OUT_MP3)]
        subprocess.run(cmd, check=True)

def main() -> None:
    print("Rendering arrangement ...", flush=True)
    mix = render()
    peak_db = 20 * np.log10(np.max(np.abs(mix)) + 1e-12)
    meter = pyln.Meter(SAMPLE_RATE)
    lufs = meter.integrated_loudness(mix.astype(np.float64))
    print(f"rendered: {len(mix)/SAMPLE_RATE:.3f}s  LUFS {lufs:.2f}  peak {peak_db:.2f} dBFS", flush=True)
    print("Encoding MP3 ...", flush=True)
    encode_and_tag(mix)
    print(f"Wrote {OUT_MP3}  ({OUT_MP3.stat().st_size/1e6:.2f} MB)", flush=True)

if __name__ == "__main__":
    main()
