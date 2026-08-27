#!/usr/bin/env python3
"""Compare generated output vs reference: bar-RMS profile, short-term LUFS, spectra."""
from __future__ import annotations
import sys
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import scipy.signal

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/analysis/ref48.wav"
NEW = sys.argv[2] if len(sys.argv) > 2 else "Ma Teho - Playmaker.mp3"

BPM = 143.55; BAR = 4 * 60.0 / BPM; T0 = 0.070

def analyze(path, label):
    cmd = ["ffmpeg_decode"]  # placeholder
    y48, sr48 = sf.read(path, dtype="float32", always_2d=True)
    dur = len(y48) / sr48
    meter = pyln.Meter(sr48)
    lufs = meter.integrated_loudness(y48.astype(np.float64))
    up = scipy.signal.resample_poly(y48, 4, 1, axis=0)
    tp = 20 * np.log10(np.max(np.abs(up)) + 1e-12)
    y, sr = librosa.load(path, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rt = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    bar_rms = []
    for b in range(112):
        s = T0 + b * BAR
        m = (rt >= s) & (rt < s + BAR)
        bar_rms.append(20 * np.log10(rms[m].mean() + 1e-9) if m.any() else -120)
    # spectra
    ym = y48.mean(axis=1)
    f_st, t_st, Z = scipy.signal.stft(ym, fs=sr48, nperseg=8192)
    Pxx = np.abs(Z) ** 2
    bands = [40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 14000]
    spec = []
    for fc in bands:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        sel = (f_st >= lo) & (f_st <= hi)
        spec.append(10 * np.log10(Pxx[sel].mean() + 1e-20))
    spec = np.array(spec)
    print(f"== {label}: {dur:.2f}s  LUFS {lufs:.2f}  TP {tp:.2f} dB")
    return dur, lufs, tp, np.array(bar_rms), spec

d1, l1, tp1, br1, sp1 = analyze(REF, "REFERENCE")
d2, l2, tp2, br2, sp2 = analyze(NEW, "NEW     ")

print("\n=== BAR RMS DIFF (new - ref, dB) ===")
for i in range(0, 112, 16):
    diff = br2[i:i+16] - br1[i:i+16]
    print(f"bars {i+1:3d}-{i+16:3d}: " + " ".join(f"{v:+5.1f}" for v in diff))
print("\n=== SPECTRUM (ref / new, dB rel own 1k) ===")
bands = [40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 14000]
i1k = bands.index(1000)
r = sp1 - sp1[i1k]; n = sp2 - sp2[i1k]
print("band    : " + " ".join(f"{b:>6}" for b in bands))
print("ref     : " + " ".join(f"{v:6.1f}" for v in r))
print("new     : " + " ".join(f"{v:6.1f}" for v in n))
print("diff    : " + " ".join(f"{v:6.1f}" for v in (n - r)))
