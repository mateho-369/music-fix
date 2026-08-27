#!/usr/bin/env python3
"""Render annotated spectrograms of the reference (structure + groove zooms)."""
import numpy as np, librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

y, sr = librosa.load("/tmp/analysis/ref48.wav", sr=22050, mono=True)
tempo = 143.55; beat = 60.0 / tempo; bar = 4 * beat; t0 = 0.070


def spec(t_start, t_end, bars_range, fname, title):
    seg = y[int(t_start * sr):int(t_end * sr)]
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(seg, n_fft=2048, hop_length=256, win_length=2048)), ref=np.max)
    fig, ax = plt.subplots(figsize=(16, 6))
    img = ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                    extent=[t_start, t_end, 0, sr / 2 / 1000], vmin=-90, vmax=0)
    ax.set_yscale("log"); ax.set_ylim(0.03, 11)
    for b in np.arange(bars_range[0] - 1, bars_range[1] + 1, 1):
        bt = t0 + b * bar
        if t_start <= bt <= t_end:
            c = "cyan" if b % 4 == 0 else "gray"
            ax.axvline(bt, color=c, lw=1.4 if b % 4 == 0 else 0.4, alpha=0.8)
            if b % 4 == 0:
                ax.text(bt + 0.02, 8.5, f"bar{b+1}", color="cyan", fontsize=8)
    ax.set_title(title); ax.set_xlabel("time (s)"); ax.set_ylabel("kHz")
    fig.colorbar(img, ax=ax, format="%+dB")
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)
    print("saved", fname)


spec(0, 187.4, (0, 112), "/tmp/analysis/spec_full.png", "FULL TRACK - bar lines every 4 bars")
spec(t0 + 8 * bar, t0 + 12 * bar, (8, 12), "/tmp/analysis/spec_groove_A.png", "GROOVE ZOOM bars 9-13")
spec(t0 + 64 * bar, t0 + 68 * bar, (64, 68), "/tmp/analysis/spec_groove_B.png", "GROOVE ZOOM bars 65-69")
spec(t0 + 39 * bar, t0 + 48 * bar, (39, 48), "/tmp/analysis/spec_break.png", "BREAKDOWN bars 40-49")
