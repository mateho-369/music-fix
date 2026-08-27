#!/usr/bin/env python3
"""Deep analysis of the reference track 'wanna play ? (Remix) (Cover) (Cover) (Cover).mp3'.

Extracts: duration, tempo/beat grid, key, beat-synced chord progression,
melodic pitch contour (pyin), drum-band pattern grids, section RMS profile,
stereo imaging, spectral balance, LUFS/true-peak metering.

Usage: python3 analyze_reference.py [ref.wav] > report.txt
"""
from __future__ import annotations

import json
import sys

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import scipy.signal

REF_WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/analysis/ref48.wav"
SR_ANALYSIS = 22050

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Chord templates: root + intervals -> (quality, intervals)
CHORD_TYPES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "sus4": [0, 5, 7],
}


def band_env(y: np.ndarray, sr: int, lo: float, hi: float, hop: int) -> np.ndarray:
    sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    f = scipy.signal.sosfilt(sos, y)
    return librosa.feature.rms(y=f, frame_length=1024, hop_length=hop)[0]


def main() -> None:
    y48, sr48 = sf.read(REF_WAV, dtype="float32", always_2d=True)
    n = len(y48)
    dur = n / sr48
    print(f"=== BASIC ===")
    print(f"duration: {dur:.3f} s ({int(dur//60)}:{dur%60:05.2f})  samples: {n}  sr: {sr48}")

    # --- Metering: integrated LUFS + true peak (4x oversampled) ---
    meter = pyln.Meter(sr48)
    lufs = meter.integrated_loudness(y48)
    up = scipy.signal.resample_poly(y48, 4, 1, axis=0)
    true_peak = float(np.max(np.abs(up)))
    short_term = []
    win = int(3.0 * sr48)
    for i in range(0, n - win, win):
        short_term.append(meter.integrated_loudness(y48[i : i + win]))
    short_term = np.array(short_term)
    print(f"integrated LUFS: {lufs:.2f}   true peak: {20*np.log10(true_peak+1e-12):.2f} dBTP   peak sample: {true_peak:.4f}")
    st_str = " ".join(f"{v:.0f}" for v in short_term[::2])
    print(f"short-term LUFS (every 6s): {st_str}")

    # --- Loudness curve (RMS dB, 0.5 s resolution) ---
    y_mono48 = y48.mean(axis=1)
    hop48 = sr48 // 2
    rms48 = librosa.feature.rms(y=y_mono48, frame_length=2048, hop_length=hop48)[0]
    rms_db = 20 * np.log10(rms48 + 1e-9)
    print("\n=== RMS PROFILE (1 s resolution, dBFS) ===")
    sec_rms = rms_db[: len(rms_db) // 2 * 2].reshape(-1, 2).mean(axis=1)
    for i in range(0, len(sec_rms), 16):
        row = sec_rms[i : i + 16]
        t0 = i
        line = " ".join(f"{v:5.1f}" for v in row)
        print(f"[{t0:3d}-{t0+len(row):3d}s] {line}")

    # --- librosa analysis at 22050 mono ---
    y, sr = librosa.load(REF_WAV, sr=SR_ANALYSIS, mono=True)

    # --- Tempo & beat grid ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=512, aggregate=None)
    tempo_med = float(np.median(tempo))
    ac = librosa.autocorrelate(onset_env, max_size=4 * SR_ANALYSIS // 512)
    est2 = 60.0 * sr / 512 / (np.argmax(ac[16:]) + 16)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    onbeat, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512, trim=False, bpm=tempo_med)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    beat_diffs = np.diff(beat_times)
    print(f"\n=== TEMPO ===")
    print(f"tempo (median tracking): {tempo_med:.2f} BPM | autocorr check: {est2:.1f} | tracker returned: {float(onbeat):.2f}")
    print(f"beat intervals: mean {beat_diffs.mean()*1000:.1f} ms  std {beat_diffs.std()*1000:.1f} ms")
    print(f"first beat at: {beat_times[0]:.3f} s, last beat at: {beat_times[-1]:.3f} s")

    # --- Key detection (global + per 8 bars) ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_octaves=5)
    chroma_mean = chroma.mean(axis=1)
    best = None
    for root in range(12):
        for prof_name, prof in (("maj", MAJOR_PROFILE), ("min", MINOR_PROFILE)):
            rotated = np.roll(prof, root)
            c = np.corrcoef(rotated, chroma_mean)[0, 1]
            if best is None or c > best[2]:
                best = (PITCH_CLASSES[root], prof_name, c)
    print(f"\n=== KEY === global: {best[0]} {best[1]} (corr {best[2]:.3f})")

    # --- Beat-synced chroma -> chord progression ---
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
    bs = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    n_beats = bs.shape[1]
    print(f"\n=== CHORDS (per beat, beat-index from first detected beat) ===")
    chord_line = []
    chord_list = []  # (beat_idx, label, score)
    for b in range(n_beats):
        v = bs[:, b]
        best_c = None
        for root in range(12):
            for qual, ivs in CHORD_TYPES.items():
                tpl = np.zeros(12)
                for iv in ivs:
                    tpl[(root + iv) % 12] = 1.0
                tpl /= tpl.sum()
                c = float(np.dot(v, tpl))
                if best_c is None or c > best_c[2]:
                    best_c = (PITCH_CLASSES[root], qual, c)
        chord_list.append((b, best_c[0] + best_c[1], best_c[2]))
        chord_line.append(f"{best_c[0]}{best_c[1]}")
    for i in range(0, len(chord_line), 16):
        b0 = i
        print(f"beat {b0:4d}-{b0+len(chord_line[i:i+16]):4d}: " + " ".join(f"{c:>5}" for c in chord_line[i:i+16]))

    # --- Melody extraction (pyin) ---
    print(f"\n=== MELODY (pyin) ===")
    f0, vflag, vprob = librosa.pyin(y, fmin=100, fmax=700, sr=sr, hop_length=320, frame_length=2048)
    f0_times = librosa.times_like(f0, sr=sr, hop_length=320)
    midi = librosa.hz_to_midi(f0)
    voiced = vflag & (vprob > 0.8)
    print(f"voiced fraction: {voiced.mean()*100:.1f}%")
    # segment into notes: median pitch over runs of >= 4 frames (~59 ms)
    notes = []
    i = 0
    nf = len(midi)
    while i < nf:
        if voiced[i] and not np.isnan(midi[i]):
            j = i
            while j < nf and voiced[j] and not np.isnan(midi[j]):
                j += 1
            run = midi[i:j]
            if len(run) >= 4:
                med = np.median(run)
                if np.percentile(run, 90) - np.percentile(run, 10) < 2.5:  # stable run
                    notes.append((f0_times[i], f0_times[j - 1], med))
            i = j
        else:
            i += 1
    # merge consecutive same-pitch notes
    merged = []
    for nt in notes:
        if merged and abs(nt[2] - merged[-1][2]) < 0.6 and nt[0] - merged[-1][1] < 0.12:
            merged[-1] = (merged[-1][0], nt[1], (merged[-1][2] + nt[2]) / 2)
        else:
            merged.append(list(nt))
    print(f"note segments: {len(merged)}")
    for t0, t1, p in merged:
        name = PITCH_CLASSES[int(round(p)) % 12] + str(int(round(p)) // 12 - 1)
        print(f"  {t0:7.3f} - {t1:7.3f}  ({t1-t0:5.2f}s)  midi {p:6.2f}  {name}")

    # --- Drum band grids ---
    print(f"\n=== DRUM BAND GRIDS (16th-note occupancy, first 32 bars) ===")
    hop = 512
    e_sub = band_env(y, sr, 30, 110, hop)      # kick / 808
    e_low = band_env(y, sr, 150, 300, hop)     # snare body
    e_noi = band_env(y, sr, 1200, 4000, hop)   # snare noise / clap
    e_hat = band_env(y, sr, 7000, 11000, hop)  # hats
    grid = np.linspace(beat_times[0], beat_times[-1], int((beat_times[-1] - beat_times[0]) / (beat_diffs.mean() / 4)) + 1)
    grid_frames = librosa.time_to_frames(grid, sr=sr, hop_length=hop)
    def occupancy(e):
        occ = np.zeros(len(grid) - 1)
        for g in range(len(grid) - 1):
            f0_, f1_ = grid_frames[g], max(grid_frames[g] + 1, grid_frames[g + 1])
            occ[g] = e[f0_:f1_].max()
        thr = occ.mean() + 0.35 * occ.std()
        return (occ > thr).astype(int)
    occs = {"sub(kick)": occupancy(e_sub), "low(snare)": occupancy(e_low), "noise(clap)": occupancy(e_noi), "hat": occupancy(e_hat)}
    ng = min(len(grid) - 1, 32 * 16)
    for name, occ in occs.items():
        occ = occ[:ng]
        for i in range(0, ng, 32):
            row = "".join("#" if v else "." for v in occ[i : i + 32])
            bar0 = i // 16 + 1
            print(f"{name:>11} bar {bar0:3d}: {row}")

    # --- Stereo image ---
    print(f"\n=== STEREO ===")
    mid = y48[:, 0] + y48[:, 1]
    side = y48[:, 0] - y48[:, 1]
    corr = float(np.corrcoef(y48[::10, 0], y48[::10, 1])[0, 1])
    print(f"L/R correlation: {corr:.3f}   side/total energy: {np.sum(side**2)/np.sum(mid**2)*100:.1f}%")

    # --- Long-term average spectrum (1/3 oct from 40 Hz) ---
    print(f"\n=== SPECTRUM (long-term avg, dB rel max) ===")
    freqs = np.array([40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 14000])
    f_stft, t_stft, Z = scipy.signal.stft(y_mono48, fs=sr48, nperseg=8192)
    spec = np.abs(Z) ** 2
    for fc in freqs:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        m = (f_stft >= lo) & (f_stft <= hi)
        val = 10 * np.log10(spec[m].mean() + 1e-20)
        print(f"  {fc:6d} Hz: {val:7.1f}")

    # --- Structure summary via RMS novelty ---
    print(f"\n=== SECTION NOVELTY (top RMS-change points) ===")
    smooth = np.convolve(sec_rms, np.ones(4) / 4, mode="same")
    nov = np.abs(np.diff(smooth, 2))
    idx = np.argsort(nov)[::-1][:24]
    pts = sorted(int(i) for i in idx)
    dedup = []
    for p in pts:
        if not dedup or p - dedup[-1] > 4:
            dedup.append(p)
    print("candidate section boundaries (s): " + ", ".join(str(p) for p in dedup))

    # save JSON summary
    summary = {
        "duration_s": dur,
        "lufs": lufs,
        "true_peak_db": 20 * np.log10(true_peak),
        "tempo": tempo_med,
        "key": f"{best[0]} {best[1]}",
        "n_notes": len(merged),
        "beats": beat_times.tolist()[:400],
        "chords": [{"beat": b, "chord": c} for b, c, _ in chord_list],
        "notes": [{"t0": t0, "t1": t1, "midi": p} for t0, t1, p in merged],
    }
    with open("/tmp/analysis/summary.json", "w") as f:
        json.dump(summary, f)
    print("\nJSON saved to /tmp/analysis/summary.json")


if __name__ == "__main__":
    main()
