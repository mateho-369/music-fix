#!/usr/bin/env python3
"""Deep analysis pass 2: melody/bass transcription, per-bar chords, drum grids,
bar-resolution structure map, tuning, section spectra."""
from __future__ import annotations

import json
import sys

import numpy as np
import librosa
import soundfile as sf
import scipy.signal

REF_WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/analysis/ref48.wav"
SR = 22050
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
CHORD_TYPES = {
    "": [0, 4, 7], "m": [0, 3, 7], "d": [0, 3, 6], "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
}


def note_name(midi: float) -> str:
    r = int(round(midi))
    return f"{PITCH_CLASSES[r % 12]}{r // 12 - 1}"


def main() -> None:
    y, sr = librosa.load(REF_WAV, sr=SR, mono=True)
    y48, sr48 = sf.read(REF_WAV, dtype="float32", always_2d=True)

    tempo = 143.55
    beat = 60.0 / tempo
    bar = 4 * beat
    n_bars = 112

    # ---------- tuning ----------
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    print(f"=== TUNING === {tuning*100:+.1f} cents vs A440")

    # ---------- find true downbeat: test 4 rotations of bar grid ----------
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_octaves=5, tuning=tuning)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
    t0 = 0.070  # first detected beat
    best_rot, best_score, best_bar_chroma = None, -1, None
    for rot in range(4):
        starts = t0 + np.arange(rot, rot + n_bars) * bar
        score = 0.0
        bc = []
        for s in starts[:-1]:
            m = (frame_times >= s) & (frame_times < s + bar)
            if m.sum() < 2:
                bc.append(np.zeros(12)); continue
            v = np.median(chroma[:, m], axis=1)
            bc.append(v)
            score += np.max(v) / (np.sum(v) + 1e-6)
        if score > best_score:
            best_score, best_rot, best_bar_chroma = score, rot, np.array(bc)
    downbeat0 = t0 + best_rot * bar
    print(f"=== GRID === downbeat offset: {downbeat0:.3f}s (rot {best_rot})  bar={bar:.4f}s  bars={n_bars}")

    # ---------- per-bar chord with bass weighting ----------
    print(f"\n=== CHORDS PER BAR (with bass root estimate) ===")
    bars_out = []
    for b in range(n_bars):
        s = downbeat0 + b * bar
        m = (frame_times >= s) & (frame_times < s + bar)
        if m.sum() < 2:
            bars_out.append(None); continue
        v = np.median(chroma[:, m], axis=1)
        v = v / (v.sum() + 1e-6)
        best_c = None
        for root in range(12):
            for qual, ivs in CHORD_TYPES.items():
                tpl = np.zeros(12)
                for iv in ivs:
                    tpl[(root + iv) % 12] = 1.0
                tpl = tpl / tpl.sum()
                # bass weighting: double weight on root in bass register proxy
                sc = float(np.dot(v, tpl)) + 0.15 * v[root]
                if best_c is None or sc > best_c[3]:
                    best_c = (root, qual, tpl, sc)
        bars_out.append({"bar": b + 1, "root": best_c[0], "qual": best_c[1], "score": best_c[3],
                         "chroma": v.tolist()})
    for i in range(0, n_bars, 8):
        row = []
        for bb in bars_out[i : i + 8]:
            row.append(f"{PITCH_CLASSES[bb['root']]}{bb['qual']:>4}" if bb else "  -- ")
        print(f"bars {i+1:3d}-{min(i+8,n_bars):3d}: " + " | ".join(f"{r:>6}" for r in row))

    # ---------- HPSS + melody & bass transcription ----------
    print(f"\n=== MELODY/BASS TRANSCRIPTION (HPSS + salience) ===")
    H, P = librosa.effects.hpss(y, margin=(3.0, 3.0))

    def transcribe(sig, fmin, fmax, label):
        S = np.abs(librosa.stft(sig, n_fft=4096, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        sal, _ = librosa.piptrack(S=S, sr=sr, fmin=fmin, fmax=fmax, threshold=0.08)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=256)
        pitches = np.zeros(S.shape[1])
        strengths = np.zeros(S.shape[1])
        for f in range(S.shape[1]):
            col = sal[:, f]
            if col.max() > 1e-4:
                k = np.argmax(col)
                pitches[f] = freqs[k]
                strengths[f] = col.max()
        midi = librosa.hz_to_midi(pitches) + tuning
        midi[pitches <= 0] = np.nan
        # median smooth 7 frames
        mf = scipy.signal.medfilt(np.nan_to_num(midi, nan=0.0), 9)
        mf[np.isnan(midi)] = np.nan
        # normalize strengths per-file
        thr = np.percentile(strengths[strengths > 0], 30) if (strengths > 0).any() else np.inf
        # segment
        notes = []
        i = 0
        N = len(mf)
        min_len = max(1, int(0.09 / (256 / sr)))
        while i < N:
            if not np.isnan(mf[i]) and strengths[i] > thr:
                j = i
                vals = []
                while j < N and not np.isnan(mf[j]) and strengths[j] > thr:
                    vals.append(mf[j]); j += 1
                if len(vals) >= min_len:
                    med = float(np.median(vals))
                    spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
                    if spread < 3.0:
                        notes.append((times[i], times[min(j, N - 1)], med, spread, float(np.median(strengths[i:j]))))
                i = j
            else:
                i += 1
        # merge consecutive same pitch
        merged = []
        for nt in notes:
            if merged and abs(nt[2] - merged[-1][2]) < 0.5 and nt[0] - merged[-1][1] < 0.10:
                merged[-1] = (merged[-1][0], nt[1], (merged[-1][2] + nt[2]) / 2, max(nt[3], merged[-1][3]), max(nt[4], merged[-1][4]))
            else:
                merged.append(list(nt))
        print(f"-- {label}: {len(merged)} note segments")
        return merged

    mel = transcribe(H, 180.0, 1200.0, "MELODY (harmonic comp, 180-1200Hz)")
    bass = transcribe(y, 30.0, 260.0, "BASS (full mix, 30-260Hz)")

    def grid_pos(t):
        tb = (t - downbeat0) / beat
        return tb

    print(f"\n-- MELODY quantized (bar:beat pitchname len_beats strength) --")
    mel_notes = []
    for t0_, t1_, m, spread, st in mel:
        g0, g1 = grid_pos(t0_), grid_pos(t1_)
        if g0 < -0.5 or g0 > n_bars * 4 + 1:
            continue
        bar_i = int(g0 // 4) + 1
        beat_in_bar = g0 % 4
        mel_notes.append({"t0": float(t0_), "t1": float(t1_), "midi": float(m),
                          "bar": bar_i, "beat": float(beat_in_bar), "len_beats": float(g1 - g0), "st": float(st)})
    for n in mel_notes:
        print(f"  bar {n['bar']:3d} beat {n['beat']:4.2f}  {note_name(n['midi']):>4} ({n['midi']:6.2f})  len {n['len_beats']:4.2f} b  str {n['st']:7.1f}")

    print(f"\n-- BASS quantized --")
    bass_notes = []
    for t0_, t1_, m, spread, st in bass:
        g0, g1 = grid_pos(t0_), grid_pos(t1_)
        if g0 < -0.5 or g0 > n_bars * 4 + 1:
            continue
        bar_i = int(g0 // 4) + 1
        bass_notes.append({"t0": float(t0_), "t1": float(t1_), "midi": float(m),
                           "bar": bar_i, "beat": float(g0 % 4), "len_beats": float(g1 - g0)})
    for n in bass_notes:
        print(f"  bar {n['bar']:3d} beat {n['beat']:4.2f}  {note_name(n['midi']):>4} ({n['midi']:6.2f})  len {n['len_beats']:4.2f} b")

    # ---------- drum grids all bars ----------
    print(f"\n=== DRUM GRIDS (all 112 bars, 16ths) ===")
    hop = 512
    def band_env(sig, lo, hi):
        sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        f = scipy.signal.sosfilt(sos, sig)
        return librosa.feature.rms(y=f, frame_length=1024, hop_length=hop)[0]
    e_sub = band_env(y, 30, 110)
    e_low = band_env(y, 150, 300)
    e_noi = band_env(y, 1200, 4000)
    e_hat = band_env(y, 7000, 11000)
    frame_times_d = librosa.frames_to_time(np.arange(len(e_sub)), sr=sr, hop_length=hop)
    sixteenth = beat / 4
    n_grid = n_bars * 16
    grids = {}
    for name, env in (("K", e_sub), ("S", e_low), ("C", e_noi), ("H", e_hat)):
        occ = np.zeros(n_grid)
        for g in range(n_grid):
            gs = downbeat0 + g * sixteenth
            m = (frame_times_d >= gs) & (frame_times_d < gs + sixteenth * 0.95)
            occ[g] = env[m].max() if m.any() else 0
        thr = occ.mean() + 0.5 * occ.std()
        grids[name] = (occ > thr).astype(int)
    for name, occ in grids.items():
        for b0 in range(0, n_bars, 16):
            rows = occ[b0 * 16 : (b0 + 16) * 16].reshape(16, 16)
            line = " ".join("".join("#" if v else "." for v in row) for row in rows)
            print(f"{name} bars {b0+1:3d}-{b0+16:3d}: {line}")
        print()

    # ---------- bar-resolution RMS map ----------
    print(f"=== BAR RMS MAP (112 bars, dBFS) ===")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    bar_rms = []
    for b in range(n_bars):
        s = downbeat0 + b * bar
        m = (rms_t >= s) & (rms_t < s + bar)
        bar_rms.append(20 * np.log10(rms[m].mean() + 1e-9) if m.any() else -120)
    for i in range(0, n_bars, 16):
        print(f"bars {i+1:3d}-{i+16:3d}: " + " ".join(f"{v:5.1f}" for v in bar_rms[i:i+16]))

    # ---------- section spectra ----------
    print(f"\n=== SECTION SPECTRA (1/3-oct bands, dB rel to 1kHz band of section A) ===")
    bands = [40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 14000]
    f_st, t_st, Z = scipy.signal.stft(y48.mean(axis=1), fs=sr48, nperseg=8192)
    Pxx = np.abs(Z) ** 2
    sections = {"intro(0-14s)": (0, 14), "A(16-66s)": (16, 66), "bridge(66-79s)": (66, 79),
                "B(80-131s)": (80, 131), "break(132-143s)": (132, 143), "C(144-168s)": (144, 168), "outro(168-187s)": (168, 186)}
    spec_out = {}
    for name, (a, b) in sections.items():
        m = (t_st >= a) & (t_st < b)
        vals = []
        for fc in bands:
            lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
            sel = (f_st >= lo) & (f_st <= hi)
            vals.append(10 * np.log10(Pxx[sel][:, m].mean() + 1e-20))
        spec_out[name] = np.array(vals)
    ref1k = spec_out["A(16-66s)"][bands.index(1000)]
    hdr = "band Hz : " + " ".join(f"{fc:>6}" for fc in bands)
    print(hdr)
    for name, vals in spec_out.items():
        print(f"{name:>14}: " + " ".join(f"{v - ref1k:6.1f}" for v in vals))

    # ---------- stereo per section ----------
    print(f"\n=== STEREO PER SECTION ===")
    L, R = y48[:, 0], y48[:, 1]
    for name, (a, b) in sections.items():
        sl = slice(int(a * sr48), int(b * sr48))
        corr = np.corrcoef(L[sl][::10], R[sl][::10])[0, 1]
        mid = L[sl] + R[sl]; side = L[sl] - R[sl]
        print(f"{name:>14}: corr {corr:5.3f}  side% {100*np.sum(side**2)/np.sum(mid**2):5.1f}")

    # ---------- save JSON ----------
    out = {
        "tempo": tempo, "bar_s": bar, "downbeat0": downbeat0, "n_bars": n_bars,
        "tuning_cents": tuning, "key": "G# minor",
        "chords": bars_out,
        "melody": mel_notes, "bass": bass_notes,
        "bar_rms": bar_rms,
        "drums": {k: v.tolist() for k, v in grids.items()},
    }
    with open("/tmp/analysis/summary2.json", "w") as f:
        json.dump(out, f, default=float)
    print("\nJSON saved to /tmp/analysis/summary2.json")


if __name__ == "__main__":
    main()
