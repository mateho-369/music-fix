#!/usr/bin/env python3
"""Generate an original electronic instrumental and encode it as a 320 kbps MP3.

The composition is deterministic and built entirely from synthesized oscillators,
noise, envelopes, and an original note arrangement. No source recording is sampled.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
BPM = 129
BEAT = 60.0 / BPM
BAR = BEAT * 4
BARS = 112
DURATION = BARS * BAR + 2.5
SEED = 369


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def adsr(n: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float32) * sustain
    a = min(n, int(attack * SAMPLE_RATE))
    d = min(max(0, n - a), int(decay * SAMPLE_RATE))
    r = min(n, int(release * SAMPLE_RATE))
    if a:
        env[:a] = np.linspace(0, 1, a, endpoint=False, dtype=np.float32)
    if d:
        env[a:a+d] = np.linspace(1, sustain, d, endpoint=False, dtype=np.float32)
    if r:
        env[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return env


def pan(signal: np.ndarray, position: float) -> np.ndarray:
    angle = (position + 1.0) * np.pi / 4.0
    return np.column_stack((signal * np.cos(angle), signal * np.sin(angle))).astype(np.float32)


def add(track: np.ndarray, signal: np.ndarray, start: float, position: float = 0.0) -> None:
    first = max(0, int(start * SAMPLE_RATE))
    last = min(len(track), first + len(signal))
    if last > first:
        track[first:last] += pan(signal[:last-first], position)


def synth_note(note: float, seconds: float, kind: str = "lead") -> np.ndarray:
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    f = midi_hz(note)
    phase = 2 * np.pi * f * t
    if kind == "pad":
        # Warm, wide-spectrum pad from detuned odd harmonics.
        sig = (0.52 * np.sin(phase * 0.997) + 0.52 * np.sin(phase * 1.003)
               + 0.18 * np.sin(phase * 2) + 0.10 * np.sin(phase * 3))
        env = adsr(n, 0.18, 0.35, 0.68, min(0.5, seconds * 0.3))
    elif kind == "bass":
        sig = 0.75 * np.sin(phase) + 0.22 * np.sin(phase * 2) + 0.08 * np.sin(phase * 3)
        sig = np.tanh(sig * 1.35)
        env = adsr(n, 0.008, 0.11, 0.60, min(0.12, seconds * 0.2))
    elif kind == "pluck":
        sig = 0.64 * np.sin(phase) + 0.24 * np.sin(phase * 2) + 0.12 * np.sin(phase * 4)
        env = np.exp(-5.2 * t / max(seconds, 0.01)).astype(np.float32)
        env *= adsr(n, 0.003, 0.02, 0.85, min(0.08, seconds * 0.2))
    else:
        vibrato = 0.012 * np.sin(2 * np.pi * 5.2 * t)
        sig = (0.64 * np.sin(phase + vibrato) + 0.20 * np.sin(phase * 2 + vibrato)
               + 0.08 * np.sin(phase * 3))
        env = adsr(n, 0.018, 0.09, 0.74, min(0.14, seconds * 0.25))
    return (sig * env).astype(np.float32)


def kick(seconds: float = 0.34) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * (48 * t + 58 * (1 - np.exp(-28 * t)) / 28)
    body = np.sin(phase) * np.exp(-10.5 * t)
    click = np.exp(-95 * t) * np.sin(2 * np.pi * 1300 * t)
    return np.tanh((body + 0.16 * click) * 1.7).astype(np.float32)


def snare(rng: np.random.Generator, seconds: float = 0.28) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = rng.standard_normal(n).astype(np.float32)
    # First difference removes much of the low-frequency noise energy.
    bright = np.r_[noise[0], np.diff(noise)].astype(np.float32)
    tone = np.sin(2 * np.pi * 190 * t)
    return ((0.28 * bright * np.exp(-17 * t) + 0.38 * tone * np.exp(-20 * t)) * 0.9).astype(np.float32)


def hat(rng: np.random.Generator, seconds: float = 0.075) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = rng.standard_normal(n).astype(np.float32)
    bright = np.r_[noise[0], np.diff(noise)].astype(np.float32)
    return (bright * np.exp(-55 * t) * 0.15).astype(np.float32)


def section_for_bar(bar: int) -> str:
    sections = [
        (8, "intro"), (24, "verse"), (32, "build"), (48, "chorus"),
        (56, "break"), (72, "verse"), (80, "build"), (96, "chorus"),
        (112, "outro"),
    ]
    return next(name for end, name in sections if bar < end)


def compose() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    total = int(DURATION * SAMPLE_RATE)
    mix = np.zeros((total, 2), dtype=np.float32)

    # Original four-bar harmonic cycle in B-flat minor.
    chords = [
        ([58, 61, 65, 70], 34),  # B-flat minor
        ([54, 58, 61, 65], 30),  # G-flat major
        ([61, 65, 68, 73], 37),  # D-flat major
        ([56, 60, 63, 68], 32),  # A-flat major
    ]
    roots = [34, 30, 37, 32]

    # A new eight-bar melody. Values are scale-degree offsets from B-flat.
    phrase = [
        [(0.0, 0, .5), (.5, 3, .5), (1.0, 7, 1.0), (2.25, 5, .5), (3.0, 3, .75)],
        [(0.0, 10, .75), (1.0, 8, .5), (1.75, 7, .75), (2.75, 3, 1.0)],
        [(0.0, 5, .5), (.5, 7, .5), (1.0, 12, 1.25), (2.5, 10, .5), (3.1, 8, .7)],
        [(0.0, 7, .75), (1.0, 3, .5), (1.75, 0, 1.5)],
        [(0.0, 0, .5), (.75, 5, .5), (1.5, 8, .5), (2.0, 7, .5), (2.75, 3, 1.0)],
        [(0.0, 10, .5), (.5, 12, .5), (1.25, 15, .75), (2.25, 12, .5), (3.0, 10, .75)],
        [(0.0, 8, .75), (1.0, 7, .5), (1.75, 5, .5), (2.5, 3, 1.25)],
        [(0.0, 5, .5), (.75, 3, .5), (1.5, 0, 2.0)],
    ]
    scale = [0, 1, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25]

    for bar in range(BARS):
        start = bar * BAR
        section = section_for_bar(bar)
        chord, bass_root = chords[bar % 4]

        # Pads establish the harmony, opening progressively through the arrangement.
        pad_gain = {"intro": .10, "verse": .13, "build": .17, "chorus": .22,
                    "break": .09, "outro": .12}[section]
        for i, note in enumerate(chord):
            signal = synth_note(note, BAR * .98, "pad") * pad_gain
            add(mix, signal, start, -0.62 + i * .42)

        # Syncopated plucks in active sections.
        if section in {"verse", "build", "chorus"}:
            pattern = [0.0, .75, 1.5, 2.0, 2.75, 3.5]
            for j, beat_pos in enumerate(pattern):
                note = chord[(j + bar) % len(chord)] + (12 if section == "chorus" else 0)
                add(mix, synth_note(note, BEAT * .34, "pluck") * .10,
                    start + beat_pos * BEAT, -.35 if j % 2 else .35)

        # Bass uses roots with a turnaround note at the end of every fourth bar.
        if section != "intro" or bar >= 4:
            bass_pattern = [(0, bass_root, .85), (1.5, bass_root, .42),
                            (2, bass_root + 12, .78), (3, bass_root + 7, .62)]
            for beat_pos, note, length in bass_pattern:
                add(mix, synth_note(note, length * BEAT, "bass") * .24,
                    start + beat_pos * BEAT)

        # Drums: restrained intro/break, full beat in chorus.
        if section not in {"intro", "break"} or (section == "intro" and bar >= 4):
            kicks = [0, 2] + ([1.5, 3.25] if section == "chorus" else ([3.5] if section == "build" else []))
            for beat_pos in kicks:
                add(mix, kick() * .38, start + beat_pos * BEAT)
            for beat_pos in [1, 3]:
                add(mix, snare(rng) * .28, start + beat_pos * BEAT, .05)
            hat_step = .25 if section in {"build", "chorus"} else .5
            for j, beat_pos in enumerate(np.arange(0, 4, hat_step)):
                add(mix, hat(rng) * (1.15 if j % 2 else .8), start + float(beat_pos) * BEAT,
                    -.28 if j % 2 else .28)
        elif section == "break":
            for beat_pos in [0, 2.5]:
                add(mix, kick() * .25, start + beat_pos * BEAT)

        # Lead is reserved for hooks, making the arrangement breathe.
        if section in {"chorus", "build"} or (section == "outro" and bar < 104):
            octave = 70 if section == "chorus" else 58
            for beat_pos, degree, length in phrase[bar % 8]:
                degree = min(degree, len(scale) - 1)
                note = octave + scale[degree]
                gain = .16 if section == "chorus" else .105
                sig = synth_note(note, length * BEAT, "lead") * gain
                add(mix, sig, start + beat_pos * BEAT, .10)
                # Quiet dotted echo, synthesized rather than sampled.
                add(mix, sig * .22, start + (beat_pos + .75) * BEAT, -.35)

    # Gentle master saturation, DC removal, fade-in/out, and true-peak headroom.
    mix -= np.mean(mix, axis=0, keepdims=True)
    mix = np.tanh(mix * 1.18)
    fade = int(1.8 * SAMPLE_RATE)
    mix[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    mix[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    peak = float(np.max(np.abs(mix)))
    mix *= (10 ** (-0.8 / 20)) / max(peak, 1e-9)
    return mix


def find_ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit("Install imageio-ffmpeg or set FFMPEG_BINARY to encode MP3.") from exc


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def encode_mp3(wav_path: Path, output: Path) -> None:
    command = [
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "320k", "-ar", str(SAMPLE_RATE),
        "-metadata", "title=Velvet Circuit",
        "-metadata", "artist=ma teho",
        "-metadata", "album_artist=ma teho",
        "-metadata", "composer=ma teho",
        "-metadata", "copyright=© 2026 ma teho. All rights reserved.",
        "-metadata", "comment=Original procedural composition; no sampled recordings.",
        "-id3v2_version", "3", str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("Ma Teho - Velvet Circuit.mp3"))
    args = parser.parse_args()
    print(f"Composing {DURATION:.1f} seconds at {BPM} BPM...")
    audio = compose()
    with tempfile.TemporaryDirectory() as directory:
        wav_path = Path(directory) / "master.wav"
        write_wav(wav_path, audio)
        encode_mp3(wav_path, args.output)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
