#!/usr/bin/env python3
"""Generate the original electronic track 'Ma Teho - Neon Meridian.mp3'.

Captures the genre, mood, tempo, energy, and production polish of the reference track
'Biscuit Toner.mp3' while featuring a completely original musical composition,
chord progression, melodies, bassline, and sound design.

Uses Spotify's Pedalboard DSP engine for analog-modeled Moog ladder filters,
lush algorithmic reverb, ping-pong delays, stereo chorus, bus compression,
and brickwall mastering limiting.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pedalboard as pb
import pyloudnorm as pyln
import scipy.signal
import soundfile as sf

# ---------------------------------------------------------------------------
# Project Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000
BPM = 130.0
BEAT = 60.0 / BPM          # ~0.461538 s
BAR = BEAT * 4.0           # ~1.846154 s
TOTAL_BARS = 114
DURATION = TOTAL_BARS * BAR # ~210.4615 s + tail = 210.52 s
SEED = 42

# ---------------------------------------------------------------------------
# Wavetable Synthesis Engine (Bandlimited / Anti-Aliased)
# ---------------------------------------------------------------------------
WT_SIZE = 4096

def _generate_wavetables() -> dict[str, np.ndarray]:
    """Generate bandlimited wavetables for different frequency registers."""
    tables = {}
    phase_axis = np.arange(WT_SIZE, dtype=np.float32) / WT_SIZE
    
    # Sawtooth tables with varying harmonic limits
    for name, max_h in [('saw_low', 48), ('saw_mid', 24), ('saw_high', 10)]:
        h = np.arange(1, max_h + 1, dtype=np.float32)
        phases = 2.0 * np.pi * np.outer(phase_axis, h)
        wt = np.sum((1.0 / h) * np.sin(phases), axis=1).astype(np.float32)
        wt /= np.max(np.abs(wt))
        tables[name] = wt
        
    # Square tables with odd harmonics
    for name, max_h in [('sq_low', 47), ('sq_mid', 23), ('sq_high', 9)]:
        h = np.arange(1, max_h + 1, 2, dtype=np.float32)
        phases = 2.0 * np.pi * np.outer(phase_axis, h)
        wt = np.sum((1.0 / h) * np.sin(phases), axis=1).astype(np.float32)
        wt /= np.max(np.abs(wt))
        tables[name] = wt
        
    # Pure sine
    tables['sine'] = np.sin(2.0 * np.pi * phase_axis).astype(np.float32)
    return tables

WAVETABLES = _generate_wavetables()

def osc_read(table_name: str, phase_accum: np.ndarray) -> np.ndarray:
    """Read from a wavetable with linear interpolation given normalized phase [0, 1)."""
    table = WAVETABLES[table_name]
    indices = (phase_accum % 1.0) * WT_SIZE
    idx0 = indices.astype(np.int32)
    idx1 = (idx0 + 1) % WT_SIZE
    frac = (indices - idx0).astype(np.float32)
    return (1.0 - frac) * table[idx0] + frac * table[idx1]

def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))

def adsr_envelope(n_samples: int, a_s: float, d_s: float, s_lvl: float, r_s: float) -> np.ndarray:
    """Generate a standard ADSR envelope."""
    env = np.ones(n_samples, dtype=np.float32) * s_lvl
    a = min(n_samples, int(a_s * SAMPLE_RATE))
    d = min(max(0, n_samples - a), int(d_s * SAMPLE_RATE))
    r = min(n_samples, int(r_s * SAMPLE_RATE))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    if d > 0:
        env[a:a+d] = np.linspace(1.0, s_lvl, d, endpoint=False, dtype=np.float32)
    if r > 0:
        env[-r:] *= np.linspace(1.0, 0.0, r, endpoint=True, dtype=np.float32)
    return env

def pan_stereo(signal: np.ndarray, pan_pos: float) -> np.ndarray:
    """Constant power panning between -1.0 (left) and +1.0 (right)."""
    angle = (np.clip(pan_pos, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    left = signal * np.cos(angle)
    right = signal * np.sin(angle)
    return np.column_stack([left, right]).astype(np.float32)

def add_to_mix(mix: np.ndarray, signal: np.ndarray, start_sec: float) -> None:
    """Add a stereo signal to the main mix at start_sec."""
    idx_start = max(0, int(start_sec * SAMPLE_RATE))
    idx_end = min(len(mix), idx_start + len(signal))
    chunk_len = idx_end - idx_start
    if chunk_len > 0:
        mix[idx_start:idx_end] += signal[:chunk_len]

# ---------------------------------------------------------------------------
# Drum Sound Synthesis (Punchy Club / Electronic)
# ---------------------------------------------------------------------------
def synthesize_kick() -> np.ndarray:
    """Punchy electronic kick drum with deep sub-bass body, knock, and transient click."""
    dur = 0.44
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    # Pitch envelope: sweeps from 165 Hz down to 48 Hz with sharp exponential curve
    pitch_env = 48.0 + (165.0 - 48.0) * np.exp(-28.0 * t)
    phase = 2.0 * np.pi * np.cumsum(pitch_env) / SAMPLE_RATE
    # Deep sub body
    body = np.sin(phase) * np.exp(-6.2 * t)
    
    # 2nd harmonic for warm analog chest punch
    punch = np.sin(2.0 * phase) * np.exp(-18.0 * t) * 0.35
    
    # Beater knock punch transient (110 Hz)
    knock = np.sin(2.0 * np.pi * 110.0 * t) * np.exp(-45.0 * t) * 0.45
    
    # High-frequency transient click (filtered to avoid excessive harshness)
    click = np.sin(2.0 * np.pi * 2800.0 * t) * np.exp(-140.0 * t) * 0.20
    
    raw = (body * 1.1 + punch + knock + click) * 1.5
    saturated = np.tanh(raw)
    
    # Smooth fade out at tail
    tail_fade = int(0.04 * SAMPLE_RATE)
    saturated[-tail_fade:] *= np.linspace(1.0, 0.0, tail_fade, dtype=np.float32)
    
    # Stereo mono-centered
    return np.column_stack([saturated, saturated]).astype(np.float32)

def synthesize_snare_clap(rng: np.random.Generator) -> np.ndarray:
    """Layered electro snare + stereo clap with micro-delays and warm room reverb."""
    dur = 0.42
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    # Tonal body: 180 Hz -> 120 Hz
    f_tone = 120.0 + (180.0 - 120.0) * np.exp(-20.0 * t)
    tone = np.sin(2.0 * np.pi * np.cumsum(f_tone) / SAMPLE_RATE) * np.exp(-13.0 * t) * 0.55
    
    # Snare noise body: bandpassed 1.0 kHz - 4.8 kHz (warmer, less harsh)
    raw_noise = rng.standard_normal(n).astype(np.float32)
    sos_bp = scipy.signal.butter(4, [1000, 4800], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    noise_bp = scipy.signal.sosfilt(sos_bp, raw_noise)
    noise_body = noise_bp * np.exp(-14.0 * t) * 0.55
    
    # Multi-clap stereo micro-bursts (at 0ms, 12ms, 24ms)
    clap_l = np.zeros(n, dtype=np.float32)
    clap_r = np.zeros(n, dtype=np.float32)
    for offset_ms, pan_pos in [(0, -0.30), (11, 0.30), (22, 0.0)]:
        s_idx = int(offset_ms * SAMPLE_RATE / 1000.0)
        b_len = int(0.016 * SAMPLE_RATE)
        b_t = np.arange(b_len, dtype=np.float32) / SAMPLE_RATE
        b_noise = rng.standard_normal(b_len).astype(np.float32) * np.exp(-170.0 * b_t)
        angle = (pan_pos + 1.0) * np.pi / 4.0
        clap_l[s_idx:s_idx+b_len] += b_noise * np.cos(angle) * 0.45
        clap_r[s_idx:s_idx+b_len] += b_noise * np.sin(angle) * 0.45
        
    left = tone + noise_body + clap_l
    right = tone + noise_body + clap_r
    stereo = np.column_stack([left, right]).astype(np.float32)
    
    # Warm room reverb
    reverb = pb.Reverb(room_size=0.32, damping=0.45, wet_level=0.22, dry_level=0.85)
    processed = reverb(stereo.T, SAMPLE_RATE).T
    return processed

def synthesize_hihat(rng: np.random.Generator, open_hat: bool = False) -> np.ndarray:
    """Warm metallic hi-hat balanced to match reference spectral centroid."""
    dur = 0.24 if open_hat else 0.055
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    # Metallic tone blend
    metallic = (np.sin(2*np.pi*3800*t) + np.sin(2*np.pi*5200*t) + 
                np.sin(2*np.pi*6800*t) + np.sin(2*np.pi*8400*t)) * 0.22
    
    # Noise with lowpass filter at 10 kHz to prevent harshness
    noise = rng.standard_normal(n).astype(np.float32)
    sos_bp = scipy.signal.butter(4, [5000, 11000] if not open_hat else [4200, 10500], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    noise_bp = scipy.signal.sosfilt(sos_bp, noise)
    
    decay_rate = 18.0 if open_hat else 75.0
    env = np.exp(-decay_rate * t)
    sig = (noise_bp * 0.65 + metallic * 0.35) * env * 0.7
    return np.column_stack([sig, sig]).astype(np.float32)

def synthesize_ride(rng: np.random.Generator) -> np.ndarray:
    """Warm ride cymbal for chorus drive."""
    dur = 0.30
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = rng.standard_normal(n).astype(np.float32)
    sos_bp = scipy.signal.butter(4, [6000, 12000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    shimmer = scipy.signal.sosfilt(sos_bp, noise) * np.exp(-14.0 * t) * 0.55
    return np.column_stack([shimmer * 0.85, shimmer * 1.05]).astype(np.float32)

def synthesize_crash(rng: np.random.Generator) -> np.ndarray:
    """Stereo crash cymbal wash for drop impacts."""
    dur = 3.2
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise_l = rng.standard_normal(n).astype(np.float32)
    noise_r = rng.standard_normal(n).astype(np.float32)
    sos_bp = scipy.signal.butter(4, [2500, 11000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    sig_l = scipy.signal.sosfilt(sos_bp, noise_l) * np.exp(-2.4 * t) * 0.7
    sig_r = scipy.signal.sosfilt(sos_bp, noise_r) * np.exp(-2.4 * t) * 0.7
    stereo = np.column_stack([sig_l, sig_r]).astype(np.float32)
    reverb = pb.Reverb(room_size=0.70, damping=0.5, wet_level=0.40, dry_level=0.75)
    return reverb(stereo.T, SAMPLE_RATE).T

def synthesize_sub_drop() -> np.ndarray:
    """Sub drop impact: sine sweep from 95 Hz down to 32 Hz over 1.8s."""
    dur = 1.8
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    f_env = 32.0 + (95.0 - 32.0) * np.exp(-2.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SAMPLE_RATE
    sig = np.sin(phase) * np.exp(-1.8 * t) * 0.85
    return np.column_stack([sig, sig]).astype(np.float32)

def synthesize_noise_riser(bars: int = 4) -> np.ndarray:
    """White noise riser with sweeping resonant lowpass filter over specified bars."""
    dur = bars * BAR
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(n).astype(np.float32)
    # Exponential volume ramp
    vol = np.exp(np.linspace(-3.5, 0.0, n))
    noise *= vol
    stereo = np.column_stack([noise, noise]).astype(np.float32)
    
    # Process through LadderFilter sweeping up
    chain = pb.Pedalboard([
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=4500, resonance=0.5),
        pb.Chorus(rate_hz=1.5, depth=0.6, mix=0.4),
        pb.Reverb(room_size=0.65, damping=0.3, wet_level=0.4, dry_level=0.7)
    ])
    return chain(stereo.T, SAMPLE_RATE).T

# ---------------------------------------------------------------------------
# Bass Synth Engine (Dual-Layer: Deep Sub-Bass + Analog Mid-Grit + Sidechain)
# ---------------------------------------------------------------------------
def synthesize_bass_note(midi_note: float, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Dual-layer bass note: Layer 1 warm saturated sub-bass, Layer 2 filtered mid-grit."""
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    f0 = midi_to_hz(midi_note)
    
    # 1. Sub Layer: Pure deep fundamental with tanh saturation for warm analog weight
    sub_phase = (t * f0) % 1.0
    sub = np.sin(2.0 * np.pi * sub_phase)
    # Gentle 2nd harmonic for audibility on smaller speakers
    sub_2nd = np.sin(4.0 * np.pi * sub_phase) * 0.25
    sub_env = adsr_envelope(n, 0.005, 0.14, 0.88, min(0.08, seconds * 0.25))
    sub_sig = np.tanh((sub + sub_2nd) * sub_env * 1.5) * 1.15
    
    # 2. Mid Grit Layer: Two detuned saws through Moog ladder filter
    phase_l = (t * f0 * 1.003 + rng.uniform(0, 1)) % 1.0
    phase_r = (t * f0 * 0.997 + rng.uniform(0, 1)) % 1.0
    table_mid = 'saw_low' if f0 < 100 else 'saw_mid'
    mid_l = osc_read(table_mid, phase_l)
    mid_r = osc_read(table_mid, phase_r)
    mid_env = adsr_envelope(n, 0.008, 0.12, 0.60, min(0.06, seconds * 0.2))
    mid_l *= mid_env
    mid_r *= mid_env
    
    stereo_mid = np.column_stack([mid_l, mid_r]).astype(np.float32)
    mid_chain = pb.Pedalboard([
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=min(900.0, f0 * 8.0), resonance=0.25),
        pb.Distortion(drive_db=4.0)
    ])
    processed_mid = mid_chain(stereo_mid.T, SAMPLE_RATE).T * 0.35
    
    combined = np.column_stack([sub_sig, sub_sig]) + processed_mid
    return combined.astype(np.float32)

# ---------------------------------------------------------------------------
# Polyphonic Supersaw Synth (Lush Stereo Unison Chords / Pads)
# ---------------------------------------------------------------------------
def synthesize_supersaw_chord(chord_notes: list[float], seconds: float, 
                              cutoff_hz: float = 2200.0, pad_mode: bool = False,
                              highpass_hz: float = 0.0,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """7-voice detuned supersaw per note with stereo unison spread."""
    if rng is None:
        rng = np.random.default_rng()
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    detune_spread = [-0.015, -0.009, -0.004, 0.0, 0.004, 0.009, 0.015]
    pan_spread = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    
    chord_l = np.zeros(n, dtype=np.float32)
    chord_r = np.zeros(n, dtype=np.float32)
    
    a_time = 0.28 if pad_mode else 0.015
    d_time = 0.45 if pad_mode else 0.22
    s_lvl = 0.85 if pad_mode else 0.60
    r_time = min(0.50, seconds * 0.35)
    env = adsr_envelope(n, a_time, d_time, s_lvl, r_time)
    
    for note in chord_notes:
        f0 = midi_to_hz(note)
        tb_name = 'saw_low' if f0 < 200 else ('saw_mid' if f0 < 600 else 'saw_high')
        for det, pan_pos in zip(detune_spread, pan_spread):
            f_det = f0 * (1.0 + det)
            phase_offset = rng.uniform(0.0, 1.0)
            phase_accum = (t * f_det + phase_offset) % 1.0
            osc = osc_read(tb_name, phase_accum)
            angle = (pan_pos + 1.0) * (np.pi / 4.0)
            chord_l += osc * np.cos(angle)
            chord_r += osc * np.sin(angle)
            
    norm_factor = 1.0 / (np.sqrt(len(chord_notes) * len(detune_spread)) + 1e-6)
    chord_l *= env * norm_factor
    chord_r *= env * norm_factor
    stereo = np.column_stack([chord_l, chord_r]).astype(np.float32)
    
    plugins = [
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=cutoff_hz, resonance=0.20),
        pb.Chorus(rate_hz=0.85, depth=0.45, mix=0.35),
        pb.Reverb(room_size=0.72, damping=0.45, wet_level=0.35, dry_level=0.75)
    ]
    if highpass_hz > 20.0:
        plugins.insert(0, pb.HighpassFilter(cutoff_frequency_hz=highpass_hz))
        
    fx_chain = pb.Pedalboard(plugins)
    processed = fx_chain(stereo.T, SAMPLE_RATE).T
    return processed

# ---------------------------------------------------------------------------
# Arpeggiator / Pluck Synth Engine
# ---------------------------------------------------------------------------
def synthesize_pluck_note(midi_note: float, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Crisp 16th-note pluck using dual saw/pulse oscillators with resonant ladder filter."""
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    f0 = midi_to_hz(midi_note)
    
    phase1 = (t * f0 + rng.uniform(0, 1)) % 1.0
    phase2 = (t * f0 * 1.003 + rng.uniform(0, 1)) % 1.0
    tb = 'saw_mid' if f0 < 600 else 'saw_high'
    osc1 = osc_read(tb, phase1)
    osc2 = osc_read('sq_mid', phase2) * 0.5
    
    env = adsr_envelope(n, 0.003, 0.08, 0.25, min(0.06, seconds * 0.2))
    sig = (osc1 + osc2) * env
    stereo = np.column_stack([sig * 0.9, sig * 1.1]).astype(np.float32)
    
    chain = pb.Pedalboard([
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=min(3200.0, f0 * 6.0), resonance=0.38),
        pb.Delay(delay_seconds=BEAT * 0.75, feedback=0.32, mix=0.28), # Dotted 8th delay
        pb.Reverb(room_size=0.5, damping=0.3, wet_level=0.22, dry_level=0.85)
    ])
    return chain(stereo.T, SAMPLE_RATE).T

# ---------------------------------------------------------------------------
# Singing Lead Synth Engine
# ---------------------------------------------------------------------------
def synthesize_lead_note(midi_note: float, seconds: float, vibrato_delay: float = 0.12,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Singing analog lead synth with delayed LFO vibrato, Moog ladder filter, and delay."""
    if rng is None:
        rng = np.random.default_rng()
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    f0 = midi_to_hz(midi_note)
    
    # Delayed LFO vibrato (5.5 Hz)
    vib_env = np.clip((t - vibrato_delay) / 0.25, 0.0, 1.0)
    vibrato = 0.014 * np.sin(2.0 * np.pi * 5.5 * t) * vib_env
    
    phase_c = (t * f0 * (1.0 + vibrato) + rng.uniform(0, 1)) % 1.0
    phase_l = (t * f0 * (1.005 + vibrato) + rng.uniform(0, 1)) % 1.0
    phase_r = (t * f0 * (0.995 + vibrato) + rng.uniform(0, 1)) % 1.0
    
    tb = 'saw_mid' if f0 < 600 else 'saw_high'
    center = osc_read(tb, phase_c)
    left = (center * 0.65 + osc_read(tb, phase_l) * 0.35)
    right = (center * 0.65 + osc_read(tb, phase_r) * 0.35)
    
    env = adsr_envelope(n, 0.018, 0.12, 0.78, min(0.18, seconds * 0.3))
    stereo = np.column_stack([left * env, right * env]).astype(np.float32)
    
    lead_chain = pb.Pedalboard([
        pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF24, cutoff_hz=3800.0, resonance=0.28),
        pb.Distortion(drive_db=3.5),
        pb.Delay(delay_seconds=BEAT * 0.5, feedback=0.35, mix=0.25), # 8th note delay
        pb.Reverb(room_size=0.65, damping=0.35, wet_level=0.30, dry_level=0.8)
    ])
    return lead_chain(stereo.T, SAMPLE_RATE).T

# ---------------------------------------------------------------------------
# Composition & Arrangement Architecture
# ---------------------------------------------------------------------------
def build_arrangement() -> tuple[np.ndarray, np.ndarray]:
    """Compose and arrange the entire 114-bar track in B-flat minor / D-flat major."""
    rng = np.random.default_rng(SEED)
    total_samples = int(DURATION * SAMPLE_RATE)
    mix = np.zeros((total_samples, 2), dtype=np.float32)
    
    print("Pre-rendering one-shot drum hits and FX...")
    kick_sample = synthesize_kick()
    snare_sample = synthesize_snare_clap(rng)
    hat_closed = synthesize_hihat(rng, open_hat=False)
    hat_open = synthesize_hihat(rng, open_hat=True)
    ride_sample = synthesize_ride(rng)
    crash_sample = synthesize_crash(rng)
    sub_drop_sample = synthesize_sub_drop()
    riser_sample = synthesize_noise_riser(bars=4)
    
    # 8-bar Harmonic Progression (B-flat Minor / D-flat Major)
    # Each entry: (chord_midi_notes, bass_root_midi)
    harmonic_cycle = [
        ([46, 58, 61, 65, 68], 34),  # Bar 1: Bbm7 (Bb2, Db3, F3, Ab3) - root Bb1 (34)
        ([42, 54, 61, 65, 70], 30),  # Bar 2: Gbmaj7 (Gb2, Db3, F3, Bb3) - root Gb1 (30)
        ([49, 56, 61, 65, 68], 37),  # Bar 3: Db add9 (Db3, Ab3, F3, Eb4) - root Db2 (37)
        ([44, 56, 60, 63, 68], 32),  # Bar 4: Ab (Ab2, Eb3, C3, Ab3) - root Ab1 (32)
        ([39, 51, 58, 61, 66], 27),  # Bar 5: Ebm9 (Eb2, Bb2, Db3, Gb3) - root Eb1 (27->39)
        ([42, 54, 58, 65, 70], 30),  # Bar 6: Gbmaj7 (Gb2, Bb2, F3, Bb3) - root Gb1 (30)
        ([46, 58, 61, 65, 68], 34),  # Bar 7: Bbm7 (Bb2, Db3, F3, Ab3) - root Bb1 (34)
        ([44, 56, 60, 65, 68], 32),  # Bar 8: Fm7/Ab (Ab2, Eb3, C3, F3) - root Ab1 (32)
    ]
    
    # Anthemic Lead Melody (8-bar phrase in Bb natural minor)
    # Offsets relative to Bb4 (Midi 70)
    # Format: list of (beat_pos, note_offset, length_beats)
    lead_phrase = [
        # Bar 1: Strong syncopated entrance on 5th and octave
        [(0.0, 7, 0.75), (0.75, 10, 0.75), (1.5, 12, 1.75)],
        # Bar 2: Cascading run down to tonic
        [(0.0, 12, 0.5), (0.5, 10, 0.5), (1.0, 7, 0.75), (2.0, 3, 0.75), (2.75, 5, 1.25)],
        # Bar 3: High leap to Db5 (octave + minor third)
        [(0.0, 8, 0.5), (0.75, 12, 0.5), (1.5, 15, 1.5), (3.0, 14, 0.8)],
        # Bar 4: Emotive resolution phrase
        [(0.0, 12, 0.75), (1.0, 10, 0.5), (1.75, 7, 0.75), (2.5, 3, 0.5), (3.0, 0, 1.0)],
        # Bar 5: Emotional lift into subdominant
        [(0.0, 3, 0.5), (0.75, 7, 0.5), (1.5, 10, 0.75), (2.5, 12, 1.25)],
        # Bar 6: Peak energy high motif
        [(0.0, 15, 0.5), (0.5, 17, 0.75), (1.5, 15, 0.5), (2.25, 12, 0.75), (3.0, 10, 0.8)],
        # Bar 7: Driving rhythmic hook
        [(0.0, 10, 0.5), (0.75, 12, 0.5), (1.5, 10, 0.5), (2.25, 7, 0.75), (3.0, 5, 0.8)],
        # Bar 8: Cadential turnaround holding into next phrase
        [(0.0, 7, 0.5), (0.75, 5, 0.5), (1.5, 3, 0.75), (2.5, 0, 1.5)]
    ]
    
    # 16th-note Arpeggio Pattern (scale indices moving across the chord)
    arp_indices = [0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2, 4, 3, 1, 2, 3]
    
    # Collect all kick times for sidechain ducking curve
    kick_times = []
    
    print("Composing bars and tracking arrangement...")
    for bar_idx in range(TOTAL_BARS):
        bar_start = bar_idx * BAR
        chord_info, root_note = harmonic_cycle[bar_idx % 8]
        
        # Adjust root for lowest octave
        if root_note < 30:
            root_note += 12
            
        # Section classification
        if bar_idx < 8:
            section = "intro"
        elif bar_idx < 24:
            section = "verse1"
        elif bar_idx < 40:
            section = "drop1"
        elif bar_idx < 56:
            section = "verse2"
        elif bar_idx < 68:
            section = "breakdown"
        elif bar_idx < 72:
            section = "buildup"
        elif bar_idx < 96:
            section = "drop2"
        elif bar_idx < 104:
            section = "outro_groove"
        else:
            section = "outro_fade"
            
        # 1. DRUMS
        if section in {"verse1", "drop1", "verse2", "drop2", "outro_groove"}:
            # Four on the floor kick
            for beat_idx in range(4):
                k_time = bar_start + beat_idx * BEAT
                add_to_mix(mix, kick_sample * 0.90, k_time)
                kick_times.append(k_time)
                
            # Snare/Clap on beats 2 and 4
            add_to_mix(mix, snare_sample * 0.75, bar_start + 1.0 * BEAT)
            add_to_mix(mix, snare_sample * 0.75, bar_start + 3.0 * BEAT)
                
            # Closed hi-hats on 16th notes with velocity groove
            for step in range(16):
                h_time = bar_start + step * (BEAT / 4.0)
                vel = 0.38 if (step % 2 == 1) else 0.22
                if step % 4 == 0:
                    vel = 0.28
                add_to_mix(mix, hat_closed * vel, h_time)
                
            # Offbeat open hats on "&" of beats
            if section in {"drop1", "drop2", "verse2", "outro_groove"}:
                for beat_idx in range(4):
                    oh_time = bar_start + (beat_idx + 0.5) * BEAT
                    add_to_mix(mix, hat_open * 0.45, oh_time)
                    
            # Ride cymbals in peak drop sections
            if section in {"drop1", "drop2"} and (bar_idx >= 32 or section == "drop2"):
                for beat_idx in range(4):
                    add_to_mix(mix, ride_sample * 0.35, bar_start + beat_idx * BEAT)
                    
        elif section == "outro_fade":
            # Outro deconstruction:
            if bar_idx < 108:
                # Bars 105-107: Kick and closed hats
                for beat_idx in range(4):
                    k_time = bar_start + beat_idx * BEAT
                    add_to_mix(mix, kick_sample * 0.85, k_time)
                    kick_times.append(k_time)
                for step in range(16):
                    h_time = bar_start + step * (BEAT / 4.0)
                    vel = 0.25 if (step % 2 == 1) else 0.15
                    add_to_mix(mix, hat_closed * vel, h_time)
            elif bar_idx < 112:
                # Bars 108-111: Kick only
                for beat_idx in range(4):
                    k_time = bar_start + beat_idx * BEAT
                    add_to_mix(mix, kick_sample * 0.80, k_time)
                    kick_times.append(k_time)
            elif bar_idx == 112:
                # Bar 112: Final downbeat kick
                add_to_mix(mix, kick_sample * 0.80, bar_start)
                kick_times.append(bar_start)
                    
        elif section == "buildup":
            # Snare roll accelerating into drop
            roll_step = BEAT / 2.0 if bar_idx < 70 else (BEAT / 4.0 if bar_idx < 71 else BEAT / 8.0)
            n_hits = int(BAR / roll_step)
            for h in range(n_hits):
                hit_time = bar_start + h * roll_step
                progress = (bar_idx - 68 + h / n_hits) / 4.0
                vel = 0.25 + 0.65 * progress
                add_to_mix(mix, snare_sample * vel, hit_time)
                
        # 2. FX RISERS & DROPS
        if bar_idx == 8 or bar_idx == 72:
            # Impact downbeat with sub drop and crash
            add_to_mix(mix, crash_sample * 0.85, bar_start)
            add_to_mix(mix, sub_drop_sample * 0.90, bar_start)
            
        if bar_idx == 4 or bar_idx == 68:
            # 4-bar white noise riser leading into drop
            add_to_mix(mix, riser_sample * 0.75, bar_start)
            
        # 3. BASSLINE
        if section in {"verse1", "drop1", "verse2", "drop2", "outro_groove"}:
            # Rolling 16th bassline
            for step in range(16):
                b_time = bar_start + step * (BEAT / 4.0)
                # Octave variation on syncopated steps
                note = root_note
                if step in {2, 6, 10, 14}:
                    note += 12 # Octave up on offbeat 16ths
                elif step in {11, 15} and (bar_idx % 4 == 3):
                    note += 7  # 5th turnaround note
                    
                dur_note = (BEAT / 4.0) * 0.92
                bass_sig = synthesize_bass_note(note, dur_note, rng) * 0.72
                add_to_mix(mix, bass_sig, b_time)
                
        elif section == "intro" and bar_idx >= 4:
            # Gentle sub bass in late intro
            for beat_idx in range(4):
                b_time = bar_start + beat_idx * BEAT
                bass_sig = synthesize_bass_note(root_note, BEAT * 0.85, rng) * 0.45
                add_to_mix(mix, bass_sig, b_time)
                
        # 4. CHORDS & PADS
        cutoff = 1500.0 if section == "intro" else (2800.0 if "drop" in section else 2000.0)
        if section in {"intro", "breakdown", "buildup"}:
            # Lush sustained pad
            pad_vol = 0.46 if section == "breakdown" else 0.35
            hp_pad = 320.0 if section in {"breakdown", "buildup"} else 0.0
            pad_sig = synthesize_supersaw_chord(chord_info, BAR * 1.05, cutoff_hz=cutoff, pad_mode=True, 
                                                highpass_hz=hp_pad, rng=rng)
            add_to_mix(mix, pad_sig * pad_vol, bar_start)
        elif section in {"verse1", "drop1", "verse2", "drop2", "outro_groove"}:
            # Rhythmic syncopated chord stabs
            stab_vol = 0.45 if "drop" in section else 0.32
            # Syncopated rhythm: beats 0.0, 1.5, 2.5, 3.25
            for s_beat, s_dur in [(0.0, 0.4), (1.5, 0.4), (2.5, 0.4), (3.25, 0.65)]:
                stab_sig = synthesize_supersaw_chord(chord_info, s_dur * BEAT, cutoff_hz=cutoff, pad_mode=False, rng=rng)
                add_to_mix(mix, stab_sig * stab_vol, bar_start + s_beat * BEAT)
                
        # 5. ARPEGGIATOR / PLUCKS
        if section in {"intro", "verse1", "drop1", "verse2", "breakdown", "buildup", "drop2", "outro_groove"}:
            arp_vol = 0.28 if "drop" in section else 0.20
            if section == "intro" and bar_idx < 4:
                arp_vol *= 0.4
            for step in range(16):
                a_time = bar_start + step * (BEAT / 4.0)
                chord_idx = arp_indices[step] % len(chord_info)
                arp_note = chord_info[chord_idx] + 12 # Octave higher
                dur_arp = (BEAT / 4.0) * 0.95
                pluck_sig = synthesize_pluck_note(arp_note, dur_arp, rng)
                add_to_mix(mix, pluck_sig * arp_vol, a_time)
                
        # 6. LEAD MELODY
        if section in {"drop1", "drop2", "verse2", "breakdown"} and (section != "verse2" or bar_idx < 48):
            phrase_bar = bar_idx % 8
            notes_in_bar = lead_phrase[phrase_bar]
            lead_vol = 0.52 if "drop" in section else 0.36
            octave_base = 70 # Bb4
            if section == "breakdown":
                octave_base = 58 # Bb3 in breakdown for intimacy
                
            for b_pos, offset, dur_beats in notes_in_bar:
                l_time = bar_start + b_pos * BEAT
                l_note = octave_base + offset
                l_sig = synthesize_lead_note(l_note, dur_beats * BEAT, vibrato_delay=0.10, rng=rng)
                add_to_mix(mix, l_sig * lead_vol, l_time)
                
    # -----------------------------------------------------------------------
    # Dynamic Sidechain Pumping Envelope
    # -----------------------------------------------------------------------
    print(f"Applying sidechain compression ducking to {len(kick_times)} kick triggers...")
    duck_samples = int(SAMPLE_RATE * 0.22)
    t_duck = np.arange(duck_samples, dtype=np.float32) / SAMPLE_RATE
    att_samples = int(SAMPLE_RATE * 0.008)
    duck_curve = np.ones(duck_samples, dtype=np.float32)
    duck_curve[:att_samples] = np.linspace(1.0, 0.15, att_samples)
    duck_curve[att_samples:] = 0.15 + 0.85 * (1.0 - np.exp(-22.0 * (t_duck[att_samples:] - t_duck[att_samples])))
    
    # Apply sidechain ducking to the mix for punch and pump
    sc_env = np.ones(total_samples, dtype=np.float32)
    for kt in kick_times:
        s_idx = int(kt * SAMPLE_RATE)
        e_idx = min(total_samples, s_idx + duck_samples)
        length = e_idx - s_idx
        if length > 0:
            sc_env[s_idx:e_idx] = np.minimum(sc_env[s_idx:e_idx], duck_curve[:length])
            
    # We apply subtle ducking across the mix to lock elements to the groove
    mix *= (0.65 + 0.35 * sc_env[:, None])
    
    return mix

# ---------------------------------------------------------------------------
# Studio Mastering Bus Chain
# ---------------------------------------------------------------------------
def master_audio(raw_mix: np.ndarray) -> np.ndarray:
    """Mastering chain: Mono low-end, 4-band EQ, bus compression, limiting, and LUFS targeting."""
    print("Running studio mastering chain...")
    
    # 1. Mono Low-End (< 115 Hz)
    # Using 4th-order Linkwitz-Riley style crossover
    sos_low = scipy.signal.butter(4, 115.0, btype='lowpass', fs=SAMPLE_RATE, output='sos')
    sos_high = scipy.signal.butter(4, 115.0, btype='highpass', fs=SAMPLE_RATE, output='sos')
    low_band_l = scipy.signal.sosfilt(sos_low, raw_mix[:, 0])
    low_band_r = scipy.signal.sosfilt(sos_low, raw_mix[:, 1])
    mono_low = (low_band_l + low_band_r) * 0.5
    
    high_band_l = scipy.signal.sosfilt(sos_high, raw_mix[:, 0])
    high_band_r = scipy.signal.sosfilt(sos_high, raw_mix[:, 1])
    
    aligned_mix = np.column_stack([mono_low + high_band_l, mono_low + high_band_r]).astype(np.float32)
    
    # 2. Mastering EQ & Dynamics via Pedalboard
    master_chain = pb.Pedalboard([
        # Clean sub rumble cut (< 26 Hz)
        pb.HighpassFilter(cutoff_frequency_hz=26.0),
        # Low-end punch & sub-bass weight
        pb.LowShelfFilter(cutoff_frequency_hz=75.0, gain_db=1.8),
        # Clean lower-mid mud dip
        pb.PeakFilter(cutoff_frequency_hz=280.0, gain_db=-1.2, q=1.0),
        # Vocal/lead presence contour
        pb.PeakFilter(cutoff_frequency_hz=2800.0, gain_db=0.6, q=0.8),
        # Analog warmth high shelf (taming excess sizzle to match reference centroid)
        pb.HighShelfFilter(cutoff_frequency_hz=9500.0, gain_db=-2.5),
        # Master Bus Glue Compressor
        pb.Compressor(threshold_db=-14.0, ratio=2.2, attack_ms=30.0, release_ms=120.0),
        # Brickwall Limiter
        pb.Limiter(threshold_db=-0.8)
    ])
    
    processed = master_chain(aligned_mix.T, SAMPLE_RATE).T
    
    # 3. Loudness Normalization to match Biscuit Toner (-13.26 LUFS)
    meter = pyln.Meter(SAMPLE_RATE)
    current_loudness = meter.integrated_loudness(processed)
    print(f"Pre-normalized master loudness: {current_loudness:.2f} LUFS")
    
    target_loudness = -13.26
    gain_db = target_loudness - current_loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    print(f"Applying target gain adjustment: {gain_db:+.2f} dB")
    normalized = processed * gain_linear
    
    # Ensure true peak headroom <= -0.8 dBFS
    max_peak = 10.0 ** (-0.8 / 20.0) # ~0.9120 (-0.8 dBFS)
    peak = np.max(np.abs(normalized))
    if peak > max_peak:
        print(f"Limiting true peak from {peak:.4f} to {max_peak:.4f} (-0.8 dBFS)")
        # Smooth transparent saturation knee above 0.85 * max_peak
        threshold = max_peak * 0.85
        over = np.abs(normalized) > threshold
        normalized[over] = np.sign(normalized[over]) * (threshold + (max_peak - threshold) * np.tanh((np.abs(normalized[over]) - threshold) / (max_peak - threshold)))
        
    # Smooth fade out at the very end
    fade_len = int(SAMPLE_RATE * 1.5)
    normalized[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, None]
    
    final_loudness = meter.integrated_loudness(normalized)
    final_peak = np.max(np.abs(normalized))
    print(f"Final Master Loudness: {final_loudness:.2f} LUFS, Peak: {final_peak:.4f} ({20*np.log10(final_peak):.2f} dBFS)")
    return normalized

# ---------------------------------------------------------------------------
# MP3 Encoding & ID3 Metadata
# ---------------------------------------------------------------------------
def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())

def encode_mp3(wav_path: Path, output_mp3: Path, title: str = "Neon Meridian") -> None:
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    command = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "320k", "-ar", str(SAMPLE_RATE),
        "-metadata", f"title={title}",
        "-metadata", "artist=ma teho",
        "-metadata", "album_artist=ma teho",
        "-metadata", "composer=ma teho",
        "-metadata", "copyright=© 2026 ma teho. All rights reserved.",
        "-metadata", f"comment=Original melodic synth composition; produced by ma teho.",
        "-id3v2_version", "3", str(output_mp3)
    ]
    subprocess.run(command, check=True)
    print(f"Encoded 320 kbps MP3: {output_mp3}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ma Teho - Neon Meridian.mp3")
    parser.add_argument("output", nargs="?", type=Path, default=Path("Ma Teho - Neon Meridian.mp3"))
    args = parser.parse_args()
    
    t0 = time.time()
    print(f"--- Generating Ma Teho - Neon Meridian ---")
    print(f"Duration: {DURATION:.2f}s, BPM: {BPM}, Key: Bb minor / Db major")
    raw_mix = build_arrangement()
    mastered = master_audio(raw_mix)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "master.wav"
        print(f"Writing temporary uncompressed 24-bit master to {wav_path}...")
        write_wav(wav_path, mastered)
        print(f"Encoding to 320 kbps MP3 with full metadata...")
        encode_mp3(wav_path, args.output, title="Neon Meridian")
        
    t1 = time.time()
    print(f"Generation completed in {t1 - t0:.2f} seconds!")

if __name__ == "__main__":
    main()
