# Deep Analysis & Re-Production Notes
## Reference: `wanna play _ (Remix) (Cover) (Cover) (Cover).mp3`
## Output: `Ma Teho - Playmaker.mp3`

All measurements below were extracted directly from the decoded reference audio
(48 kHz float) using `tools/analyze_reference*.py`. The re-production
(`generate_ma_teho_playmaker.py`) is driven by these numbers.

---

## 1. Container / format (reference)

| Property | Reference | Notes |
|---|---|---|
| Duration | 187.36 s decoded / 3:07.39 container | 18.96 s of leading/following silence-free padding handled by T0 offset |
| Codec | MP3, ~197 kb/s VBR | contains 360x360 embedded JPEG cover |
| Sample rate | 48 000 Hz | |
| Channels | 2 (stereo) | |
| Metadata | title "wanna play ? (Remix) (Cover) (Cover) (Cover)", artist "unyieldinghocket381" | owner's own Suno-style account name |

## 2. Tempo / meter / tuning

| Property | Value |
|---|---|
| Tempo | **143.55 BPM** (median beat tracker; beat intervals 411.5 ± 11.3 ms) |
| Meter | 4/4 (bar = 1.6719 s; downbeat rotation test confirmed beat 1) |
| First downbeat (T0) | **0.070 s** |
| Tuning | +2 cents vs A440 (standard tuning within measurement error) |
| Total bars | 112 (music ends at T0 + 112×BAR = 187.318 s; tail fades to silence) |

## 3. Key / harmony

- **Key: G# natural minor** (global Krumhansl correlation 0.761; E major 0.681 = relative-color partner).
- Chord loop family (per-bar chroma + sub-bass root detection):
  - Main loop: **G#m | E | E | D#sus4** (hook) and **E | G#m | E | D#sus4** (riff section, E-lean)
  - Intro/breakdowns/outro: **G#m | E | G#m | D#sus4**
  - B-section variation: **Emaj7 | G#m | E | G#m**
  - Turnarounds land on **D#sus4 (V)** with G#sus2 colorings
- Harmonic rhythm: one chord per bar (fast 4-bar loops).
- Sub-bass register: **E1 (41 Hz) / G#1 (52 Hz) / D#1 (39 Hz)** roots with a continuous 40 Hz energy floor.

## 4. Structure (bars of 4/4 @ 143.55 BPM)

| Bars | Time | Section | Character |
|---|---|---|---|
| 1–8 | 0.0–13.4 s | Intro | Filtered chords, riser, hats enter bar 4, pickup phrase bar 7–8 |
| 9–24 | 13.4–40.1 s | Hook A1 | Full groove, lead melody (A#5/G#5/B5 phrases), snare 2+4 |
| 25–40 | 40.1–66.9 s | Riff A2 | Same loop, E-lean, riff takes the lead, extra kick on 14th slot |
| 41–47 | 66.9–78.6 s | Breakdown 1 | −24 dB dip, arp + pad + bass root only, drum fill into bar 48 |
| 48–79 | 78.6–130.6 s | Main B | 32 bars; long "sky" lead notes bars 71–79; grooves per 4 bars |
| 80–86 | 130.6–140.6 s | Breakdown 2 | Same dip recipe as 41–47 |
| 87 | 140.6–142.3 s | Riser | 1-bar white-noise riser + roll |
| 88–102 | 142.3–167.3 s | Final drop C | Highest density: extra kicks (slots 3/14), clap layer, bells |
| 103–112 | 167.3–187.3 s | Outro | Gradual layer strip and long fade, silent by ~186.6 s |

Section RMS (bars): intro ≈ −26 dBFS, hook/riff/B/C ≈ −14 dBFS,
breakdowns ≈ −26 dBFS, outro ramps down to silence.

## 5. Groove (16th-note grid, slots 0–15)

- Kick: syncopated **3-per-bar pattern at slots 0 / 6 / 11** (visible as bright
  vertical stripes in the groove-zoom spectrograms); variations: +slot 14 in A2,
  +slots 3 & 14 alternating in C, bar-4 fill variation in B.
- Snare: **beats 2 & 4 (slots 4, 12)** with ghost notes; clap doubles beat 4 in B/C.
- Hats: **dense 16ths**, accented offbeats (2/6/10/14), open hat on slot 10 of odd bars, ride 8ths in C.
- Fills: 16th snare rolls at section boundaries (bars 8/16/24/32/40/47/48/56/64/72/80/86/87/88/96/102).
- Quantization: essentially on-grid (±11 ms beat jitter is source noise, not swing).

## 6. Melody & bass (transcription summary)

- **No vocals** (pyin voiced fraction 0.0%); the "vocal-like" line is an instrumental lead (pitch range D#5–C#6).
- Lead hook contour (bars 9–16, quantized): A#5 sustained phrases stepping
  G#5↔A#5↔B5 with F#5/G#5 pickups; long G#5 "sky" notes in bars 71–79;
  final-drop answers around D#5–C#6. Full quantized note list embedded in
  `generate_ma_teho_playmaker.py` as `MELODY`.
- Bass: per-16th syncopated riff around G#2–C#3 fundamentals with octave jumps,
  following the chord roots (G#m→G#, E→E, D#sus4→D#). Full list embedded as
  `BASS_PAT_A/B/BREAK` patterns per loop.

## 7. Sound palette (reference, for contrast)

Supersaw-style synth chords/pads, plucky synth riff, sine-sub 808-style bass,
programmed electronic drums (metallic hats, mid snare), white-noise risers,
sub-drop impacts. Dense 16th hats and heavy 40 Hz energy are signature
elements that the new production preserves.

## 8. New instrumentation (output)

| Reference element | Playmaker replacement |
|---|---|
| Supersaw chords | FM tine electric piano (2-op, 3:1 ratio) + 3-osc analog pad |
| Plucky riff lead | Karplus-Strong plucked electric guitar |
| Singing lead | 3-osc hybrid formant lead (dual resonant peaks + sub-osc, delayed vibrato, duophonic fifth) |
| 808 sine sub bass | Round analog mono bass (detuned saws → Moog LP → tanh) + dedicated sub layer + continuous sub bed |
| Programmed kit | Hybrid punchy acoustic-EDM kit (pitch-swept kick w/ beater knock, crack + room snare, 4-burst clap, metallic hats, ride, crash/reverse-crash) |
| Risers/drops | Swept 3-band noise risers, sub-drop sweeps, downlifter, reverse crashes |

## 9. Mix / master (measured targets)

| Metric | Reference | Playmaker |
|---|---|---|
| Integrated loudness | −13.73 LUFS | **−13.68 LUFS** |
| True peak | +0.04 dBTP | **−4.4 dBTP** (deliberate headroom, no clipping) |
| Long-term spectrum (rel 1 kHz) | 40 Hz: +29.6 / 250 Hz: +9.4 / 2.5 k: −3.7 | 40 Hz: +29.0 / 250 Hz: +7.8 / 2.5 k: −7.0 |
| Bar-RMS section profile | see §4 | matched within ±2.5 dB across main sections |
| Stereo correlation | 0.87 full / wide breakdowns | 0.95 full / wide breakdowns |
| Duration | 187.36 s decoded | 187.36 s decoded (3:07.39 container) |

## 10. Reproduction

```bash
pip install -r requirements.txt
python3 tools/decode_ref.py                    # decode reference to /tmp/analysis/ref48.wav
python3 tools/analyze_reference.py             # pass 1: metering, tempo, key, chords
python3 tools/analyze_reference2.py            # pass 2: transcription, drums, structure
python3 tools/analyze_reference3.py            # pass 3: sub roots, groove histograms
python3 tools/make_spectrograms.py             # visual verification
python3 generate_ma_teho_playmaker.py          # render + master + tag the output
python3 tools/compare_mastering.py             # bar-RMS/spectrum diff vs reference
```

Rendering is seeded (`SEED = 20260827`) and deterministic.

## 11. Files

- `Ma Teho - Playmaker.mp3` — final master (320 kb/s CBR, 48 kHz, stereo,
  embedded cover, ID3v2.3: title/artist/composer/copyright/comment)
- `generate_ma_teho_playmaker.py` — full production (arrangement data + synthesis + mix + master)
- `cover_playmaker.png` — cover art source
- `index.html` + `serve.py` — private audition/download page (A/B decks, section
  jump buttons, spectrum visualizer; serve the repo with `python3 serve.py 8123`)
- `tools/` — analysis/verification scripts (+ `tools/make_project_zip.py`, which
  rebuilds the un-committed download bundle `Playmaker_project.zip`)
- `ANALYSIS.md` — this document
