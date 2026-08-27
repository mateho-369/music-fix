#!/usr/bin/env python3
"""Analysis pass 3 (memory-lean): sub-bass roots per bar, chroma details, drum pattern histograms."""
from __future__ import annotations
import json
import numpy as np
import librosa
import scipy.signal

REF = "/tmp/analysis/ref48.wav"
SR = 22050
PC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def nname(m): r=int(round(m)); return f"{PC[r%12]}{r//12-1}"

y, sr = librosa.load(REF, sr=SR, mono=True)
tempo=143.55; beat=60.0/tempo; bar=4*beat; t0=0.070
n_bars=112

# ---- top-3 chroma per bar from saved summary ----
print("=== TOP-3 CHROMA PER BAR (selected) ===")
chords = json.load(open("/tmp/analysis/summary2.json"))["chords"]
for b in list(range(1,17))+[25,26,33,40,57,65,70,88,96,105]:
    v = np.array(chords[b-1]["chroma"])
    top=np.argsort(v)[::-1][:3]
    print(f" bar {b:3d}: " + ", ".join(f"{PC[t]} {v[t]*100:.0f}%" for t in top))

# ---- sub bass roots, per-bar slices ----
print("\n=== SUB-BASS ROOTS (25-110Hz band, per bar median pitch when active) ===")
sos = scipy.signal.butter(4, [25, 110], btype="bandpass", fs=sr, output="sos")
freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
bass_roots={}
for b in range(1,n_bars+1):
    s=t0+(b-1)*bar
    i0,i1=int(s*sr),int((s+bar)*sr)
    seg=scipy.signal.sosfilt(sos, y[i0:i1])
    S=np.abs(librosa.stft(seg, n_fft=4096, hop_length=512))
    sal,_=librosa.piptrack(S=S, sr=sr, fmin=25, fmax=110, threshold=0.05)
    est=[]
    for c in range(sal.shape[1]):
        col=sal[:,c]
        if col.max()>2e-3:
            k=np.argmax(col); est.append(freqs[k])
    if len(est)>=6:
        med=float(np.median(est))
        midi=librosa.hz_to_midi(med)
        bass_roots[b]=midi
json.dump(bass_roots, open("/tmp/analysis/bass_roots.json","w"), default=float)
for b,m in bass_roots.items():
    if b<=16 or 39<=b<=50 or 79<=b<=90 or b>=103:
        print(f" bar {b:3d}: {librosa.midi_to_hz(m):6.1f} Hz  midi {m:5.2f}  {nname(m)}")

# ---- drum onset histograms per 16th slot ----
print("\n=== DRUM ONSET HISTOGRAMS (per 16th slot 0-15, % of bars in section) ===")
hop=512
def onset_slots(lo,hi,label):
    sosb=scipy.signal.butter(4,[lo,hi],btype="bandpass",fs=sr,output="sos")
    f=scipy.signal.sosfilt(sosb,y)
    env=librosa.onset.onset_strength(y=f,sr=sr,hop_length=hop)
    onf=librosa.onset.onset_detect(onset_envelope=env,sr=sr,hop_length=hop,delta=0.12,backtrack=False)
    ont=librosa.frames_to_time(onf,sr=sr,hop_length=hop)
    sections={"A(9-40)":(9,41),"B(48-80)":(48,81),"C(88-103)":(88,104)}
    for name,(b0,b1) in sections.items():
        hist=np.zeros(16)
        for t in ont:
            g=(t-t0)/(beat/4)
            if b0*4-4 <= g < b1*4-4:
                slot=int(round((g%4)*4))
                if 0<=slot<16: hist[slot]+=1
        norm=hist/(b1-b0)*100
        print(f" {label:>7} {name}: " + " ".join(f"{v:4.0f}" for v in norm))
onset_slots(30,110,"KICK")
onset_slots(150,300,"SNRlow")
onset_slots(1200,4000,"SNRnoi")
onset_slots(6000,11000,"HAT")

# ---- energy envelope of intro ----
print("\n=== INTRO DETAIL (band RMS dB by 0.5s, bars 1-8) ===")
for lo,hi,lab in [(30,110,"sub"),(150,300,"low"),(1200,4000,"noise"),(6000,11000,"hat")]:
    sosb=scipy.signal.butter(4,[lo,hi],btype="bandpass",fs=sr,output="sos")
    f=scipy.signal.sosfilt(sosb,y)
    e=librosa.feature.rms(y=f,frame_length=1024,hop_length=hop)[0]
    et=librosa.frames_to_time(np.arange(len(e)),sr=sr,hop_length=hop)
    vals=[]
    for i in range(27):
        m=(et>=i*0.5)&(et<(i+1)*0.5)
        vals.append(20*np.log10(e[m].mean()+1e-9) if m.any() else -99)
    print(f" {lab:>5}: " + " ".join(f"{v:5.0f}" for v in vals))
