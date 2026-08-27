#!/usr/bin/env python3
"""Decode the reference MP3 to /tmp/analysis/ref48.wav for analysis tools."""
import os
import subprocess

import imageio_ffmpeg

ff = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs("/tmp/analysis", exist_ok=True)
subprocess.run(
    [ff, "-y", "-hide_banner", "-loglevel", "error",
     "-i", "wanna play _ (Remix) (Cover) (Cover) (Cover).mp3",
     "-vn", "-acodec", "pcm_f32le", "/tmp/analysis/ref48.wav"],
    check=True,
)
print("decoded -> /tmp/analysis/ref48.wav")
