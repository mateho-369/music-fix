#!/usr/bin/env python3
"""Rebuild Playmaker_project.zip (download bundle for the audition page).

The zip is a convenience bundle of files that are all version-controlled
individually, so it is intentionally NOT committed to git — regenerate it with:

    python3 tools/make_project_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Playmaker_project.zip"

INCLUDE = [
    "Ma Teho - Playmaker.mp3",
    "cover_playmaker.png",
    "generate_ma_teho_playmaker.py",
    "ANALYSIS.md",
    "requirements.txt",
    "serve.py",
    "index.html",
    "tools/analyze_reference.py",
    "tools/analyze_reference2.py",
    "tools/analyze_reference3.py",
    "tools/compare_mastering.py",
    "tools/decode_ref.py",
    "tools/make_spectrograms.py",
    "tools/make_project_zip.py",
]


def main() -> None:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in INCLUDE:
            p = ROOT / name
            if not p.exists():
                print("warning: missing", name)
                continue
            z.write(p, arcname=f"playmaker_project/{name}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
