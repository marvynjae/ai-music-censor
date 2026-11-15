"""Vocal separation with Demucs (two stems: vocals / no_vocals)."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

def separate_vocals(audio_path: str, workdir: str):
    """Return (vocals_path, instrumental_path)."""
    src = Path(audio_path)
    tmp_in = Path(workdir) / f"track{src.suffix}"
    shutil.copy(src, tmp_in)
    out_root = Path(workdir) / "demucs_out"
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "-n", "htdemucs", "-o", str(out_root), str(tmp_in)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{proc.stderr[-1500:]}")
    d = out_root / "htdemucs" / "track"
    return str(d / "vocals.wav"), str(d / "no_vocals.wav")
