from pathlib import Path
import librosa
import soundfile as sf
import streamlit.components.v1 as components

FRONTEND = Path(__file__).parent / "trimmer_frontend"
_component = components.declare_component("section_trimmer", path=str(FRONTEND))


def prepare_preview(audio, sr, track_key):

    name = f"track_{track_key}.wav"
    for old in FRONTEND.glob("track_*.wav"):
        if old.name != name:
            try:
                old.unlink()
            except OSError:
                pass
    path = FRONTEND / name
    if not path.exists():
        mono = librosa.resample(audio.mean(axis=0), orig_sr=sr, target_sr=22050)
        sf.write(str(path), mono, 22050, subtype="PCM_16")
    return name


def section_trimmer(filename, window, key):

    value = _component(url=filename, window=float(window), key=key, default=0.0)
    return float(value or 0.0)