from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CACHE = {}


def _load_demucs(model_name: str = "htdemucs"):
    if model_name not in _MODEL_CACHE:
        model = get_model(model_name)
        model.to(DEVICE)
        model.eval()
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


try:
    import streamlit as st
    get_demucs_model = st.cache_resource(_load_demucs)
except Exception:
    get_demucs_model = _load_demucs


def separate_vocals(audio_path: str, workdir: str, model_name: str = "htdemucs"):
    model = get_demucs_model(model_name)

    wav_np, _ = librosa.load(audio_path, sr=model.samplerate, mono=False)
    wav_np = np.atleast_2d(wav_np).astype(np.float32)

    if wav_np.shape[0] == 1:
        wav_np = np.repeat(wav_np, 2, axis=0)
    elif wav_np.shape[0] > 2:
        wav_np = wav_np[:2, :]

    wav = torch.from_numpy(wav_np)
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav_norm[None].to(DEVICE),
            device=DEVICE,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]

    sources = (sources * ref.std() + ref.mean()).cpu().numpy()

    vocal_idx = model.sources.index("vocals")
    vocals = sources[vocal_idx]
    instrumental = sum(
        sources[i] for i in range(len(model.sources)) if i != vocal_idx
    )

    out_dir = Path(workdir) / "demucs_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = out_dir / "vocals.wav"
    inst_path = out_dir / "no_vocals.wav"

    sf.write(str(vocals_path), vocals.T, model.samplerate)
    sf.write(str(inst_path), instrumental.T, model.samplerate)

    return str(vocals_path), str(inst_path)
