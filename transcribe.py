import librosa
import soundfile as sf
import torch
from faster_whisper import WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
_MODEL_CACHE = {}


def _load_whisper(model_name: str = "base"):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
    return _MODEL_CACHE[model_name]


try:
    import streamlit as st
    get_whisper_model = st.cache_resource(_load_whisper)
except Exception:
    get_whisper_model = _load_whisper


def transcribe_words(audio_path: str, model_name: str = "base", language: str = "en"):
    model = get_whisper_model(model_name)

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    segments, _ = model.transcribe(
        audio,
        word_timestamps=True,
        language=language or "en",
        condition_on_previous_text=False,
        vad_filter=False,
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                })
    return words
