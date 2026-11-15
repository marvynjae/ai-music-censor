import librosa
import whisper

_cache = {}

def transcribe_words(audio_path: str, model_name: str = "base", language=None):
    if model_name not in _cache:
        _cache[model_name] = whisper.load_model(model_name)
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)   # Whisper expects 16 kHz mono
    result = _cache[model_name].transcribe(
        audio, word_timestamps=True, language=language,
        condition_on_previous_text=False, fp16=False,
    )
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(),
                          "start": float(w["start"]), "end": float(w["end"])})
    return words