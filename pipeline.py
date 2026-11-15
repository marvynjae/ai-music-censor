
"""Audio -> (Demucs) -> Whisper -> profanity detection -> censor."""
import tempfile
import time

from separate import separate_vocals
from transcribe import transcribe_words
from profanity import detect
import censor as C


def render(source, flagged, method="mute", pad=0.1):
    """Apply censoring to the already-loaded audio. No AI processing is repeated."""
    segments = [(w["start"], w["end"]) for w in flagged]

    if source["separated"]:
        return C.censor_stems(
            source["vocals"],
            source["inst"],
            C.SR,
            segments,
            method,
            pad
        )

    return C.censor_full_mix(
        source["mix"],
        C.SR,
        segments,
        method,
        pad
    )


def run(in_path, out_path=None, method="mute", model_name="base", pad=0.1,
        use_separation=True, language=None, clf=None):
    """Transcribe and detect profanity, then return the audio sources and results."""
    timings = {}

    with tempfile.TemporaryDirectory() as workdir:
        t0 = time.time()

        if use_separation:
            vocals_path, instrumental_path = separate_vocals(in_path, workdir)
            timings["separation_s"] = round(time.time() - t0, 1)
            transcribe_src = vocals_path
        else:
            transcribe_src = in_path

        t0 = time.time()
        words = transcribe_words(transcribe_src, model_name, language)
        timings["transcription_s"] = round(time.time() - t0, 1)

        flagged = detect(words, clf)

        # Load audio into memory before the temporary Demucs directory is removed.
        if use_separation:
            source = {
                "separated": True,
                "vocals": C.load(vocals_path),
                "inst": C.load(instrumental_path)
            }
        else:
            source = {
                "separated": False,
                "mix": C.load(in_path)
            }

    return {
        "transcript": words,
        "flagged": flagged,
        "timings": timings,
        "output": None,
        "source": source
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)

    result = run(sys.argv[1])

    print(result["timings"])

    for word in result["flagged"]:
        print(
            f"censored {word['word']!r} "
            f"{word['start']:.2f}-{word['end']:.2f}s"
        )
