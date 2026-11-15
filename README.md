# AI-Based Profanity Detection and Automatic Censoring in Music (SWE414)

Pipeline: audio -> Demucs (vocals / instrumental) -> Whisper (word timestamps) -> profanity detection
-> mute/beep on vocal stem -> recombine with instrumental -> censored WAV.

## Setup
Install FFmpeg, then: `pip install -r requirements.txt`

## Run
- CLI: `python pipeline.py song.mp3`
- UI: `streamlit run app.py`
- Tests: `pytest -q`

## AI-use declaration
Initial code scaffolding was AI-assisted (Claude). Declared per the course academic-integrity policy.
