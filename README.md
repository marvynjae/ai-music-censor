# AI-Based Profanity Detection and Automatic Censoring in Music

**Course:** SWE414 – Artificial Intelligence and Intelligent Systems

## Overview

An AI-powered application that detects profanity in music and automatically censors identified words. It combines audio source separation, speech recognition and profanity detection to produce a censored version of the original track.

## Processing Pipeline

Audio Input → Demucs → Whisper Transcription → Profanity Detection → Audio Censoring → Output

1. **Audio input:** Load the song for processing.
2. **Source separation:** Demucs separates vocals from the instrumental track.
3. **Speech recognition:** Faster-Whisper transcribes vocals and identifies individual word timestamps.
4. **Profanity detection:** Identifies explicit words using a profanity detection library and an additional local word list.
5. **Audio censoring:** Mutes or replaces detected words with beeps.
6. **Audio reconstruction:** Combines the processed vocals with the instrumental track.

## Features

- Upload audio files for processing.
- Preview and scan a selected 30-second section.
- Detect profanity with word-level timestamps.
- Choose between mute and beep censoring.
- Adjust censoring padding.
- Preserve the instrumental while censoring vocals.
- Process and export the full song.
- Download the censored track as an MP3.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core application |
| Streamlit | User interface |
| Demucs | Vocal and instrumental separation |
| Faster-Whisper | Speech recognition and word timestamps |
| Better Profanity | Profanity detection |
| NumPy | Audio data processing |
| Librosa | Audio loading and resampling |
| SoundFile | Audio file handling |
| Lameenc | MP3 encoding |

## Setup

### Requirements

- Python 3.10 or later
- FFmpeg
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/marvynjae/ai-music-censor.git
cd ai-music-censor
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure FFmpeg is installed and accessible from your system PATH.

## Running the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

The command-line pipeline can also be used to analyse a song and print detected profanity and processing timings:

```bash
python pipeline.py song.mp3
```

Run tests:

```bash
pytest -q
```

## AI Use Declaration

Initial code scaffolding was AI-assisted using Claude. This use of AI is declared in accordance with the course's academic-integrity policy.