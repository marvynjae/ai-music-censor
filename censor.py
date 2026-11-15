
"""Audio censoring: mute vocal stem, recombine with instrumental, optional beep."""
import io
import numpy as np
import librosa
import soundfile as sf
import lameenc

SR = 44100

def load(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=False)
    return np.atleast_2d(y).astype(np.float32)           # (channels, n)

def merge_segments(segs, gap=0.05):
    merged = []
    for s, e in sorted(segs):
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def _regions(segments, sr, n, pad):
    padded = [(max(0.0, s - pad), e + pad) for s, e in segments]
    for s, e in merge_segments(padded):
        i0, i1 = int(s * sr), min(int(e * sr), n)
        if i1 > i0:
            yield i0, i1

def _envelope(length, sr, fade=0.01):
    env = np.ones(length, dtype=np.float32)
    f = min(int(fade * sr), length // 2)
    if f > 0:
        env[:f] = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
    return env

def mute_segments(y, sr, segments, pad=0.1):
    out = y.copy()
    for i0, i1 in _regions(segments, sr, out.shape[-1], pad):
        out[..., i0:i1] *= (1 - _envelope(i1 - i0, sr))
    return out

def add_beeps(y, sr, segments, pad=0.1, freq=1000, gain=0.25):
    out = y.copy()
    for i0, i1 in _regions(segments, sr, out.shape[-1], pad):
        t = np.arange(i1 - i0) / sr
        out[..., i0:i1] += gain * np.sin(2 * np.pi * freq * t) * _envelope(i1 - i0, sr)
    return out

def censor_stems(vocals, instrumental, sr, segments, method="mute", pad=0.1):
    n = min(vocals.shape[-1], instrumental.shape[-1])
    mix = mute_segments(vocals[..., :n], sr, segments, pad) + instrumental[..., :n]
    if method == "beep":
        mix = add_beeps(mix, sr, segments, pad)
    return np.clip(mix, -1, 1)

def censor_full_mix(y, sr, segments, method="mute", pad=0.1):
    out = mute_segments(y, sr, segments, pad)
    if method == "beep":
        out = add_beeps(out, sr, segments, pad)
    return np.clip(out, -1, 1)

def save(path, y, sr=SR):
    sf.write(path, y.T, sr)


def to_wav_bytes(y, sr=SR):
    """Encode audio (channels, n) as in-memory WAV bytes, for st.audio / downloads."""
    buf = io.BytesIO()
    sf.write(buf, y.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def to_mp3_bytes(y, sr=SR):
    channels = y.shape[0]

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(channels)
    encoder.set_bit_rate(192)
    encoder.set_quality(2)

    audio = np.ascontiguousarray(
        np.clip(y.T, -1.0, 1.0) * 32767,
        dtype="<i2"
    )

    return bytes(encoder.encode(audio.tobytes()) + encoder.flush())
