import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import censor as C

SR = 44100

def tone(sec=3.0, freq=440):
    t = np.arange(int(SR * sec)) / SR
    return np.stack([0.5 * np.sin(2 * np.pi * freq * t)] * 2).astype(np.float32)

def rms(y): return float(np.sqrt(np.mean(y ** 2)))

def test_mute_silences_segment_only():
    y = tone()
    out = C.mute_segments(y, SR, [(1.0, 1.5)], pad=0.0)
    assert rms(out[:, int(1.05*SR):int(1.45*SR)]) < 1e-3     # muted
    assert rms(out[:, :int(0.9*SR)]) > 0.3                    # untouched before
    assert rms(out[:, int(1.6*SR):]) > 0.3                    # untouched after

def test_recombine_keeps_instrumental():
    vocals, inst = tone(freq=440), tone(freq=220)
    out = C.censor_stems(vocals, inst, SR, [(1.0, 1.5)], pad=0.0)
    assert rms(out[:, int(1.05*SR):int(1.45*SR)]) > 0.3       # instrumental still plays
    assert out.shape == vocals.shape

def test_beep_adds_signal_over_muted_silence():
    y = np.zeros((2, SR * 2), dtype=np.float32)
    out = C.add_beeps(y, SR, [(0.5, 1.0)], pad=0.0)
    assert rms(out[:, int(0.6*SR):int(0.9*SR)]) > 0.1

def test_overlapping_segments_merge():
    assert C.merge_segments([(0, 1), (0.98, 2)]) == [[0, 2]]

def test_length_mismatch_handled():
    out = C.censor_stems(tone(3.0), tone(2.5), SR, [], pad=0.0)
    assert out.shape[-1] == int(SR * 2.5)
