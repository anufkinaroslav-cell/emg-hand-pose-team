"""Frame-level EMG features at 50 Hz (one frame every 10 samples).

Per channel:
  * log moving RMS over 25 / 50 / 100 / 250 samples (50-500 ms), centred
  * log RMS (100-sample window) of three frequency bands avoiding the mains notches
Frame features are normalised per recording (minus the recording median), so
electrode gain changes between sessions cancel out. The per-recording medians
are returned separately as a small "global" vector.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 500
HOP = 10
WINDOWS = (25, 50, 100, 250)
BANDS = ((12, 45), (55, 95), (105, 195))
_SOS = [butter(4, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]


def _moving_ms(x2: np.ndarray, w: int) -> np.ndarray:
    """Centred moving mean of x2 along axis 0 (edges use the available samples)."""
    T = x2.shape[0]
    c = np.concatenate([np.zeros((1, x2.shape[1])), np.cumsum(x2, 0)], 0)
    lo = np.clip(np.arange(T) - w // 2, 0, T)
    hi = np.clip(np.arange(T) - w // 2 + w, 0, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def frame_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (T, 6) raw EMG. Returns (frames (F, D) normalised, global (G,))."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    idx = np.arange(0, x.shape[0], HOP)
    eps = 1e-14
    feats = []
    x2 = x ** 2
    for w in WINDOWS:
        feats.append(0.5 * np.log(_moving_ms(x2, w)[idx] + eps))
    for sos in _SOS:
        xb = sosfiltfilt(sos, x, axis=0)
        feats.append(0.5 * np.log(_moving_ms(xb ** 2, 100)[idx] + eps))
    F = np.concatenate(feats, 1)  # (frames, 6 * 7)
    med = np.median(F, 0)
    Fn = F - med
    # global: per-channel median log-RMS(100) and band medians, absolute (volts, log)
    g = med.copy()
    return Fn.astype(np.float32), g.astype(np.float32)


def frame_targets(y: np.ndarray) -> np.ndarray:
    return y[::HOP].astype(np.float32)


def upsample(pred_frames: np.ndarray, T: int) -> np.ndarray:
    """Linear interpolation from frame centres (0, 10, 20, ...) to every sample."""
    centres = np.arange(pred_frames.shape[0]) * HOP
    t = np.arange(T)
    return np.stack([np.interp(t, centres, pred_frames[:, k]) for k in range(pred_frames.shape[1])], 1)
