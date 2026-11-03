"""Per-frame features from the tracking.

Speeds are computed from positions: the speed field in the raw file is not reliable (the ball's median is 1.3 m/s there,
5.1 m/s from the positions). All positions are normalised 0-1 and converted to metres before any distance is taken.
"""
import numpy as np
import pandas as pd

from tracking import FPS, PITCH_L, PITCH_W, graded_mask, halves  # noqa: F401  (re-exported for the scripts)


def player_speed(P, max_speed=12.0):
    """Median speed (m/s) over the visible track slots, one value per frame.

    Slot-to-slot jumps above max_speed are identity switches in the tracker, so they are dropped. The median over about
    20 players is robust to the few bad slots; it is low when the whole team is standing or walking (a stoppage)."""
    x = P[:, :, 0] * PITCH_L
    y = P[:, :, 1] * PITCH_W
    v = np.full(x.shape, np.nan, np.float32)
    v[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS   # metres per frame -> m/s
    v[v > max_speed] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(v, axis=1).astype(np.float32)


def ball_speed(B, max_speed=45.0):
    """Ball speed (m/s) from consecutive positions. NaN across tracking gaps and for impossible jumps."""
    x = B[:, 0] * PITCH_L
    y = B[:, 1] * PITCH_W
    v = np.full(len(B), np.nan, np.float32)
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    v[v > max_speed] = np.nan
    return v


def rolling_median(a, seconds):
    """Centred rolling median over `seconds` (window forced to an odd number of frames)."""
    w = max(1, int(seconds * FPS) | 1)
    return pd.Series(a).rolling(w, center=True, min_periods=1).median().to_numpy(np.float32)


def rolling_mean(a, seconds):
    """Centred rolling mean over `seconds`."""
    w = max(1, int(seconds * FPS) | 1)
    return pd.Series(a.astype(np.float32)).rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32)


def ball_outside(B, margin_m=0.8):
    """True where the ball is seen outside the touchlines / goal lines (plus a small margin for tracking noise)."""
    mx, my = margin_m / PITCH_L, margin_m / PITCH_W
    seen = ~np.isnan(B[:, 0])
    return seen & ((B[:, 0] < -mx) | (B[:, 0] > 1 + mx) | (B[:, 1] < -my) | (B[:, 1] > 1 + my))
