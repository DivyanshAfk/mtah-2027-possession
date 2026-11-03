"""Load the tactical tracking (ATD) into numpy arrays and cache them outside the repo.

Arrays (frame index = video frame, 0 .. n-1; the ATD frame_id is video frame + 1):
    P[n, slots, 3]  x, y (normalised 0-1) and speed (m/s) per track slot, NaN where unseen
    B[n, 3]         same for the ball
    team[slots]     0 = home (Barcelona), 1 = away (Athletic)
The cache lives in MTAH_WORK_DIR (default C:\\mtah-work) so no derived tracking data enters the repo.

Every other script starts with:  P, B, team = load()
"""
import os
import sys

import numpy as np

# --- make the organisers' starter code importable (games.py in the repo root, baseline.py in starter/) ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402  (organisers' helpers)
import baseline as _baseline  # noqa: E402  (we reuse its slot -> team mapping)

FPS = 25
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]   # pitch length / width in metres (105.3 x 68.0 measured by the organisers)
# Derived arrays are stored here, never inside the repo (the match data terms forbid publishing derived data).
WORK = os.environ.get("MTAH_WORK_DIR", r"C:\mtah-work" if os.name == "nt" else os.path.expanduser("~/mtah-work"))


def halves():
    """[(first_frame, last_frame), ...] of the two graded halves, inclusive."""
    return [(a, b) for _, a, b in half_video_blocks(GAME)]


def graded_mask(n):
    """Boolean mask over n frames: True inside the two halves (the only frames that are scored)."""
    m = np.zeros(n, bool)
    for a, b in halves():
        m[a:b + 1] = True
    return m


def _parse(raw_path, n, slots):
    """Parse the raw ATD text file. Each line is  '<frame_id>:<slot>;<slot>;...:<ball>'  with 'x,y,speed' groups."""
    P = np.full((n, slots, 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1               # frame_id counts from 1, our frames from 0
            if not 0 <= i < n:
                continue
            vals = parts[1].replace(";", ",").split(",")
            if len(vals) == slots * 3:          # 'NaN' strings convert to float nan, so unseen slots stay NaN
                P[i] = np.array(vals, dtype=np.float32).reshape(slots, 3)
            b = parts[2].split(",")
            if len(b) >= 3:
                B[i] = np.array(b[:3], dtype=np.float32)
    return P, B


def load(refresh=False):
    """Return (P, B, team). Parses the raw file once (about a minute) and then reads the cached .npz."""
    os.makedirs(WORK, exist_ok=True)
    cache = os.path.join(WORK, "tracking.npz")
    meta, raw = require(["atd_meta", "atd_raw"], GAME, None)   # paths come from games.py / MTAH_DATA_DIR
    team = _baseline.team_of_each_track(meta)                   # 46 slots -> 0 home / 1 away
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        return z["P"], z["B"], team
    n = source_frames("tactical", GAME)
    P, B = _parse(raw, n, len(team))
    np.savez(cache, P=P, B=B)
    return P, B, team
