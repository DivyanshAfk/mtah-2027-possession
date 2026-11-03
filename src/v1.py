"""Version 1: possession by sustained ball control, with an explicit dead-ball detector.

Pipeline (no labels from this match are used anywhere):
  1. Dead ball. A frame is dead if any of these hold, then short blips are cleaned up:
       a. it lies in [start - tag_pre, start + tag_post] of a Smart Tagging restart (corner, throw-in, goal kick, free kick, kick-off);
       b. the players (median speed over the visible tracks) are almost standing still for a sustained stretch;
       c. the ball is seen outside the pitch lines.
  2. Possession. In every live stretch each frame where the ball is seen near a player is evidence for that player's team
     (strong if within ctrl_near metres, weaker within ctrl_far, none while the ball is in flight). The team state is the
     minimum-cost labelling: cost = evidence against the state + switch_cost per change of team. Where the ball is
     missing there is no evidence, so the state is held, never guessed.
  3. Ungraded frames (before kick-off, half-time, after the whistle) are X.
All constants are in PARAMS, are fixed in advance by football reasoning (not tuned on the public ground truth), and
the sensitivity runs below show what each one does.

Run:  python src/v1.py                  writes evaluation/results/v1.csv
      python src/v1.py --sensitivity    also writes the ablation / switch-cost variants
"""
import argparse
import os
import sys  # noqa: F401
import xml.etree.ElementTree as ET

import numpy as np

from tracking import *
from features import *
import baseline as _baseline   # only for write_submission(); tracking.py already put starter/ on the path

# Every tunable number lives here so the technical document can list them in one place.
PARAMS = dict(
    ctrl_near=2.0,      # metres: a player this close to the ball is treated as controlling it (strong evidence)
    ctrl_far=4.0,       # metres: weaker evidence out to here (tracking error is a metre or two)
    w_near=1.0,         # evidence weight inside ctrl_near
    w_far=0.4,          # evidence weight between ctrl_near and ctrl_far
    ball_fast=15.0,     # m/s: a ball moving faster than this is in flight, so nobody controls it
    switch_cost=40.0,   # price of changing team, in units of strong-evidence frames (the main knob: higher = stickier)
    tag_pre=2.0,        # seconds before a tagged restart that are already dead
    tag_post=8.0,       # seconds after the tag start that stay dead (the tag marks the start of a stoppage, see explore2/3)
    slow_thr=0.8,       # m/s: median player speed below this means the players are standing or walking
    slow_min_s=6.0,     # ...for at least this long, otherwise it is just a slow phase of play
    out_pad_s=1.0,      # seconds of padding around a ball-outside-the-pitch run
    merge_gap_s=3.0,    # live gaps shorter than this between two dead runs are filled in
    min_dead_s=3.0,     # dead runs shorter than this are dropped (tracking blips)
    use_tags=True, use_slow=True, use_out=True,   # switches used by the ablation runs
)


def tagged_restarts():
    """Start frames of every Smart Tagging 'SET PIECES' instance (kick-offs, throw-ins, corners, goal kicks, free kicks)."""
    tag = require(["smart_tagging"], GAME, None)[0]
    starts = []
    for inst in ET.parse(tag).getroot().iter("instance"):
        if (inst.findtext("code") or "").upper() == "SET PIECES":
            starts.append(int(round(float(inst.findtext("start")) * FPS)))   # seconds -> video frame
    return starts


def runs_of(mask):
    """Start (inclusive) and end (exclusive) indices of every run of True in a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def clean(mask, merge_gap, min_len):
    """Fill short False gaps between True runs, then drop True runs shorter than min_len (both in frames)."""
    m = mask.copy()
    s, e = runs_of(~m)
    for a, b in zip(s, e):  # fill short live gaps between dead runs
        if 0 < a and b < len(m) and (b - a) < merge_gap:
            m[a:b] = True
    s, e = runs_of(m)
    for a, b in zip(s, e):  # drop short dead blips
        if (b - a) < min_len:
            m[a:b] = False
    return m


def dead_mask(P, B, g, p):
    """Boolean mask of graded frames judged dead-ball (see the module docstring for the three signals)."""
    n = len(B)
    dead = np.zeros(n, bool)
    if p["use_tags"]:                              # (a) tagged restarts
        for a in tagged_restarts():
            dead[max(0, a - int(p["tag_pre"] * FPS)):min(n, a + int(p["tag_post"] * FPS))] = True
    if p["use_slow"]:                              # (b) players standing still for a while
        slow = rolling_median(player_speed(P), 3.0) < p["slow_thr"]
        s, e = runs_of(slow & g)
        for a, b in zip(s, e):
            if (b - a) >= p["slow_min_s"] * FPS:
                dead[a:b] = True
    if p["use_out"]:                               # (c) ball seen outside the lines
        bo = ball_outside(B)
        k = int(p["out_pad_s"] * FPS)
        s, e = runs_of(bo)
        for a, b in zip(s, e):
            dead[max(0, a - k):min(n, b + k)] = True
    dead = clean(dead, int(p["merge_gap_s"] * FPS), int(p["min_dead_s"] * FPS))
    return dead & g


def control_evidence(P, B, team, p):
    """Per frame: team of the nearest player to the ball (-1 if no usable evidence) and the evidence weight."""
    n = len(B)
    dx = (P[:, :, 0] - B[:, None, 0]) * PITCH_L      # metres from every slot to the ball
    dy = (P[:, :, 1] - B[:, None, 1]) * PITCH_W
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)             # unseen slots can never be the nearest
    j = np.argmin(d, axis=1)
    dmin = d[np.arange(n), j]
    e = team[j].astype(np.int8)
    w = np.where(dmin <= p["ctrl_near"], p["w_near"], np.where(dmin <= p["ctrl_far"], p["w_far"], 0.0)).astype(np.float32)
    bs = ball_speed(B)
    w[np.nan_to_num(bs, nan=0.0) > p["ball_fast"]] = 0.0   # ball in flight: the nearest player is not in control
    w[np.isnan(B[:, 0])] = 0.0                              # no ball, no evidence
    e[w == 0] = -1
    return e, w


def decode(e, w, a, b, switch_cost):
    """Two-state minimum-cost labelling (Viterbi) of frames a..b-1. Returns 0 = home / 1 = away per frame.

    Cost of being in state s at a frame = the evidence weight if that frame's evidence points at the OTHER team, else 0.
    Changing state costs switch_cost. Frames without evidence cost nothing in either state, so the label is simply held."""
    m = b - a
    cost = np.zeros((m, 2), np.float32)
    ev = e[a:b]; ww = w[a:b]
    cost[:, 0] = np.where(ev == 1, ww, 0.0)   # evidence for away costs the home state
    cost[:, 1] = np.where(ev == 0, ww, 0.0)   # evidence for home costs the away state
    back = np.zeros((m, 2), np.int8)           # back-pointers for the traceback
    c0, c1 = cost[0, 0], cost[0, 1]
    for t in range(1, m):
        s0 = c0; s1 = c1
        n0, b0 = (s0, 0) if s0 <= s1 + switch_cost else (s1 + switch_cost, 1)
        n1, b1 = (s1, 1) if s1 <= s0 + switch_cost else (s0 + switch_cost, 0)
        back[t, 0], back[t, 1] = b0, b1
        c0 = n0 + cost[t, 0]; c1 = n1 + cost[t, 1]
    state = 0 if c0 <= c1 else 1
    out = np.empty(m, np.int8)
    for t in range(m - 1, -1, -1):             # traceback from the cheaper end state
        out[t] = state
        state = back[t, state]
    return out


def build(P, B, team, **over):
    """Full pipeline. Returns (labels as an array of 'H'/'A'/'D'/'X' per frame, the parameters used)."""
    p = dict(PARAMS); p.update(over)
    n = len(B); g = graded_mask(n)
    dead = dead_mask(P, B, g, p)
    e, w = control_evidence(P, B, team, p)
    lab = np.full(n, "X", dtype="<U1")             # ungraded frames stay X
    lab[g & dead] = "D"
    live = g & ~dead
    s, t = runs_of(live)
    for a, b in zip(s, t):                         # decode each live stretch on its own
        lab[a:b] = np.where(decode(e, w, a, b, p["switch_cost"]) == 0, "H", "A")
    return lab, p


def summarise(lab, name):
    """Print frame counts per label and how many possessions (maximal H or A runs) the labelling contains."""
    cnt = {k: int((lab == k).sum()) for k in "HADX"}
    poss, dur = 0, []
    for a, b in halves():
        x = lab[a:b + 1]
        ch = np.concatenate(([True], x[1:] != x[:-1])); idx = np.where(ch)[0]; end = np.concatenate((idx[1:], [len(x)]))
        for i, j in zip(idx, end):
            if x[i] in "HA":
                poss += 1; dur.append((j - i) / FPS)
    print("%-22s H %6d A %6d D %6d (%.1f%% dead)  possessions %4d  mean %.1fs  median %.1fs" % (
        name, cnt["H"], cnt["A"], cnt["D"], 100 * cnt["D"] / max(1, cnt["H"] + cnt["A"] + cnt["D"]), poss, np.mean(dur), np.median(dur)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"), help="folder for the CSVs")
    ap.add_argument("--sensitivity", action="store_true", help="also write the switch-cost and ablation variants")
    a = ap.parse_args()
    P, B, team = load()
    variants = [("v1", {})]
    if a.sensitivity:
        variants += [("v1_cost20", dict(switch_cost=20.0)), ("v1_cost80", dict(switch_cost=80.0)), ("v1_cost160", dict(switch_cost=160.0)),
                     ("v1_notags", dict(use_tags=False)), ("v1_noslow", dict(use_slow=False)), ("v1_noout", dict(use_out=False)),
                     ("v1_nodead", dict(use_tags=False, use_slow=False, use_out=False))]
    for name, over in variants:
        lab, p = build(P, B, team, **over)
        _baseline.write_submission(os.path.join(a.out, name + ".csv"), lab)
        summarise(lab, name)
