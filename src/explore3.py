"""Exploration 3: how long are stoppages around tagged restarts, and how slow are the players during them?

General patterns only (no frame-specific rules). Findings: most tagged windows are 35 s long (kick-offs 5 s), so the tag
marks roughly where a stoppage begins, not its exact extent; lead-in times vary a lot by restart type; players are
almost standing (<0.8 m/s) for 6 s or more on about 16% of graded time. These informed the constants in v1.py.

Run:  python src/explore3.py
"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from tracking import *
from features import *

P, B, team = load()
n = len(B); g = graded_mask(n)
ps = rolling_median(player_speed(P), 3.0)           # 3 s median of the team's median player speed
live_ref = np.nanmedian(ps[g])
print("smoothed player speed (3 s median) on graded: p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f" % tuple(np.nanpercentile(ps[g], [10, 25, 50, 75, 90])))
tag = require(["smart_tagging"], GAME, None)[0]
sets = []                                           # (restart type, start frame, end frame)
for inst in ET.parse(tag).getroot().iter("instance"):
    if (inst.findtext("code") or "").upper() != "SET PIECES": continue
    typ = ""
    for lab in inst.iter("label"):
        if (lab.findtext("group") or "") == "Type": typ = lab.findtext("text") or ""
    sets.append((typ, int(round(float(inst.findtext("start")) * FPS)), int(round(float(inst.findtext("end")) * FPS))))
print("window length by type (s):")
L = defaultdict(list)
for t, a, b in sets: L[t].append((b - a) / FPS)
for t, v in L.items(): print("  %-22s n=%2d  min %.0f med %.0f max %.0f" % (t, len(v), min(v), np.median(v), max(v)))
# Lead-in: walk backwards from each restart start until the players are moving at the typical live pace again.
thr = live_ref
lead = defaultdict(list)
for t, a, b in sets:
    if not (0 <= a < n and g[a]): continue
    k = 0
    while k < 40 * FPS and a - k > 0 and g[a - k] and not (ps[a - k] >= thr): k += 1
    lead[t].append(k / FPS)
print("lead-in: seconds back from tag start until smoothed player speed returns to the typical live pace (%.2f m/s)" % thr)
for t, v in lead.items(): print("  %-22s n=%2d  p25 %.1f  median %.1f  p75 %.1f" % (t, len(v), *np.percentile(v, [25, 50, 75])))
inwin = np.concatenate([ps[a:b + 1] for t, a, b in sets if g[a]])
print("player speed inside tagged windows: p50 %.2f p75 %.2f p90 %.2f" % tuple(np.nanpercentile(inwin, [50, 75, 90])))
bo = ball_outside(B)
print("ball outside the pitch: %d graded frames, in %d separate runs" % ((bo & g).sum(), ((np.diff(np.concatenate(([0], (bo & g).astype(int)))) == 1).sum())))
# How much of the game is "players nearly standing still for a while" at different speed thresholds?
for thr2 in (0.8, 1.0, 1.2):
    slow = (ps < thr2) & g
    d = np.diff(np.concatenate(([0], slow.astype(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = (e - s) / FPS
    print("slow-player runs (<%.1f m/s): %d runs; >=6 s: %d runs covering %.1f%% of graded time" % (thr2, len(runs), (runs >= 6).sum(), 100 * runs[runs >= 6].sum() * FPS / g.sum()))
