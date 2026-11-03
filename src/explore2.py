"""Exploration 2: ball and player statistics, and what the tracking looks like around Smart Tagging restarts.

Findings (see README): the ball-speed field in the file is unreliable; 4.2% of ball sightings are outside the pitch;
mean player speed dips to about 1 m/s around restarts and the ball is seen on only a quarter to a third of frames there.

Run:  python src/explore2.py     (prints summaries only)
"""
import xml.etree.ElementTree as ET
import numpy as np
from tracking import *
from features import *

P, B, team = load()
n = len(B)
g = graded_mask(n)
print("P", P.shape, "B", B.shape, "home slots", int((team == 0).sum()), "away slots", int((team == 1).sum()))
bx, by, bs = B[:, 0], B[:, 1], B[:, 2]
vis = ~np.isnan(bx)
print("ball seen on graded: %.1f%%" % (100 * vis[g].mean()))
print("ball x range %.3f..%.3f  y range %.3f..%.3f" % (np.nanmin(bx), np.nanmax(bx), np.nanmin(by), np.nanmax(by)))
# Ball positions outside the 0-1 box are the tracker seeing the ball beyond a touchline or goal line.
out = vis & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
print("ball outside 0-1 box on graded frames: %d (%.2f%% of seen)" % ((out & g).sum(), 100 * (out & g).sum() / max(1, (vis & g).sum())))
# Is the speed field in the file believable? Compare it with the speed from consecutive positions.
dx = np.diff(bx) * PITCH_L; dy = np.diff(by) * PITCH_W
fd = np.hypot(dx, dy) * FPS
both = vis[1:] & vis[:-1] & g[1:]
print("ball speed field median %.2f m/s, finite-diff median %.2f m/s, p95 field %.1f / fd %.1f" % (np.nanmedian(bs[1:][both]), np.median(fd[both]), np.nanpercentile(bs[1:][both], 95), np.percentile(fd[both], 95)))
seen_players = (~np.isnan(P[:, :, 0])).sum(1)
print("players tracked per graded frame: median %d, p5 %d, p95 %d" % (np.median(seen_players[g]), np.percentile(seen_players[g], 5), np.percentile(seen_players[g], 95)))
ps = P[:, :, 2]
mean_ps = np.nanmean(ps, axis=1)
print("mean player speed (m/s) on graded: p10 %.2f p50 %.2f p90 %.2f" % tuple(np.nanpercentile(mean_ps[g], [10, 50, 90])))

# --- profile around the start of every tagged restart (offset in seconds, medians over the restarts) ---
tag = require(["smart_tagging"], GAME, None)[0]
root = ET.parse(tag).getroot()
sets = []
for inst in root.iter("instance"):
    if (inst.findtext("code") or "").upper() != "SET PIECES": continue
    typ = ""
    for lab in inst.iter("label"):
        if (lab.findtext("group") or "") == "Type": typ = lab.findtext("text") or ""
    sets.append((typ, int(round(float(inst.findtext("start")) * FPS)), int(round(float(inst.findtext("end")) * FPS))))
print("tagged window lengths (s):", sorted({round((b - a) / FPS, 1) for _, a, b in sets}))
offs = list(range(-20, 16, 2))
print("profile around the START of tagged restarts (offset in s): medians over %d restarts" % len(sets))
print("offset  ball_seen%  ball_speed  mean_player_speed  ball_outside%")
for o in offs:
    seen, bsp, mps, ob = [], [], [], []
    for typ, a, b in sets:
        i = a + o * FPS
        if not (0 <= i < n - FPS) or not g[i]: continue
        w = slice(i, i + FPS)  # a 1 s window starting at the offset
        seen.append(vis[w].mean())
        if vis[w].any(): bsp.append(np.nanmedian(bs[w]))
        mps.append(np.nanmedian(mean_ps[w]))
        ob.append(out[w].mean())
    print("%+4d     %5.0f      %6.2f       %6.2f            %5.0f" % (o, 100 * np.mean(seen), np.median(bsp) if bsp else np.nan, np.median(mps), 100 * np.mean(ob)))
for t in ("Throw-in", "Goal Kick", "Corner Kick"):
    sub = [(a, b) for typ, a, b in sets if typ == t and g[a]]
    print("%-12s ball speed med 4s before start: %.2f, first 2s of window: %.2f" % (t, np.nanmedian([np.nanmedian(bs[max(0,a-4*FPS):a]) for a, b in sub]), np.nanmedian([np.nanmedian(bs[a:a+2*FPS]) for a, b in sub])))
