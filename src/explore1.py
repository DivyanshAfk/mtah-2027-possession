"""Exploration 1: how often the ball is missing, how many possessions simple rules give, and how the Smart Tagging restarts
look. Prints aggregate numbers only; derived arrays stay in the work dir (not in the repo).

Run:  python src/explore1.py
"""
import os, xml.etree.ElementTree as ET
import numpy as np
from tracking import *            # also puts the repo root and starter/ on sys.path
import baseline as B               # organisers' baseline: reused for the nearest-player labelling

os.makedirs(r"C:\mtah-work", exist_ok=True)
meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
n = source_frames("tactical", GAME)
cache = r"C:\mtah-work\codes.npy"
if os.path.exists(cache):
    codes = np.load(cache)
else:
    codes = B.nearest_player_labels(raw, B.team_of_each_track(meta), n, GAMES[GAME]["pitch"])
    np.save(cache, codes)

halves = half_video_blocks(GAME)  # [(1, a, b), (2, a, b)]
graded = np.zeros(n, bool)
for _, a, b in halves:
    graded[a:b + 1] = True
vis = codes != -1

def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]

print("== 1. Ball visibility (graded frames)")
print("graded frames %d, ball seen %.1f%%" % (graded.sum(), 100 * vis[graded].mean()))
s, e = runs(~vis & graded)
L = (e - s) / FPS
print("missing-ball gaps: %d, median %.2fs, p90 %.1fs, p99 %.1fs, max %.1fs" % (len(L), np.median(L), np.percentile(L, 90), np.percentile(L, 99), L.max()))
tot = L.sum()
for t in (1, 2, 5, 10):
    print("  share of missing time in gaps > %2ds: %.1f%%  (%d gaps)" % (t, 100 * L[L > t].sum() / tot, (L > t).sum()))
s2, e2 = runs(vis & graded)
V = (e2 - s2) / FPS
print("visible stretches: %d, median %.2fs, p90 %.1fs" % (len(V), np.median(V), np.percentile(V, 90)))

print("== 2. Possession counts under simple rules (counts only)")
def label_runs(lab):
    cnt, dur = 0, []
    for _, a, b in halves:
        x = lab[a:b + 1]
        keep = x >= 0
        xs = x[keep]
        if len(xs) == 0:
            continue
        ch = np.concatenate(([True], xs[1:] != xs[:-1]))
        idx = np.where(ch)[0]
        ends = np.concatenate((idx[1:], [len(xs)]))
        cnt += len(idx); dur += list((ends - idx) / FPS)
    return cnt, (np.mean(dur) if dur else 0)

def ffill(c):
    out = c.copy()
    for _, a, b in halves:
        last = -1
        for i in range(a, b + 1):
            if out[i] >= 0: last = out[i]
            else: out[i] = last
    return out

def majority(lab, w):
    out = lab.copy()
    for k in (0, 1):
        pass
    h = np.concatenate(([0], np.cumsum(lab == 0)))
    a_ = np.concatenate(([0], np.cumsum(lab == 1)))
    i = np.arange(len(lab)); lo = np.maximum(0, i - w); hi = np.minimum(len(lab), i + w + 1)
    hh = h[hi] - h[lo]; aa = a_[hi] - a_[lo]
    res = np.where(hh >= aa, 0, 1).astype(np.int8)
    res[(hh + aa) == 0] = -1
    return res

def min_run(lab, mn):
    """Absorb runs shorter than mn frames into the previous run (per half)."""
    out = lab.copy()
    for _, a, b in halves:
        x = out[a:b + 1].copy()
        changed = True
        while changed:
            changed = False
            ch = np.concatenate(([True], x[1:] != x[:-1])); idx = np.where(ch)[0]; ends = np.concatenate((idx[1:], [len(x)]))
            for j in range(1, len(idx)):
                if ends[j] - idx[j] < mn:
                    x[idx[j]:ends[j]] = x[idx[j - 1]]; changed = True; break
        out[a:b + 1] = x
    return out

c, d = label_runs(codes); print("nearest-player, ball-gaps left empty : %4d possessions, mean %.1fs" % (c, d))
f = ffill(codes); c, d = label_runs(f); print("+ hold last team through ball gaps    : %4d possessions, mean %.1fs" % (c, d))
for w in (50, 125, 250):
    m = majority(f, w); c, d = label_runs(m); print("+ majority vote +/-%3d frames         : %4d possessions, mean %.1fs" % (w, c, d))
for sec in (3, 5, 8):
    m = min_run(f, sec * FPS); c, d = label_runs(m); print("+ drop possessions shorter than %ds    : %4d possessions, mean %.1fs" % (sec, c, d))
print("(reference: the public 5-minute window has 17 possessions, mean 14.2s)")

print("== 3. Smart Tagging restarts vs ball visibility")
tag = require(["smart_tagging"], GAME, None)[0]
root = ET.parse(tag).getroot()
rows = []
for inst in root.iter("instance"):
    if (inst.findtext("code") or "").upper() != "SET PIECES": continue
    typ = ""; team = ""
    for lab in inst.iter("label"):
        g = lab.findtext("group") or ""
        if g == "Type": typ = lab.findtext("text") or ""
        if g == "Team": team = lab.findtext("text") or ""
    rows.append((typ, team, int(round(float(inst.findtext("start")) * FPS)), int(round(float(inst.findtext("end")) * FPS))))
from collections import Counter, defaultdict
print("set pieces:", dict(Counter(r[0] for r in rows)))
by = defaultdict(list)
for typ, team, a, b in rows:
    a = max(a, 0); b = min(b, n - 1)
    by[typ].append((vis[a:b + 1].mean(), a, b))
for typ, v in by.items():
    print("  %-22s n=%2d  ball seen in tagged window: mean %.0f%%, in graded play: %d" % (typ, len(v), 100 * np.mean([x[0] for x in v]), sum(1 for x in v if graded[x[1]])))
print("EXPLORE-DONE")
