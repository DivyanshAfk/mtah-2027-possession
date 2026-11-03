"""Ball-fill model: estimate where the ball is from where the players are, for frames where the ball is not tracked.

Training targets are the ball positions the tracker already gives us on ~59% of graded frames, so no human labels are
involved. Features are permutation-invariant (the 46 track slots are anonymous): per-team centroid, spread, extremes,
coarse occupancy grid and 1 s mean displacement. Model: gradient boosting, one regressor per axis.

Validation is time-blocked (5 contiguous chunks of graded play; train on four, test on the fifth) and judged on how well
the estimate would serve possession, not just on distance error.

Result so far (see README): median error 10.4 m (20.3 m for the player centroid), but the estimate does NOT identify
the team of the ball carrier (agreement with the true nearest player's team is at chance level), so version 1 does not use it.

Run:  python src/ballfill.py        prints the evaluation and saves the filled ball track to <work dir>/ball_hat.npy
"""
import argparse
import os
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tracking import *
from features import *

warnings.filterwarnings("ignore")   # all-NaN slices are expected (frames with no visible players on one side)
GX, GY = 8, 5                        # coarse occupancy grid over the pitch: 8 columns along the length, 5 across


def _team_block(P, idx):
    """Permutation-invariant features of one team (slots `idx`): centroid, spread, extremes, count, grid, 1 s motion."""
    x = P[:, idx, 0]; y = P[:, idx, 1]
    n = len(P)
    seen = ~np.isnan(x)
    cnt = seen.sum(1)
    feats = [np.nanmean(x, 1), np.nanmean(y, 1), np.nanstd(x, 1), np.nanstd(y, 1),
             np.nanmin(x, 1), np.nanmax(x, 1), np.nanmin(y, 1), np.nanmax(y, 1), cnt.astype(np.float32)]
    # occupancy grid: how many of the team's visible players stand in each cell (as a fraction of visible players)
    cx = np.clip((np.nan_to_num(x) * GX).astype(int), 0, GX - 1)
    cy = np.clip((np.nan_to_num(y) * GY).astype(int), 0, GY - 1)
    cell = cx * GY + cy
    flat = (np.arange(n)[:, None] * (GX * GY) + cell)[seen]       # one bin index per (frame, visible player)
    grid = np.bincount(flat, minlength=n * GX * GY).reshape(n, GX * GY).astype(np.float32)
    grid /= np.maximum(cnt, 1)[:, None]
    # mean displacement over the last second, in metres: where the team is moving (the ball goes with the play)
    k = FPS
    dxs = np.full(n, np.nan, np.float32); dys = np.full(n, np.nan, np.float32)
    dxs[k:] = np.nanmean((x[k:] - x[:-k]) * PITCH_L, 1); dys[k:] = np.nanmean((y[k:] - y[:-k]) * PITCH_W, 1)
    return np.column_stack(feats + [dxs, dys]), grid


def make_features(P, team):
    """Feature matrix, one row per frame: home block, away block, and the centroid of everyone."""
    out = []
    for t in (0, 1):
        f, g = _team_block(P, np.where(team == t)[0])
        out += [f, g]
    allx = np.nanmean(P[:, :, 0], 1); ally = np.nanmean(P[:, :, 1], 1)
    out.append(np.column_stack([allx, ally]))
    return np.column_stack(out).astype(np.float32)


def fit(X, Y, rows):
    """Fit one gradient-boosting regressor per axis (x, y) on the given training rows. Deterministic (random_state=0)."""
    ms = []
    for j in (0, 1):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[rows], Y[rows, j]); ms.append(m)
    return ms


def predict(ms, X):
    """Predicted (x, y), normalised 0-1, for every row of X."""
    return np.column_stack([m.predict(X) for m in ms]).astype(np.float32)


def smooth_pred(pred, g, seconds=1.0):
    """Rolling-median smoothing of the predicted track, inside each half."""
    out = pred.copy()
    for a, b in halves():
        for j in (0, 1):
            out[a:b + 1, j] = rolling_median(pred[a:b + 1, j], seconds)
    return out


def nearest_team(P, team, xy):
    """Team of the player nearest to the point xy (per frame) and that distance in metres."""
    dx = (P[:, :, 0] - xy[:, None, 0]) * PITCH_L; dy = (P[:, :, 1] - xy[:, None, 1]) * PITCH_W
    d = np.hypot(dx, dy); d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(1)
    return team[j], d[np.arange(len(P)), j]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=5, help="number of contiguous time blocks")
    a = ap.parse_args()
    P, B, team = load(); n = len(B); g = graded_mask(n)
    X = make_features(P, team)
    seen = ~np.isnan(B[:, 0])
    nvis = (~np.isnan(P[:, :, 0])).sum(1)
    ok = g & (nvis >= 14)                      # need most of the players visible for the features to mean anything
    Y = np.clip(B[:, :2], -0.1, 1.1)           # ball targets (normalised); clipped so a stray point cannot dominate
    gi = np.where(g)[0]
    chunks = np.array_split(gi, a.folds)       # contiguous blocks of graded play
    oof = np.full((n, 2), np.nan, np.float32)  # out-of-fold predictions: every frame is predicted by a model that never saw its block
    for k, test in enumerate(chunks):
        train = np.setdiff1d(gi, test)
        rows = train[seen[train] & ok[train]][::4]   # every 4th frame: neighbours are near-duplicates, this just saves time
        ms = fit(X, Y, rows)
        oof[test] = predict(ms, X[test])
        print("fold %d: trained on %d frames, tested on %d" % (k + 1, len(rows), len(test)))
    oof_s = smooth_pred(np.nan_to_num(oof, nan=0.5), g)
    ev = seen & ok                             # frames where we can score the estimate against the tracked ball

    def err(est):
        return np.hypot((est[:, 0] - B[:, 0]) * PITCH_L, (est[:, 1] - B[:, 1]) * PITCH_W)

    centroid = np.column_stack([np.nanmean(P[:, :, 0], 1), np.nanmean(P[:, :, 1], 1)])
    centre = np.full((n, 2), 0.5, np.float32)
    print("\nerror of the estimated ball position vs the tracked ball, held-out frames only (metres)")
    print("%-26s %7s %7s %9s" % ("method", "median", "mean", "within 5m"))
    for name, est in (("pitch centre", centre), ("centroid of all players", centroid), ("model, raw", oof), ("model, 1 s smoothed", oof_s)):
        e = err(est)[ev]
        print("%-26s %7.1f %7.1f %8.0f%%" % (name, np.nanmedian(e), np.nanmean(e), 100 * np.nanmean(e <= 5)))
    # Downstream test: would the estimate pick the same team's player as the true ball does? (50% would be chance.)
    tt, td = nearest_team(P, team, B[:, :2])
    print("\nteam of the nearest player: estimated ball vs tracked ball (held-out frames)")
    for lab, mask in (("all frames with ball", ev), ("ball within 2 m of a player", ev & (td <= 2.0)), ("ball within 4 m of a player", ev & (td <= 4.0))):
        for name, est in (("centroid", centroid), ("model 1 s smoothed", oof_s)):
            et, _ = nearest_team(P, team, np.nan_to_num(est, nan=0.5))
            print("  %-30s %-20s agree %.1f%% (n=%d)" % (lab, name, 100 * np.mean(et[mask] == tt[mask]), mask.sum()))
    # Final model: fit on everything seen, predict every frame, save outside the repo (derived data).
    rows = np.where(seen & ok)[0][::4]
    ms = fit(X, Y, rows)
    full = smooth_pred(predict(ms, X), g)
    np.save(os.path.join(WORK, "ball_hat.npy"), full)
    gap = g & ~seen
    print("\nfilled %d gap frames (%.1f%% of graded) with the full model; saved to %s" % (gap.sum(), 100 * gap.sum() / g.sum(), os.path.join(WORK, "ball_hat.npy")))
    dens = ((np.where(np.isnan(td), np.inf, td) <= 2.0) & seen & g).sum() / g.sum()
    print("control-evidence density on graded frames today (ball seen and a player within 2 m): %.1f%%" % (100 * dens))


if __name__ == "__main__":
    main()
