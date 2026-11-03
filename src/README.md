# Our code: possession labelling for More Than A Hack 2027

This folder holds everything we have written on top of the organisers' starter repo. It lives next to the starter
files (`games.py`, `starter/`, `evaluation/`) and imports them, so it must stay inside that repo (`<repo>/src/`).

Status: **version 1 works and is scored on the public window; the ball-fill model is trained and evaluated
but is not used by version 1.** Nothing here is a final submission yet.

## What is where

| File | What it does |
|---|---|
| `tracking.py` | Loads the tracking into numpy arrays (`P` players, `B` ball, `team`), caches them outside the repo. Every script starts with `P, B, team = load()`. |
| `features.py` | Per-frame features: player speed, ball speed, rolling medians, "ball outside the pitch". Speeds come from positions (the speed field in the file is unreliable). |
| `v1.py` | **Version 1.** Dead-ball detector + possession by sustained ball control. Writes `evaluation/results/v1.csv`. |
| `ballfill.py` | **Ball-fill model.** Predicts the ball position from the player positions where the ball is not tracked. Prints its evaluation and saves the filled track. |
| `score_table.py` | Runs the organisers' scorer over every CSV in `evaluation/results` and prints a compact table. |
| `explore1.py`, `explore2.py`, `explore3.py` | Exploration scripts that print summaries (ball gaps, possession counts, what the tracking looks like around restarts). |
| `check_env.py` | Prints the package versions. |
| `requirements.txt` | Pinned package versions. |

## Where things live

- Code: this folder (`src/`), inside a clone of the organisers' starter repo.
- Match data (never in git): the folder named by the environment variable `MTAH_DATA_DIR` (see `DATA.md`).
- Derived arrays (never in git): the folder named by `MTAH_WORK_DIR` (default `C:\mtah-work` on Windows, `~/mtah-work` elsewhere): `tracking.npz` (parsed tracking), `codes.npy`, `ball_hat.npy`.
- Result CSVs: `evaluation/results/` (the organisers' four baselines; our own `v1*.csv` files are generated there and are not committed).

## How to run (Windows PowerShell; on Linux/macOS use `python` and `export` instead)

```powershell
cd <path-to-this-repo>
$env:MTAH_DATA_DIR = '<path-to-the-match-data>'   # folder that contains tactical\
.\.venv\Scripts\python.exe -m pip install -r src\requirements.txt   # once

.\.venv\Scripts\python.exe src\check_env.py           # environment check
.\.venv\Scripts\python.exe src\v1.py                  # writes evaluation\results\v1.csv (first run parses the tracking, about a minute)
.\.venv\Scripts\python.exe src\v1.py --sensitivity    # also the switch-cost and ablation variants
.\.venv\Scripts\python.exe src\score_table.py         # score every CSV in evaluation\results on the public 5-minute window
.\.venv\Scripts\python.exe src\ballfill.py            # train and cross-validate the ball-fill model
.\.venv\Scripts\python.exe src\explore1.py            # also explore2.py, explore3.py
.\.venv\Scripts\python.exe evaluation\validate_submission.py evaluation\results\v1.csv   # format check before any upload
.\.venv\Scripts\python.exe starter\baseline.py        # regenerate the organisers' four baselines
```

## What version 1 does (`v1.py`)

1. **Dead ball.** A frame is dead if it is within `[start - 2 s, start + 8 s]` of a Smart Tagging restart, or the players have been almost standing still (median speed under 0.8 m/s for 6 s or more), or the ball is seen outside the pitch lines. Short gaps are filled and short blips dropped.
2. **Possession.** In each live stretch, a frame where the ball is seen near a player is evidence for that player's team (strong within 2 m, weaker within 4 m, none while the ball is in flight). A two-state minimum-cost labelling picks the team sequence; changing team costs `switch_cost` (40) in units of strong-evidence frames. With no ball there is no evidence, so the previous label is held.
3. **Ungraded frames** (before kick-off, half-time, after the whistle) are `X`.

All constants are in `PARAMS` at the top of `v1.py`, with a comment on each. They were fixed in advance by football reasoning, not tuned on the public ground truth.

## Results so far

Public 5-minute window only (17 possessions: a pipeline check, not a ranking; the intervals are wide).

| | macroF1 | spellF1 | spellF1 interval | possessions (true: 17) |
|---|---|---|---|---|
| best organiser baseline (`smooth50`) | 0.621 | 0.326 | 0.22 to 0.45 | 26 |
| **v1** | **0.694** | 0.224 | 0.06 to 0.39 | 8 |
| v1 without the dead-ball detector | 0.526 | 0.191 | 0.07 to 0.31 | 6 |

Whole match (graded frames): v1 labels 24.8% dead, 170 possessions, median 22.7 s.

- v1 is better frame by frame (the dead-ball detector is the reason) but **too sticky between teams**: far fewer possessions than a real match has. Control evidence exists on only 30.5% of graded frames, so a switch cost of 40 needs about 5 s of uninterrupted evidence.
- **Ball-fill model:** median error 10.4 m against 20.3 m for the player centroid (5-fold, time-blocked), but the team of the player nearest to the estimated ball matches the true one only 43 to 47% of the time (chance level). It is a coarse location estimate, not possession evidence, so v1 does not use it.

## Rules we must keep (summary of `RULES.md`)

- Labels come only from an algorithm. No hand labels on this match, no frame-specific rules.
- The public 5-minute ground truth is for checking that the pipeline lines up, not for training or tuning. Report it with its intervals.
- No human-made data about this match from any source.
- Never commit or publish the match data or anything derived from it (`C:\mtah-work` stays out of git).
- The final repo must regenerate the exact submission file. v1 has no randomness; the ball-fill model uses `random_state=0`.

## Not done yet

- Version 1.1: a lower switch cost (about 15, from the evidence density), or a better evidence source such as touch detection from the ball's direction changes.
- A dead-ball detector validated beyond the public window (there is nothing else to validate on, so the design rests on football reasoning and the aggregate patterns in `explore2/3.py`).
- The experiment log and the technical document.
