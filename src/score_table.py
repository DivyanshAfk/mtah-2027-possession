"""Run the organisers' scorer over every CSV in evaluation/results and print a compact table.

The scorer only knows the public 5-minute window (17 possessions), so this is a pipeline check, never a way to choose
between two versions (see RULES.md). The bootstrap columns show how wide the uncertainty is.

Run:  python src/score_table.py
"""
import os, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The scorer prints long per-file reports; we only want the scoreboard it writes to evaluation/scoreboard.csv.
subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"), "--bootstrap", "300"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
d = pd.read_csv(os.path.join(ROOT, "evaluation", "scoreboard.csv"))
cols = [c for c in ("submission", "status", "macro_f1", "spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "poss") if c in d.columns]
print(d[cols].round(3).to_string(index=False))
