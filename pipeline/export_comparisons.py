"""
export_comparisons.py
──────────────────────
Enrichissement bronze : calcule, pour chaque run ayant une `initial_map`,
la comparaison tour par tour avec l'agent optimal (BFS + TSP de npc_solver).

Ce calcul est algorithmique (pathfinding), pas exprimable en SQL/dbt -> il est
fait ici en Python et le résultat est déposé comme une table bronze de plus
(data/bronze/comparisons.parquet), que dbt viendra ensuite lire comme une
source parmi d'autres.

Colonnes : run_id, turn, opt_dir, match, opt_steps_remaining, regret

Lancer (après export_bronze.py) :
    python pipeline/export_comparisons.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from npc_solver import compare_run

BRONZE_DIR = Path("data/bronze")


def main():
    runs_df = pd.read_parquet(BRONZE_DIR / "runs.parquet")
    decisions_df = pd.read_parquet(BRONZE_DIR / "decisions.parquet")

    rows = []
    n_skipped = 0
    for _, run in runs_df[runs_df["has_initial_map"]].iterrows():
        rid = run["run_id"]
        initial_map = np.array(json.loads(run["initial_map"]))
        decision_log = (
            decisions_df[decisions_df["run_id"] == rid]
            .sort_values("turn")
            .to_dict("records")
        )
        if not decision_log:
            n_skipped += 1
            continue

        _, cmp_rows = compare_run(initial_map, decision_log)
        for r in cmp_rows:
            rows.append({
                "run_id":              rid,
                "turn":                r["turn"],
                "opt_dir":             r["opt_dir"],
                "match":               bool(r["match"]),
                "opt_steps_remaining": r["opt_steps_remaining"],
                "regret":              r["regret"],
            })

    comparisons_df = pd.DataFrame(rows)
    comparisons_df.to_parquet(BRONZE_DIR / "comparisons.parquet", index=False)

    print(f"comparisons.parquet : {len(comparisons_df)} lignes ({n_skipped} runs sans decision_log ignorés)")


if __name__ == "__main__":
    main()
