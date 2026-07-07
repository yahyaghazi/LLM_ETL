"""
export_bronze.py
────────────────
Couche BRONZE : convertit les runs JSON bruts (runs/*.json) en tables parquet.

Deux tables produites dans data/bronze/ :
- runs.parquet       : une ligne par run (métadonnées + carte initiale + solution optimale)
- decisions.parquet  : une ligne par tour joué (decision_log aplati, run_id en clé étrangère)

Aucun calcul métier ici : on ne fait que typer/aplatir le JSON, comme demandé
pour une couche bronze ("résultats bruts de la simulation").

Lancer :
    python pipeline/export_bronze.py
"""

import glob
import json
from pathlib import Path

import pandas as pd

RUNS_DIR = Path("runs")
BRONZE_DIR = Path("data/bronze")


def _run_id(path: Path) -> str:
    return path.stem  # nom de fichier = identifiant unique du run


def load_runs():
    runs_rows = []
    decisions_rows = []

    for p in sorted(glob.glob(str(RUNS_DIR / "*.json"))):
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[ignoré] {p} — JSON invalide")
            continue
        if "decision_log" not in data:
            print(f"[ignoré] {p} — pas de decision_log")
            continue

        rid = _run_id(path)
        has_map = "initial_map" in data
        optimal = data.get("optimal", {})

        runs_rows.append({
            "run_id":         rid,
            "model":          data.get("model"),
            "typology":       data.get("typology", "aleatoire"),  # runs anciens = généré aléatoirement
            "final_score":    data.get("final_score"),
            "total_gold":     data.get("total_gold"),
            "turns_used":     data.get("turns_used"),
            "status":         data.get("status"),
            "has_initial_map": has_map,
            "initial_map":    json.dumps(data["initial_map"]) if has_map else None,
            "optimal_steps":  optimal.get("steps"),
            "optimal_directions": json.dumps(optimal.get("directions")) if optimal else None,
        })

        for e in data["decision_log"]:
            decisions_rows.append({
                "run_id":    rid,
                "turn":      e.get("turn"),
                "direction": e.get("direction"),
                "reason":    e.get("reason"),
                "blocked":   bool(e.get("blocked", False)),
                "gold":      bool(e.get("gold", False)),
            })

    return pd.DataFrame(runs_rows), pd.DataFrame(decisions_rows)


def main():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    runs_df, decisions_df = load_runs()

    runs_df.to_parquet(BRONZE_DIR / "runs.parquet", index=False)
    decisions_df.to_parquet(BRONZE_DIR / "decisions.parquet", index=False)

    print(f"runs.parquet      : {len(runs_df)} runs")
    print(f"decisions.parquet : {len(decisions_df)} tours")


if __name__ == "__main__":
    main()
