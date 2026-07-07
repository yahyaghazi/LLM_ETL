"""
run_pipeline.py
────────────────
Point d'entrée unique : reconstruit tout le pipeline data à partir de runs/*.json.

    bronze (parquet)  ->  dbt run (silver + gold, duckdb)

Lancer depuis la racine du projet :
    python pipeline/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
DBT = ROOT / ".venv" / "Scripts" / "dbt.exe"
DBT_PROJECT_DIR = ROOT / "dbt" / "npc_benchmark"


def run(cmd, cwd=None, env=None):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    import os

    run([PYTHON, "pipeline/export_bronze.py"], cwd=ROOT)
    run([PYTHON, "pipeline/export_comparisons.py"], cwd=ROOT)

    env = os.environ.copy()
    env["DBT_PROFILES_DIR"] = str(DBT_PROJECT_DIR)
    run([str(DBT), "run"], cwd=DBT_PROJECT_DIR, env=env)

    print("\nPipeline terminé. Tables gold dans data/npc.duckdb "
          "(schémas main_silver / main_gold).")


if __name__ == "__main__":
    main()
