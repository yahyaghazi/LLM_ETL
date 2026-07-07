# Tuto : utiliser le projet NPC Brain

Ce guide décrit le déroulé complet, du lancement d'une partie jusqu'aux
graphiques de reporting. Pour l'architecture détaillée, voir [README.md](README.md).

## 1. Prérequis

- Le venv `.venv` avec les dépendances installées : `numpy`, `pygame`, `openai`,
  `pydantic`, `python-dotenv`, `pandas`, `matplotlib`, `pyarrow`, `duckdb`, `dbt-duckdb`.
- Un fichier `.env` à la racine avec :
  ```
  LLM_API_URL=...
  LLM_API_TOKEN=...
  ```
  pointant vers un serveur LLM compatible OpenAI (LM Studio, Ollama...).

## 2. Lancer une partie simulée (notebook)

1. Ouvrir `npc_brain.ipynb` et exécuter toutes les cellules dans l'ordre.
2. La dernière cellule lance `npc_viewer.py` en subprocess : une fenêtre
   Pygame s'ouvre et s'actualise en temps réel pendant que le LLM joue.
3. Chaque partie est sauvegardée automatiquement dans
   `runs/run_<modele>_<date>.json`.

## 3. Rejouer une partie a posteriori

```
python npc_replay.py                      # rejoue le dernier run
python npc_replay.py runs/run_xxx.json    # rejoue un run précis
```

Affiche le déplacement tour par tour avec la flèche du coup optimal
(calculé par `npc_solver.py`, BFS + TSP).

## 4. Benchmark headless (sans Pygame ni notebook)

Comparer plusieurs modèles LLM d'un coup :

```
python benchmark_runner.py --models qwen/qwen3-vl-4b microsoft/phi-4-mini-reasoning \
    --n-games 2 --max-turns 15
```

Comparer sur des typologies de carte différentes (obstacle direct, pièce
leurre...) — voir `GAME_DESIGN.md` :

```
python benchmark_runner.py --models mistralai/ministral-3-3b \
    --typologies aleatoire chemin_bloque piece_leurre --n-games 3 --max-turns 25
```

## 5. Pipeline data (bronze → silver → gold)

Une fois des runs accumulés dans `runs/*.json`, reconstruire tout le
pipeline en une commande :

```
python pipeline/run_pipeline.py
```

Étapes internes :
1. `pipeline/export_bronze.py` — aplatit les JSON en parquet (`data/bronze/{runs,decisions}.parquet`)
2. `pipeline/export_comparisons.py` — calcule BFS+TSP par tour (`data/bronze/comparisons.parquet`)
3. `dbt run` (dbt-duckdb) — construit silver (`stg_runs`, `stg_decisions`) puis
   gold (`agg_run_kpi`, `agg_model_comparison`, `agg_typology_kpi`) dans `data/npc.duckdb`

Consulter les résultats :

```python
import duckdb
con = duckdb.connect("data/npc.duckdb", read_only=True)
con.execute("select * from main_gold.agg_run_kpi").df()
con.execute("select * from main_gold.agg_model_comparison").df()
con.execute("select * from main_gold.agg_typology_kpi").df()
```

`data/bronze/*.parquet` et `data/npc.duckdb` sont régénérés à chaque run
du pipeline (non versionnés) — seule la source de vérité `runs/*.json`
est conservée.

## 6. Graphiques de reporting

```
python dataviz/report.py
```

Lit les tables gold et régénère dans `reports/` :
- `model_comparison.png` — taux de victoire, accord avec l'optimal, efficacité, taux de blocage, par modèle LLM
- `per_run_efficiency.png` — nuage de points coups joués vs coups optimaux, par run
- `typology_comparison.png` — mêmes KPI, par typologie de carte

Pas de mise à jour temps réel : à relancer après chaque
`python pipeline/run_pipeline.py`.

## Ordre typique d'utilisation

1. Générer des runs (notebook ou `benchmark_runner.py`)
2. `python pipeline/run_pipeline.py`
3. `python dataviz/report.py`
4. Regarder les PNG dans `reports/`, ou requêter `data/npc.duckdb` directement

## KPI disponibles

- `match_rate` — le LLM a-t-il joué le coup optimal ?
- `efficiency` — coups optimaux / coups réellement joués (1.0 = parfait)
- `regret` — coups qui n'ont pas rapproché du but
- `blocked_rate` — proportion de coups joués contre un mur
- `wasted_turns` — coups en trop par rapport à l'optimal (si victoire)

Détails et hypothèses du benchmark : voir `BENCHMARK.md`.
