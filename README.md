# NPC Brain

Un LLM contrôle un joueur sur une grille pour ramasser de l'or, en évitant des
ennemis-obstacles. Le projet couvre la simulation (notebook + Pygame), un
agent optimal de référence pour benchmarker le LLM, et un pipeline de data
ingénierie (bronze/silver/gold) pour analyser la qualité des runs.

## Structure du projet

```
├── npc_brain.ipynb      # cerveau : logique de jeu + appels LLM (le notebook principal)
├── npc_viewer.py         # affichage Pygame en temps réel (subprocess, lit npc_state.json)
├── npc_replay.py         # replay pas-à-pas d'un run sauvegardé, avec flèche LLM vs optimal
├── npc_solver.py         # agent optimal (BFS + TSP) + comparaison LLM/optimal
├── benchmark_runner.py   # harnais headless : fait jouer plusieurs modèles LLM sans Pygame
├── map_typologies.py     # scénarios de carte délibérés (aléatoire / chemin bloqué / pièce leurre)
├── doc.md                # documentation cellule par cellule du notebook
├── GAME_DESIGN.md        # règles de jeu tranchées et justifiées (§1.2 SPECS.md)
├── BENCHMARK.md          # objectifs, axes et KPI du benchmark
│
├── runs/                 # logs bruts d'un run (JSON), écrits par le notebook ou benchmark_runner.py
├── pipeline/             # scripts d'export JSON -> parquet (couche bronze)
├── dbt/npc_benchmark/    # projet dbt-duckdb : silver (staging) + gold (marts)
├── data/                 # bronze (parquet) + base duckdb, générés par le pipeline
├── dataviz/report.py     # génère les graphiques de reporting depuis les tables gold
└── reports/              # graphiques PNG produits par dataviz/report.py
```

## Faire tourner la simulation

1. Avoir un `.env` avec `LLM_API_URL` et `LLM_API_TOKEN` (serveur OpenAI-compatible : LM Studio, Ollama...).
2. Ouvrir `npc_brain.ipynb` et exécuter toutes les cellules dans l'ordre.
3. La dernière cellule lance `npc_viewer.py` en subprocess et démarre la partie :
   la fenêtre Pygame s'actualise en temps réel via `npc_state.json`.
4. Chaque run est aussi sauvegardé dans `runs/run_<modele>_<date>.json`
   (carte initiale, décisions du LLM, solution optimale).

Pour rejouer un run a posteriori, tour par tour, avec la flèche optimale en plus :

```
python npc_replay.py                      # dernier run avec carte
python npc_replay.py runs/run_xxx.json    # run précis
```

## Benchmark : LLM vs agent optimal

`npc_solver.py` calcule le meilleur chemin possible (BFS pour les distances,
TSP par permutation pour l'ordre de visite des ors) et le compare au
comportement réel du LLM, coup par coup :

- `match_rate` — le LLM a-t-il joué le coup optimal ?
- `efficiency` — coups optimaux / coups réellement joués (1.0 = parfait)
- `regret` — coups qui n'ont pas rapproché du but
- `blocked_rate` — proportion de coups joués contre un mur
- `wasted_turns` — coups en trop par rapport à l'optimal (si victoire)

Ces KPI sont utilisables directement dans le notebook (`compare_run`) ou via
les tables gold du pipeline (voir plus bas). Les objectifs et hypothèses du
benchmark (axes charge algorithmique / qualité LLM / modèles) sont
formalisés dans `BENCHMARK.md`.

### Comparer plusieurs modèles LLM (headless)

`benchmark_runner.py` fait jouer un ou plusieurs modèles sans lancer Pygame
ni le notebook — pratique pour accumuler des runs pour l'axe "Modèles de
LLM" du benchmark. Il réutilise les mêmes constantes et le même solveur que
le notebook (`npc_solver.py`) et écrit dans `runs/*.json` avec le schéma
exact attendu par le pipeline :

```
python benchmark_runner.py --models qwen/qwen3-vl-4b microsoft/phi-4-mini-reasoning \
    --n-games 2 --max-turns 15
```

### Comparer plusieurs typologies de carte (layout de départ)

En plus de la génération aléatoire, `map_typologies.py` définit des scénarios
délibérés (obstacle direct, pièce leurre) — voir `GAME_DESIGN.md` pour le
détail et la justification de chacun :

```
python benchmark_runner.py --models mistralai/ministral-3-3b \
    --typologies aleatoire chemin_bloque piece_leurre --n-games 3 --max-turns 25
```

## Pipeline de data ingénierie (bronze → silver → gold)

Architecture en médaillon, avec parquet / duckdb / dbt-duckdb :

```
runs/*.json
    │  pipeline/export_bronze.py        (aplatit le JSON, aucun calcul)
    ▼
data/bronze/{runs,decisions}.parquet
    │  pipeline/export_comparisons.py   (BFS+TSP de npc_solver, par tour)
    ▼
data/bronze/comparisons.parquet
    │  dbt run  (dbt-duckdb)
    ▼
silver : stg_runs, stg_decisions          (types nettoyés, jointures)
gold   : agg_run_kpi                      (KPI par run)
         agg_model_comparison             (KPI agrégés par modèle LLM)
         agg_typology_kpi                 (KPI agrégés par typologie de carte)
```

Reconstruire tout le pipeline en une commande, après une ou plusieurs
nouvelles parties :

```
python pipeline/run_pipeline.py
```

Les tables gold sont alors disponibles dans `data/npc.duckdb`
(schémas `main_silver` / `main_gold`), consultables par exemple avec :

```python
import duckdb
con = duckdb.connect("data/npc.duckdb", read_only=True)
con.execute("select * from main_gold.agg_run_kpi").df()
con.execute("select * from main_gold.agg_model_comparison").df()
con.execute("select * from main_gold.agg_typology_kpi").df()
```

`data/bronze/*.parquet` et `data/npc.duckdb` sont régénérés à chaque run du
pipeline (non versionnés, voir `.gitignore`) — seule la source de vérité
`runs/*.json` est conservée.

## Dataviz (reporting)

`dataviz/report.py` lit les tables gold (`main_gold.agg_run_kpi`,
`main_gold.agg_model_comparison`, `main_gold.agg_typology_kpi`) et régénère
les graphiques dans `reports/` :

- `model_comparison.png` — taux de victoire, accord avec l'optimal, efficacité,
  taux de blocage, par modèle LLM.
- `per_run_efficiency.png` — nuage de points coups joués vs coups optimaux, par run.
- `typology_comparison.png` — mêmes KPI, par typologie de carte (layout de départ).

Pas de mise à jour temps réel : à relancer après chaque `pipeline/run_pipeline.py`.

```
python pipeline/run_pipeline.py
python dataviz/report.py
```

## Dépendances

Installées dans `.venv` : `numpy`, `pygame`, `openai`, `pydantic`, `python-dotenv`,
`pandas`, `matplotlib`, `pyarrow`, `duckdb`, `dbt-duckdb`.
# LLM_ETL
