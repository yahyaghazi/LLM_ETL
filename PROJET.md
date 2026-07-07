# Le projet, fichier par fichier

## 1. Le cœur de la simulation

**[npc_brain.ipynb](npc_brain.ipynb)** — le fichier principal, le "cerveau". Un notebook Jupyter avec 17 cellules qui, dans l'ordre :

- configure le client LLM (lit `.env`, pointe vers ton serveur OpenAI-compatible local) ;
- modélise le monde (grille numpy, codes `VOID=0/PLAYER=1/ENEMY=2/GOLD=3`, génération de carte aléatoire) ;
- définit le contrat de sortie du LLM (`PlayerDecision` en Pydantic : `reason` + `direction`) ;
- implémente la perception (delta vers l'or le plus proche + directions franchissables, sans jamais dire quoi faire) et le déplacement ;
- construit le prompt et appelle le LLM (`decide()`) ;
- fait tourner la partie tour par tour (`run_simulation()`), écrit l'état à chaque coup, et sauvegarde le run complet dans `runs/` ;
- les dernières cellules (18+) rechargent un run et affichent des stats/graphiques ad hoc (distribution des directions, timeline, comparaison à l'optimal).

**[npc_viewer.py](npc_viewer.py)** — l'affichage. Un vrai process Python séparé (lancé en subprocess par le notebook) qui ouvre une fenêtre Pygame et relit `npc_state.json` en boucle pour afficher la partie en temps réel. Séparé du notebook pour éviter un crash SDL dans le kernel Jupyter sous Windows.

**[npc_replay.py](npc_replay.py)** — rejoue a posteriori un run sauvegardé (`runs/*.json`), tour par tour, avec navigation clavier (←/→, espace pour l'autoplay). Affiche en plus la flèche du coup optimal à côté de celle jouée par le LLM, pour visualiser où il s'est trompé.

## 2. Le benchmark (agent optimal + comparaison)

**[npc_solver.py](npc_solver.py)** — la référence absolue du projet. Calcule le meilleur chemin possible pour ramasser tout l'or (BFS pour les distances/chemins exacts, TSP par force brute pour l'ordre optimal de visite des pièces). Fournit aussi `compare_run()`, qui rejoue un run du LLM coup par coup contre cet optimal et calcule les KPI (`match_rate`, `efficiency`, `regret`, etc.). C'est le module que tout le reste du projet réutilise pour ne jamais recalculer deux fois la même logique.

**[map_typologies.py](map_typologies.py)** — les 3 scénarios de carte : `aleatoire` (génération procédurale), `chemin_bloque` (un mur d'ennemis bloque le passage évident), `piece_leurre` (une pièce très proche en ligne droite mais très coûteuse en vrai chemin — un piège pour un agent qui suivrait bêtement le signal de proximité). Chaque fonction retourne une grille numpy compatible avec `npc_solver.py`.

**[benchmark_runner.py](benchmark_runner.py)** — le harnais headless : fait jouer un ou plusieurs modèles LLM, sur une ou plusieurs typologies, sans Pygame ni notebook. C'est ce qui a permis de générer les runs pour `qwen3-vl-4b`, `phi-4-mini-reasoning`, `deepseek-r1` et les typologies `chemin_bloque`/`piece_leurre`, tout en écrivant des fichiers `runs/*.json` avec exactement le même schéma que le notebook.

## 3. Le pipeline de données (bronze → silver → gold)

**[pipeline/export_bronze.py](pipeline/export_bronze.py)** — lit tous les `runs/*.json` et les aplatit en deux tables parquet : `data/bronze/runs.parquet` (une ligne par partie) et `data/bronze/decisions.parquet` (une ligne par tour). Aucun calcul, juste du typage — c'est la couche bronze.

**[pipeline/export_comparisons.py](pipeline/export_comparisons.py)** — calcule, pour chaque run avec une carte initiale, la comparaison tour par tour avec l'agent optimal (réutilise `compare_run()` de `npc_solver.py`). Écrit `data/bronze/comparisons.parquet`. C'est fait en Python (pas en SQL) parce que le BFS/TSP n'est pas exprimable proprement en dbt.

**[pipeline/run_pipeline.py](pipeline/run_pipeline.py)** — le point d'entrée unique : enchaîne les deux scripts ci-dessus puis `dbt run`. C'est la commande à relancer après chaque nouvelle partie.

**[dbt/npc_benchmark/](dbt/npc_benchmark/)** — le projet dbt-duckdb :

- `dbt_project.yml` / `profiles.yml` : config du projet et connexion à `data/npc.duckdb`.
- `models/staging/sources.yml` : déclare les 3 fichiers parquet bronze comme sources.
- `models/staging/stg_runs.sql`, `stg_decisions.sql` : couche **silver** — types nettoyés, jointure decisions↔comparisons.
- `models/marts/agg_run_kpi.sql` : couche **gold**, un run = une ligne de KPI.
- `models/marts/agg_model_comparison.sql` : gold, KPI agrégés par modèle LLM.
- `models/marts/agg_typology_kpi.sql` : gold, KPI agrégés par typologie de carte.

**[data/](data/)** — sorties générées par le pipeline (parquet bronze + `npc.duckdb` contenant silver/gold). Non versionné dans git (voir `.gitignore`), régénéré à volonté depuis `runs/*.json`.

## 4. Dataviz

**[dataviz/report.py](dataviz/report.py)** — lit les tables gold via duckdb et produit 3 graphiques dans `reports/` : comparaison inter-modèles, nuage de points coups joués vs optimaux par run, comparaison inter-typologies. Palette catégorielle validée (skill dataviz), gère explicitement les valeurs "non définies" (ex. efficacité d'un modèle qui n'a jamais gagné) pour ne jamais afficher un 0 trompeur.

**[reports/](reports/)** — les 3 PNG produits par le script ci-dessus.

## 5. Documentation

- **[README.md](README.md)** — vue d'ensemble technique : structure, comment lancer chaque script, architecture du pipeline.
- **[GAME_DESIGN.md](GAME_DESIGN.md)** — tranche et justifie les règles de jeu (plusieurs pièces, pas de combat, layout de départ) en réponse directe à `SPECS.md` §1.2.
- **[BENCHMARK.md](BENCHMARK.md)** — objectifs et 4 axes du benchmark (charge algo, qualité LLM, modèles, typologies), hypothèses testées, définitions des KPI.
- **[doc.md](doc.md)** — documentation cellule par cellule du notebook.
- **[TUTO.md](TUTO.md)** — guide pas-à-pas d'utilisation, renvoie vers le README pour l'architecture détaillée.
- **[SPECS.md](SPECS.md)** — l'énoncé du projet fourni par le formateur.
- **[Tree.md](Tree.md)** — arbre de fichiers généré par une extension VS Code ("FileTree Pro"), simple instantané non maintenu.

## 6. Données et config

- **[runs/](runs/)** — la source de vérité : un JSON par partie jouée (carte initiale, décisions du LLM, solution optimale), qu'elle vienne du notebook ou de `benchmark_runner.py`.
- **`.env`** — `LLM_API_URL` / `LLM_API_TOKEN`, non versionné.
- **`.python-version`** — fixe la version Python du projet.
- **`.gitignore`** — exclut le venv, les artefacts générés (parquet, duckdb, logs dbt, `__pycache__`) et les logs temporaires de benchmark.
