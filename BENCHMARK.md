# Benchmark — NPC Brain

## Objectif

L'objectif n'est **pas** de maximiser la performance du LLM, mais de trouver
un point d'équilibre entre charge cognitive algorithmique (ce que fait le
code Python) et charge cognitive du LLM (ce qu'on lui laisse décider), et de
mesurer cet équilibre de façon reproductible.

Le solveur optimal (`npc_solver.py`, BFS + TSP) sert de référence absolue :
il donne le nombre de coups minimal pour ramasser tout l'or sur une carte
donnée. Toute la mesure de qualité du LLM se fait par écart à cette
référence, pas dans l'absolu.

## Axes de benchmark

### Axe 1 — Charge cognitive algorithmique

`perception()` calcule le delta vers l'or le plus proche et les directions
franchissables, mais ne dit jamais quelle direction prendre : la navigation
reste entièrement à la charge du LLM. C'est le choix retenu (documenté dans
`SPECS.md` : "si vous mettez un algo de Manhattan en place, le LLM n'a plus
grand chose à faire").

- **Hypothèse testée** : donner une distance signée par axe (au lieu d'une
  direction toute faite) suffit à un LLM de quelques milliards de paramètres
  pour naviguer correctement la majorité du temps, sans lui mâcher le
  travail de décision.
- **KPI associés** : `match_rate` (accord avec le coup optimal), `blocked_rate`
  (coups joués contre un mur — un algo Manhattan aurait un taux de 0%,
  donc ce KPI mesure directement le "reste à charge" laissé au LLM).

### Axe 2 — Qualité de la charge du LLM

Pour un même niveau de charge algorithmique (perception fixe), on regarde si
le LLM navigue efficacement : évite les boucles, exploite le feedback
"coup bloqué", termine la partie.

- **Hypothèse testée** : le feedback anti-boucle (rappel des 5 derniers
  coups + signal explicite "ton dernier coup était bloqué") réduit les
  comportements dégénérés (oscillations, blocages répétés) sans changer le
  contrat de perception.
- **KPI associés** : `efficiency` (coups optimaux / coups joués, 1.0 =
  parfait), `regret` (coups qui n'ont pas réduit la distance restante),
  `wasted_turns`, taux de victoire (`win_rate`).

### Axe 3 — Modèles de LLM

Le contrat (perception, prompt, contrainte de sortie structurée) est
strictement identique quel que soit le modèle : seul le modèle change. Ça
isole l'effet "capacité du modèle" du reste du système.

- **Hypothèse testée** : à charge algorithmique et prompt égaux, un modèle
  plus gros / plus orienté raisonnement obtient un meilleur `match_rate` et
  `efficiency`.
- **KPI associés** : toutes les métriques ci-dessus, agrégées par modèle
  dans la table gold `agg_model_comparison` (voir `README.md`).
- **Modèles comparés** : `mistralai/ministral-3-3b` (référence, le plus de
  runs accumulés), `qwen/qwen3-vl-4b`, `microsoft/phi-4-mini-reasoning`,
  `deepseek/deepseek-r1-0528-qwen3-8b` — tous servis localement via le même
  serveur OpenAI-compatible (LM Studio), donc mêmes conditions matérielles.

### Axe 4 — Typologies de layout de départ

Au-delà de la génération aléatoire, trois scénarios délibérés sont définis
dans `map_typologies.py` (justifiés dans `GAME_DESIGN.md`) : `aleatoire`,
`chemin_bloque` (obstacle direct sur le chemin le plus évident) et
`piece_leurre` (pièce la plus proche en ligne droite mais la plus coûteuse
en vrai coût de chemin — un piège pour toute stratégie fondée sur la seule
proximité euclidienne).

- **Hypothèse testée** : le LLM performe bien sur `aleatoire` (cas moyen)
  mais se dégrade sur les scénarios adverses — en particulier
  `piece_leurre`, où suivre naïvement `nearest_gold_delta` (le seul signal
  de perception fourni) mène à un chemin très sous-optimal, alors que
  l'agent optimal (BFS+TSP) l'évite correctement.
- **KPI associés** : les mêmes que l'axe 2, agrégés par typologie plutôt que
  par modèle, dans la table gold `agg_typology_kpi`.

> **Point de méthode** : `--max-turns` doit toujours être strictement
> supérieur au nombre de coups optimal de la typologie (`optimal_collect`),
> sans quoi une victoire est mathématiquement impossible et le `win_rate`
> mesure le budget de tours, pas la qualité du LLM. `piece_leurre` a un
> optimal de 26 coups : un premier run à `--max-turns 25` a dû être écarté
> et rejoué à 40 pour cette raison.

## KPI — définitions

| KPI | Définition | Ce qu'il mesure |
| --- | --- | --- |
| `match_rate` | % de coups où le LLM joue exactement la direction optimale du moment | Qualité de navigation instantanée |
| `efficiency` | coups optimaux / coups réellement joués (parties gagnées uniquement) | Efficacité globale du parcours, 1.0 = parfait |
| `wasted_turns` | coups joués − coups optimaux (si victoire) | Coût en tours de la non-optimalité |
| `regret` | somme des tours où le coup joué n'a pas réduit la distance restante optimale | Décisions contre-productives |
| `blocked_rate` | % de coups joués contre un mur/bord | Charge de navigation mal gérée par le LLM |
| `win_rate` | % de parties où tout l'or est ramassé avant `max_turns` | Réussite globale de la tâche |

## Comment le benchmark est produit

1. Chaque partie (notebook interactif ou `benchmark_runner.py` en headless
   pour les runs multi-modèles) écrit un run dans `runs/*.json` avec le même
   schéma : carte initiale, décisions tour par tour, statut final.
2. Le pipeline (`pipeline/run_pipeline.py`) calcule la comparaison à
   l'optimal pour chaque tour et agrège les KPI par run et par modèle dans
   les tables gold `agg_run_kpi` / `agg_model_comparison` (voir `README.md`).
3. `dataviz/report.py` lit ces tables gold et régénère le rapport après
   chaque `dbt run` — pas de mise à jour temps réel, conformément aux specs.
