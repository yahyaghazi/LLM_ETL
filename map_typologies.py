"""
map_typologies.py
──────────────────
Les "typologies" de carte demandées par SPECS.md §1.2 : plusieurs layouts de
départ distincts, en plus de la génération aléatoire, pour vérifier que le
LLM ne réussit pas seulement sur des cartes faciles/moyennes en moyenne,
mais aussi sur des scénarios délibérément adverses.

- "aleatoire"     : génération procédurale (baseline, make_random_map existant)
- "chemin_bloque" : un mur d'ennemis bloque le passage direct le plus évident
                    -> teste si le LLM sait contourner plutôt que de
                    s'acharner contre le mur (cf. KPI blocked_rate)
- "piece_leurre"  : une pièce très proche EN LIGNE DROITE (donc désignée par
                    nearest_gold_delta) mais en réalité très coûteuse à
                    atteindre (encerclée, un seul accès à l'autre bout de la
                    carte) -> teste si le LLM se laisse piéger par le signal
                    de proximité au lieu de prioriser une pièce moins
                    "proche" mais réellement plus rapide à atteindre. C'est
                    le seul scénario où même l'agent optimal (BFS+TSP, qui
                    raisonne en vrai coût de chemin) diverge nettement de ce
                    qu'un LLM naïf suivant le delta ferait.

Toutes les cartes utilisent les mêmes codes que npc_solver.py (VOID=0,
PLAYER=1, ENEMY=2, GOLD=3), pour rester compatibles avec le solveur, le
replay et le pipeline sans aucune adaptation.
"""

import numpy as np

from npc_solver import VOID, PLAYER, ENEMY, GOLD


def random_map(n_rows=8, n_cols=8, n_gold=6, n_enemy=10, seed=None):
    rng = np.random.default_rng(seed)
    grid = np.zeros((n_rows, n_cols), dtype=int)
    total = n_rows * n_cols
    n_entities = 1 + n_gold + n_enemy
    flat_positions = rng.choice(total, size=n_entities, replace=False)
    coords = [(p // n_cols, p % n_cols) for p in flat_positions]
    grid[coords[0]] = PLAYER
    for c in coords[1:1 + n_gold]:
        grid[c] = GOLD
    for c in coords[1 + n_gold:]:
        grid[c] = ENEMY
    return grid


def blocked_path_map(**_ignored):
    """Mur d'ennemis (colonne 3, lignes 2-6) qui bloque le passage horizontal
    direct entre le joueur et deux des pièces. Contournable par le haut
    (lignes 0-1) ou par le bas (ligne 7)."""
    O, P, E, G = VOID, PLAYER, ENEMY, GOLD
    grid = [
        [O, O, O, O, O, O, O, G],
        [O, O, O, O, O, O, O, O],
        [O, O, O, E, O, O, O, O],
        [O, O, O, E, O, O, G, O],
        [P, O, O, E, O, O, O, O],
        [O, O, O, E, O, O, O, O],
        [O, O, O, E, O, O, O, O],
        [G, O, O, O, O, O, O, O],
    ]
    return np.array(grid, dtype=int)


def decoy_gold_map(**_ignored):
    """Pièce leurre en (5,0) : très proche du joueur en ligne droite (delta
    euclidien minimal), mais enfermée derrière un mur d'ennemis (ligne 6,
    colonnes 0-6) dont l'unique ouverture est à l'autre bout de la carte
    (colonne 7). Le vrai coût pour l'atteindre est ~16 pas, alors que
    d'autres pièces sont bien plus rapides d'accès."""
    O, P, E, G = VOID, PLAYER, ENEMY, GOLD
    grid = [
        [G, O, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, O],
        [O, O, O, O, G, O, O, O],
        [O, O, O, O, O, O, O, O],
        [O, O, O, O, O, O, O, O],
        [G, O, O, O, O, O, O, G],   # (5,0) = pièce leurre
        [E, E, E, E, E, E, E, O],   # mur, ouverture uniquement en (6,7)
        [P, O, O, O, O, O, O, G],
    ]
    return np.array(grid, dtype=int)


TYPOLOGIES = {
    "aleatoire":      random_map,
    "chemin_bloque":  blocked_path_map,
    "piece_leurre":   decoy_gold_map,
}


def make_map(typology, **kwargs):
    if typology not in TYPOLOGIES:
        raise ValueError(f"Typologie inconnue : {typology!r}. Choix : {list(TYPOLOGIES)}")
    return TYPOLOGIES[typology](**kwargs)
