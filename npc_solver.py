"""
npc_solver.py
─────────────
Agent OPTIMAL (référence algorithmique) pour NPC Brain, + comparaison LLM/optimal.

Le "meilleur chemin" = ramasser TOUT l'or en un minimum de coups.
C'est un TSP à chemin ouvert : ordre optimal de visite des ors, chaque tronçon
étant un plus court chemin BFS (4-connexe) qui contourne l'ennemi (case bloquante).

- BFS  -> distances/chemins exacts entre points-clés (joueur + chaque or)
- TSP  -> brute-force des permutations d'ordre de visite (ok jusqu'à ~8 ors ;
          au-delà, passer à Held-Karp : DP O(2^n · n^2))

Aucune dépendance au notebook : le module est autonome (client LLM, pygame exclus).

    from npc_solver import optimal_collect, compare_run
"""

from itertools import permutations
from collections import deque

import numpy as np

VOID, PLAYER, ENEMY, GOLD = 0, 1, 2, 3

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
DELTA_TO_DIR = {v: k for k, v in MOVES.items()}

# ── Graphe de grille (l'ennemi bloque, l'or et le vide sont franchissables) ────
def _passable(world_map, r, c):
    nr, nc = world_map.shape
    return 0 <= r < nr and 0 <= c < nc and world_map[r, c] != ENEMY

def _bfs_from(world_map, start):
    """BFS 4-connexe. Retourne (dist, parent) sur toutes les cases atteignables."""
    dist, parent = {start: 0}, {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            nb = (r + dr, c + dc)
            if _passable(world_map, *nb) and nb not in dist:
                dist[nb] = dist[(r, c)] + 1
                parent[nb] = (r, c)
                q.append(nb)
    return dist, parent

def _reconstruct(parent, target):
    """Chemin start..target (inclus) à partir du dict de parents."""
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]

# ── Solveur optimal ────────────────────────────────────────────────────────────
def optimal_collect(world_map):
    """Meilleur chemin ramassant tout l'or atteignable depuis le joueur.

    Retour (types JSON-safe) :
        steps       : int   nombre de coups optimal
        directions  : list[str]        séquence UP/DOWN/LEFT/RIGHT complète
        cells       : list[[r, c]]     cases traversées (départ inclus)
        order       : list[[r, c]]     joueur puis ors dans l'ordre de visite
        unreachable : list[[r, c]]     ors enfermés (ex. cernés par l'ennemi)
    """
    world_map = np.asarray(world_map)
    player = tuple(int(x) for x in np.argwhere(world_map == PLAYER)[0])
    golds  = [tuple(int(x) for x in g) for g in np.argwhere(world_map == GOLD)]

    if not golds:
        return {"steps": 0, "directions": [], "cells": [list(player)],
                "order": [list(player)], "unreachable": []}

    # Distances/chemins entre chaque point-clé
    dist, par = {}, {}
    for k in [player] + golds:
        dist[k], par[k] = _bfs_from(world_map, k)

    reachable   = [g for g in golds if g in dist[player]]
    unreachable = [g for g in golds if g not in dist[player]]

    if not reachable:
        return {"steps": 0, "directions": [], "cells": [list(player)],
                "order": [list(player)], "unreachable": [list(g) for g in unreachable]}

    # TSP chemin ouvert : départ fixe (joueur), on visite tous les ors atteignables.
    # Brute-force des permutations — n_or petit (défaut 4 -> 24 ordres).
    if len(reachable) > 8:
        raise ValueError(
            f"{len(reachable)} ors : brute-force trop coûteux. "
            "Basculer sur Held-Karp (DP bitmask) pour n > 8."
        )

    best = None
    for perm in permutations(reachable):
        seq, total, ok = [player] + list(perm), 0, True
        for a, b in zip(seq, seq[1:]):
            if b not in dist[a]:
                ok = False
                break
            total += dist[a][b]
        if ok and (best is None or total < best[0]):
            best = (total, seq)

    total, seq = best

    # Reconstruction de la trajectoire pas-à-pas
    cells = [player]
    for a, b in zip(seq, seq[1:]):
        seg = _reconstruct(par[a], b)      # a..b inclus
        cells.extend(seg[1:])              # on évite de dupliquer le sommet

    dirs = [DELTA_TO_DIR[(r2 - r1, c2 - c1)]
            for (r1, c1), (r2, c2) in zip(cells, cells[1:])]

    return {
        "steps":       int(total),
        "directions":  dirs,
        "cells":       [list(c) for c in cells],
        "order":       [list(c) for c in seq],
        "unreachable": [list(g) for g in unreachable],
    }

# ── Rejouer un coup (mêmes règles que le notebook) ─────────────────────────────
def _allowed(world_map, pos):
    r, c = pos
    nr, nc = world_map.shape
    return 0 <= r < nr and 0 <= c < nc and world_map[r, c] in (VOID, GOLD)

def _apply(world_map, pos, d):
    dr, dc = MOVES[d]
    nb = (pos[0] + dr, pos[1] + dc)
    if not _allowed(world_map, nb):
        return pos, False                              # bloqué : pas de déplacement
    gold = bool(world_map[nb] == GOLD)
    world_map[pos] = VOID
    world_map[nb]  = PLAYER
    return nb, gold

# ── Comparaison LLM vs optimal ─────────────────────────────────────────────────
def compare_run(initial_map, decision_log):
    """Rejoue le run du LLM sur la carte et le confronte à l'agent optimal.

    À chaque tour on recalcule l'optimal DEPUIS la position courante du LLM :
    - opt_dir            : ce qu'aurait joué l'optimal ici
    - match              : le LLM a-t-il joué ce coup ?
    - opt_steps_remaining: coups optimaux restants avant ce coup
    - regret             : +1 si le coup n'a pas rapproché du but (coup gâché),
                            0 si productif  (delta du restant, corrigé du -1 attendu)

    Retour : (summary: dict, rows: list[dict])
    """
    initial_map = np.asarray(initial_map)
    opt = optimal_collect(initial_map)

    m   = initial_map.copy()
    pos = tuple(int(x) for x in np.argwhere(m == PLAYER)[0])

    rows = []
    prev_remaining = None
    for e in decision_log:
        d = e.get("direction", "?")

        opt_here = optimal_collect(m)
        opt_dir  = opt_here["directions"][0] if opt_here["directions"] else None
        remaining = opt_here["steps"]

        if d == "?":
            moved_pos, gold, blocked = pos, False, False
        else:
            moved_pos, gold = _apply(m, pos, d)
            blocked = (moved_pos == pos)

        # regret : un coup parfait fait passer restant de R à R-1.
        # On mesure après coup au tour suivant ; ici on stocke le restant courant
        # et on calcule le regret du coup PRÉCÉDENT.
        if prev_remaining is not None:
            rows[-1]["regret"] = int(remaining - (prev_remaining - 1))
        prev_remaining = remaining

        rows.append({
            "turn":                e.get("turn"),
            "llm_dir":             d,
            "opt_dir":             opt_dir,
            "match":               (d == opt_dir),
            "blocked":             blocked,
            "gold":                gold,
            "opt_steps_remaining": remaining,
            "regret":              None,      # rempli au tour suivant
        })
        pos = moved_pos

    # dernier coup : regret mesuré sur l'état final
    if rows:
        final_remaining = optimal_collect(m)["steps"]
        rows[-1]["regret"] = int(final_remaining - (prev_remaining - 1))

    n         = len(rows)
    gold_left = int((m == GOLD).sum())
    llm_won   = gold_left == 0
    n_match   = sum(r["match"] for r in rows)
    n_blocked = sum(r["blocked"] for r in rows)

    summary = {
        "optimal_steps":  opt["steps"],
        "llm_turns":      n,
        "llm_won":        llm_won,
        "gold_left":      gold_left,
        "match_rate":     round(n_match / n, 2) if n else 0.0,
        "blocked":        n_blocked,
        # efficacité : coups optimaux / coups réellement joués (1.0 = parfait)
        "efficiency":     round(opt["steps"] / n, 2) if (llm_won and n) else None,
        "wasted_turns":   (n - opt["steps"]) if llm_won else None,
        "total_regret":   int(sum(r["regret"] or 0 for r in rows)),
    }
    return summary, rows