"""
benchmark_runner.py
────────────────────
Harnais de benchmark HEADLESS (sans Pygame/notebook) pour faire jouer
plusieurs modèles LLM sur des parties indépendantes, et produire des runs
dans runs/*.json avec exactement le même schéma que npc_brain.ipynb
(model, final_score, total_gold, turns_used, status, initial_map, optimal,
decision_log) — pour rester compatible avec le pipeline data (pipeline/).

Contrairement au notebook (démo interactive avec affichage Pygame), ce
script sert uniquement à peupler le benchmark multi-modèles : pas de
viewer, pas de npc_state.json, juste les logs de run.

Les constantes de la grille (VOID/PLAYER/ENEMY/GOLD, MOVES) et le calcul de
la solution optimale sont importés de npc_solver.py — seule source de
vérité déjà utilisée par le notebook et le replay, pour garantir la
cohérence des données entre les deux façons de jouer.

Lancer :
    python benchmark_runner.py --models mistralai/ministral-3-3b \
        --typologies aleatoire chemin_bloque piece_leurre --n-games 2 --max-turns 20
"""

import argparse
import datetime
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from npc_solver import VOID, PLAYER, ENEMY, GOLD, MOVES, optimal_collect
from map_typologies import make_map, TYPOLOGIES

RUNS_DIR = Path("runs")


# ── Contrat de sortie du LLM (identique au notebook, cellule 7) ────────────────
class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class PlayerDecision(BaseModel):
    reason: str
    direction: Direction


# ── Moteurs (identiques au notebook, cellule 9) ─────────────────────────────────
def localize(world_map, entity):
    return np.argwhere(world_map == entity)


def compute_distances(entities_positions, reference_pos):
    if len(entities_positions) == 0:
        return np.array([])
    v = entities_positions - reference_pos
    return np.round(np.linalg.norm(v, axis=1), 2)


def allowed_move(world_map, pos):
    r, c = pos
    n_rows, n_cols = world_map.shape
    if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
        return False
    return world_map[r, c] in (VOID, GOLD)


def move(world_map, old_pos, new_pos):
    if not allowed_move(world_map, new_pos):
        return tuple(old_pos), False
    gold_collected = bool(world_map[new_pos[0], new_pos[1]] == GOLD)
    entity = world_map[old_pos[0], old_pos[1]]
    world_map[old_pos[0], old_pos[1]] = VOID
    world_map[new_pos[0], new_pos[1]] = entity
    return tuple(new_pos), gold_collected


def perception(world_map):
    player_pos = localize(world_map, PLAYER)[0]
    golds_pos = localize(world_map, GOLD)
    golds_dist = compute_distances(golds_pos, player_pos)

    if len(golds_dist) > 0:
        nearest_idx = np.argmin(golds_dist)
        delta = golds_pos[nearest_idx] - player_pos
        nearest_gold_delta = {"row": int(delta[0]), "col": int(delta[1])}
    else:
        nearest_gold_delta = {"row": 0, "col": 0}

    valid_directions = [
        name for name, (dr, dc) in MOVES.items()
        if allowed_move(world_map, (player_pos[0] + dr, player_pos[1] + dc))
    ]

    return {
        "player_pos": [int(player_pos[0]), int(player_pos[1])],
        "nearest_gold_delta": nearest_gold_delta,
        "valid_directions": valid_directions,
        "golds_count": int(len(golds_dist)),
    }


# ── Décision LLM (identique au notebook, cellule 11 ; modèle paramétrable) ────
def decide(client, model, player_perception, move_history, feedback="") -> Optional[PlayerDecision]:
    delta = player_perception["nearest_gold_delta"]
    valid = player_perception["valid_directions"]

    prompt = f"""/no_think
Tu controles un joueur sur une grille. Objectif : atteindre l'or.

# Ou est l'or le plus proche (par rapport a toi)
- {abs(delta['row'])} case(s) vers {'le DOWN' if delta['row'] > 0 else 'le UP' if delta['row'] < 0 else '(aligne verticalement)'}
- {abs(delta['col'])} case(s) vers {'la DROITE' if delta['col'] > 0 else 'la LEFT' if delta['col'] < 0 else '(aligne horizontalement)'}

# Directions ou tu peux avancer (les seules autorisees)
{valid}

# Tes 5 derniers coups
{move_history[-5:] if move_history else "aucun"}
{feedback}

# Consignes
- Choisis UNE direction parmi {valid}.
- Reduis d'abord le plus grand ecart (vertical ou horizontal).
- Si tu tournes en rond, essaie une autre direction que d'habitude.

Ta reponse : une seule direction."""

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=PlayerDecision,
            temperature=1,
        )
        return response.choices[0].message.parsed or None
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None


# ── Game loop headless (identique au notebook, cellule 15 ; sans écriture d'état live) ─
def run_simulation(client, model, initial_map, max_turns=30):
    world_map = initial_map.copy()
    move_history, decision_log = [], []
    score = 0
    total_gold = int(np.sum(initial_map == GOLD))
    status = "running"
    feedback = ""
    turn = 0

    for turn in range(1, max_turns + 1):
        p = perception(world_map)
        if p["golds_count"] == 0:
            status = "won"
            break

        decision = decide(client, model, p, move_history, feedback)
        if decision is None:
            decision_log.append({"turn": turn, "direction": "?", "reason": "(pas de decision LLM)",
                                  "blocked": False, "gold": False})
            continue

        dir_str = decision.direction.value
        reason = decision.reason.strip()
        player_pos = localize(world_map, PLAYER)[0]
        old_pos = tuple(player_pos)
        d_row, d_col = MOVES[dir_str]
        new_pos = (player_pos[0] + d_row, player_pos[1] + d_col)
        new_pos, gold_collected = move(world_map, player_pos, new_pos)
        move_history.append(dir_str)
        blocked = (new_pos == old_pos)

        decision_log.append({
            "turn": turn, "direction": dir_str, "reason": reason,
            "blocked": blocked, "gold": bool(gold_collected),
        })

        feedback = f"# ATTENTION\nTon dernier coup {dir_str} etait BLOQUE (mur). Change de direction." if blocked else ""
        if gold_collected:
            score += 1
        status = "won" if score >= total_gold else "running"
        tag = "+OR" if gold_collected else "bloque" if blocked else ""
        print(f"  [{turn:02d}] {dir_str:7} {tag:8} score={score}/{total_gold}")
    else:
        status = "timeout"

    return {
        "world_map": world_map, "score": score, "total_gold": total_gold,
        "turn": turn, "status": status, "decision_log": decision_log,
    }


def save_run_log(model, typology, result, initial_map):
    safe_model = re.sub(r"[^0-9A-Za-z._-]", "_", model)
    path = RUNS_DIR / f"run_{safe_model}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    optimal = optimal_collect(initial_map)
    full = {
        "model": model,
        "typology": typology,
        "final_score": int(result["score"]),
        "total_gold": int(result["total_gold"]),
        "turns_used": int(result["turn"]),
        "status": result["status"],
        "initial_map": initial_map.tolist(),
        "optimal": {"steps": optimal["steps"], "directions": optimal["directions"]},
        "decision_log": result["decision_log"],
    }
    RUNS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="Benchmark headless multi-modèles pour NPC Brain.")
    ap.add_argument("--models", nargs="+", required=True, help="IDs de modèles (server OpenAI-compatible)")
    ap.add_argument("--typologies", nargs="+", default=["aleatoire"], choices=list(TYPOLOGIES),
                     help="Typologies de carte à jouer (cf. map_typologies.py / GAME_DESIGN.md)")
    ap.add_argument("--n-games", type=int, default=2, help="Parties par (modèle, typologie)")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--n-rows", type=int, default=8)
    ap.add_argument("--n-cols", type=int, default=8)
    ap.add_argument("--n-gold", type=int, default=6)
    ap.add_argument("--n-enemy", type=int, default=10)
    args = ap.parse_args()

    load_dotenv()
    client = OpenAI(base_url=os.environ["LLM_API_URL"], api_key=os.environ["LLM_API_TOKEN"])

    for model in args.models:
        for typology in args.typologies:
            for g in range(args.n_games):
                t0 = time.time()
                seed = hash((model, typology, g)) & 0xFFFFFFFF
                initial_map = make_map(typology, n_rows=args.n_rows, n_cols=args.n_cols,
                                        n_gold=args.n_gold, n_enemy=args.n_enemy, seed=seed)
                print(f"\n=== {model}  ·  {typology}  (partie {g + 1}/{args.n_games}) ===")
                result = run_simulation(client, model, initial_map, max_turns=args.max_turns)
                path = save_run_log(model, typology, result, initial_map)
                dt = time.time() - t0
                print(f"-> {result['status']}  {result['score']}/{result['total_gold']} or  "
                      f"en {result['turn']} tours  ({dt:.0f}s)  -> {path.name}")


if __name__ == "__main__":
    main()
