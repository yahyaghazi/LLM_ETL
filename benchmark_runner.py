"""
benchmark_runner.py
────────────────────
Harnais de benchmark HEADLESS pour faire jouer plusieurs modèles LLM
(via LM Studio / endpoint OpenAI-compatible) sur des parties indépendantes.

Corrections clés :
- max_tokens=16000 : le modèle a TOUT le temps de réfléchir PUIS de conclure.
- Extraction du JSON depuis reasoning_content (là où les modèles "thinking"
  écrivent réellement leur réponse).
- Pas de /no_think : la réflexion est libre et encouragée.
- Timeout HTTP allongé (600s).
- Grid croisé : modèles × prompts × températures × parties.
"""

import datetime
import json
import os
import re
import time
from enum import Enum
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from npc_solver import VOID, PLAYER, ENEMY, GOLD, MOVES, optimal_collect
from map_typologies import make_map, TYPOLOGIES as ALL_TYPOLOGIES

RUNS_DIR = Path("runs")

DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_API_TOKEN = "lm-studio"

# Budget de raisonnement LARGE : on ne bride JAMAIS la réflexion.
MAX_TOKENS = 16000
# Timeout HTTP généreux : on laisse le temps réel au modèle.
HTTP_TIMEOUT = 600

DEBUG_LLM = True


# ── Contrat de sortie du LLM ───────────────────────────────────────────────────
class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class PlayerDecision(BaseModel):
    reason: str
    direction: Direction


# ── Utilitaires carte ───────────────────────────────────────────────────────────
def localize(world_map, entity):
    return np.argwhere(world_map == entity)


def player_position(world_map):
    pos = localize(world_map, PLAYER)
    return tuple(pos[0]) if len(pos) > 0 else None


def count_gold(world_map):
    return int(np.sum(world_map == GOLD))


def render_map(world_map):
    symbols = {VOID: ".", PLAYER: "P", ENEMY: "E", GOLD: "G"}
    lines = []
    for row in world_map:
        lines.append(" ".join(symbols.get(int(c), "?") for c in row))
    return "\n".join(lines)


def apply_move(world_map, direction):
    world_map = world_map.copy()
    pos = player_position(world_map)
    if pos is None:
        return world_map, "wall"

    r, c = pos
    dr, dc = MOVES[direction]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < world_map.shape[0] and 0 <= nc < world_map.shape[1]):
        return world_map, "wall"

    target = world_map[nr, nc]
    if target == ENEMY:
        return world_map, "enemy"

    event = "gold" if target == GOLD else "move"
    world_map[r, c] = VOID
    world_map[nr, nc] = PLAYER
    return world_map, event


# ── Prompts ─────────────────────────────────────────────────────────────────────
def build_prompt(world_map):
    """Prompt de BASE : vision locale + delta vers l'or le plus proche."""
    grid = render_map(world_map)
    pos = player_position(world_map)
    gold_left = count_gold(world_map)

    gold_coords = localize(world_map, GOLD)
    hint = ""
    if len(gold_coords) > 0 and pos is not None:
        dists = [abs(int(gr) - pos[0]) + abs(int(gc) - pos[1]) for gr, gc in gold_coords]
        idx = int(np.argmin(dists))
        gr, gc = gold_coords[idx]
        dr, dc = int(gr) - pos[0], int(gc) - pos[1]
        hint = (f"\nL'or le plus proche est à un delta de "
                f"(lignes: {dr:+d}, colonnes: {dc:+d}) par rapport à toi.")

    return f"""Tu joues à un jeu de collecte d'or sur une grille. Voici la carte :

{grid}

Légende :
- P = toi (le joueur)
- G = or à ramasser
- E = ennemi (MORTEL : tu meurs si tu marches dessus)
- . = case vide (sûre)

Ta position actuelle : ligne {pos[0]}, colonne {pos[1]}.
Or restant à collecter : {gold_left}.{hint}

Règles de déplacement (une case à la fois) :
- UP = monter (ligne - 1)
- DOWN = descendre (ligne + 1)
- LEFT = gauche (colonne - 1)
- RIGHT = droite (colonne + 1)

Objectif : ramasse l'or en évitant les ennemis. Ne marche jamais sur une case E.
Indique le PROCHAIN déplacement (une seule direction).
Réponds avec ta raison (reason) et la direction (direction)."""


def build_prompt_adventurer(world_map):
    """Prompt AVENTURIER : vision globale de tout l'or et de tous les ennemis."""
    grid = render_map(world_map)
    pos = player_position(world_map)
    gold_left = count_gold(world_map)

    gold_coords = [tuple(map(int, c)) for c in localize(world_map, GOLD)]
    enemy_coords = [tuple(map(int, c)) for c in localize(world_map, ENEMY)]

    gold_list = ", ".join(f"(ligne {r}, col {c})" for r, c in gold_coords) or "aucun"
    enemy_list = ", ".join(f"(ligne {r}, col {c})" for r, c in enemy_coords) or "aucun"

    return f"""Tu es un AVENTURIER équipé d'une carte magique qui te donne une VISION
GLOBALE et en TEMPS RÉEL de tout le territoire. Voici la carte complète :

{grid}

Légende :
- P = toi (l'aventurier)
- G = or à ramasser
- E = camp ennemi (MORTEL : tu meurs instantanément si tu marches dessus)
- . = case vide (sûre)

Ta position actuelle : ligne {pos[0]}, colonne {pos[1]}.
Or restant à collecter : {gold_left}.

━━━ VISION GLOBALE (grâce à ta carte) ━━━
Positions de TOUT l'or : {gold_list}
Positions de TOUS les camps ennemis : {enemy_list}

Règles de déplacement (une case à la fois) :
- UP = monter (ligne - 1)
- DOWN = descendre (ligne + 1)
- LEFT = gauche (colonne - 1)
- RIGHT = droite (colonne + 1)

━━━ TA MISSION ━━━
Exploite ta vision complète de la carte pour PLANIFIER un itinéraire qui :
1. Récupère la TOTALITÉ de l'or sur la grille.
2. CONTOURNE systématiquement les camps ennemis : ne marche JAMAIS sur une case E.
3. Emprunte le chemin le plus court et le plus sûr possible.

Réfléchis d'abord à l'ordre optimal de collecte de l'or et au trajet global, puis
indique UNIQUEMENT le PROCHAIN déplacement (une seule direction) qui fait avancer
ton plan. Réponds avec ta raison (reason) et la direction (direction)."""


PROMPT_BUILDERS = {
    "base": build_prompt,
    "adventurer": build_prompt_adventurer,
}


# ── Extraction JSON robuste ─────────────────────────────────────────────────────
def extract_json(text):
    """Extrait une PlayerDecision depuis un texte libre contenant un objet JSON."""
    if not text:
        return None

    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
            direction = str(data.get("direction", "")).strip().upper()
            if direction in Direction.__members__:
                return PlayerDecision(
                    reason=str(data.get("reason", "")),
                    direction=Direction[direction],
                )
        except (json.JSONDecodeError, ValueError):
            continue

    m = re.search(r"\b(UP|DOWN|LEFT|RIGHT)\b", text.upper())
    if m:
        return PlayerDecision(reason=text[:200], direction=Direction[m.group(1)])

    return None


def extract_decision(msg):
    """
    Récupère la décision où qu'elle soit :
    1. structured .parsed
    2. .content classique
    3. reasoning_content ← POINT CLÉ pour les modèles "thinking"
    """
    if getattr(msg, "parsed", None):
        return msg.parsed
    if getattr(msg, "content", None):
        d = extract_json(msg.content)
        if d:
            return d
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        d = extract_json(reasoning)
        if d:
            return d
    return None


# ── Debug de la sortie LLM ──────────────────────────────────────────────────────
def debug_dump_completion(tag, completion):
    if not DEBUG_LLM:
        return
    try:
        choice = completion.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        parsed = getattr(msg, "parsed", None)
        reasoning = getattr(msg, "reasoning_content", None)
        usage = getattr(completion, "usage", None)

        content_len = len(content) if content else 0
        reasoning_len = len(reasoning) if reasoning else 0

        print(f"  ┌─ [DEBUG {tag}] ──────────────────────────────")
        print(f"  │ finish_reason   : {choice.finish_reason!r}")
        print(f"  │ parsed          : {parsed!r}")
        print(f"  │ content is None : {content is None}")
        print(f"  │ content len     : {content_len}")
        print(f"  │ reasoning len   : {reasoning_len}")
        if usage is not None:
            print(f"  │ tokens          : prompt={getattr(usage, 'prompt_tokens', '?')} "
                  f"completion={getattr(usage, 'completion_tokens', '?')} "
                  f"total={getattr(usage, 'total_tokens', '?')}")
        if content_len:
            print(f"  │ content preview : {content[:150]!r}")
        elif reasoning_len:
            print(f"  │ reasoning tail  : {reasoning[-150:]!r}")
        print(f"  └──────────────────────────────────────────────")
    except Exception as e:
        print(f"  [DEBUG] Impossible d'inspecter la completion : {e}")


# ── Appels LLM ──────────────────────────────────────────────────────────────────
# Système SANS /no_think : la réflexion est libre et encouragée.
SYSTEM_STRUCTURED = (
    "Tu es un agent de jeu stratégique. Prends tout le temps nécessaire pour "
    "réfléchir, PUIS termine impérativement par ta décision au format demandé "
    "(reason + direction)."
)
SYSTEM_FALLBACK = (
    "Tu es un agent de jeu stratégique. Prends tout le temps nécessaire pour "
    "réfléchir, PUIS termine OBLIGATOIREMENT par un objet JSON valide, seul, au "
    'format : {"reason": "...", "direction": "UP"} où direction ∈ '
    "{UP, DOWN, LEFT, RIGHT}."
)


def ask_llm(client, model, world_map, prompt_mode="base", temperature=0.2, retries=2):
    """Interroge le LLM et retourne une PlayerDecision (ou None)."""
    prompt = PROMPT_BUILDERS[prompt_mode](world_map)

    for attempt in range(retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_STRUCTURED},
                    {"role": "user", "content": prompt},
                ],
                response_format=PlayerDecision,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            debug_dump_completion(f"STRUCTURED attempt={attempt+1}", completion)

            decision = extract_decision(completion.choices[0].message)
            if decision:
                return decision

            print(f"  [WARN] Rien d'exploitable (tentative {attempt+1}/{retries+1}), fallback JSON...")
            decision = ask_llm_fallback(client, model, prompt, temperature=temperature)
            if decision:
                return decision
        except Exception as e:
            print(f"  [WARN] Échec structuré ({e}), fallback JSON (tentative {attempt+1})...")
            decision = ask_llm_fallback(client, model, prompt, temperature=temperature)
            if decision:
                return decision

    return None


def ask_llm_fallback(client, model, prompt, temperature=0.2):
    """Fallback : parsing JSON robuste, en lisant aussi le reasoning_content."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_FALLBACK},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        debug_dump_completion("FALLBACK", completion)

        decision = extract_decision(completion.choices[0].message)
        if decision is None:
            finish = completion.choices[0].finish_reason
            print(f"  [ERREUR] Aucun JSON exploitable (finish={finish}).")
        return decision
    except Exception as e:
        print(f"  [ERREUR] Fallback JSON échoué : {e}")
        return None


# ── Simulation ──────────────────────────────────────────────────────────────────
def run_simulation(client, model, initial_map, prompt_mode="base",
                   temperature=0.2, max_turns=20):
    world_map = initial_map.copy()
    total_gold = count_gold(world_map)
    score = 0
    turn = 0
    status = "in_progress"
    history = []

    while turn < max_turns:
        turn += 1

        if count_gold(world_map) == 0:
            status = "win"
            break

        decision = ask_llm(client, model, world_map,
                           prompt_mode=prompt_mode, temperature=temperature)
        if decision is None:
            status = "error_no_decision"
            break

        world_map, event = apply_move(world_map, decision.direction.value)

        history.append({
            "turn": turn,
            "direction": decision.direction.value,
            "reason": decision.reason,
            "event": event,
        })

        print(f"  Tour {turn}: {decision.direction.value:5s} -> {event}  "
              f"({decision.reason[:60]})")

        if event == "enemy":
            status = "dead"
            break
        if event == "gold":
            score += 1
            if count_gold(world_map) == 0:
                status = "win"
                break

    if status == "in_progress":
        status = "timeout"

    return {
        "status": status,
        "score": score,
        "total_gold": total_gold,
        "turn": turn,
        "history": history,
    }


# ── Sauvegarde ──────────────────────────────────────────────────────────────────
def save_run_log(model, typology, result, initial_map, prompt_mode, temperature):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = re.sub(r"[^\w\-.]", "_", model)
    temp_tag = f"t{temperature:.1f}".replace(".", "")
    filename = f"{safe_model}__{typology}__{prompt_mode}__{temp_tag}__{timestamp}.json"
    path = RUNS_DIR / filename
    payload = {
        "model": model,
        "typology": typology,
        "prompt_mode": prompt_mode,
        "temperature": temperature,
        "timestamp": timestamp,
        "initial_map": initial_map.tolist(),
        "result": result,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ── Découverte des modèles ──────────────────────────────────────────────────────
def get_available_models(client):
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"[ERREUR] Impossible de lister les modèles : {e}")
        return []


# ── Point d'entrée ──────────────────────────────────────────────────────────────
def main():
    # ═══════════════════════════════════════════════════════════════════
    # ⚙️  CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    TYPOLOGIES = ["aleatoire"]
    PROMPT_MODES = ["base", "adventurer"]
    TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
    N_GAMES = 2
    MAX_TURNS = 20
    N_ROWS = 8
    N_COLS = 8
    N_GOLD = 6
    N_ENEMY = 10

    EXCLUDE_EMBEDDINGS = True
    API_URL = "http://localhost:1234/v1"
    API_TOKEN = "lm-studio"
    # ═══════════════════════════════════════════════════════════════════

    load_dotenv()
    api_url = os.environ.get("LLM_API_URL", API_URL)
    api_token = os.environ.get("LLM_API_TOKEN", API_TOKEN)

    print(f"[INFO] Connexion au serveur LLM : {api_url}")
    # timeout allongé : on laisse le temps réel au modèle de réfléchir.
    client = OpenAI(base_url=api_url, api_key=api_token, timeout=HTTP_TIMEOUT)

    models = get_available_models(client)
    print(f"[INFO] {len(models)} modèle(s) trouvé(s) : {models}")

    if EXCLUDE_EMBEDDINGS:
        models = [m for m in models if "embed" not in m.lower()]

    if not models:
        print("[ERREUR] Aucun modèle disponible. Vérifiez que LM Studio est démarré.")
        return

    print(f"[INFO] Modèles retenus pour le benchmark : {models}")

    total_runs = (len(models) * len(TYPOLOGIES) * len(PROMPT_MODES)
                  * len(TEMPERATURES) * N_GAMES)
    print(f"[INFO] Grid de tests : {len(models)} modèles × {len(PROMPT_MODES)} prompts "
          f"× {len(TEMPERATURES)} températures × {len(TYPOLOGIES)} typologies "
          f"× {N_GAMES} parties = {total_runs} runs\n")

    run_idx = 0
    for model in models:
        for typology in TYPOLOGIES:
            for prompt_mode in PROMPT_MODES:
                for temperature in TEMPERATURES:
                    for g in range(N_GAMES):
                        run_idx += 1
                        t0 = time.time()
                        seed = hash((model, typology, prompt_mode,
                                     temperature, g)) & 0xFFFFFFFF
                        initial_map = make_map(typology, n_rows=N_ROWS, n_cols=N_COLS,
                                               n_gold=N_GOLD, n_enemy=N_ENEMY, seed=seed)
                        print(f"\n=== [{run_idx}/{total_runs}] {model}  ·  {typology}  ·  "
                              f"prompt={prompt_mode}  ·  T={temperature}  "
                              f"(partie {g + 1}/{N_GAMES}) ===")
                        result = run_simulation(client, model, initial_map,
                                                prompt_mode=prompt_mode,
                                                temperature=temperature,
                                                max_turns=MAX_TURNS)
                        path = save_run_log(model, typology, result, initial_map,
                                            prompt_mode, temperature)
                        dt = time.time() - t0
                        print(f"-> {result['status']}  {result['score']}/{result['total_gold']} or  "
                              f"en {result['turn']} tours  ({dt:.0f}s)  -> {path.name}")

    print(f"\n[INFO] Benchmark terminé : {total_runs} runs sauvegardés dans {RUNS_DIR}/")


if __name__ == "__main__":
    main()
