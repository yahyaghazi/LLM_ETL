"""
npc_replay.py
─────────────
Replay pas-à-pas d'un run sauvegardé (runs/*.json) pour NPC Brain.
Rejoue la partie tour par tour avec la flèche du coup joué par le LLM, et —
si le run contient `initial_map` — la flèche du coup OPTIMAL depuis la même case.

Lancer :
    python npc_replay.py                      # dernier run avec carte
    python npc_replay.py runs/run_xxx.json    # run précis

Navigation :
    ← / →   ou  A / D    reculer / avancer d'un tour
    ESPACE               lecture auto (on/off)
    ↑ / ↓                vitesse de lecture auto
    HOME / END           premier / dernier tour
    ESC / fermer         quitter

Nécessite `initial_map` dans le run (les vieux logs sans carte ne sont pas
rejouables : impossible de reconstituer les positions).
"""

import sys, glob, json
from pathlib import Path

import numpy as np
import pygame

# npc_solver est optionnel : sans lui, on perd juste la flèche optimale.
try:
    from npc_solver import compare_run
    HAS_SOLVER = True
except Exception:
    HAS_SOLVER = False

VOID, PLAYER, ENEMY, GOLD = 0, 1, 2, 3
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

# ── Palette (alignée sur npc_viewer.py) ───────────────────────────────────────
CELL, SIDEBAR, FPS = 96, 340, 60
C_BG         = (15,  15,  20)
C_GRID_DARK  = (25,  28,  38)
C_GRID_LIGHT = (30,  33,  46)
C_GRID_LINE  = (45,  50,  70)
C_PLAYER     = (80,  160, 255); C_PLAYER_SH = (20, 60, 130)
C_ENEMY      = (220,  55,  55); C_ENEMY_SH  = (100, 15, 15)
C_GOLD       = (255, 210,  40); C_GOLD_SH   = (140, 100, 5)
C_SIDEBAR_BG = (18,  20,  30)
C_TEXT       = (200, 205, 220); C_TEXT_DIM  = (90, 95, 115)
C_ACCENT     = (255, 210,  40)
C_LLM        = (80,  160, 255)     # flèche du LLM
C_OPT        = (80,  220, 120)     # flèche optimale
C_SUCCESS    = (80,  220, 120); C_DANGER = (220, 80, 80)

# ── Reconstruction des frames (pure, testable sans pygame) ────────────────────
def _allowed(m, pos):
    r, c = pos
    nr, nc = m.shape
    return 0 <= r < nr and 0 <= c < nc and m[r, c] in (VOID, GOLD)

def build_frames(run):
    """État AVANT chaque coup + coup LLM + coup optimal. None si pas de carte."""
    if "initial_map" not in run:
        return None
    m0 = np.array(run["initial_map"])

    opt_by_turn = {}
    if HAS_SOLVER:
        _, cmp_rows = compare_run(m0, run["decision_log"])
        opt_by_turn = {r["turn"]: r for r in cmp_rows}

    m   = m0.copy()
    pos = tuple(int(x) for x in np.argwhere(m == PLAYER)[0])
    score = 0
    frames = []
    for e in run["decision_log"]:
        d = e["direction"]
        o = opt_by_turn.get(e["turn"], {})
        frames.append({
            "turn": e["turn"], "map": m.copy(), "pos": pos, "dir": d,
            "reason": e.get("reason", ""), "blocked": e.get("blocked", False),
            "gold": e.get("gold", False), "score": score,
            "opt_dir": o.get("opt_dir"), "match": o.get("match"),
            "regret": o.get("regret"),
        })
        if d in MOVES:
            dr, dc = MOVES[d]
            nb = (pos[0] + dr, pos[1] + dc)
            if _allowed(m, nb):
                if m[nb] == GOLD:
                    score += 1
                m[pos] = VOID; m[nb] = PLAYER; pos = nb

    frames.append({
        "turn": run["decision_log"][-1]["turn"] + 1, "map": m.copy(), "pos": pos,
        "dir": None, "reason": "— fin de partie —", "blocked": False, "gold": False,
        "score": score, "opt_dir": None, "match": None, "regret": None,
    })
    return frames

# ── Dessin ────────────────────────────────────────────────────────────────────
def draw_arrow(surf, cx, cy, direction, color, length, width, head, outline=False):
    dr, dc = MOVES[direction]
    ux, uy = dc, dr                                   # (x, y) écran
    tail = (cx - ux * length * 0.35, cy - uy * length * 0.35)
    tip  = (cx + ux * length * 0.55, cy + uy * length * 0.55)
    # perpendiculaire pour la pointe
    px, py = -uy, ux
    base = (tip[0] - ux * head, tip[1] - uy * head)
    left  = (base[0] + px * head * 0.7, base[1] + py * head * 0.7)
    right = (base[0] - px * head * 0.7, base[1] - py * head * 0.7)
    if outline:
        pygame.draw.line(surf, color, tail, base, max(1, width // 2))
        pygame.draw.polygon(surf, color, [tip, left, right], 2)
    else:
        pygame.draw.line(surf, color, tail, base, width)
        pygame.draw.polygon(surf, color, [tip, left, right])

def draw_grid(surf, m, frame):
    nr, nc = m.shape
    gw, gh = nc * CELL, nr * CELL
    for r in range(nr):
        for c in range(nc):
            col = C_GRID_LIGHT if (r + c) % 2 == 0 else C_GRID_DARK
            pygame.draw.rect(surf, col, (c*CELL, r*CELL, CELL, CELL))
    for r in range(nr + 1):
        pygame.draw.line(surf, C_GRID_LINE, (0, r*CELL), (gw, r*CELL), 1)
    for c in range(nc + 1):
        pygame.draw.line(surf, C_GRID_LINE, (c*CELL, 0), (c*CELL, gh), 1)

    for r in range(nr):
        for c in range(nc):
            cx, cy = c*CELL + CELL//2, r*CELL + CELL//2
            val = m[r][c]
            if val == GOLD:
                rad = int(CELL * 0.28)
                pygame.draw.circle(surf, C_GOLD_SH, (cx+3, cy+4), rad)
                pygame.draw.circle(surf, C_GOLD,    (cx, cy),     rad)
                pygame.draw.circle(surf, (255,245,160), (cx-rad//4, cy-rad//4), rad//4)
            elif val == ENEMY:
                rad = int(CELL * 0.28)
                pygame.draw.circle(surf, C_ENEMY_SH, (cx+3, cy+5), rad)
                pygame.draw.circle(surf, C_ENEMY,    (cx, cy),     rad)
                pygame.draw.circle(surf, (255,200,200), (cx-5, cy-4), 4)
                pygame.draw.circle(surf, (255,200,200), (cx+5, cy-4), 4)
            elif val == PLAYER:
                rad = int(CELL * 0.30)
                pygame.draw.circle(surf, C_PLAYER_SH, (cx+3, cy+5), rad)
                pygame.draw.circle(surf, C_PLAYER,    (cx, cy),     rad)
                pygame.draw.circle(surf, (180,220,255), (cx-5, cy-6), rad//4)

    # Flèches sur la case du joueur (état courant de la frame)
    pr, pc = frame["pos"]
    cx, cy = pc*CELL + CELL//2, pr*CELL + CELL//2
    opt, llm = frame["opt_dir"], frame["dir"]
    # optimal d'abord (dessous), en vert, seulement s'il diffère
    if opt and opt != llm:
        draw_arrow(surf, cx, cy, opt, C_OPT, CELL*0.9, 5, 16, outline=True)
    if llm in MOVES:
        col = C_DANGER if (opt and opt != llm) else C_LLM
        draw_arrow(surf, cx, cy, llm, col, CELL*0.8, 7, 18)

def wrap(text, font, max_w):
    out = []
    for para in text.split("\n"):
        words, line = para.split(), ""
        for w in words:
            test = f"{line} {w}".strip()
            if font.size(test)[0] <= max_w:
                line = test
            else:
                out.append(line); line = w
        out.append(line)
    return out

def draw_sidebar(surf, fonts, grid_w, win_h, frame, idx, n, run, autoplay, speed):
    f_title, f_ui, f_small = fonts
    x0 = grid_w
    pygame.draw.rect(surf, C_SIDEBAR_BG, (x0, 0, SIDEBAR, win_h))
    pygame.draw.line(surf, C_GRID_LINE, (x0, 0), (x0, win_h), 2)
    pad = 18; x, y = x0 + pad, 18

    def txt(font, s, color, dy):
        nonlocal y
        surf.blit(font.render(s, True, color), (x, y)); y += dy
    def sep():
        nonlocal y
        pygame.draw.line(surf, C_GRID_LINE, (x, y), (x0+SIDEBAR-pad, y), 1); y += 12

    txt(f_title, "Replay", C_ACCENT, 30)
    txt(f_small, run.get("model", ""), C_TEXT_DIM, 20)
    txt(f_small, f"{run.get('status','?')}  ·  {run.get('final_score','?')}/{run.get('total_gold','?')} or",
        C_TEXT_DIM, 22)
    sep()

    # progression
    bar_w = SIDEBAR - 2*pad
    pygame.draw.rect(surf, C_GRID_LINE, (x, y, bar_w, 8), border_radius=4)
    filled = int(bar_w * idx / max(1, n-1))
    pygame.draw.rect(surf, C_ACCENT, (x, y, filled, 8), border_radius=4)
    y += 16
    txt(f_ui, f"Tour {frame['turn']}  ({idx+1}/{n})", C_TEXT, 30)
    sep()

    # coup LLM vs optimal
    llm, opt = frame["dir"], frame["opt_dir"]
    ARROW = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→", None: "—"}
    txt(f_small, "Coup LLM", C_TEXT_DIM, 18)
    tag = "  +OR" if frame["gold"] else "  (bloqué)" if frame["blocked"] else ""
    lcol = C_SUCCESS if frame["gold"] else C_DANGER if frame["blocked"] else C_LLM
    txt(f_ui, f"{ARROW.get(llm,'?')}  {llm or '—'}{tag}", lcol, 28)

    if opt is not None:
        txt(f_small, "Coup optimal", C_TEXT_DIM, 18)
        mcol = C_SUCCESS if frame["match"] else C_DANGER
        mark = "  = LLM" if frame["match"] else "  ≠ LLM"
        txt(f_ui, f"{ARROW.get(opt,'?')}  {opt}{mark}", C_OPT, 26)
        if frame["regret"] is not None:
            rcol = C_DANGER if frame["regret"] else C_TEXT_DIM
            txt(f_small, f"regret : {frame['regret']}", rcol, 22)
    sep()

    # justification
    txt(f_small, "Justification", C_TEXT_DIM, 18)
    reason = frame["reason"] or "—"
    space = win_h - 70 - y
    max_lines = max(3, space // 16)
    for line in wrap(reason, f_small, SIDEBAR - 2*pad)[:max_lines]:
        surf.blit(f_small.render(line, True, C_TEXT), (x, y)); y += 16

    # contrôles
    yb = win_h - 46
    pygame.draw.line(surf, C_GRID_LINE, (x, yb), (x0+SIDEBAR-pad, yb), 1)
    play = f"▶ x{speed}" if autoplay else "❚❚ pause"
    surf.blit(f_small.render(f"← →  tour     ESPACE  {play}", True, C_TEXT_DIM), (x, yb+8))
    surf.blit(f_small.render("↑↓ vitesse   HOME/END   ESC quitter", True, C_TEXT_DIM), (x, yb+24))

# ── Sélection du run ──────────────────────────────────────────────────────────
def pick_run(arg):
    if arg:
        return Path(arg)
    candidates = sorted(glob.glob("runs/*.json"), reverse=True)
    for c in candidates:
        try:
            if "initial_map" in json.loads(Path(c).read_text(encoding="utf-8")):
                return Path(c)
        except json.JSONDecodeError:
            continue
    return Path(candidates[0]) if candidates else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = pick_run(arg)
    if path is None or not path.exists():
        print("Aucun run trouvé. Lance une simulation d'abord (dossier runs/).")
        sys.exit(1)

    run = json.loads(path.read_text(encoding="utf-8"))
    frames = build_frames(run)
    if frames is None:
        print(f"{path.name} : pas de 'initial_map' -> run non rejouable "
              "(relance avec les patchs qui enregistrent la carte).")
        sys.exit(1)
    if not HAS_SOLVER:
        print("npc_solver introuvable : flèche optimale désactivée (replay LLM seul).")

    print(f"Replay : {path.name}  ({len(frames)} frames)")

    pygame.init()
    nr, nc = frames[0]["map"].shape
    grid_w = nc * CELL
    win_w, win_h = grid_w + SIDEBAR, nr * CELL
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"NPC Brain — Replay — {path.name}")
    clock = pygame.time.Clock()
    try:
        fonts = (pygame.font.SysFont("Segoe UI", 24, bold=True),
                 pygame.font.SysFont("Segoe UI", 17),
                 pygame.font.SysFont("Segoe UI", 13))
    except Exception:
        fonts = (pygame.font.Font(None, 28), pygame.font.Font(None, 22),
                 pygame.font.Font(None, 16))

    idx, autoplay, speed = 0, False, 2
    acc = 0.0
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE,):
                    running = False
                elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = min(idx + 1, len(frames) - 1); autoplay = False
                elif ev.key in (pygame.K_LEFT, pygame.K_a):
                    idx = max(idx - 1, 0); autoplay = False
                elif ev.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif ev.key == pygame.K_UP:
                    speed = min(speed + 1, 10)
                elif ev.key == pygame.K_DOWN:
                    speed = max(speed - 1, 1)
                elif ev.key == pygame.K_HOME:
                    idx = 0; autoplay = False
                elif ev.key == pygame.K_END:
                    idx = len(frames) - 1; autoplay = False

        if autoplay:
            acc += dt * speed
            if acc >= 1.0:
                acc = 0.0
                idx += 1
                if idx >= len(frames) - 1:
                    idx = len(frames) - 1; autoplay = False

        fr = frames[idx]
        screen.fill(C_BG)
        draw_grid(screen, fr["map"], fr)
        draw_sidebar(screen, fonts, grid_w, win_h, fr, idx, len(frames), run, autoplay, speed)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()