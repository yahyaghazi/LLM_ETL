"""
npc_viewer.py
─────────────
Viewer Pygame pour NPC Brain.
Lit npc_state.json écrit par le notebook à chaque tour.

Lancer via le notebook (subprocess) ou directement :
    python npc_viewer.py
"""

import sys, time, json
from pathlib import Path

import numpy as np
import pygame

STATE_FILE = Path("npc_state.json")

# ── Palette ───────────────────────────────────────────────────────────────────
CELL    = 96
SIDEBAR = 300
FPS     = 30   # pas besoin de 60, le LLM est bien plus lent

C_BG         = (15,  15,  20)
C_GRID_DARK  = (25,  28,  38)
C_GRID_LIGHT = (30,  33,  46)
C_GRID_LINE  = (45,  50,  70)
C_PLAYER     = (80,  160, 255);  C_PLAYER_SH = (20,  60, 130)
C_ENEMY      = (220,  55,  55);  C_ENEMY_SH  = (100,  15,  15)
C_GOLD       = (255, 210,  40);  C_GOLD_SH   = (140, 100,   5)
C_SIDEBAR_BG = (18,  20,  30)
C_TEXT       = (200, 205, 220);  C_TEXT_DIM  = (90,  95, 115)
C_ACCENT     = (255, 210,  40)
C_SUCCESS    = (80,  220, 120);  C_DANGER    = (220,  80,  80)
C_THINKING   = (160, 100, 255)

VOID, PLAYER, ENEMY, GOLD = 0, 1, 2, 3

# ── Lecture état ──────────────────────────────────────────────────────────────
def read_state(last_good=None):
    """Lit npc_state.json. Si le fichier est en cours d'écriture (JSON partiel)
    ou absent, on garde le dernier état valide au lieu de crasher / clignoter."""
    try:
        txt = STATE_FILE.read_text(encoding="utf-8")
        if not txt.strip():
            return last_good
        return json.loads(txt)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError, OSError):
        return last_good

# ── Rendu grille ──────────────────────────────────────────────────────────────
def draw_grid(surface, world_map, t):
    n_rows, n_cols = len(world_map), len(world_map[0])
    pulse = 0.10 * np.sin(t * 3.5)
    gw, gh = n_cols * CELL, n_rows * CELL

    for r in range(n_rows):
        for c in range(n_cols):
            col = C_GRID_LIGHT if (r + c) % 2 == 0 else C_GRID_DARK
            pygame.draw.rect(surface, col, (c*CELL, r*CELL, CELL, CELL))

    for r in range(n_rows + 1):
        pygame.draw.line(surface, C_GRID_LINE, (0, r*CELL), (gw, r*CELL), 1)
    for c in range(n_cols + 1):
        pygame.draw.line(surface, C_GRID_LINE, (c*CELL, 0), (c*CELL, gh), 1)

    for r in range(n_rows):
        for c in range(n_cols):
            cx, cy = c*CELL + CELL//2, r*CELL + CELL//2
            val = world_map[r][c]

            if val == GOLD:
                rad = int(CELL * (0.28 + pulse))
                s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                pygame.draw.circle(s, (255,210,40,35), (CELL//2, CELL//2), rad+10)
                surface.blit(s, (c*CELL, r*CELL))
                pygame.draw.circle(surface, C_GOLD_SH, (cx+3, cy+4), rad)
                pygame.draw.circle(surface, C_GOLD,    (cx,   cy),   rad)
                pygame.draw.circle(surface, (255,245,160), (cx - rad//4, cy - rad//4), rad//4)

            elif val == PLAYER:
                rad = int(CELL * 0.32)
                s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                pygame.draw.circle(s, (80,160,255,30), (CELL//2, CELL//2), rad+14)
                surface.blit(s, (c*CELL, r*CELL))
                pygame.draw.circle(surface, C_PLAYER_SH, (cx+3, cy+5), rad)
                pygame.draw.circle(surface, C_PLAYER,    (cx,   cy),   rad)
                pygame.draw.circle(surface, (180,220,255), (cx-5, cy-6), rad//4)

            elif val == ENEMY:
                rad = int(CELL * 0.28)
                pygame.draw.circle(surface, C_ENEMY_SH, (cx+3, cy+5), rad)
                pygame.draw.circle(surface, C_ENEMY,    (cx,   cy),   rad)
                pygame.draw.circle(surface, (255,200,200), (cx-5, cy-4), 4)
                pygame.draw.circle(surface, (255,200,200), (cx+5, cy-4), 4)
                pygame.draw.circle(surface, (80,0,0),      (cx-5, cy-4), 2)
                pygame.draw.circle(surface, (80,0,0),      (cx+5, cy-4), 2)

# ── Rendu sidebar ─────────────────────────────────────────────────────────────
def draw_sidebar(surface, fonts, grid_w, win_h, t, state):
    font_title, font_ui, font_small = fonts
    x0 = grid_w
    pygame.draw.rect(surface, C_SIDEBAR_BG, (x0, 0, SIDEBAR, win_h))
    pygame.draw.line(surface, C_GRID_LINE, (x0, 0), (x0, win_h), 2)

    score      = state.get("score", 0)
    total_gold = state.get("total_gold", 0)
    turn       = state.get("turn", 0)
    status     = state.get("status", "running")
    dec_log    = state.get("decision_log", [])
    move_hist  = [e.get("direction", "?") for e in dec_log]   # derive du decision_log
    max_turns  = 30

    pad = 18
    x, y = x0 + pad, 20

    def txt(font, text, color, dy):
        nonlocal y
        surface.blit(font.render(text, True, color), (x, y))
        y += dy

    def sep():
        nonlocal y
        pygame.draw.line(surface, C_GRID_LINE, (x, y), (x0+SIDEBAR-pad, y), 1)
        y += 14

    txt(font_title, "NPC Brain",          C_ACCENT,   38)
    txt(font_small, "LLM  ·  Pygame",     C_TEXT_DIM, 26)
    sep()

    bar_w = SIDEBAR - 2*pad
    pygame.draw.rect(surface, C_GRID_LINE, (x, y, bar_w, 10), border_radius=5)
    if total_gold > 0:
        filled = int(bar_w * score / total_gold)
        if filled > 0:
            pygame.draw.rect(surface, C_GOLD, (x, y, filled, 10), border_radius=5)
    y += 18

    txt(font_ui,    f"Or   {score} / {total_gold}",   C_GOLD,     26)
    txt(font_small, f"Tour  {turn} / {max_turns}",    C_TEXT_DIM, 24)

    if status == "thinking":
        col = C_THINKING if int(t*4) % 2 == 0 else C_TEXT_DIM
        lbl, fc = "⟳  Réflexion LLM…", col
    elif status == "won":
        lbl, fc = "🏆  Victoire !",     C_SUCCESS
    elif status == "timeout":
        lbl, fc = "⏱️  Temps écoulé",  C_DANGER
    else:
        last = move_hist[-1] if move_hist else ""
        lbl  = f"→  {last}" if last else "En attente…"
        fc   = C_TEXT
    txt(font_ui, lbl, fc, 30)
    sep()

    txt(font_small, "Historique", C_TEXT_DIM, 18)
    hist_str = "  ".join(move_hist[-8:]) if move_hist else "—"
    txt(font_small, hist_str[:36], C_TEXT, 22)
    sep()

    # Justification de chaque coup (tour + direction + raison)
    txt(font_small, "Justifications par coup", C_TEXT_DIM, 18)

    def wrap(text, width=32):
        words, line, lines = text.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line = f"{line} {w}".strip()
            else:
                lines.append(line); line = w
        if line:
            lines.append(line)
        return lines

    if dec_log:
        # On affiche les derniers coups qui rentrent dans l'espace restant
        space_left = (win_h - 130) - y      # marge avant la legende
        max_lines  = max(2, space_left // 16)
        used = 0
        for entry in reversed(dec_log):     # plus recent en haut
            tour = entry.get("turn", "?")
            dr   = entry.get("direction", "?")
            rsn  = entry.get("reason", "")
            tag  = " +OR" if entry.get("gold") else " (bloque)" if entry.get("blocked") else ""

            head_col = C_SUCCESS if entry.get("gold") else C_DANGER if entry.get("blocked") else C_ACCENT
            surface.blit(font_small.render(f"[{tour:>2}] {dr}{tag}", True, head_col), (x, y))
            y += 16; used += 1

            for l in wrap(rsn):
                if used >= max_lines:
                    break
                surface.blit(font_small.render(l, True, C_THINKING), (x + 8, y))
                y += 15; used += 1

            y += 4; used += 0.25
            if used >= max_lines:
                break
    else:
        txt(font_small, "—", C_TEXT_DIM, 16)

    # Légende
    y_leg = win_h - 100
    pygame.draw.line(surface, C_GRID_LINE, (x, y_leg), (x0+SIDEBAR-pad, y_leg), 1)
    y_leg += 12
    surface.blit(font_small.render("Légende", True, C_TEXT_DIM), (x, y_leg)); y_leg += 18
    for fc, label in [(C_PLAYER,"Joueur (LLM)"), (C_GOLD,"Or"), (C_ENEMY,"Ennemi")]:
        pygame.draw.circle(surface, fc, (x+7, y_leg+7), 7)
        surface.blit(font_small.render(label, True, C_TEXT), (x+20, y_leg))
        y_leg += 20

    surface.blit(font_small.render("ESC  quitter", True, C_TEXT_DIM), (x, win_h-22))

# ── Écran d'attente ───────────────────────────────────────────────────────────
def draw_waiting(surface, fonts, win_w, win_h, t):
    surface.fill(C_BG)
    _, font_ui, font_small = fonts
    col = (160, 100, int(min(255, 180 + 70*np.sin(t*3))))
    for text, fc, dy in [
        ("NPC Brain",                     C_ACCENT,  -50),
        ("En attente de npc_state.json…", col,         0),
        ("Lance la simulation depuis le notebook.", C_TEXT_DIM, 40),
    ]:
        s = font_ui.render(text, True, fc)
        surface.blit(s, (win_w//2 - s.get_width()//2, win_h//2 + dy))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()

    # Taille initiale — sera redimensionnée dès qu'on lit l'état
    n_rows, n_cols = 7, 7
    grid_w = n_cols * CELL
    win_w, win_h = grid_w + SIDEBAR, n_rows * CELL
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("NPC Brain — Viewer")
    clock = pygame.time.Clock()

    try:
        fonts = (
            pygame.font.SysFont("Segoe UI", 22, bold=True),
            pygame.font.SysFont("Segoe UI", 16),
            pygame.font.SysFont("Segoe UI", 13),
        )
    except Exception:
        fonts = (pygame.font.Font(None, 26), pygame.font.Font(None, 20), pygame.font.Font(None, 16))

    t0 = time.time()
    state = None
    running = True

    while running:
        clock.tick(FPS)
        t = time.time() - t0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Lecture du fichier d'état (garde le dernier valide si lecture partielle)
        state = read_state(last_good=state)

        screen.fill(C_BG)

        if state is None:
            draw_waiting(screen, fonts, win_w, win_h, t)
        else:
            wm = state["world_map"]
            nr, nc = len(wm), len(wm[0])
            if nr != n_rows or nc != n_cols:
                n_rows, n_cols = nr, nc
                grid_w = nc * CELL
                win_w, win_h = grid_w + SIDEBAR, nr * CELL
                screen = pygame.display.set_mode((win_w, win_h))

            draw_grid(screen, wm, t)
            draw_sidebar(screen, fonts, grid_w, win_h, t, state)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()