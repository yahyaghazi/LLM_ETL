# NPC Brain — Documentation des cellules

Documentation cellule par cellule du notebook `npc_brain.ipynb`.
Le principe : un LLM contrôle un joueur sur une grille pour ramasser de l'or. Le notebook = le cerveau (logique + LLM), `npc_viewer.py` = l'affichage (Pygame en subprocess).

---

## Cellule 1 — Imports

Charge les dépendances : `numpy` (grille), `openai` (client LLM), `pydantic` (validation de la réponse du modèle), `subprocess`/`sys` (lancement du viewer), `pathlib` (fichier d'état).

---

## Cellule 3 — Config LLM

Lit les variables d'environnement (`LLM_API_URL`, `LLM_API_TOKEN`) depuis le `.env`, fixe le modèle à utiliser (`MODEL`), et instancie le client OpenAI pointant vers ton serveur local (LM Studio / Ollama). C'est ici que tu changes de modèle.

---

## Cellule 5 — Modélisation du monde

Définit les codes des entités (`VOID=0`, `PLAYER=1`, `ENEMY=2`, `GOLD=3`) et la carte de départ `INITIAL_MAP` sous forme de matrice numpy 7×7. Chaque chiffre = une case.

---

## Cellule 7 — Couche de contrat

Définit le "contrat" de sortie du LLM :
- `Direction` : enum des 4 directions autorisées (HAUT/BAS/GAUCHE/DROITE)
- `PlayerDecision` : le schéma pydantic que le LLM doit remplir (`{direction: ...}`)
- `MOVES` : la traduction de chaque direction en delta `(Δrow, Δcol)` sur la grille

C'est ce qui force le LLM à répondre proprement au lieu de texte libre.

---

## Cellule 9 — Moteurs (perception + déplacement)

Le cœur logique, quatre fonctions :

- **`localize`** : trouve les coordonnées d'un type d'entité sur la carte
- **`compute_distances`** : distance euclidienne entre des entités et une position de référence
- **`allowed_move`** : renvoie `True` si une case est franchissable (vide ou or), `False` si mur/bord/ennemi
- **`move`** : déplace le joueur si le coup est permis, ramasse l'or au passage, renvoie `(nouvelle_position, or_ramassé)`
- **`perception`** : construit ce que "voit" le joueur → position, delta vers l'or le plus proche, **directions franchissables**, nombre d'or/ennemis restants. Volontairement, elle ne dit PAS quelle direction prendre : c'est au LLM de naviguer.

---

## Cellule 11 — Moteur de décision (LLM)

`decide()` construit le prompt envoyé au modèle et parse sa réponse. Le prompt contient :
- la position de l'or traduite en langage naturel ("2 cases vers le BAS, 3 vers la DROITE")
- les directions autorisées
- les 5 derniers coups (pour éviter les boucles)
- un `feedback` optionnel ("ton dernier coup était bloqué")

Le préfixe `/no_think` désactive le raisonnement verbeux des modèles de type Qwen. La réponse est validée contre `PlayerDecision` via le structured output. En cas d'erreur API, renvoie `None`.

---

## Cellule 13 — Fichier d'état partagé

`write_state()` sérialise l'état complet de la partie (carte, score, tour, statut, historique, logs) dans `npc_state.json`. C'est le pont entre le notebook et le viewer Pygame : le notebook écrit, le viewer lit en boucle. C'est aussi ta future couche **bronze** pour le pipeline data.

---

## Cellule 15 — Game loop

`run_simulation()` orchestre la partie tour par tour :

1. calcule la perception
2. vérifie la condition de victoire (plus d'or) → sort
3. appelle `decide()` pour obtenir la direction du LLM
4. applique le mouvement via `move()`
5. calcule le **feedback** : si `new_pos == old_pos`, le coup était bloqué → on le signale au LLM au tour suivant (c'est le correctif anti-boucle)
6. met à jour le score, écrit l'état, recommence

Si `max_turns` est atteint sans tout ramasser → statut `timeout`.

---

## Cellule 17 — Lancement

Vérifie que `npc_viewer.py` existe dans le dossier, le lance en **subprocess** (vrai process Python = pas de crash SDL dans le kernel Jupyter), attend 1.5s que Pygame s'initialise, puis démarre `run_simulation()`. La fenêtre affiche la partie en temps réel via `npc_state.json`.

---

## Flux global

```
Cellule 17 (lancement)
    │
    ├──> subprocess ──> npc_viewer.py  ──lit──> npc_state.json  ──> affichage Pygame
    │
    └──> run_simulation (boucle)
             │
             ├─ perception()  ── ce que voit le joueur
             ├─ decide()      ── le LLM choisit une direction
             ├─ move()        ── application du coup
             └─ write_state() ── écrit npc_state.json (lu par le viewer)
```