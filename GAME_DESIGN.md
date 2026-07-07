# Game Design — NPC Brain

Ce document tranche explicitement les questions de game design posées par
`SPECS.md` (§1.2), et justifie chaque choix par rapport à l'objectif du
projet : mesurer la qualité de navigation d'un LLM à charge algorithmique
constante, pas construire un jeu complet.

## Le joueur doit ramasser UNE pièce ou PLUSIEURS ?

**Plusieurs** (6 par défaut, `n_gold` paramétrable dans `make_random_map`).

**Pourquoi** : une seule pièce ne teste qu'un aller simple ; plusieurs pièces
forcent un ordre de visite (le LLM doit re-décider une cible après chaque
ramassage) et permettent de comparer sa trajectoire à un vrai problème
d'optimisation (TSP à chemin ouvert, résolu par `npc_solver.py`) plutôt qu'à
un simple pathfinding point à point. C'est ce qui rend les KPI `efficiency`
et `wasted_turns` significatifs : sur une seule pièce, il n'y aurait presque
rien à distinguer entre un bon et un mauvais joueur.

## Les combats sont-ils autorisés ? Y'a-t-il des combats sur la carte ?

**Non.** Les ennemis (`ENEMY`) sont des **obstacles statiques** : ils
bloquent une case comme le ferait un mur (`allowed_move` les traite
exactement comme une case hors-grille), mais n'attaquent pas, ne se
déplacent pas, et le joueur ne peut pas les éliminer.

**Pourquoi** : le projet mesure une seule capacité du LLM — la navigation
vers un objectif sous contrainte de directions valides. Ajouter du combat
introduirait une seconde compétence (décision tactique) qui se mélangerait
aux KPI de navigation et rendrait le benchmark impossible à interpréter
proprement (un mauvais `match_rate` voudrait dire "mauvais en navigation"
*ou* "mauvais en combat", sans pouvoir trancher). Isoler une seule variable
à la fois est le principe directeur retenu dans `SPECS.md` ("garder un
équilibre en charge cognitive algorithmique et charge du LLM").

## Quel layout de départ ?

**Carte générée aléatoirement à chaque run** (`make_random_map`) : position
du joueur, des pièces et des ennemis tirées sans collision sur une grille
8×8 (8 lignes, 8 colonnes, 6 pièces, 10 ennemis par défaut).

**Pourquoi l'aléatoire plutôt qu'un layout fixe** : un layout unique et fixe
biaiserait le benchmark vers "un LLM peut-il résoudre CETTE carte
précise" — au risque de sur-interpréter un résultat qui tient à la carte
plutôt qu'au modèle. Randomiser à chaque partie (avec une seed dérivée du
modèle + numéro de partie dans `benchmark_runner.py`, pour rester
reproductible) donne une distribution de difficulté représentative plutôt
qu'un seul point de mesure.

Un layout à la main existait initialement (carte 7×7 commentée dans la
cellule 5 du notebook, avec un ennemi placé volontairement entre le joueur
et une pièce) : il a été conservé en commentaire comme exemple de "carte
dirigée", mais n'est plus le mode par défaut, pour la raison ci-dessus.

## Ce qui reste volontairement hors scope

- **Déplacement des ennemis** : les ennemis ne bougent jamais. Un ennemi
  mobile transformerait le problème en poursuite/évitement dynamique, une
  autre compétence encore, hors du périmètre "navigation vers un objectif".
- **Pièces à valeur variable ou pièges** : toutes les pièces valent 1 point,
  il n'y a pas de fausse pièce ni de case piège. Ça garderait le calcul de
  l'optimal (BFS + TSP) directement comparable à ce que joue le LLM, sans
  ajouter une couche de risque/récompense que le prompt actuel ne modélise
  pas.
