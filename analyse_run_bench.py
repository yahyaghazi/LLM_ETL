"""
analyze_runs.py
────────────────
Analyse comparative des runs produits par benchmark_runner.py.
Agrège les métriques par modèle, prompt_mode et température, et produit :
- un tableau récapitulatif dans la console,
- un export CSV (runs_summary.csv),
- des graphiques comparatifs (si matplotlib est disponible).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Console Windows par défaut en cp1252 : force l'UTF-8 pour les caractères
# de mise en forme (═, ─, ·) sans quoi print() lève UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUNS_DIR = Path("runs")
CSV_OUT = Path("runs_summary.csv")


# ── Chargement des runs ─────────────────────────────────────────────────────────
def load_runs(runs_dir=RUNS_DIR):
    """Charge tous les fichiers JSON de runs.

    Ignore les runs à l'ancien format plat (status/final_score/turns_used à la
    racine, produits par une version antérieure de benchmark_runner.py ou par
    le notebook) : ce script analyse l'axe prompt_mode × température, qui
    n'existe que dans le nouveau format ({"result": {...}} imbriqué)."""
    runs = []
    n_skipped = 0
    if not runs_dir.exists():
        print(f"[ERREUR] Le dossier {runs_dir}/ n'existe pas.")
        return runs

    for path in sorted(runs_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Impossible de lire {path.name} : {e}")
            continue

        if "result" not in data:
            n_skipped += 1
            continue

        data["_file"] = path.name
        runs.append(data)

    if n_skipped:
        print(f"[INFO] {n_skipped} run(s) au format précédent ignoré(s) "
              f"(pas de clé 'result' — pas de prompt_mode/température associés).")
    return runs


# ── Agrégation ──────────────────────────────────────────────────────────────────
def aggregate(runs, keys):
    """
    Agrège les runs selon un tuple de clés (ex. ("model", "prompt_mode")).
    Retourne un dict {clé: métriques}.
    """
    groups = defaultdict(list)
    for r in runs:
        key = tuple(r.get(k) for k in keys)
        groups[key] = groups.get(key, [])
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        n = len(group)
        wins = sum(1 for r in group if r["result"]["status"] == "win")
        deaths = sum(1 for r in group if r["result"]["status"] == "dead")
        timeouts = sum(1 for r in group if r["result"]["status"] == "timeout")
        errors = sum(1 for r in group
                     if r["result"]["status"].startswith("error")
                     or r["result"]["status"] == "error_no_decision")

        total_score = sum(r["result"]["score"] for r in group)
        total_gold = sum(r["result"]["total_gold"] for r in group)
        avg_score = total_score / n if n else 0.0
        gold_ratio = total_score / total_gold if total_gold else 0.0
        avg_turns = sum(r["result"]["turn"] for r in group) / n if n else 0.0

        summary[key] = {
            "n": n,
            "wins": wins,
            "win_rate": wins / n if n else 0.0,
            "deaths": deaths,
            "death_rate": deaths / n if n else 0.0,
            "timeouts": timeouts,
            "errors": errors,
            "avg_score": avg_score,
            "gold_ratio": gold_ratio,
            "avg_turns": avg_turns,
        }
    return summary


# ── Affichage console ───────────────────────────────────────────────────────────
def print_table(summary, key_names):
    """Affiche un tableau formaté dans la console."""
    if not summary:
        print("  (aucune donnée)")
        return

    key_hdr = " · ".join(key_names)
    header = (f"{key_hdr:<45} | {'N':>3} | {'Win%':>6} | {'Mort%':>6} | "
              f"{'TO':>3} | {'Err':>3} | {'Or moy':>7} | {'Or%':>6} | {'Tours':>6}")
    print(header)
    print("─" * len(header))

    for key in sorted(summary.keys(), key=lambda k: tuple(str(x) for x in k)):
        m = summary[key]
        key_str = " · ".join(str(x) for x in key)
        print(f"{key_str:<45} | {m['n']:>3} | "
              f"{m['win_rate']*100:>5.1f}% | {m['death_rate']*100:>5.1f}% | "
              f"{m['timeouts']:>3} | {m['errors']:>3} | "
              f"{m['avg_score']:>7.2f} | {m['gold_ratio']*100:>5.1f}% | "
              f"{m['avg_turns']:>6.1f}")
    print()


# ── Export CSV ──────────────────────────────────────────────────────────────────
def export_csv(runs, path=CSV_OUT):
    """Exporte tous les runs (ligne par run) en CSV."""
    import csv

    fields = ["file", "model", "typology", "prompt_mode", "temperature",
              "status", "score", "total_gold", "turn"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            res = r["result"]
            writer.writerow({
                "file": r.get("_file", ""),
                "model": r.get("model", ""),
                "typology": r.get("typology", ""),
                "prompt_mode": r.get("prompt_mode", ""),
                "temperature": r.get("temperature", ""),
                "status": res["status"],
                "score": res["score"],
                "total_gold": res["total_gold"],
                "turn": res["turn"],
            })
    print(f"[INFO] Export CSV : {path}")


# ── Graphiques (optionnels) ─────────────────────────────────────────────────────
def plot_comparisons(runs):
    """Produit des graphiques comparatifs (win_rate par prompt et température)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib non installé → graphiques ignorés "
              "(pip install matplotlib pour les activer).")
        return

    # 1) Win rate par prompt_mode
    by_prompt = aggregate(runs, ("prompt_mode",))
    prompts = sorted(by_prompt.keys(), key=lambda k: str(k[0]))
    labels = [str(k[0]) for k in prompts]
    win_rates = [by_prompt[k]["win_rate"] * 100 for k in prompts]
    gold_ratios = [by_prompt[k]["gold_ratio"] * 100 for k in prompts]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], win_rates, width=0.4, label="Win %")
    ax.bar([i + 0.2 for i in x], gold_ratios, width=0.4, label="Or collecté %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_title("Performance par prompt")
    ax.legend()
    fig.tight_layout()
    fig.savefig("perf_par_prompt.png", dpi=120)
    print("[INFO] Graphique : perf_par_prompt.png")

    # 2) Or collecté % par température (courbes par prompt_mode)
    by_temp_prompt = aggregate(runs, ("prompt_mode", "temperature"))
    prompt_modes = sorted({k[0] for k in by_temp_prompt})
    temps = sorted({k[1] for k in by_temp_prompt if k[1] is not None})

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for pm in prompt_modes:
        ys = []
        for t in temps:
            key = (pm, t)
            ys.append(by_temp_prompt[key]["gold_ratio"] * 100
                      if key in by_temp_prompt else None)
        ax2.plot(temps, ys, marker="o", label=str(pm))
    ax2.set_xlabel("Température")
    ax2.set_ylabel("Or collecté %")
    ax2.set_title("Or collecté selon la température")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("or_par_temperature.png", dpi=120)
    print("[INFO] Graphique : or_par_temperature.png")


# ── Point d'entrée ──────────────────────────────────────────────────────────────
def main():
    runs = load_runs()
    print(f"[INFO] {len(runs)} run(s) chargé(s) depuis {RUNS_DIR}/\n")
    if not runs:
        return

    print("═" * 70)
    print("  RÉCAPITULATIF PAR PROMPT_MODE")
    print("═" * 70)
    print_table(aggregate(runs, ("prompt_mode",)), ["prompt_mode"])

    print("═" * 70)
    print("  RÉCAPITULATIF PAR TEMPÉRATURE")
    print("═" * 70)
    print_table(aggregate(runs, ("temperature",)), ["temperature"])

    print("═" * 70)
    print("  RÉCAPITULATIF PAR PROMPT × TEMPÉRATURE")
    print("═" * 70)
    print_table(aggregate(runs, ("prompt_mode", "temperature")),
                ["prompt_mode", "temperature"])

    print("═" * 70)
    print("  RÉCAPITULATIF PAR MODÈLE × PROMPT")
    print("═" * 70)
    print_table(aggregate(runs, ("model", "prompt_mode")),
                ["model", "prompt_mode"])

    print("═" * 70)
    print("  DÉTAIL COMPLET : MODÈLE × PROMPT × TEMPÉRATURE")
    print("═" * 70)
    print_table(aggregate(runs, ("model", "prompt_mode", "temperature")),
                ["model", "prompt_mode", "temp"])

    export_csv(runs)
    plot_comparisons(runs)


if __name__ == "__main__":
    main()
