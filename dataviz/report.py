"""
dataviz/report.py
──────────────────
Rapport (reporting) du benchmark, généré depuis les tables GOLD du pipeline
(data/npc.duckdb : main_gold.agg_run_kpi, main_gold.agg_model_comparison).

Pas de mise à jour temps réel (conforme aux specs) : à relancer après chaque
`python pipeline/run_pipeline.py`.

Lancer :
    python dataviz/report.py
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

DB_PATH = Path("data/npc.duckdb")
OUT_DIR = Path("reports")

# Palette catégorielle (ordre fixe, jamais recyclé — cf. skill dataviz)
MODEL_COLORS = {
    "mistralai/ministral-3-3b":        "#2a78d6",  # slot 1 blue
    "qwen/qwen3-vl-4b":                "#1baf7a",  # slot 2 aqua
    "microsoft/phi-4-mini-reasoning":  "#eda100",  # slot 3 yellow
    "deepseek/deepseek-r1-0528-qwen3-8b": "#008300",  # slot 4 green
}
TYPOLOGY_COLORS = {
    "aleatoire":     "#2a78d6",  # slot 1 blue
    "chemin_bloque": "#e34948",  # slot 6 red (obstacle direct)
    "piece_leurre":  "#4a3aa7",  # slot 5 violet (piège)
}
FALLBACK_COLOR = "#4a3aa7"  # slot 5 violet, si une catégorie non prévue apparaît

INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"
SURFACE       = "#fcfcfb"


def _color_for(key, color_map=MODEL_COLORS):
    return color_map.get(key, FALLBACK_COLOR)


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _bar_panel(ax, df, col, title, pct=True, key_col="model", color_map=MODEL_COLORS, short_label=True):
    keys = df[key_col].tolist()
    raw = df[col].tolist()
    is_na = [v is None or (isinstance(v, float) and v != v) for v in raw]
    values = [0 if na else v for v, na in zip(raw, is_na)]
    colors = [_color_for(k, color_map) for k in keys]
    bar_colors = [GRIDLINE if na else c for na, c in zip(is_na, colors)]
    bars = ax.bar(range(len(keys)), values, color=bar_colors, width=0.6, zorder=3)
    ax.set_xticks(range(len(keys)))
    labels = [k.split("/")[-1] if short_label else k for k in keys]
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    ymax = max(values) if values else 1
    ax.set_ylim(0, max(ymax * 125, 1) / 100 if pct else ymax * 1.25)
    for bar, v, na in zip(bars, values, is_na):
        label = "N/A" if na else (f"{v:.0%}" if pct else f"{v:.2f}")
        color = INK_MUTED if na else INK_SECONDARY
        y = ax.get_ylim()[1] * 0.03 if na else bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, label,
                 ha="center", va="bottom", fontsize=9, color=color)
    _style_axis(ax)


def model_comparison_chart(con):
    df = con.execute("""
        select model, win_rate, avg_match_rate, avg_efficiency, avg_blocked_rate, n_runs
        from main_gold.agg_model_comparison
        order by n_runs desc
    """).df()

    if df.empty:
        print("agg_model_comparison est vide — lance des runs puis pipeline/run_pipeline.py.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("NPC Brain — comparaison inter-modèles", color=INK_PRIMARY,
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.945,
              "Agrégats gold (main_gold.agg_model_comparison) — mêmes règles de jeu, même prompt, modèle seul variable",
              color=INK_SECONDARY, fontsize=9)

    _bar_panel(axes[0, 0], df, "win_rate", "Taux de victoire", pct=True)
    _bar_panel(axes[0, 1], df, "avg_match_rate", "Accord moyen avec le coup optimal", pct=True)
    _bar_panel(axes[1, 0], df, "avg_efficiency", "Efficacité moyenne (1.0 = parfait)", pct=False)
    _bar_panel(axes[1, 1], df, "avg_blocked_rate", "Taux de coups bloqués (mur)", pct=True)

    fig.text(0.02, 0.02,
              "n = nombre de runs par modèle : " +
              ", ".join(f"{m.split('/')[-1]}={n}" for m, n in zip(df["model"], df["n_runs"])),
              color=INK_MUTED, fontsize=8)
    fig.text(0.02, 0.005,
              "N/A (efficacité) = modèle qui n'a jamais gagné sur ses runs — la métrique n'est pas définie, ce n'est pas un 0.",
              color=INK_MUTED, fontsize=8, style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = OUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


def per_run_efficiency_chart(con):
    df = con.execute("""
        select model, run_id, turns_used, optimal_steps, status
        from main_gold.agg_run_kpi
        where optimal_steps is not null
        order by model, run_id
    """).df()

    if df.empty:
        print("agg_run_kpi est vide.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(SURFACE)

    lim = max(df["turns_used"].max(), df["optimal_steps"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=BASELINE, linewidth=1, linestyle="--", zorder=2,
            label="parcours optimal (référence)")

    for model, g in df.groupby("model"):
        marker = "o" if g["status"].iloc[0] != "timeout" else "o"
        ax.scatter(g["optimal_steps"], g["turns_used"], color=_color_for(model),
                   s=70, alpha=0.85, edgecolor=SURFACE, linewidth=0.5, zorder=3,
                   label=model.split("/")[-1])

    ax.set_xlabel("Coups optimaux (BFS+TSP)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Coups joués par le LLM", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Chaque run : coups joués vs coups optimaux\n(sur la diagonale = parfait, au-dessus = coups gâchés)",
                  color=INK_PRIMARY, fontsize=12, loc="left")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    out = OUT_DIR / "per_run_efficiency.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


def typology_comparison_chart(con):
    df = con.execute("""
        select typology, win_rate, avg_match_rate, avg_efficiency, avg_blocked_rate, n_runs
        from main_gold.agg_typology_kpi
        order by typology
    """).df()

    if df.empty:
        print("agg_typology_kpi est vide — lance des runs puis pipeline/run_pipeline.py.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("NPC Brain — comparaison inter-typologies de carte", color=INK_PRIMARY,
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.945,
              "Agrégats gold (main_gold.agg_typology_kpi) — même modèle, même prompt, layout de départ seul variable "
              "(cf. GAME_DESIGN.md)",
              color=INK_SECONDARY, fontsize=9)

    kwargs = dict(key_col="typology", color_map=TYPOLOGY_COLORS, short_label=False)
    _bar_panel(axes[0, 0], df, "win_rate", "Taux de victoire", pct=True, **kwargs)
    _bar_panel(axes[0, 1], df, "avg_match_rate", "Accord moyen avec le coup optimal", pct=True, **kwargs)
    _bar_panel(axes[1, 0], df, "avg_efficiency", "Efficacité moyenne (1.0 = parfait)", pct=False, **kwargs)
    _bar_panel(axes[1, 1], df, "avg_blocked_rate", "Taux de coups bloqués (mur)", pct=True, **kwargs)

    fig.text(0.02, 0.02,
              "n = nombre de runs par typologie : " +
              ", ".join(f"{t}={n}" for t, n in zip(df["typology"], df["n_runs"])),
              color=INK_MUTED, fontsize=8)
    fig.text(0.02, 0.005,
              "N/A (efficacité) = aucune victoire sur cette typologie — métrique non définie, pas un 0.",
              color=INK_MUTED, fontsize=8, style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = OUT_DIR / "typology_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


def main():
    if not DB_PATH.exists():
        print(f"{DB_PATH} introuvable — lance d'abord python pipeline/run_pipeline.py")
        return

    OUT_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    model_comparison_chart(con)
    per_run_efficiency_chart(con)
    typology_comparison_chart(con)
    con.close()


if __name__ == "__main__":
    main()
