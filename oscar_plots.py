import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from oscar_config import BG, PANEL, GOLD, BLUE, WHITE, GRID, RED, TEAL, DIM


def set_dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=10)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(GOLD)
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    ax.grid(color=GRID, lw=0.6, ls="--", alpha=0.7)
    if title:
        ax.set_title(title, color=GOLD, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=WHITE, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=WHITE, fontsize=11)


def plot_top5(res_df, output_path="wyniki/oscar_night_top5.png"):
    top5 = res_df.nlargest(5, "F1").sort_values("F1", ascending=True)
    metrics  = ["Bal.Acc", "Precision", "Recall", "F1", "AUC"]
    m_labels = ["Zbal.Dok.", "Precyzja", "Czułość", "F1", "AUC"]
    m_colors = [WHITE, TEAL, BLUE, GOLD, RED]
    has_std  = f"{metrics[0]}_std" in res_df.columns

    n = len(top5)
    bar_h = 0.12
    spacing = 0.85
    pos = np.arange(n) * spacing
    off = np.linspace(-(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics)) * bar_h

    fig, ax = plt.subplots(figsize=(10, 5))
    title = "Porównanie modeli — Top 5" + (" (śr. ± std, K-fold)" if has_std else "")
    set_dark(ax, title=title, xlabel="Wartość metryki")

    for i, (m, lm, c) in enumerate(zip(metrics, m_labels, m_colors)):
        vals = top5[m].astype(float).values
        errs = top5[f"{m}_std"].astype(float).values if has_std else None
        bars = ax.barh(pos + off[i], vals, height=bar_h * 0.88,
                       color=c, alpha=0.88, label=lm, edgecolor=BG, lw=0.4,
                       xerr=errs, error_kw={"ecolor": WHITE, "elinewidth": 0.8,
                                            "capsize": 2, "alpha": 0.7} if errs is not None else {})
        for b, v in zip(bars, vals):
            if v > 0.05:
                ax.text(v + 0.007, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                        va="center", ha="left", fontsize=8, color=c)

    ax.set_yticks(pos)
    ax.set_yticklabels(top5["Model"].tolist(), fontsize=10, color=WHITE)
    ax.set_xlim(0, 1.15)

    best = top5["F1"].values.argmax()
    ax.axhspan(pos[best] - spacing / 2 + 0.06, pos[best] + spacing / 2 - 0.06,
               color=GOLD, alpha=0.09, zorder=0)
    ax.legend(loc="lower right", fontsize=10, facecolor="#1A1A1A",
              edgecolor="#444", labelcolor=WHITE, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[wykres] {output_path}")


def plot_predictions(year_results, best_name, pred_years,
                     output_path="wyniki/oscar_night_predictions.png"):
    fig, axes = plt.subplots(1, len(pred_years), figsize=(18, 7),
                             gridspec_kw={"wspace": 0.5})
    fig.patch.set_facecolor(BG)

    for ax, yr in zip(axes, pred_years):
        full = year_results[yr].sort_values("prob", ascending=False)

        top4 = full.head(4)
        winner_row = full[full["winner"] == 1]
        if len(winner_row) and winner_row.index[0] not in top4.index:
            top4 = pd.concat([top4, winner_row]).drop_duplicates()

        yd = top4.sort_values("prob", ascending=True)
        ax.set_facecolor(PANEL)

        cols = [
            GOLD if r["winner"] == 1
            else (BLUE if r["prob"] == full["prob"].max() else DIM)
            for _, r in yd.iterrows()
        ]
        short_names = yd["Movie"].apply(lambda x: x if len(x) <= 23 else x[:21] + "…")

        bars = ax.barh(range(len(yd)), yd["prob"].values,
                       color=cols, edgecolor=BG, lw=0.4, height=0.65)
        for b, v in zip(bars, yd["prob"].values):
            if v > 0.12:
                ax.text(v - 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                        va="center", ha="right", fontsize=9, color=BG, fontweight="bold")
            elif v > 0.02:
                ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                        va="center", ha="left", fontsize=8, color=WHITE)

        ax.set_yticks(range(len(yd)))
        ax.set_yticklabels(short_names.values, fontsize=8.5, color=WHITE)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_visible(False)
        ax.tick_params(colors=WHITE)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        correct = yd["winner"].iloc[-1] == 1
        verdict = "✓ Trafna predykcja" if correct else "✗ Błędna predykcja"
        ax.set_title(f"{yr}\n{verdict}",
                     color=TEAL if correct else RED,
                     fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        f"Predykcja Oscara za Najlepszy Film — {best_name}\n"
        f"Model trenowany bez roku przewidywanego (leave-one-year-out)",
        color=GOLD, fontsize=13, fontweight="bold", y=1.0,
    )
    patches = [
        mpatches.Patch(color=GOLD, label="✓  Rzeczywisty zwycięzca"),
        mpatches.Patch(color=BLUE, label="★  Faworyt modelu (błędna pred.)"),
        mpatches.Patch(color=DIM,  label="Pozostałe nominacje"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11,
               facecolor="#1A1A1A", edgecolor="#555", labelcolor=WHITE,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[wykres] {output_path}")


def plot_features(imp, best_name, ame_series,
                  output_path="wyniki/oscar_night_features.png"):
    def _pick_color(f):
        if f.startswith("Gatunek:"):
            return TEAL
        if any(x in f for x in ["BAFTA", "Złote", "PGA", "Oscar", "Łączna"]):
            return GOLD
        return BLUE

    fig, ax = plt.subplots(figsize=(9, 7))
    set_dark(ax,
             title=f"Wpływ cech na decyzję modelu — {best_name} (top 20)",
             xlabel="Średni wzrost szansy na Oscara [punkty procentowe, pp]")

    colors = [_pick_color(f) for f in imp.index]
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor=BG, lw=0.4, alpha=0.9)
    for b, v in zip(bars, imp.values):
        ax.text(v + imp.values.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"+{v:.1f} pp", va="center", ha="left", fontsize=8.5, color=WHITE)

    ax.set_xlim(0, imp.values.max() * 1.28)
    ax.tick_params(axis="y", labelsize=9)

    legend_patches = [
        mpatches.Patch(color=GOLD, label="Nagrody branżowe"),
        mpatches.Patch(color=BLUE, label="Cechy numeryczne / recenzje"),
        mpatches.Patch(color=TEAL, label="Gatunek filmowy"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              facecolor="#1A1A1A", edgecolor="#444", labelcolor=WHITE,
              bbox_to_anchor=(0.99, 0.08))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[wykres] {output_path}")

    print("\nTop 10 cech wg AME:")
    for feat, val in ame_series.nlargest(10).items():
        print(f"  {val:+.1f} pp  {feat}")


def plot_ttest(res_df, ttest_df, best_name,
               output_path="wyniki/oscar_night_ttest.png"):
    sig_color = {"***": GOLD, "**": TEAL, "*": BLUE, "ns": DIM}
    sig_label = {
        "***": "p < 0.001",
        "**":  "p < 0.01",
        "*":   "p < 0.05",
        "ns":  "p ≥ 0.05  (brak istotności)",
    }

    sig_map  = dict(zip(ttest_df["Model"], ttest_df["istotność"]))
    pval_map = dict(zip(ttest_df["Model"], ttest_df["p-wartość"]))

    df = res_df[["Model", "F1", "F1_std"]].copy()
    df["sig"]   = df["Model"].map(sig_map).fillna("best")
    df["p_val"] = df["Model"].map(pval_map).fillna(np.nan)
    df = df.sort_values("F1", ascending=True).reset_index(drop=True)

    best_f1 = res_df.loc[res_df["Model"] == best_name, "F1"].values[0]

    bar_colors = [
        RED if row["sig"] == "best" else sig_color.get(row["sig"], DIM)
        for _, row in df.iterrows()
    ]

    n = len(df)
    fig, (ax_main, ax_sig) = plt.subplots(
        1, 2, figsize=(13, n * 0.38 + 1.6),
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.04},
    )
    fig.patch.set_facecolor(BG)

    ax_main.set_facecolor(PANEL)
    for sp in ax_main.spines.values():
        sp.set_edgecolor("#333")

    y = np.arange(n)
    ax_main.barh(
        y, df["F1"].values,
        xerr=df["F1_std"].values,
        color=bar_colors, edgecolor=BG, lw=0.3, alpha=0.88, height=0.72,
        error_kw={"ecolor": WHITE, "elinewidth": 1.0, "capsize": 3, "alpha": 0.55},
    )

    for i, (_, row) in enumerate(df.iterrows()):
        v = row["F1"]
        c = RED if row["sig"] == "best" else sig_color.get(row["sig"], DIM)
        ax_main.text(
            v + row["F1_std"] + 0.005, i,
            f"{v:.3f}", va="center", ha="left", fontsize=8.5,
            color=c if c != DIM else WHITE,
        )

    ax_main.axvline(best_f1, color=RED, lw=1.6, ls="--", alpha=0.75, zorder=3)
    ax_main.text(
        best_f1 + 0.003, -0.9,
        f"F1 = {best_f1:.3f}\n({best_name})",
        color=RED, fontsize=8, va="top", ha="left",
    )

    ns_f1s = df.loc[df["sig"] == "ns", "F1"].values
    if len(ns_f1s):
        ax_main.axvspan(ns_f1s.min() - 0.005, best_f1,
                        color=WHITE, alpha=0.04, zorder=0)

    ax_main.set_yticks(y)
    ax_main.set_yticklabels(df["Model"].tolist(), fontsize=9.5, color=WHITE)
    ax_main.tick_params(colors=WHITE, labelsize=9)
    ax_main.set_xlabel("Średni F1-score (5-fold CV)", color=WHITE, fontsize=11)
    ax_main.xaxis.label.set_color(WHITE)
    ax_main.grid(axis="x", color=GRID, lw=0.6, ls="--", alpha=0.7)
    ax_main.set_xlim(0, df["F1"].max() + df["F1_std"].max() + 0.07)
    ax_main.set_title(
        f"Testy statystyczne — sparowany t-test (5-fold CV)\n"
        f"Referencja: {best_name}",
        color=GOLD, fontsize=13, fontweight="bold", pad=10,
    )

    ax_sig.set_facecolor(PANEL)
    for sp in ax_sig.spines.values():
        sp.set_edgecolor("#333")
    ax_sig.set_yticks([])
    ax_sig.set_xticks([])
    ax_sig.set_xlim(0, 1)
    ax_sig.set_ylim(-0.5, n - 0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        sig = row["sig"]
        if sig == "best":
            label, color, fw = "★ best", RED, "bold"
        else:
            label = sig
            color = sig_color.get(sig, WHITE)
            fw = "bold" if sig != "ns" else "normal"
        ax_sig.text(0.5, i, label, va="center", ha="center",
                    fontsize=10, color=color, fontweight=fw)

    ax_sig.set_title("Istotność", color=WHITE, fontsize=10, pad=10)

    patches = [
        mpatches.Patch(color=RED,  label=f"★  {best_name} (model referencyjny)"),
    ] + [
        mpatches.Patch(color=sig_color[k], label=f"{k}  {sig_label[k]}")
        for k in ["***", "**", "*", "ns"]
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3, fontsize=9.5,
        facecolor="#1A1A1A", edgecolor="#444", labelcolor=WHITE,
        framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[wykres] {output_path}")
