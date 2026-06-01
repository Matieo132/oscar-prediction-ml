"""
ablation_preprocessing.py
--------------------------
Badanie wpływu kolejnych kroków preprocessingu na jakość predykcji.

Porównuje 3 modele (LR, RF, XGBoost) na 6 etapach:
  0. Baza:           tylko cechy numeryczne + data leakage w total_awards
  1. Leakage fix:    naprawa total_awards (bez Best Picture)
  2. + Pre-Oscar:    dodajemy BAFTA / Golden Globes / PGA
  3. + Oscar Night:  inne Oscary tej samej nocy
  4. + Gatunki:      kodowanie gatunków one-hot
  5. Pełny:          + cecha krytycy vs. widzowie  ← aktualny pipeline

Uruchomienie:
    python ablation_preprocessing.py
"""

import os, sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from oscar_config import BG, PANEL, GOLD, BLUE, WHITE, GRID, RED, TEAL, DIM
from oscar_data import PRE_OSCAR, OSCAR_NIGHT, NUMERIC, TOP_GENRES, TARGET

os.makedirs("wyniki", exist_ok=True)

# ── Etapy ablacji ────────────────────────────────────────────────────────
STAGES = [
    {
        "label":           "0. Baza\n(numeryczne\n+ data leakage)",
        "short":           "0. Baza\n(data leakage)",
        "fix_leakage":     False,
        "use_pre_oscar":   False,
        "use_oscar_night": False,
        "use_genres":      False,
        "use_critic":      False,
    },
    {
        "label":           "1. Naprawa\ndata leakage",
        "short":           "1. Naprawa\nleakage",
        "fix_leakage":     True,
        "use_pre_oscar":   False,
        "use_oscar_night": False,
        "use_genres":      False,
        "use_critic":      False,
    },
    {
        "label":           "2. + Nagrody\nPre-Oscar",
        "short":           "2. + Pre-Oscar\n(BAFTA/GG/PGA)",
        "fix_leakage":     True,
        "use_pre_oscar":   True,
        "use_oscar_night": False,
        "use_genres":      False,
        "use_critic":      False,
    },
    {
        "label":           "3. + Oscar Night\n(inne Oscary)",
        "short":           "3. + Oscar\nNight",
        "fix_leakage":     True,
        "use_pre_oscar":   True,
        "use_oscar_night": True,
        "use_genres":      False,
        "use_critic":      False,
    },
    {
        "label":           "4. + Gatunki\n(one-hot)",
        "short":           "4. + Gatunki\n(one-hot)",
        "fix_leakage":     True,
        "use_pre_oscar":   True,
        "use_oscar_night": True,
        "use_genres":      True,
        "use_critic":      False,
    },
    {
        "label":           "5. Pełny pipeline\n(+ krytycy vs.\nwidzowie)",
        "short":           "5. Pełny\npipeline",
        "fix_leakage":     True,
        "use_pre_oscar":   True,
        "use_oscar_night": True,
        "use_genres":      True,
        "use_critic":      True,
    },
]

# ── Modele porównywane ───────────────────────────────────────────────────
SELECTED_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, C=1.0),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=200, random_state=42),
}
if XGB_AVAILABLE:
    SELECTED_MODELS["XGBoost"] = XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0
    )

MODEL_COLORS = {
    "Logistic Regression": GOLD,
    "Random Forest":       BLUE,
    "Extra Trees":         TEAL,
    "XGBoost":             RED,
}


# ── Ładowanie danych wg etapu ────────────────────────────────────────────
def load_stage(stage, data_path="oscary_dane.csv"):
    """Zwraca (X, y, n_numeric) dla danego etapu ablacji."""
    df = pd.read_csv(data_path, sep=";")

    # Naprawa typów (zawsze)
    df["IMDb_Rating"] = df["IMDb_Rating"].apply(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan
    )
    for col in ["Metascore", "Rotten_Tomatoes", "Runtime_min", "IMDb_Votes", "total_awards"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PRE_OSCAR + OSCAR_NIGHT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Data leakage fix (opcjonalnie)
    if stage["fix_leakage"]:
        df["total_awards"] = (
            pd.to_numeric(df["total_awards"], errors="coerce").fillna(0)
            - pd.to_numeric(df[TARGET], errors="coerce").fillna(0)
        ).clip(lower=0)

    # Cecha inżynierska (opcjonalnie)
    if stage["use_critic"]:
        df["critic_vs_audience"] = df["Metascore"].fillna(50) / 10 - df["IMDb_Rating"].fillna(7)

    # Gatunki (opcjonalnie)
    if stage["use_genres"]:
        for g in TOP_GENRES:
            df[f"Genre_{g}"] = df["Genre"].fillna("").apply(
                lambda x: int(bool(re.search(rf"\b{g}\b", x, re.IGNORECASE)))
            )

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Buduj listę cech
    feature_cols = NUMERIC.copy()
    if stage["use_pre_oscar"]:
        feature_cols += [c for c in PRE_OSCAR if c in df.columns]
    if stage["use_oscar_night"]:
        feature_cols += [c for c in OSCAR_NIGHT if c in df.columns]
    if stage["use_critic"]:
        feature_cols += ["critic_vs_audience"]
    if stage["use_genres"]:
        feature_cols += [f"Genre_{g}" for g in TOP_GENRES]

    for col in NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].values.astype(float)
    y = df[TARGET].values
    n_numeric = len(NUMERIC)   # pierwsze n_numeric kolumn to skalowalne numeryczne
    return X, y, n_numeric, len(feature_cols)


# ── Cross-validation ─────────────────────────────────────────────────────
def evaluate_stage(X, y, n_numeric, model):
    """5-fold stratified CV, zwraca dict z mean/std F1, AUC, Bal.Acc."""
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), list(range(n_numeric)))],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        pipe, X, y, cv=kfold,
        scoring={"f1": "f1", "auc": "roc_auc", "bal_acc": "balanced_accuracy"},
        return_train_score=False,
    )
    return {
        "F1":      scores["test_f1"].mean(),
        "F1_std":  scores["test_f1"].std(),
        "AUC":     scores["test_auc"].mean(),
        "AUC_std": scores["test_auc"].std(),
        "BalAcc":  scores["test_bal_acc"].mean(),
        "BalAcc_std": scores["test_bal_acc"].std(),
    }


# ════════════════════════════════════════════════════════════════════════
# Główna pętla ablacji
# ════════════════════════════════════════════════════════════════════════
all_results = {}   # {model_name: [dict_stage0, dict_stage1, ...]}

for model_name, model in SELECTED_MODELS.items():
    print(f"\n── {model_name} ──")
    all_results[model_name] = []
    for i, stage in enumerate(STAGES):
        X, y, n_num, n_feat = load_stage(stage)
        res = evaluate_stage(X, y, n_num, model)
        res["n_features"] = n_feat
        all_results[model_name].append(res)
        print(f"  Etap {i}  n_feat={n_feat:2d}  "
              f"F1={res['F1']:.3f}±{res['F1_std']:.3f}  "
              f"AUC={res['AUC']:.3f}±{res['AUC_std']:.3f}  "
              f"BalAcc={res['BalAcc']:.3f}±{res['BalAcc_std']:.3f}")


# ── Tabela podsumowująca ─────────────────────────────────────────────────
print("\n\n══ TABELA WYNIKÓW (F1 mean±std) ══")
header = f"{'Etap':<35}" + "".join(f"{m:>22}" for m in SELECTED_MODELS)
print(header)
print("─" * len(header))
for i, stage in enumerate(STAGES):
    row = f"{stage['short'].replace(chr(10), ' '):<35}"
    for model_name in SELECTED_MODELS:
        r = all_results[model_name][i]
        row += f"  {r['F1']:.3f} ± {r['F1_std']:.3f}   "
    print(row)

print("\n══ DELTA F1 wg etapu ══ (+ = poprawa vs poprzedni etap)")
print(f"{'Zmiana':<35}" + "".join(f"{m:>22}" for m in SELECTED_MODELS))
print("─" * (35 + 22 * len(SELECTED_MODELS)))
for i in range(1, len(STAGES)):
    label = f"Etap {i-1} → {i}"
    row = f"{label:<35}"
    for model_name in SELECTED_MODELS:
        delta = all_results[model_name][i]["F1"] - all_results[model_name][i-1]["F1"]
        sign = "+" if delta >= 0 else ""
        row += f"  {sign}{delta:+.3f}              "
    print(row)


# ════════════════════════════════════════════════════════════════════════
# Wykres 1: Liniowy — F1 i AUC przez kolejne etapy
# ════════════════════════════════════════════════════════════════════════
def set_dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL); ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(GOLD)
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.grid(color=GRID, lw=0.5, ls="--", alpha=0.6)
    if title:  ax.set_title(title,  color=GOLD,  fontsize=12, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=10)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor(BG)
stage_labels = [s["short"] for s in STAGES]
x = np.arange(len(STAGES))

for metric, ax, title in [("F1", ax1, "F1-score (5-fold CV)"), ("AUC", ax2, "AUC (5-fold CV)")]:
    set_dark(ax, title=title, xlabel="Etap preprocessingu", ylabel=metric)
    for model_name, color in MODEL_COLORS.items():
        if model_name not in all_results:
            continue
        vals     = [r[metric]          for r in all_results[model_name]]
        stds     = [r[f"{metric}_std"] for r in all_results[model_name]]
        ax.plot(x, vals, color=color, marker="o", lw=2, ms=7,
                label=model_name, zorder=3)
        ax.fill_between(x,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        color=color, alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=7.5, color=WHITE, ha="center")
    ax.set_ylim(0, 1.05)

    # Strzałki z wartościami przy ostatnim etapie
    for model_name, color in MODEL_COLORS.items():
        if model_name not in all_results:
            continue
        v = all_results[model_name][-1][metric]
        ax.annotate(f"{v:.3f}", xy=(len(STAGES)-1, v),
                    xytext=(len(STAGES)-1 + 0.15, v),
                    color=color, fontsize=8, va="center",
                    arrowprops=None)

    leg = ax.legend(fontsize=9, facecolor="#1A1A1A", edgecolor="#444",
                    labelcolor=WHITE, framealpha=0.9, loc="upper left")

fig.suptitle("Wpływ etapów preprocessingu na jakość modeli\n(5-fold stratified CV)",
             color=GOLD, fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
path1 = "wyniki/ablation_progression.png"
fig.savefig(path1, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n[wykres] {path1}")


# ════════════════════════════════════════════════════════════════════════
# Wykres 2: Słupkowy — delta F1 przy każdym kroku (co daje ile)
# ════════════════════════════════════════════════════════════════════════
delta_labels = [
    "Naprawa\ndata leakage",
    "+ Pre-Oscar\n(BAFTA/GG/PGA)",
    "+ Oscar Night\n(inne Oscary)",
    "+ Gatunki\n(one-hot)",
    "+ Krytycy\nvs. widzowie",
]

fig2, ax3 = plt.subplots(figsize=(12, 6))
set_dark(ax3,
         title="Przyrost F1 po każdym kroku preprocessingu",
         xlabel="Krok",
         ylabel="ΔF1 (zmiana vs. poprzedni etap)")
fig2.patch.set_facecolor(BG)

n_models = len([m for m in SELECTED_MODELS if m in all_results])
bar_w = 0.18
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w
step_x = np.arange(len(delta_labels))

for idx, (model_name, color) in enumerate(MODEL_COLORS.items()):
    if model_name not in all_results:
        continue
    deltas = [
        all_results[model_name][i]["F1"] - all_results[model_name][i-1]["F1"]
        for i in range(1, len(STAGES))
    ]
    bars = ax3.bar(step_x + offsets[idx], deltas, width=bar_w * 0.88,
                   color=color, alpha=0.85, label=model_name,
                   edgecolor=BG, lw=0.4)
    for b, v in zip(bars, deltas):
        va = "bottom" if v >= 0 else "top"
        offset_y = 0.003 if v >= 0 else -0.003
        ax3.text(b.get_x() + b.get_width() / 2, v + offset_y,
                 f"{v:+.3f}", ha="center", va=va, fontsize=7.5,
                 color=color, fontweight="bold")

ax3.axhline(0, color=WHITE, lw=0.8, alpha=0.5)
ax3.set_xticks(step_x)
ax3.set_xticklabels(delta_labels, color=WHITE, fontsize=9)
ax3.legend(fontsize=9, facecolor="#1A1A1A", edgecolor="#444",
           labelcolor=WHITE, framealpha=0.9, loc="upper right")

fig2.tight_layout()
path2 = "wyniki/ablation_delta_f1.png"
fig2.savefig(path2, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[wykres] {path2}")


# ════════════════════════════════════════════════════════════════════════
# Wykres 3: Heatmapa F1 — modele × etapy
# ════════════════════════════════════════════════════════════════════════
fig3, ax4 = plt.subplots(figsize=(12, 4))
fig3.patch.set_facecolor(BG)
ax4.set_facecolor(PANEL)

models_list = [m for m in SELECTED_MODELS if m in all_results]
matrix = np.array([
    [all_results[m][s]["F1"] for s in range(len(STAGES))]
    for m in models_list
])

im = ax4.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=1.0)
cbar = fig3.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors=WHITE, labelsize=8)
cbar.set_label("F1", color=WHITE, fontsize=9)

ax4.set_xticks(range(len(STAGES)))
ax4.set_xticklabels([s["short"] for s in STAGES], color=WHITE, fontsize=8)
ax4.set_yticks(range(len(models_list)))
ax4.set_yticklabels(models_list, color=WHITE, fontsize=9)
ax4.set_title("Heatmapa F1: modele × etapy preprocessingu",
              color=GOLD, fontsize=12, fontweight="bold", pad=10)
for i in range(len(models_list)):
    for j in range(len(STAGES)):
        v = matrix[i, j]
        ax4.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=8, color="black" if v > 0.65 else WHITE,
                 fontweight="bold")

fig3.tight_layout()
path3 = "wyniki/ablation_heatmap.png"
fig3.savefig(path3, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[wykres] {path3}")

print("\n✓ Gotowe! Wyniki ablacji zapisane w ./wyniki/")
print("  ablation_progression.png — liniowy F1 i AUC przez etapy")
print("  ablation_delta_f1.png    — przyrost F1 po każdym kroku")
print("  ablation_heatmap.png     — heatmapa F1: modele × etapy")
