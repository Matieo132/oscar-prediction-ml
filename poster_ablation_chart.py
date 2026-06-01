"""
poster_ablation_chart.py
------------------------
Generuje jeden, zwięzły wykres ablation study do plakatu.
Zastępuje 'Porównanie modeli — Top 5' (poziome słupki).

Uruchomienie:
    python poster_ablation_chart.py
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

# ── Definicje etapów ─────────────────────────────────────────────────────
STAGES = [
    {"label": "Baza\n(data leakage)",  "fix_leakage": False, "use_pre_oscar": False,
     "use_oscar_night": False, "use_genres": False, "use_critic": False},
    {"label": "Naprawa\nleakage",      "fix_leakage": True,  "use_pre_oscar": False,
     "use_oscar_night": False, "use_genres": False, "use_critic": False},
    {"label": "+ Pre-Oscar\n(BAFTA/GG/PGA)", "fix_leakage": True, "use_pre_oscar": True,
     "use_oscar_night": False, "use_genres": False, "use_critic": False},
    {"label": "+ Oscar Night\n(inne Oscary)", "fix_leakage": True, "use_pre_oscar": True,
     "use_oscar_night": True,  "use_genres": False, "use_critic": False},
    {"label": "+ Gatunki\n(one-hot)",   "fix_leakage": True,  "use_pre_oscar": True,
     "use_oscar_night": True,  "use_genres": True,  "use_critic": False},
    {"label": "+ Krytycy\nvs. widzowie", "fix_leakage": True, "use_pre_oscar": True,
     "use_oscar_night": True,  "use_genres": True,  "use_critic": True},
]

DELTA_LABELS = [
    "Korekta\nbłędnych danych",
    "Nagrody\nprzed galą",
    "Wyniki\ntej samej nocy",
    "Gatunek\nfilmu",
    "Opinie\nkrytyków",
]

MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), GOLD),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, random_state=42), BLUE),
    "Extra Trees":         (ExtraTreesClassifier(n_estimators=200, random_state=42), TEAL),
}
if XGB_AVAILABLE:
    MODELS["XGBoost"] = (XGBClassifier(n_estimators=200, random_state=42,
                                        eval_metric="logloss", verbosity=0), RED)


# ── Ładowanie danych wg etapu ────────────────────────────────────────────
def load_stage(stage, data_path="oscary_dane.csv"):
    df = pd.read_csv(data_path, sep=";")
    df["IMDb_Rating"] = df["IMDb_Rating"].apply(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan)
    for col in ["Metascore", "Rotten_Tomatoes", "Runtime_min", "IMDb_Votes", "total_awards"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PRE_OSCAR + OSCAR_NIGHT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if stage["fix_leakage"]:
        df["total_awards"] = (
            pd.to_numeric(df["total_awards"], errors="coerce").fillna(0)
            - pd.to_numeric(df[TARGET], errors="coerce").fillna(0)
        ).clip(lower=0)

    if stage["use_critic"]:
        df["critic_vs_audience"] = df["Metascore"].fillna(50) / 10 - df["IMDb_Rating"].fillna(7)
    if stage["use_genres"]:
        for g in TOP_GENRES:
            df[f"Genre_{g}"] = df["Genre"].fillna("").apply(
                lambda x: int(bool(re.search(rf"\b{g}\b", x, re.IGNORECASE))))

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

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
    return X, y, len(NUMERIC), len(feature_cols)


def evaluate_stage(X, y, n_numeric, model):
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), list(range(n_numeric)))],
        remainder="passthrough")
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=kfold,
                            scoring={"f1": "f1", "auc": "roc_auc"},
                            return_train_score=False)
    return scores["test_f1"].mean(), scores["test_f1"].std()


# ── Obliczenia ───────────────────────────────────────────────────────────
print("Obliczam ablation study (5-fold CV)…")
results = {}   # {model_name: [(f1, std), ...] per stage}
for name, (model, _) in MODELS.items():
    print(f"  {name}")
    results[name] = []
    for stage in STAGES:
        X, y, n_num, _ = load_stage(stage)
        f1, std = evaluate_stage(X, y, n_num, model)
        results[name].append((f1, std))


# ════════════════════════════════════════════════════════════════════════
# Wykres: tylko delta F1
# ════════════════════════════════════════════════════════════════════════
fig, ax_bot = plt.subplots(figsize=(11, 5.5), facecolor=BG)
ax_bot.set_facecolor(PANEL)
for sp in ax_bot.spines.values():
    sp.set_edgecolor("#333")
ax_bot.tick_params(colors=WHITE, labelsize=10)
ax_bot.grid(color=GRID, lw=0.5, ls="--", alpha=0.5)

n_m = len(MODELS)
bar_w = 0.17
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w
dx = np.arange(len(DELTA_LABELS))

for idx, (name, (_, color)) in enumerate(MODELS.items()):
    deltas = [results[name][i][0] - results[name][i - 1][0]
              for i in range(1, len(STAGES))]
    bars = ax_bot.bar(dx + offsets[idx], deltas, width=bar_w * 0.88,
                      color=color, alpha=0.88, edgecolor=BG, lw=0.4,
                      label=name)
    for b, v in zip(bars, deltas):
        va = "bottom" if v >= 0 else "top"
        dy = 0.006 if v >= 0 else -0.006
        ax_bot.text(b.get_x() + b.get_width() / 2, v + dy,
                    f"{v:+.2f}", ha="center", va=va,
                    fontsize=8.5, color=color, fontweight="bold")

ax_bot.axhline(0, color=WHITE, lw=1.0, alpha=0.5)

# Czerwone tło dla kroków bez poprawy
ax_bot.axvspan(2.5, 4.5, color=RED, alpha=0.06, zorder=0)
ax_bot.text(3.5, -0.19, "brak poprawy", color=RED,
            fontsize=9, ha="center", style="italic", alpha=0.85)

ax_bot.set_xticks(dx)
ax_bot.set_xticklabels(DELTA_LABELS, color=WHITE, fontsize=11)
ax_bot.set_ylabel("ΔF1 vs. poprzedni krok", color=WHITE, fontsize=11)
ax_bot.yaxis.label.set_color(WHITE)
ax_bot.set_ylim(-0.22, 0.26)
ax_bot.set_title(
    "Przyrost F1 po każdym kroku budowania zbioru cech\n"
    "Więcej cech nie zawsze oznacza lepszy wynik (5-fold CV, n=469)",
    color=GOLD, fontsize=12, fontweight="bold", pad=10)
ax_bot.legend(fontsize=9.5, facecolor="#1A1A1A", edgecolor="#444",
              labelcolor=WHITE, framealpha=0.9, loc="upper left")

fig.tight_layout()
path = "wyniki/poster_ablation.png"
fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n[wykres] {path}")
print("Gotowe — wstaw wyniki/poster_ablation.png do plakatu w miejsce 'Porównanie modeli — Top 5'.")
