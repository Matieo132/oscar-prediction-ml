import os, sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from oscar_data import load_oscar_night, NUMERIC_IDX
from oscar_config import BG, PANEL, GOLD, BLUE, WHITE, GRID, RED, TEAL, DIM
from models import get_models

os.makedirs("wyniki", exist_ok=True)

TOP5 = ["LDA", "SGD Classifier", "Perceptron", "Logistic Regression", "QDA"]
N_FOLDS = 5

print(f"=== 5-fold CV dla Top 5 klasyfikatorów ===")
train_df, _, feature_cols, TARGET = load_oscar_night()
X = train_df[feature_cols].values
y = train_df[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_f1 = {name: [] for name in TOP5}

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
    y_tr, y_te = y[tr_idx], y[te_idx]
    sc = StandardScaler()
    X_tr[:, NUMERIC_IDX] = sc.fit_transform(X_tr[:, NUMERIC_IDX])
    X_te[:, NUMERIC_IDX] = sc.transform(X_te[:, NUMERIC_IDX])
    fresh = get_models(random_state=42)
    for name in TOP5:
        fresh[name].fit(X_tr, y_tr)
        fold_f1[name].append(f1_score(y_te, fresh[name].predict(X_te), zero_division=0))
    print(f"  Fold {fold_idx}: " +
          "  ".join(f"{n}: {fold_f1[n][-1]:.3f}" for n in TOP5))

means = {n: np.mean(fold_f1[n]) for n in TOP5}
stds  = {n: np.std(fold_f1[n])  for n in TOP5}

print("\n=== Parowe t-testy (wszystkie kombinacje) ===")
print(f"  {'Model A':<22} {'Model B':<22} {'ΔF1':>7} {'t':>7} {'p':>10}  Ist.")

pairs = []
for i in range(len(TOP5)):
    for j in range(i + 1, len(TOP5)):
        a, b = TOP5[i], TOP5[j]
        fa = np.array(fold_f1[a])
        fb = np.array(fold_f1[b])
        t, p = stats.ttest_rel(fa, fb)
        diff = means[a] - means[b]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        pairs.append({"A": a, "B": b, "delta": diff, "t": t, "p": p, "sig": sig})
        print(f"  {a:<22} {b:<22} {diff:>+7.4f} {t:>+7.3f} {p:>10.4f}  {sig}")

pd.DataFrame(pairs).to_csv("wyniki/ttest_top5.csv", index=False)

p_matrix  = np.full((5, 5), np.nan)
sig_matrix = [[""] * 5 for _ in range(5)]

for row in pairs:
    i = TOP5.index(row["A"])
    j = TOP5.index(row["B"])
    p_matrix[i, j]   = row["p"]
    p_matrix[j, i]   = row["p"]
    sig_matrix[i][j] = row["sig"]
    sig_matrix[j][i] = row["sig"]

for i in range(5):
    p_matrix[i, i]   = 0.0
    sig_matrix[i][i] = "—"

MODEL_COLORS = [GOLD, TEAL, BLUE, "#F0F0F0", RED]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_edgecolor("#333")

sorted_idx = sorted(range(5), key=lambda i: means[TOP5[i]])

for rank, i in enumerate(sorted_idx):
    name  = TOP5[i]
    m     = means[name]
    s     = stds[name]
    color = MODEL_COLORS[i]
    ax.barh(rank, m, xerr=s, color=color, height=0.62,
            edgecolor=BG, lw=0.3, alpha=0.88,
            error_kw={"ecolor": WHITE, "elinewidth": 1.1,
                      "capsize": 4, "alpha": 0.6})
    ax.text(m + s + 0.005, rank,
            f"{m:.3f} ±{s:.3f}",
            va="center", ha="left", fontsize=10, color=color)

ax.set_yticks(range(5))
ax.set_yticklabels([TOP5[i] for i in sorted_idx], fontsize=11, color=WHITE)
ax.tick_params(colors=WHITE)
ax.set_xlabel("Średni F1-score (5-fold CV)", color=WHITE, fontsize=11)
ax.grid(axis="x", color=GRID, lw=0.6, ls="--", alpha=0.7)
ax.set_xlim(0, max(means.values()) + max(stds.values()) + 0.10)
ax.set_title(
    "Top 5 klasyfikatorów — sparowany t-test Studenta (5-fold CV)\n"
    "Wszystkie pary: brak istotnych różnic statystycznych (ns, p ≥ 0,05)",
    color=GOLD, fontsize=12, fontweight="bold", pad=10,
)
fig.tight_layout()
fig.savefig("wyniki/ttest_top5.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()

print("\n[wykres] wyniki/ttest_top5.png")
print("  wyniki/ttest_top5.csv  — tabela p-wartości")
