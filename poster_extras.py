import os, sys, warnings, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from oscar_data import PRE_OSCAR, OSCAR_NIGHT, NUMERIC, TOP_GENRES, TARGET
from oscar_config import BG, PANEL, GOLD, BLUE, WHITE, GRID, RED, TEAL, DIM
from models import get_models

os.makedirs("wyniki", exist_ok=True)

N_FOLDS   = 5
NUMERIC_IDX = list(range(len(NUMERIC)))


def _preprocess(df_raw, fix_leakage=True, include_genres=False, include_critic=False):
    df = df_raw.copy()
    df["IMDb_Rating"] = df["IMDb_Rating"].apply(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan
    )
    for col in ["Metascore", "Rotten_Tomatoes", "Runtime_min", "IMDb_Votes", "total_awards"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PRE_OSCAR + OSCAR_NIGHT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if fix_leakage:
        df["total_awards"] = (
            pd.to_numeric(df["total_awards"], errors="coerce").fillna(0)
            - pd.to_numeric(df[TARGET], errors="coerce").fillna(0)
        ).clip(lower=0)

    if include_critic:
        df["critic_vs_audience"] = (
            df["Metascore"].fillna(50) / 10 - df["IMDb_Rating"].fillna(7)
        )
    if include_genres:
        for g in TOP_GENRES:
            df[f"Genre_{g}"] = df["Genre"].fillna("").apply(
                lambda x, g=g: int(bool(re.search(rf"\b{g}\b", x, re.IGNORECASE)))
            )

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def run_kfold_f1(X, y, model_names):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = {n: [] for n in model_names}
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]
        sc = StandardScaler()
        X_tr[:, NUMERIC_IDX] = sc.fit_transform(X_tr[:, NUMERIC_IDX])
        X_te[:, NUMERIC_IDX] = sc.transform(X_te[:, NUMERIC_IDX])
        fresh = get_models(random_state=42)
        for name in model_names:
            fresh[name].fit(X_tr, y_tr)
            results[name].append(
                f1_score(y_te, fresh[name].predict(X_te), zero_division=0)
            )
    return results


print("=" * 60)
print(f"Ablation study ({N_FOLDS}-fold CV)")
print("=" * 60)

df_raw = pd.read_csv("oscary_dane.csv", sep=";")

all_models   = get_models(random_state=42)
ABL_NAMES    = ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost"]
abl_names    = [n for n in ABL_NAMES if n in all_models]
GENRE_COLS   = [f"Genre_{g}" for g in TOP_GENRES]

df_leaky  = _preprocess(df_raw, fix_leakage=False)
df_fixed  = _preprocess(df_raw, fix_leakage=True)
df_genres = _preprocess(df_raw, fix_leakage=True, include_genres=True)
df_critic = _preprocess(df_raw, fix_leakage=True, include_genres=True, include_critic=True)

pre_cols  = [c for c in PRE_OSCAR   if c in df_fixed.columns]
night_cols= [c for c in OSCAR_NIGHT if c in df_fixed.columns]
y         = df_fixed[TARGET].values

stages_spec = [
    ("Korekta\nbłędnych danych",  df_leaky,  NUMERIC),
    ("Korekta\nbłędnych danych",  df_fixed,  NUMERIC),
    ("Nagrody\nprzed galą",       df_fixed,  NUMERIC + pre_cols),
    ("Wyniki\ntej samej nocy",    df_fixed,  NUMERIC + pre_cols + night_cols),
    ("Gatunek\nfilmu",            df_genres, NUMERIC + pre_cols + night_cols + GENRE_COLS),
    ("Opinie\nkrytyków",          df_critic, NUMERIC + pre_cols + night_cols + GENRE_COLS + ["critic_vs_audience"]),
]

fold_stages = []
for label, df_s, fcols in stages_spec:
    X_s = df_s[[c for c in fcols if c in df_s.columns]].values
    n_feats = X_s.shape[1]
    print(f"  {label.replace(chr(10),' '):<30} ({n_feats} cech) …", end=" ", flush=True)
    res = run_kfold_f1(X_s, y, abl_names)
    fold_stages.append(res)
    means = {n: np.mean(v) for n, v in res.items()}
    print("  ".join(f"{n}: {v:.3f}" for n, v in means.items()))

# ΔF1 (current - previous): pary (0→1), (1→2), …, (4→5)
STAGE_LABELS = [
    "Korekta\nbłędnych danych",
    "Nagrody\nprzed galą",
    "Wyniki\ntej samej nocy",
    "Gatunek\nfilmu",
    "Opinie\nkrytyków",
]
n_stages  = len(STAGE_LABELS)
delta_m   = {n: [] for n in abl_names}
delta_s   = {n: [] for n in abl_names}

for prev, curr in zip(fold_stages[:-1], fold_stages[1:]):
    for n in abl_names:
        diff = np.array(curr[n]) - np.array(prev[n])
        delta_m[n].append(float(diff.mean()))
        delta_s[n].append(float(diff.std()))

bar_w   = 0.17
colors  = [GOLD, BLUE, TEAL, RED]
x       = np.arange(n_stages)
n_m     = len(abl_names)
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor(PANEL)
fig.patch.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_edgecolor("#333")

for i, (name, color) in enumerate(zip(abl_names, colors)):
    bars = ax.bar(
        x + offsets[i], delta_m[name], width=bar_w * 0.88,
        yerr=delta_s[name],
        error_kw={"ecolor": WHITE, "elinewidth": 0.9, "capsize": 3, "alpha": 0.55},
        color=color, edgecolor=BG, lw=0.3, alpha=0.88, label=name,
    )
    for b, v in zip(bars, delta_m[name]):
        va  = "bottom" if v >= 0 else "top"
        pad = 0.006 if v >= 0 else -0.006
        ax.text(b.get_x() + b.get_width() / 2, v + pad,
                f"{v:+.2f}", ha="center", va=va,
                fontsize=8, color=color, fontweight="bold")

ax.axhline(0, color=WHITE, lw=0.9, alpha=0.5)

ax.axvspan(2.5, n_stages - 0.5, color="#300000", alpha=0.40, zorder=0)
y_min = ax.get_ylim()[0]
ax.text(n_stages - 0.55, y_min * 0.80, "brak poprawy",
        color=RED, fontsize=10, style="italic", ha="right", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(STAGE_LABELS, color=WHITE, fontsize=11)
ax.tick_params(colors=WHITE, labelsize=10)
ax.set_ylabel("ΔF1 vs. poprzedni krok", color=WHITE, fontsize=11)
ax.grid(axis="y", color=GRID, lw=0.6, ls="--", alpha=0.7)
ax.set_title(
    "Przyrost F1 po każdym kroku budowania zbioru cech\n"
    f"Więcej cech nie zawsze oznacza lepszy wynik  ({N_FOLDS}-fold CV, n=469)",
    color=GOLD, fontsize=13, fontweight="bold", pad=10,
)
ax.legend(loc="upper left", fontsize=10, facecolor="#1A1A1A",
          edgecolor="#444", labelcolor=WHITE, framealpha=0.9)
fig.tight_layout()
fig.savefig("wyniki/poster_ablation.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n[wykres] wyniki/poster_ablation.png")


print("\nGeneruję tabela_top5.png …")

metryki_path = "wyniki/oscar_night_metryki.csv"
if not os.path.exists(metryki_path):
    sys.exit(f"BRAK {metryki_path} — najpierw uruchom oscar_night_2026.py")

mdf  = pd.read_csv(metryki_path)
top5 = mdf.nlargest(5, "F1").reset_index(drop=True)

METRICS   = ["Bal.Acc", "Precision", "Recall", "F1",     "AUC"]
STD_COLS  = ["Bal.Acc_std", "Precision_std", "Recall_std", "F1_std", "AUC_std"]
COL_HDR   = ["Bal.Acc.",  "Precision", "Recall", "F1", "AUC"]
has_std   = all(c in top5.columns for c in STD_COLS)

COL_W     = [0.29, 0.142, 0.142, 0.118, 0.142, 0.166]
COL_X     = np.cumsum([0.0] + COL_W[:-1])
N_ROWS    = 1 + len(top5)
TITLE_H   = 0.12
ROW_H     = (1.0 - TITLE_H - 0.02) / N_ROWS

best_f1   = top5["F1"].max()
best_auc  = top5["AUC"].max()

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.97,
        f"Porównanie Top 5 klasyfikatorów  (n=469, {N_FOLDS}-fold CV, stratyfikowany)",
        ha="center", va="top", fontsize=13, color=GOLD, fontweight="bold")

TBL_TOP = 1.0 - TITLE_H


def _cell(ax, x, y, w, h, facecolor, edgecolor="#444", lw=0.5):
    ax.add_patch(plt.Rectangle((x, y), w, h,
                                facecolor=facecolor, edgecolor=edgecolor,
                                lw=lw, zorder=1))


hy = TBL_TOP - ROW_H
for j, (cx, cw, lbl) in enumerate(zip(COL_X, COL_W, ["Model"] + COL_HDR)):
    _cell(ax, cx, hy, cw, ROW_H, "#2A2000", edgecolor="#666", lw=0.6)
    ax.text(cx + cw / 2, hy + ROW_H / 2, lbl,
            ha="center", va="center", fontsize=11, color=GOLD,
            fontweight="bold", zorder=2)

for i, row in top5.iterrows():
    ry = TBL_TOP - ROW_H * (i + 2)
    row_bg = "#141414" if i % 2 == 0 else "#1C1C1C"

    for j, (cx, cw) in enumerate(zip(COL_X, COL_W)):
        _cell(ax, cx, ry, cw, ROW_H, row_bg)

    ax.text(COL_X[0] + COL_W[0] / 2, ry + ROW_H / 2, row["Model"],
            ha="center", va="center", fontsize=10,
            color=GOLD if i == 0 else WHITE,
            fontweight="bold" if i == 0 else "normal", zorder=2)

    for j, (metric, std_col, hdr) in enumerate(zip(METRICS, STD_COLS, COL_HDR), 1):
        val = row[metric]
        std = row[std_col] if has_std else None
        cx  = COL_X[j]
        cw  = COL_W[j]

        is_best_f1  = metric == "F1"  and abs(val - best_f1)  < 1e-6
        is_best_auc = metric == "AUC" and abs(val - best_auc) < 1e-6

        if is_best_f1:
            hi_color, txt_color = TEAL, BG
        elif is_best_auc:
            hi_color, txt_color = GOLD, BG
        else:
            hi_color, txt_color = None, WHITE

        if hi_color:
            _cell(ax, cx + 0.003, ry + 0.018, cw - 0.006, ROW_H - 0.036,
                  hi_color, edgecolor="none", lw=0)

        main = f"{val:.2f}"
        if std is not None:
            ax.text(cx + cw / 2, ry + ROW_H * 0.63, main,
                    ha="center", va="center", fontsize=10.5,
                    color=txt_color,
                    fontweight="bold" if hi_color else "normal", zorder=3)
            ax.text(cx + cw / 2, ry + ROW_H * 0.26, f"±{std:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=txt_color, alpha=0.72, zorder=3)
        else:
            ax.text(cx + cw / 2, ry + ROW_H / 2, main,
                    ha="center", va="center", fontsize=10.5,
                    color=txt_color,
                    fontweight="bold" if hi_color else "normal", zorder=3)

patches = [
    mpatches.Patch(color=GOLD, label="Najlepszy AUC"),
    mpatches.Patch(color=TEAL, label="Najlepszy F1"),
]
ax.legend(handles=patches, loc="lower right", fontsize=9,
          facecolor="#1A1A1A", edgecolor="#444", labelcolor=WHITE,
          framealpha=0.9, bbox_to_anchor=(0.99, 0.02))

fig.tight_layout()
fig.savefig("wyniki/tabela_top5.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("[wykres] wyniki/tabela_top5.png")

print("\n✓ Gotowe!")
print("  wyniki/poster_ablation.png  — ablacja cech (K-fold + słupki błędu)")
print("  wyniki/tabela_top5.png      — Top 5 klasyfikatorów (mean ± std)")
