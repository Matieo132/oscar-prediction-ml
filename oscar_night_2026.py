"""
oscar_night_2026.py
-------------------
Scenariusz "Oscar night":
  - Podczas gali znane są już wyniki innych kategorii Oscarów
    (Reżyseria, Scenariusz, Montaż, Zdjęcia)
  - Model przewiduje który film zdobędzie Oscara za NAJLEPSZY FILM

Generuje 3 wykresy:
  1. wyniki/oscar_night_top5.png         — porównanie top 5 klasyfikatorów
  2. wyniki/oscar_night_predictions.png  — predykcja 2024/2025/2026
  3. wyniki/oscar_night_features.png     — ważność cech

Uruchomienie:
    python oscar_night_2026.py
"""

import os, sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from models import get_models
from oscar_config import lbl
from oscar_data import load_oscar_night, NUMERIC_IDX
from oscar_features import get_importances, compute_ame, fill_numeric_ame
from oscar_plots import plot_top5, plot_predictions, plot_features

os.makedirs("wyniki", exist_ok=True)

# ── 1. Trenuj modele na całym zbiorze ────────────────────────────────────
print("=== Trenuję modele (tryb Oscar night, cały zbiór) ===")
train_df, _, feature_cols, TARGET = load_oscar_night()

X = train_df[feature_cols].values
y = train_df[TARGET].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr[:, NUMERIC_IDX] = scaler.fit_transform(X_tr[:, NUMERIC_IDX])
X_te[:, NUMERIC_IDX] = scaler.transform(X_te[:, NUMERIC_IDX])

results = []
for name, model in get_models(random_state=42).items():
    print(f"  {name}")
    model.fit(X_tr, y_tr)
    yp    = model.predict(X_te)
    yprob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    results.append({
        "Model":     name,
        "Bal.Acc":   round(balanced_accuracy_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_te, yprob), 4) if yprob is not None else np.nan,
    })

res_df = pd.DataFrame(results)
res_df.to_csv("wyniki/oscar_night_metryki.csv", index=False)
print("\nTop 10 wg F1:")
print(res_df.nlargest(10, "F1")[["Model", "Bal.Acc", "F1", "AUC"]].to_string(index=False))

plot_top5(res_df)

# ── 2. Predykcja 2024 / 2025 / 2026 (leave-one-year-out) ────────────────
best_name  = res_df.nlargest(1, "F1")["Model"].values[0]
PRED_YEARS = [2024, 2025, 2026]
print(f"\n=== Predykcja {'/'.join(map(str, PRED_YEARS))} — model: {best_name} ===")

year_results = {}
for yr in PRED_YEARS:
    train_yr, pred_yr, fc_yr, _ = load_oscar_night(exclude_year=yr)

    Xtr = train_yr[fc_yr].values
    ytr = train_yr[TARGET].values
    Xpr = pred_yr[fc_yr].values

    sc_yr = StandardScaler()
    Xtr[:, NUMERIC_IDX] = sc_yr.fit_transform(Xtr[:, NUMERIC_IDX])
    Xpr[:, NUMERIC_IDX] = sc_yr.transform(Xpr[:, NUMERIC_IDX])

    model_yr = get_models(random_state=42)[best_name]
    model_yr.fit(Xtr, ytr)
    probs = model_yr.predict_proba(Xpr)[:, 1]

    pred_yr = pred_yr.copy()
    pred_yr["prob"]   = probs
    pred_yr["winner"] = pred_yr[TARGET]
    year_results[yr]  = pred_yr.sort_values("prob", ascending=False)

    correct = year_results[yr].iloc[0]["winner"] == 1
    print(f"\n  {yr} — {'✓ TRAFNA' if correct else '✗ BŁĘDNA'} predykcja:")
    for _, r in year_results[yr].iterrows():
        tag = " ✓ ZWYCIĘZCA" if r["winner"] == 1 else ""
        fav = " ★ FAWORYT"   if r["prob"] == probs.max() else ""
        print(f"    {r['prob']:.1%}  {r['Movie']}{tag}{fav}")

plot_predictions(year_results, best_name, PRED_YEARS)

# ── 3. Feature importance ────────────────────────────────────────────────
fi_model = get_models(random_state=42)[best_name]
fi_model.fit(X_tr, y_tr)

importances, imp_label, method_note = get_importances(fi_model, X_te, y_te, feature_cols)
print(f"\n[feature importance] metoda: {method_note} dla modelu: {best_name}")

ame_dict = compute_ame(fi_model, X_tr, feature_cols, NUMERIC_IDX, importances)
ame_dict = fill_numeric_ame(ame_dict, importances, feature_cols, NUMERIC_IDX)

ame_series = pd.Series({lbl(k): v for k, v in ame_dict.items()
                        if v is not None and v > 0})
imp = ame_series.nlargest(20).sort_values()

plot_features(imp, best_name, ame_series)

print("\n✓ Gotowe! Wykresy zapisane w ./wyniki/")
print("  oscar_night_top5.png         — porównanie top 5 klasyfikatorów")
print("  oscar_night_predictions.png  — predykcja 2024/2025/2026")
print("  oscar_night_features.png     — ważność cech")
print("  oscar_night_metryki.csv      — pełna tabela metryk")
