import os, sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
from oscar_plots import plot_top5, plot_predictions, plot_features, plot_ttest

os.makedirs("wyniki", exist_ok=True)

N_FOLDS = 5
print(f"=== Trenuję modele (tryb Oscar night, {N_FOLDS}-fold CV) ===")
train_df, _, feature_cols, TARGET = load_oscar_night()

X = train_df[feature_cols].values
y = train_df[TARGET].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_scores = {
    name: {"Bal.Acc": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
    for name in get_models(random_state=42)
}

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
    y_tr, y_te = y[tr_idx], y[te_idx]

    scaler = StandardScaler()
    X_tr[:, NUMERIC_IDX] = scaler.fit_transform(X_tr[:, NUMERIC_IDX])
    X_te[:, NUMERIC_IDX] = scaler.transform(X_te[:, NUMERIC_IDX])

    print(f"\n  [Fold {fold_idx}/{N_FOLDS}]")
    for name, model in get_models(random_state=42).items():
        print(f"    {name}", end=" … ", flush=True)
        model.fit(X_tr, y_tr)
        yp    = model.predict(X_te)
        yprob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        fold_scores[name]["Bal.Acc"].append(balanced_accuracy_score(y_te, yp))
        fold_scores[name]["Precision"].append(precision_score(y_te, yp, zero_division=0))
        fold_scores[name]["Recall"].append(recall_score(y_te, yp, zero_division=0))
        fold_scores[name]["F1"].append(f1_score(y_te, yp, zero_division=0))
        fold_scores[name]["AUC"].append(
            roc_auc_score(y_te, yprob) if yprob is not None else np.nan
        )
        print("✓")

results = []
for name, scores in fold_scores.items():
    row = {"Model": name}
    for m, vals in scores.items():
        arr = np.array(vals, dtype=float)
        row[m]          = round(float(np.nanmean(arr)), 4)
        row[f"{m}_std"] = round(float(np.nanstd(arr)),  4)
    results.append(row)

res_df = pd.DataFrame(results)
res_df.to_csv("wyniki/oscar_night_metryki.csv", index=False)

print("\nTop 10 wg F1 (średnia ± odch.std po foldach):")
cols_show = ["Model", "Bal.Acc", "Bal.Acc_std", "F1", "F1_std", "AUC", "AUC_std"]
print(res_df.nlargest(10, "F1")[cols_show].to_string(index=False))

plot_top5(res_df)

best_name = res_df.nlargest(1, "F1")["Model"].values[0]
best_f1   = np.array(fold_scores[best_name]["F1"], dtype=float)

print(f"\n=== Sparowany t-test (F1, {N_FOLDS} foldów): {best_name} vs pozostałe ===")
print(f"  {'Model':<35}  {'ΔF1':>7}  {'t':>7}  {'p-wartość':>10}  Istotność")
print("  " + "-" * 70)

ttest_rows = []
for name, scores in fold_scores.items():
    if name == best_name:
        continue
    other_f1 = np.array(scores["F1"], dtype=float)
    t_stat, p_val = stats.ttest_rel(best_f1, other_f1)
    diff = float(np.nanmean(best_f1)) - float(np.nanmean(other_f1))
    sig  = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ttest_rows.append({
        "Model":          name,
        "ΔF1 (vs best)":  round(diff,   4),
        "t-statystyka":   round(t_stat,  4),
        "p-wartość":      round(p_val,   6),
        "istotność":      sig,
    })
    print(f"  {name:<35}  {diff:>+7.4f}  {t_stat:>+7.3f}  {p_val:>10.4f}  {sig}")

ttest_df = pd.DataFrame(ttest_rows).sort_values("p-wartość")
ttest_df.to_csv("wyniki/oscar_night_ttest.csv", index=False)
print(f"\n  Legenda: * p<0.05  ** p<0.01  *** p<0.001  ns = brak istotności statystycznej")
print(f"  Wyniki → wyniki/oscar_night_ttest.csv")

plot_ttest(res_df, ttest_df, best_name)

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

scaler_full = StandardScaler()
X_full = X.copy()
X_full[:, NUMERIC_IDX] = scaler_full.fit_transform(X_full[:, NUMERIC_IDX])

fi_model = get_models(random_state=42)[best_name]
fi_model.fit(X_full, y)

importances, imp_label, method_note = get_importances(fi_model, X_full, y, feature_cols)
print(f"\n[feature importance] metoda: {method_note} dla modelu: {best_name}")

ame_dict = compute_ame(fi_model, X_full, feature_cols, NUMERIC_IDX, importances)
ame_dict = fill_numeric_ame(ame_dict, importances, feature_cols, NUMERIC_IDX)

ame_series = pd.Series({lbl(k): v for k, v in ame_dict.items()
                        if v is not None and v > 0})
imp = ame_series.nlargest(20).sort_values()

plot_features(imp, best_name, ame_series)

print("\n✓ Gotowe! Wykresy zapisane w ./wyniki/")
print("  oscar_night_top5.png         — porównanie top 5 klasyfikatorów")
print("  oscar_night_predictions.png  — predykcja 2024/2025/2026")
print("  oscar_night_features.png     — ważność cech")
print("  oscar_night_metryki.csv      — pełna tabela metryk (śr. z K-fold)")
print("  oscar_night_ttest.csv        — sparowany t-test (najlepszy vs reszta)")
print("  oscar_night_ttest.png        — wizualizacja testów statystycznych")
