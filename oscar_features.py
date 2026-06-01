"""
oscar_features.py
-----------------
Wyciąganie ważności cech i obliczanie Average Marginal Effect (AME).
"""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid


def get_importances(model, X_te, y_te, feature_cols):
    """
    Zwraca (importances, imp_label, method_note) — działa dla drzew,
    modeli liniowych i fallbacku permutation importance.
    """
    if hasattr(model, "feature_importances_"):
        return (
            model.feature_importances_,
            "Ważność cechy (Gini)",
            "feature_importances_",
        )

    if hasattr(model, "coef_"):
        return (
            np.abs(model.coef_[0]),
            "|Współczynnik| modelu (po standaryzacji)",
            "|coef_|",
        )

    if hasattr(model, "estimators_"):
        try:
            imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
            return (
                imps,
                "Ważność cechy (uśr. Gini, bazowe estymatory)",
                "mean(estimators_.feature_importances_)",
            )
        except AttributeError:
            pass

    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_te, y_te, n_repeats=20,
                               random_state=42, scoring="f1")
    return (
        r.importances_mean,
        "Ważność cechy (permutation importance, F1)",
        "permutation_importance",
    )


def compute_ame(model, X, feature_cols, numeric_idx, importances):
    """
    Average Marginal Effect — zmiana P(win) w pp przy zmianie cechy 0→1.
    Dla cech numerycznych używa znormalizowanych importances jako proxy.
    """
    ame = {}

    try:
        coef  = model.coef_[0]
        inter = model.intercept_[0]
    except AttributeError:
        total = importances.sum()
        for i, f in enumerate(feature_cols):
            ame[f] = importances[i] / total * 100 if total > 0 else 0
        return ame

    log_odds_base = X @ coef + inter
    p_base = sigmoid(log_odds_base)

    for i, feat in enumerate(feature_cols):
        if i in numeric_idx:
            ame[feat] = None
        else:
            X_flip = X.copy()
            X_flip[:, i] = 1 - X_flip[:, i]
            p_flip = sigmoid(X_flip @ coef + inter)
            delta = np.where(
                X[:, i] == 0,
                p_flip - p_base,
                p_base - p_flip,
            )
            ame[feat] = float(np.mean(delta) * 100)

    return ame


def fill_numeric_ame(ame_dict, importances, feature_cols, numeric_idx):
    """
    Wypełnia None dla cech numerycznych znormalizowanymi importances
    przeskalowanymi do zakresu binarnych AME.
    """
    num_imps = {feature_cols[i]: importances[i] for i in numeric_idx}
    num_total = sum(num_imps.values())
    max_ame = max((v for v in ame_dict.values() if v is not None), default=5)

    for i in numeric_idx:
        f = feature_cols[i]
        ame_dict[f] = (
            num_imps[f] / num_total * max_ame * 0.6
            if num_total > 0 else 0
        )

    return ame_dict
