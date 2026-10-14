"""Random forest and XGBoost learners on the fused vector z (Section 3.7).

Both are tuned on the validation partition by AUC over a small grid; the
selected settings are returned so they can be written to Supplementary Table S2.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def fit_rf(z_tr, y_tr, z_va, y_va, grid: dict, seed: int = 0):
    best = (-1.0, None, None)
    for mf, leaf in itertools.product(grid["max_features"], grid["min_samples_leaf"]):
        rf = RandomForestClassifier(n_estimators=int(grid["n_estimators"]), max_features=mf,
                                    min_samples_leaf=int(leaf), oob_score=True, n_jobs=-1,
                                    random_state=seed, class_weight="balanced")
        rf.fit(z_tr, y_tr)
        auc = roc_auc_score(y_va, rf.predict_proba(z_va)[:, 1])
        if auc > best[0]:
            best = (auc, rf, {"max_features": mf, "min_samples_leaf": leaf, "n_estimators": grid["n_estimators"]})
    return best[1], best[2], best[0]


def rf_train_probability(rf: RandomForestClassifier) -> np.ndarray:
    """Out-of-bag probability for the training rows (Section 3.7, 'OOB-calibrated')."""
    p = rf.oob_decision_function_[:, 1]
    return np.where(np.isnan(p), 0.5, p)


def fit_xgb(z_tr, y_tr, z_va, y_va, grid: dict, seed: int = 0):
    from xgboost import XGBClassifier
    best = (-1.0, None, None)
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    for depth, lr, sub, col in itertools.product(grid["max_depth"], grid["learning_rate"],
                                                 grid["subsample"], grid["colsample_bytree"]):
        clf = XGBClassifier(n_estimators=int(grid["n_estimators"]), max_depth=int(depth),
                            learning_rate=float(lr), subsample=float(sub), colsample_bytree=float(col),
                            objective="binary:logistic", eval_metric="logloss", random_state=seed,
                            scale_pos_weight=spw, early_stopping_rounds=int(grid.get("early_stopping_rounds", 30)),
                            n_jobs=-1, verbosity=0)
        clf.fit(z_tr, y_tr, eval_set=[(z_va, y_va)], verbose=False)
        auc = roc_auc_score(y_va, clf.predict_proba(z_va)[:, 1])
        if auc > best[0]:
            best = (auc, clf, {"max_depth": depth, "learning_rate": lr, "subsample": sub,
                               "colsample_bytree": col, "best_iteration": int(clf.best_iteration)})
    return best[1], best[2], best[0]
