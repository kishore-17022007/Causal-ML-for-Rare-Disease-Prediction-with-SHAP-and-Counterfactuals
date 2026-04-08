"""
model_training.py
─────────────────
Train and evaluate Logistic Regression, Random Forest, and XGBoost models.
Handles class imbalance via class_weight / scale_pos_weight where applicable.
"""

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    """
    Train three classifiers and return them in a dict.

    Parameters
    ----------
    X_train : array-like, preprocessed training features
    y_train : array-like, binary labels

    Returns
    -------
    dict  {model_name: fitted_estimator}
    """
    # Class-imbalance weight for Logistic Regression / Random Forest
    pos      = int(np.sum(y_train == 1))
    neg      = int(np.sum(y_train == 0))
    scale_pw = neg / pos           # for XGBoost

    models = {
        # ── Logistic Regression ───────────────────────────────────────────
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),

        # ── Random Forest ─────────────────────────────────────────────────
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pw,      # handles imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        print(
            "       Skipping XGBoost: optional dependency unavailable "
            f"({XGBOOST_IMPORT_ERROR})"
        )

    for name, model in models.items():
        print(f"       Training {name} …")
        model.fit(X_train, y_train)

    # ── Persist models ────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/trained_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print("       Models saved → outputs/trained_models.pkl")

    return models


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluate all models at default threshold (0.5).

    Returns
    -------
    results   : dict  {model_name: metrics_dict}
    best_model: the model with the highest ROC-AUC
    """
    results   = {}
    best_name = None
    best_auc  = -1.0

    for name, model in models.items():
        y_pred     = model.predict(X_test)
        y_proba    = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy" : accuracy_score(y_test,  y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test,    y_pred),
            "f1"       : f1_score(y_test,         y_pred),
            "roc_auc"  : roc_auc_score(y_test,   y_proba),
            "conf_mat" : confusion_matrix(y_test, y_pred),
            "y_proba"  : y_proba,
        }
        results[name] = metrics

        print(f"\n       ── {name} ──")
        print(f"         Accuracy  : {metrics['accuracy']:.4f}")
        print(f"         Precision : {metrics['precision']:.4f}")
        print(f"         Recall    : {metrics['recall']:.4f}")
        print(f"         F1-Score  : {metrics['f1']:.4f}")
        print(f"         ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"         Confusion Matrix:\n{metrics['conf_mat']}")

        if metrics["roc_auc"] > best_auc:
            best_auc  = metrics["roc_auc"]
            best_name = name

    print(f"\n       ✓ Best model by ROC-AUC: {best_name} ({best_auc:.4f})")

    # Save evaluation summary to CSV
    summary = {
        name: {k: v for k, v in m.items() if k not in ("conf_mat", "y_proba")}
        for name, m in results.items()
    }
    pd.DataFrame(summary).T.to_csv("outputs/evaluation_summary.csv")

    return results, models[best_name]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic_data, preprocess_data
    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fnames = preprocess_data(df)
    mdls = train_models(X_tr, y_tr)
    res, best = evaluate_models(mdls, X_te, y_te)
