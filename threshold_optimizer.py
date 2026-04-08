"""
threshold_optimizer.py
──────────────────────
Dynamic / adaptive threshold selection for rare-disease classification.

Instead of using the default 0.5 cut-off, we sweep over candidate
thresholds and choose the one that maximises either:
  • Recall  (minimise false negatives – missed diagnoses)
  • F-beta  (β > 1  weights recall more heavily than precision)
  • Youden's J  (balanced sensitivity + specificity)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    fbeta_score, roc_curve
)


# ─────────────────────────────────────────────────────────────────────────────
def optimize_threshold(
    model,
    X_test,
    y_test,
    strategy: str = "fbeta",      # "recall" | "fbeta" | "youden"
    beta: float   = 2.0,          # β=2 → recall twice as important as precision
    n_thresholds: int = 200,
):
    """
    Sweep over `n_thresholds` candidate thresholds and return the best one.

    Parameters
    ----------
    model       : fitted classifier with predict_proba()
    X_test      : test features (preprocessed)
    y_test      : true binary labels
    strategy    : optimisation criterion
    beta        : F-beta weight (only used when strategy="fbeta")
    n_thresholds: number of points in the sweep

    Returns
    -------
    best_threshold : float
    df_results     : pd.DataFrame with per-threshold metrics
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, n_thresholds)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        # guard against degenerate predictions
        if y_pred.sum() == 0:
            rec = prec = f1 = fbeta = youden = 0.0
        else:
            rec    = recall_score(y_test,    y_pred, zero_division=0)
            prec   = precision_score(y_test, y_pred, zero_division=0)
            f1     = f1_score(y_test,        y_pred, zero_division=0)
            fbeta  = fbeta_score(y_test,     y_pred, beta=beta, zero_division=0)

            # Youden's J = sensitivity + specificity − 1
            tn = int(((y_pred == 0) & (y_test == 0)).sum())
            fp = int(((y_pred == 1) & (y_test == 0)).sum())
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            youden = rec + specificity - 1.0

        rows.append({
            "threshold" : t,
            "recall"    : rec,
            "precision" : prec,
            "f1"        : f1,
            "fbeta"     : fbeta,
            "youden_j"  : youden,
        })

    df = pd.DataFrame(rows)

    # ── Pick best threshold ───────────────────────────────────────────────
    score_col = {
        "recall" : "recall",
        "fbeta"  : "fbeta",
        "youden" : "youden_j",
    }.get(strategy, "fbeta")

    best_idx       = df[score_col].idxmax()
    best_threshold = float(df.loc[best_idx, "threshold"])
    best_row       = df.loc[best_idx]

    print(f"       Strategy        : {strategy}  (β={beta})")
    print(f"       Optimal threshold: {best_threshold:.3f}")
    print(f"       At threshold     : recall={best_row['recall']:.3f} | "
          f"precision={best_row['precision']:.3f} | "
          f"F{beta}={best_row['fbeta']:.3f}")

    # ── Save results ─────────────────────────────────────────────────────
    df.to_csv("outputs/threshold_sweep.csv", index=False)
    _save_threshold_report(best_threshold, best_row, y_test, y_proba, strategy, beta)

    return best_threshold, df


# ─────────────────────────────────────────────────────────────────────────────
def _save_threshold_report(threshold, row, y_test, y_proba, strategy, beta):
    y_pred = (y_proba >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())

    report = (
        f"ADAPTIVE THRESHOLD REPORT\n"
        f"=========================\n"
        f"Strategy          : {strategy}\n"
        f"Beta              : {beta}\n"
        f"Optimal Threshold : {threshold:.4f}\n\n"
        f"Confusion Matrix  :\n"
        f"  TP={tp}  FP={fp}\n"
        f"  FN={fn}  TN={tn}\n\n"
        f"Recall            : {row['recall']:.4f}\n"
        f"Precision         : {row['precision']:.4f}\n"
        f"F1                : {row['f1']:.4f}\n"
        f"F-beta            : {row['fbeta']:.4f}\n"
        f"False Negatives   : {fn}  ← missed diagnoses (want LOW)\n"
    )

    with open("outputs/threshold_report.txt", "w") as f:
        f.write(report)
    print("       Report saved → outputs/threshold_report.txt")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic_data, preprocess_data
    from model_training  import train_models
    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fn = preprocess_data(df)
    mdls = train_models(X_tr, y_tr)
    xgb  = mdls["XGBoost"]
    best_t, df_sweep = optimize_threshold(xgb, X_te, y_te)
    print(df_sweep.head())
