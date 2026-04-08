"""
visualizations.py
─────────────────
All project visualisations in one module:
  1. ROC Curves (all 3 models)
  2. Confusion Matrices (best model – default vs adaptive threshold)
  3. SHAP Feature Importance (bar)
  4. Threshold vs Recall/Precision/F-beta sweep
  5. Combined Feature Importance (RF built-in + SHAP)
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)


PALETTE = {
    "Logistic Regression": "#3498db",
    "Random Forest"      : "#2ecc71",
    "XGBoost"            : "#e74c3c",
}
NEUTRAL  = "#95a5a6"
ACCENT   = "#e74c3c"
FIGSIZE  = (12, 5)


# ─────────────────────────────────────────────────────────────────────────────
def plot_all(models: dict, best_model, X_test, y_test,
             shap_values, feature_names, threshold_df, best_threshold):
    """Entry point – calls every individual plot function."""
    os.makedirs("outputs", exist_ok=True)

    plot_roc_curves(models,      X_test, y_test)
    plot_confusion_matrices(best_model, X_test, y_test, best_threshold)
    plot_threshold_sweep(threshold_df, best_threshold)
    plot_shap_bar(shap_values, feature_names)
    plot_feature_importance_comparison(models, shap_values, feature_names)

    print("       All visualisations saved to outputs/")


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_proba     = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=PALETTE.get(name, NEUTRAL),
                lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate",  fontsize=12)
    ax.set_ylabel("True Positive Rate",   fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14, pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")

    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("       Plot saved → outputs/roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(model, X_test, y_test, best_threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default  = (y_proba >= 0.50).astype(int)
    y_pred_adaptive = (y_proba >= best_threshold).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    labels = ["Low Risk", "High Risk"]

    for ax, y_pred, title in zip(
        axes,
        [y_pred_default, y_pred_adaptive],
        [f"Default Threshold (0.50)", f"Adaptive Threshold ({best_threshold:.3f})"]
    ):
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=11, pad=8)
        # Highlight false negatives (FN) cell in red
        ax.texts[2].set_color("white")   # TP

    fig.suptitle("Confusion Matrices – XGBoost", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("       Plot saved → outputs/confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_threshold_sweep(threshold_df: pd.DataFrame, best_threshold: float):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(threshold_df["threshold"], threshold_df["recall"],
            color="#e74c3c", lw=2, label="Recall (Sensitivity)")
    ax.plot(threshold_df["threshold"], threshold_df["precision"],
            color="#3498db", lw=2, label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["f1"],
            color="#2ecc71", lw=2, label="F1-Score")
    ax.plot(threshold_df["threshold"], threshold_df["fbeta"],
            color="#9b59b6", lw=2, linestyle="--", label="F-beta (β=2)")

    ax.axvline(best_threshold, color="black", lw=1.5, linestyle=":",
               label=f"Optimal threshold = {best_threshold:.3f}")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Sweep – Recall / Precision / F-scores", fontsize=14, pad=10)
    ax.legend(fontsize=10)
    ax.set_xlim([0.05, 0.95]);  ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    # Shade the "safe zone" (recall > 0.80)
    recall_vals = threshold_df["recall"].values
    thresh_vals = threshold_df["threshold"].values
    mask = recall_vals >= 0.80
    if mask.any():
        ax.axvspan(thresh_vals[mask].min(), thresh_vals[mask].max(),
                   alpha=0.08, color="#e74c3c",
                   label="Recall ≥ 0.80 zone")

    plt.tight_layout()
    plt.savefig("outputs/threshold_sweep.png", dpi=150)
    plt.close()
    print("       Plot saved → outputs/threshold_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_bar(shap_values, feature_names, top_n: int = 15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx_sort = np.argsort(mean_abs)[-top_n:][::-1]

    feat_sorted  = [feature_names[i] for i in idx_sort]
    shap_sorted  = mean_abs[idx_sort]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = [ACCENT if i < 5 else "#aab7b8" for i in range(len(feat_sorted))]
    ax.barh(feat_sorted[::-1], shap_sorted[::-1],
            color=colors[::-1], edgecolor="none", height=0.7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/shap_feature_importance.png", dpi=150)
    plt.close()
    print("       Plot saved → outputs/shap_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance_comparison(models: dict, shap_values, feature_names,
                                        top_n: int = 12):
    """
    Side-by-side: Random Forest built-in importance vs XGBoost SHAP.
    """
    rf    = models.get("Random Forest")
    xgb   = models.get("XGBoost")

    if rf is None or xgb is None:
        return

    rf_imp   = pd.Series(rf.feature_importances_,  index=feature_names)
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)

    # Normalise to [0, 1]
    rf_imp   = rf_imp   / rf_imp.max()
    shap_imp = shap_imp / shap_imp.max()

    top_feats = shap_imp.nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, series, title, color in zip(
        axes,
        [rf_imp[top_feats],   shap_imp[top_feats]],
        ["Random Forest\n(Built-in Importance)", "XGBoost\n(Mean |SHAP|)"],
        ["#3498db",           "#e74c3c"],
    ):
        sorted_series = series.sort_values()
        ax.barh(sorted_series.index, sorted_series.values,
                color=color, edgecolor="none", height=0.7, alpha=0.85)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Normalised Importance", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim([0, 1.1])

    fig.suptitle("Feature Importance Comparison: RF vs SHAP", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("       Plot saved → outputs/feature_importance_comparison.png")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator    import generate_synthetic_data, preprocess_data
    from model_training    import train_models, evaluate_models
    from threshold_optimizer import optimize_threshold
    from explainability    import run_shap_analysis

    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fn = preprocess_data(df)
    mdls = train_models(X_tr, y_tr)
    res, best = evaluate_models(mdls, X_te, y_te)
    bt, tdf   = optimize_threshold(best, X_te, y_te)
    sv, ex    = run_shap_analysis(best, X_tr, X_te, fn)
    plot_all(mdls, best, X_te, y_te, sv, fn, tdf, bt)
