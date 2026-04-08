"""
explainability.py
─────────────────
SHAP-based global and local explanations for the trained XGBoost model.

Outputs
───────
  outputs/shap_values.pkl          – raw SHAP value array
  outputs/shap_summary_bar.png     – global bar chart
  outputs/shap_summary_beeswarm.png– beeswarm plot
  outputs/shap_local_<i>.png       – waterfall plot for first 3 test samples
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_train, X_test, feature_names: list,
                       max_display: int = 15, n_background: int = 100):
    """
    Compute SHAP values and save plots.

    Parameters
    ----------
    model        : fitted XGBoost (or tree-based) classifier
    X_train      : preprocessed training features (used for background)
    X_test       : preprocessed test features
    feature_names: list of feature name strings
    max_display  : how many top features to show in summary plots
    n_background : number of background samples for TreeExplainer

    Returns
    -------
    shap_values : np.ndarray  shape (n_test, n_features)
    explainer   : SHAP explainer instance
    """
    os.makedirs("outputs", exist_ok=True)

    # ── Build explainer ───────────────────────────────────────────────────
    # Use a representative background sample for speed
    bg_size  = min(n_background, X_train.shape[0])
    bg_idx   = np.random.choice(X_train.shape[0], bg_size, replace=False)
    background = X_train[bg_idx]

    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback for non-tree models (e.g., LogisticRegression)
        try:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_test)
        except Exception:
            # Last-resort model-agnostic path
            explainer = shap.Explainer(model.predict_proba, background)
            shap_values = explainer(X_test).values

    # Binary classifiers may return class-wise SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3 and shap_values.shape[-1] == 2:
        shap_values = shap_values[:, :, 1]

    print(f"       SHAP values shape : {shap_values.shape}")

    # ── Convert to DataFrames for analysis ────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv("outputs/shap_values.csv", index=False)

    # Mean absolute SHAP per feature
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    print("       Top-5 features by mean |SHAP|:")
    for feat, val in mean_abs.head(5).items():
        print(f"         {feat:<32} {val:.5f}")

    # ── Global plots ──────────────────────────────────────────────────────
    _plot_bar_summary(shap_values, X_test, feature_names, max_display)
    _plot_beeswarm(shap_values, X_test, feature_names, max_display)

    # ── Local (per-sample) waterfall plots ───────────────────────────────
    _plot_local_waterfall(shap_values, X_test, feature_names,
                           explainer, n_samples=3)

    # ── Persist ───────────────────────────────────────────────────────────
    with open("outputs/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    with open("outputs/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    return shap_values, explainer


# ─────────────────────────────────────────────────────────────────────────────
def _plot_bar_summary(shap_values, X_test, feature_names, max_display):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type="bar",
                      max_display=max_display,
                      show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_bar.png", dpi=150)
    plt.close()
    print("       Plot saved → outputs/shap_summary_bar.png")


def _plot_beeswarm(shap_values, X_test, feature_names, max_display):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      max_display=max_display,
                      show=False)
    plt.title("SHAP Beeswarm – Feature Impact Distribution", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_beeswarm.png", dpi=150)
    plt.close()
    print("       Plot saved → outputs/shap_summary_beeswarm.png")


def _plot_local_waterfall(shap_values, X_test, feature_names, explainer, n_samples=3):
    """Save per-sample waterfall (force-style) plots."""
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    for i in range(min(n_samples, len(shap_values))):
        sv   = shap_values[i]
        vals = dict(zip(feature_names, X_test[i]))

        # manual waterfall using matplotlib for broad compatibility
        sorted_idx = np.argsort(np.abs(sv))[::-1][:10]
        feat_lab   = [feature_names[j] for j in sorted_idx]
        feat_shap  = sv[sorted_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ["#e74c3c" if s > 0 else "#3498db" for s in feat_shap]
        bars    = ax.barh(feat_lab[::-1], feat_shap[::-1], color=colors[::-1], edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
        ax.set_title(f"Local Explanation – Patient {i+1}\n"
                     f"Base value: {base_value:.3f}", fontsize=12)
        for bar, val in zip(bars, feat_shap[::-1]):
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=8)
        plt.tight_layout()
        plt.savefig(f"outputs/shap_local_patient{i+1}.png", dpi=150)
        plt.close()

    print(f"       Local SHAP plots saved for {min(n_samples, len(shap_values))} patients.")


# ─────────────────────────────────────────────────────────────────────────────
def get_top_shap_features(shap_values_row, feature_names, top_n=5):
    """
    Return the top-n features and their SHAP values for a single prediction.
    Useful for the predictor module.
    """
    pairs = list(zip(feature_names, shap_values_row))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic_data, preprocess_data
    from model_training  import train_models
    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fn = preprocess_data(df)
    mdls  = train_models(X_tr, y_tr)
    sv, ex = run_shap_analysis(mdls["XGBoost"], X_tr, X_te, fn)
    print("Top features:", get_top_shap_features(sv[0], fn))
