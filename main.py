"""
=============================================================================
Causal-Aware ML Framework with Counterfactual Explainability
and Adaptive Thresholding for Rare Disease Risk Prediction
=============================================================================
DISCLAIMER: This system is for research/decision-support purposes only.
It is NOT a substitute for professional medical diagnosis or treatment.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# ── Ensure all modules are importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_generator      import preprocess_data, load_kaggle_data
from model_training      import train_models, evaluate_models
from threshold_optimizer import optimize_threshold
from explainability      import run_shap_analysis
from counterfactuals     import generate_counterfactuals
from causal_analysis     import run_causal_analysis
from visualizations      import plot_all
from predictor           import predict_patient

import pandas as pd


def main():
    print("=" * 70)
    print("  CAUSAL-AWARE ML FRAMEWORK – RARE DISEASE RISK PREDICTION")
    print("=" * 70)

    # ── 1. REAL DATA LOAD & PREPROCESSING ─────────────────────────────────
    data_path = os.getenv("REAL_DATA_PATH", "data/kaggle_data.csv")
    print(f"\n[1/7]  Loading Kaggle dataset from {data_path} …")
    df_raw = load_kaggle_data(data_path)
    df_raw.to_csv("data/raw_data.csv", index=False)
    print(f"       Dataset shape : {df_raw.shape}")
    print(f"       Class balance  : {df_raw['disease_risk'].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test, preprocessor, feature_names = \
        preprocess_data(df_raw, measurement_noise_std=2.2)

    # ── 2. MODEL TRAINING ─────────────────────────────────────────────────
    print("\n[2/7]  Training models (LR / RF / XGBoost) …")
    models = train_models(X_train, y_train)

    # ── 3. EVALUATION ─────────────────────────────────────────────────────
    print("\n[3/7]  Evaluating all models …")
    results, best_model = evaluate_models(models, X_test, y_test)

    print("\n       ── Per-Model Results ──")
    for name, metrics in results.items():
        print(f"       {name:<20} | AUC={metrics['roc_auc']:.3f} "
              f"| Recall={metrics['recall']:.3f} "
              f"| F1={metrics['f1']:.3f}")

    # ── 4. ADAPTIVE THRESHOLD ─────────────────────────────────────────────
    print("\n[4/7]  Optimising decision threshold (recall-focused) …")
    best_threshold, threshold_df = optimize_threshold(best_model, X_test, y_test)
    print(f"       Optimal threshold : {best_threshold:.3f}")

    # ── 5. SHAP EXPLAINABILITY ────────────────────────────────────────────
    print("\n[5/7]  Running SHAP global + local explanations …")
    shap_values, explainer = run_shap_analysis(
        best_model, X_train, X_test, feature_names
    )

    # ── 6. COUNTERFACTUAL EXPLANATIONS ───────────────────────────────────
    print("\n[6/7]  Generating DiCE counterfactual recommendations …")
    cf_examples = generate_counterfactuals(
        best_model, preprocessor, df_raw, feature_names
    )

    # ── 7. CAUSAL ANALYSIS ───────────────────────────────────────────────
    print("\n[7/7]  Running basic causal analysis (DoWhy) …")
    causal_results = run_causal_analysis(df_raw)

    # ── VISUALISATIONS ────────────────────────────────────────────────────
    print("\n[+]   Generating visualisations …")
    plot_all(models, best_model, X_test, y_test,
             shap_values, feature_names, threshold_df, best_threshold)

    # ── DEMO PREDICTION ───────────────────────────────────────────────────
    print("\n[+]   Demo prediction for a sample patient …")
    sample_patient = {
        "age": 55,
        "gender": 1,
        "systolic_bp": 148,
        "diastolic_bp": 92,
        "cholesterol": 240,
        "glucose": 130,
        "bmi": 29.5,
        "fatigue_score": 7,
        "pain_score": 5,
        "family_history": 1,
        "smoking": 1,
        "physical_activity": 2,
    }

    prediction = predict_patient(
        sample_patient, best_model, preprocessor,
        feature_names, shap_values, explainer,
        best_threshold, X_test
    )

    print("\n" + "─" * 60)
    print("  PATIENT RISK REPORT")
    print("─" * 60)
    print(f"  Decision         : {prediction['decision']}")
    print(f"  Risk Probability : {prediction['probability']:.1%}")
    print(f"  Threshold Used   : {prediction['threshold']:.3f}")
    print("\n  Top Contributing Factors:")
    for feat, val in prediction["top_features"]:
        direction = "↑ risk" if val > 0 else "↓ risk"
        print(f"    • {feat:<28} SHAP={val:+.4f}  [{direction}]")
    print("─" * 60)

    print("\n✓  All outputs saved to outputs/")
    print("✓  Pipeline complete.\n")


if __name__ == "__main__":
    os.makedirs("data",    exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    main()
