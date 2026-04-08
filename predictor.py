"""
predictor.py
────────────
Patient-level prediction function.

Given a dict of raw patient features, produces:
  • Risk decision     (High Risk / Low Risk)
  • Probability score
  • Adaptive threshold used
  • Top SHAP feature contributions
  • A plain-text clinical summary
"""

import numpy as np
import pandas as pd
from explainability import get_top_shap_features


# ─────────────────────────────────────────────────────────────────────────────
def predict_patient(
    patient_dict: dict,
    model,
    preprocessor,
    feature_names: list,
    shap_values_test: np.ndarray,
    explainer,
    threshold: float,
    X_test: np.ndarray,
    top_n_features: int = 8,
) -> dict:
    """
    Generate a full prediction report for a single patient.

    Parameters
    ----------
    patient_dict      : raw feature values (un-preprocessed)
    model             : fitted XGBoost (or any) classifier
    preprocessor      : fitted ColumnTransformer
    feature_names     : list of feature name strings
    shap_values_test  : SHAP values for the test set (used for reference)
    explainer         : shap.TreeExplainer instance
    threshold         : adaptive decision threshold
    X_test            : preprocessed test array (for background reference)
    top_n_features    : number of features to include in report

    Returns
    -------
    dict with keys:
        decision, probability, threshold,
        top_features, clinical_summary
    """
    # ── 1. Build raw DataFrame ───────────────────────────────────────────
    raw = pd.DataFrame([patient_dict])

    # ── 2. Feature engineering (mirrors data_generator) ──────────────────
    raw = _apply_feature_engineering(raw)

    # ── 3. Preprocess ────────────────────────────────────────────────────
    X_proc = preprocessor.transform(raw[feature_names])

    # ── 4. Predict ───────────────────────────────────────────────────────
    probability = float(model.predict_proba(X_proc)[0, 1])
    decision    = "⚠️  HIGH RISK" if probability >= threshold else "✅  LOW RISK"

    # ── 5. Local SHAP ────────────────────────────────────────────────────
    try:
        shap_vals = explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        local_shap = shap_vals[0]
    except Exception:
        # Fallback: find nearest test sample by L2 distance
        dists      = np.linalg.norm(X_test - X_proc, axis=1)
        nearest_i  = int(np.argmin(dists))
        local_shap = shap_values_test[nearest_i]

    top_features = get_top_shap_features(local_shap, feature_names, top_n=top_n_features)

    # ── 6. Clinical summary (human-readable) ─────────────────────────────
    summary = _build_clinical_summary(patient_dict, probability, decision,
                                       threshold, top_features)

    # Save to file
    with open("outputs/patient_prediction_report.txt", "w") as f:
        f.write(summary)

    return {
        "decision"        : decision,
        "probability"     : probability,
        "threshold"       : threshold,
        "top_features"    : top_features,
        "clinical_summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_med = {
        "cholesterol" : 200, "glucose": 95, "bmi": 25,
        "systolic_bp" : 120, "diastolic_bp": 80,
    }
    for col, med in num_med.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    df["metabolic_risk_index"] = (
        (df["cholesterol"] / 200) +
        (df["glucose"]     / 100) +
        (df["bmi"]         / 25)
    ) / 3

    df["cv_stress"] = (
        df["systolic_bp"] / 120 +
        df["diastolic_bp"] / 80
    ) / 2

    df["symptom_burden"]        = df["fatigue_score"] + df["pain_score"]
    df["genetic_lifestyle_risk"] = (
        df["family_history"] * 2 +
        df["smoking"] +
        (3 - df["physical_activity"])
    )

    cat_cols = ["gender", "family_history", "smoking",
                "physical_activity", "genetic_lifestyle_risk"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _build_clinical_summary(patient: dict, prob: float, decision: str,
                              threshold: float, top_features: list) -> str:
    activity_map = {0: "None", 1: "Low", 2: "Moderate", 3: "High"}
    lines = [
        "=" * 60,
        "  PATIENT RISK PREDICTION REPORT",
        "=" * 60,
        "",
        "  Patient Profile:",
        f"    Age                 : {patient.get('age', 'N/A')}",
        f"    Gender              : {'Male' if patient.get('gender') == 1 else 'Female'}",
        f"    BMI                 : {patient.get('bmi', 'N/A')}",
        f"    Systolic BP         : {patient.get('systolic_bp', 'N/A')} mmHg",
        f"    Cholesterol         : {patient.get('cholesterol', 'N/A')} mg/dL",
        f"    Glucose             : {patient.get('glucose', 'N/A')} mg/dL",
        f"    Fatigue Score       : {patient.get('fatigue_score', 'N/A')} / 10",
        f"    Pain Score          : {patient.get('pain_score', 'N/A')} / 10",
        f"    Family History      : {'Yes' if patient.get('family_history') else 'No'}",
        f"    Smoking             : {'Yes' if patient.get('smoking') else 'No'}",
        f"    Physical Activity   : {activity_map.get(patient.get('physical_activity'), 'N/A')}",
        "",
        "─" * 60,
        f"  RISK DECISION      : {decision}",
        f"  Risk Probability   : {prob:.1%}",
        f"  Decision Threshold : {threshold:.3f}  (adaptive, recall-optimised)",
        "",
        "─" * 60,
        "  Key Contributing Factors (SHAP):",
    ]

    for feat, shap_val in top_features:
        bar_len   = int(abs(shap_val) * 100)
        bar_str   = ("█" * min(bar_len, 20)).ljust(20)
        direction = "↑ INCREASES risk" if shap_val > 0 else "↓ decreases risk"
        lines.append(f"    {feat:<28} {bar_str}  {direction}  ({shap_val:+.4f})")

    lines += [
        "",
        "─" * 60,
        "  DISCLAIMER:",
        "  This prediction is generated by an ML model for DECISION SUPPORT",
        "  ONLY. It is NOT a medical diagnosis. All clinical decisions must",
        "  be made by a qualified healthcare professional.",
        "=" * 60,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
def batch_predict(patients: list, model, preprocessor, feature_names,
                  threshold: float) -> pd.DataFrame:
    """
    Predict risk for a list of patient dicts and return a summary DataFrame.
    Useful for batch screening scenarios.
    """
    rows = []
    for i, patient in enumerate(patients):
        raw = pd.DataFrame([patient])
        raw = _apply_feature_engineering(raw)
        X   = preprocessor.transform(raw[feature_names])
        prob = float(model.predict_proba(X)[0, 1])
        rows.append({
            "patient_id"  : i + 1,
            "probability" : round(prob, 4),
            "decision"    : "High Risk" if prob >= threshold else "Low Risk",
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/batch_predictions.csv", index=False)
    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator    import generate_synthetic_data, preprocess_data
    from model_training    import train_models
    from threshold_optimizer import optimize_threshold
    from explainability    import run_shap_analysis

    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fn = preprocess_data(df)
    mdls = train_models(X_tr, y_tr)
    bt, _ = optimize_threshold(mdls["XGBoost"], X_te, y_te)
    sv, ex = run_shap_analysis(mdls["XGBoost"], X_tr, X_te, fn)

    sample = {
        "age": 58, "gender": 1, "systolic_bp": 155, "diastolic_bp": 95,
        "cholesterol": 260, "glucose": 140, "bmi": 31, "fatigue_score": 8,
        "pain_score": 6, "family_history": 1, "smoking": 1, "physical_activity": 1,
    }
    result = predict_patient(sample, mdls["XGBoost"], pre, fn, sv, ex, bt, X_te)
    print(result["clinical_summary"])
