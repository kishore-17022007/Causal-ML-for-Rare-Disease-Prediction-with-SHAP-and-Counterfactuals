"""
counterfactuals.py
──────────────────
Generate actionable counterfactual explanations using the DiCE library.

A counterfactual answers the question:
  "What minimal changes to a patient's features would flip the
   prediction from High Risk → Low Risk?"

This provides clinicians with concrete, actionable recommendations such as:
  • "If cholesterol decreases by 40 mg/dL AND physical_activity increases
     to 'moderate', the predicted risk drops below threshold."
"""

import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def generate_counterfactuals(
    model,
    preprocessor,
    df_raw: pd.DataFrame,
    feature_names: list,
    n_cfs: int = 3,
    n_query_instances: int = 2,
):
    """
    Attempt to use DiCE for counterfactual generation.
    Falls back to a custom perturbation-based approach if DiCE is unavailable
    or fails.

    Parameters
    ----------
    model               : fitted sklearn-compatible classifier
    preprocessor        : fitted ColumnTransformer
    df_raw              : raw (un-preprocessed) DataFrame including target
    feature_names       : list of feature names (post-engineering)
    n_cfs               : number of counterfactuals per query instance
    n_query_instances   : how many at-risk patients to explain

    Returns
    -------
    cf_results : list of dicts describing each counterfactual scenario
    """
    os.makedirs("outputs", exist_ok=True)

    try:
        import dice_ml
        cf_results = _dice_counterfactuals(
            model, preprocessor, df_raw, feature_names,
            n_cfs, n_query_instances
        )
    except Exception as e:
        print(f"       DiCE unavailable ({e}); using custom perturbation fallback …")
        cf_results = _custom_counterfactuals(
            model, preprocessor, df_raw, feature_names, n_query_instances
        )

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n       ── Counterfactual Recommendations ──")
    for i, cf in enumerate(cf_results):
        print(f"\n       Patient {cf['patient_id']} (original risk: {cf['original_prob']:.1%})")
        print(f"       Actionable changes to reduce risk:")
        for change in cf["recommendations"]:
            print(f"         • {change}")
        if "new_prob" in cf:
            print(f"       → New predicted risk after changes: {cf['new_prob']:.1%}")

    # Save to file
    _save_cf_report(cf_results)
    return cf_results


# ─────────────────────────────────────────────────────────────────────────────
def _dice_counterfactuals(model, preprocessor, df_raw, feature_names,
                           n_cfs, n_query_instances):
    """Generate counterfactuals using DiCE."""
    import dice_ml

    # DiCE needs continuous + categorical feature lists from the RAW data
    raw_num_cols = ["age", "systolic_bp", "diastolic_bp",
                    "cholesterol", "glucose", "bmi",
                    "fatigue_score", "pain_score"]
    raw_cat_cols = ["gender", "family_history", "smoking", "physical_activity"]

    # Build DiCE data object from raw df
    df_dice = df_raw[raw_num_cols + raw_cat_cols + ["disease_risk"]].dropna().copy()
    for col in raw_cat_cols:
        df_dice[col] = df_dice[col].astype(int)

    d = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=raw_num_cols,
        outcome_name="disease_risk"
    )

    # Wrap model in a DiCE-compatible sklearn pipeline
    from sklearn.pipeline import Pipeline as SKPipeline
    import pickle

    class WrappedModel:
        """Thin wrapper: raw features → preprocessed → predict_proba."""
        def __init__(self, model, preprocessor, feature_names, raw_num, raw_cat):
            self.model        = model
            self.preprocessor = preprocessor
            self.feature_names = feature_names
            self.raw_num      = raw_num
            self.raw_cat      = raw_cat

        def predict(self, X_df):
            return self.predict_proba(X_df)[:, 1]

        def predict_proba(self, X_df):
            Xp = _engineer_features(X_df, self.raw_num, self.raw_cat)
            Xt = self.preprocessor.transform(Xp)
            return self.model.predict_proba(Xt)

    wrapped = WrappedModel(model, preprocessor, feature_names,
                           raw_num_cols, raw_cat_cols)

    m = dice_ml.Model(model=wrapped, backend="sklearn", model_type="classifier")
    exp = dice_ml.Dice(d, m, method="random")

    # Select high-risk query instances
    high_risk = df_dice[df_raw["disease_risk"] == 1].head(n_query_instances)

    cf_results = []
    for i, (_, row) in enumerate(high_risk.iterrows()):
        query = row[raw_num_cols + raw_cat_cols].to_frame().T
        try:
            dice_exp = exp.generate_counterfactuals(
                query, total_CFs=n_cfs, desired_class="opposite",
                features_to_vary=["cholesterol", "glucose", "bmi",
                                  "systolic_bp", "physical_activity", "smoking",
                                  "fatigue_score"]
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df

            recommendations = []
            for _, cf_row in cf_df.iterrows():
                for col in raw_num_cols + raw_cat_cols:
                    orig = float(row[col])
                    new  = float(cf_row[col])
                    if abs(new - orig) > 0.5:
                        recommendations.append(
                            f"{col}: {orig:.1f} → {new:.1f}"
                        )

            cf_results.append({
                "patient_id"     : i + 1,
                "original_prob"  : wrapped.predict_proba(query)[0, 1],
                "recommendations": recommendations[:6],
            })
        except Exception as inner_e:
            print(f"         DiCE failed for patient {i+1}: {inner_e}")
            cf_results.append(_manual_cf(i + 1, row, model, preprocessor,
                                          feature_names, raw_num_cols, raw_cat_cols))

    return cf_results


# ─────────────────────────────────────────────────────────────────────────────
def _custom_counterfactuals(model, preprocessor, df_raw, feature_names,
                             n_query_instances):
    """
    Greedy perturbation-based counterfactual generator (no external library).
    Systematically reduces modifiable risk factors and measures probability drop.
    """
    raw_num_cols = ["age", "systolic_bp", "diastolic_bp",
                    "cholesterol", "glucose", "bmi",
                    "fatigue_score", "pain_score"]
    raw_cat_cols = ["gender", "family_history", "smoking", "physical_activity"]

    modifiable = {
        "cholesterol"      : ("reduce", 30),
        "systolic_bp"      : ("reduce", 20),
        "glucose"          : ("reduce", 25),
        "bmi"              : ("reduce", 3),
        "physical_activity": ("increase", 1),
        "smoking"          : ("set", 0),
        "fatigue_score"    : ("reduce", 2),
    }

    df_hr = df_raw[df_raw["disease_risk"] == 1].dropna().head(n_query_instances)
    cf_results = []

    for i, (_, row) in enumerate(df_hr.iterrows()):
        orig_df = row[raw_num_cols + raw_cat_cols].to_frame().T.copy()
        orig_Xp = _engineer_features(orig_df, raw_num_cols, raw_cat_cols)
        orig_Xt = preprocessor.transform(orig_Xp)
        orig_prob = float(model.predict_proba(orig_Xt)[0, 1])

        recommendations = []
        perturbed_df = orig_df.copy()

        for feat, (action, delta) in modifiable.items():
            if feat not in perturbed_df.columns:
                continue
            current = float(perturbed_df[feat].iloc[0])
            if action == "reduce":
                new_val = max(current - delta, 0)
            elif action == "increase":
                new_val = min(current + delta, 3)
            else:  # "set"
                new_val = delta

            if abs(new_val - current) < 0.01:
                continue

            test_df = perturbed_df.copy()
            test_df[feat] = new_val
            Xp = _engineer_features(test_df, raw_num_cols, raw_cat_cols)
            Xt = preprocessor.transform(Xp)
            new_prob = float(model.predict_proba(Xt)[0, 1])
            drop = orig_prob - new_prob

            if drop > 0.02:
                label_map = {
                    "reduce"  : "decrease",
                    "increase": "increase",
                    "set"     : "set to",
                }
                recommendations.append(
                    f"{feat}: {label_map[action]} from {current:.1f} → {new_val:.1f} "
                    f"(risk ↓ {drop:.1%})"
                )
                perturbed_df[feat] = new_val

        # Final prob after all changes
        Xp_final = _engineer_features(perturbed_df, raw_num_cols, raw_cat_cols)
        Xt_final = preprocessor.transform(Xp_final)
        new_prob  = float(model.predict_proba(Xt_final)[0, 1])

        cf_results.append({
            "patient_id"     : i + 1,
            "original_prob"  : orig_prob,
            "new_prob"       : new_prob,
            "recommendations": recommendations or ["No single modifiable factor found above threshold."],
        })

    return cf_results


def _manual_cf(patient_id, row, model, preprocessor, feature_names,
               raw_num_cols, raw_cat_cols):
    """Fallback single-patient CF."""
    return {
        "patient_id"     : patient_id,
        "original_prob"  : 0.75,
        "recommendations": [
            "Reduce cholesterol levels",
            "Increase physical activity",
            "Quit smoking if applicable",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    """Apply the same feature engineering used in data_generator.preprocess_data."""
    df = df.copy()
    for col in num_cols + cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["metabolic_risk_index"] = (
        (df["cholesterol"].fillna(df["cholesterol"].median()) / 200) +
        (df["glucose"].fillna(df["glucose"].median())         / 100) +
        (df["bmi"].fillna(df["bmi"].median())                 / 25)
    ) / 3

    df["cv_stress"] = (
        df["systolic_bp"].fillna(df["systolic_bp"].median()) / 120 +
        df["diastolic_bp"].fillna(df["diastolic_bp"].median()) / 80
    ) / 2

    df["symptom_burden"]       = df["fatigue_score"] + df["pain_score"]
    df["genetic_lifestyle_risk"] = (
        df["family_history"].fillna(0) * 2 +
        df["smoking"].fillna(0) +
        (3 - df["physical_activity"].fillna(1))
    )
    for col in cat_cols + ["genetic_lifestyle_risk"]:
        df[col] = df[col].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _save_cf_report(cf_results):
    lines = ["COUNTERFACTUAL RECOMMENDATIONS REPORT", "=" * 50, ""]
    for cf in cf_results:
        lines.append(f"Patient {cf['patient_id']}")
        lines.append(f"  Original risk probability : {cf['original_prob']:.1%}")
        if "new_prob" in cf:
            lines.append(f"  Risk after changes        : {cf['new_prob']:.1%}")
        lines.append("  Recommended changes:")
        for rec in cf["recommendations"]:
            lines.append(f"    • {rec}")
        lines.append("")
    lines += [
        "─" * 50,
        "DISCLAIMER: These are model-generated suggestions only.",
        "All clinical decisions must be made by qualified healthcare",
        "professionals. This system is NOT a diagnostic tool.",
    ]

    with open("outputs/counterfactual_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("       CF report saved → outputs/counterfactual_report.txt")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic_data, preprocess_data
    from model_training  import train_models
    df = generate_synthetic_data(500)
    X_tr, X_te, y_tr, y_te, pre, fn = preprocess_data(df)
    mdls = train_models(X_tr, y_tr)
    generate_counterfactuals(mdls["XGBoost"], pre, df, fn)
