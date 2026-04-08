"""
causal_analysis.py
──────────────────
Basic causal inference using DoWhy.

Goals
─────
1. Define a causal graph over the clinical features.
2. Estimate the Average Treatment Effect (ATE) for key modifiable factors.
3. Compare causal estimates vs simple correlation-based importances.
4. Save a causal report to outputs/.

If DoWhy is not installed, falls back to a correlation-based analysis
with a clear notice to the user.
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def run_causal_analysis(df: pd.DataFrame) -> dict:
    """
    Run causal analysis on the raw dataframe.

    Returns
    -------
    dict with keys:
        causal_effects   : {treatment: ATE}   (DoWhy) or {}
        correlations     : pd.Series
        comparison_table : pd.DataFrame
    """
    os.makedirs("outputs", exist_ok=True)

    df_clean = df.dropna().copy()
    outcome   = "disease_risk"
    treatments = [
        "smoking", "physical_activity",
        "cholesterol", "glucose", "systolic_bp", "bmi",
    ]

    # ── Correlation-based baseline ────────────────────────────────────────
    correlations = df_clean[treatments + [outcome]].corr()[outcome].drop(outcome).abs()
    correlations = correlations.sort_values(ascending=False)

    print("\n       ── Correlation with disease_risk ──")
    for feat, corr in correlations.items():
        print(f"         {feat:<28} |r|={corr:.4f}")

    # ── Attempt DoWhy ─────────────────────────────────────────────────────
    causal_effects = {}
    try:
        import dowhy
        causal_effects = _dowhy_ate(df_clean, outcome, treatments)
    except ImportError:
        print("\n       DoWhy not installed; using regression-based ATE estimates.")
        causal_effects = _regression_ate(df_clean, outcome, treatments)
    except Exception as e:
        print(f"\n       DoWhy error ({e}); falling back to regression-based ATE.")
        causal_effects = _regression_ate(df_clean, outcome, treatments)

    # ── Comparison table: causal vs correlation ────────────────────────────
    comp_rows = []
    for feat in treatments:
        comp_rows.append({
            "feature"          : feat,
            "correlation_rank" : int(correlations.rank(ascending=False)[feat]),
            "|correlation|"    : round(correlations.get(feat, 0.0), 4),
            "causal_ATE"       : round(causal_effects.get(feat, float("nan")), 4),
        })
    comp_df = pd.DataFrame(comp_rows).set_index("feature")
    comp_df.to_csv("outputs/causal_vs_correlation.csv")

    print("\n       ── Causal ATE vs Correlation ──")
    print(comp_df.to_string())

    # ── Save report ───────────────────────────────────────────────────────
    _save_causal_report(comp_df, causal_effects)

    return {
        "causal_effects"  : causal_effects,
        "correlations"    : correlations,
        "comparison_table": comp_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _dowhy_ate(df: pd.DataFrame, outcome: str, treatments: list) -> dict:
    """
    Use DoWhy to estimate ATE for each binary/continuous treatment variable.
    Binary treatments → do-calculus; continuous → linear IV approximation.
    """
    from dowhy import CausalModel

    results = {}

    # Define a common causal graph (simplified DAG)
    # Confounders: age, gender, family_history
    gml_base = """
        graph [
            node [id "age"               label "age"]
            node [id "gender"            label "gender"]
            node [id "family_history"    label "family_history"]
            node [id "smoking"           label "smoking"]
            node [id "physical_activity" label "physical_activity"]
            node [id "cholesterol"       label "cholesterol"]
            node [id "glucose"           label "glucose"]
            node [id "systolic_bp"       label "systolic_bp"]
            node [id "bmi"               label "bmi"]
            node [id "disease_risk"      label "disease_risk"]
            edge [source "age"               target "cholesterol"]
            edge [source "age"               target "glucose"]
            edge [source "age"               target "disease_risk"]
            edge [source "gender"            target "disease_risk"]
            edge [source "family_history"    target "disease_risk"]
            edge [source "smoking"           target "cholesterol"]
            edge [source "smoking"           target "disease_risk"]
            edge [source "physical_activity" target "bmi"]
            edge [source "physical_activity" target "cholesterol"]
            edge [source "physical_activity" target "disease_risk"]
            edge [source "cholesterol"       target "disease_risk"]
            edge [source "glucose"           target "disease_risk"]
            edge [source "systolic_bp"       target "disease_risk"]
            edge [source "bmi"               target "disease_risk"]
        ]
    """

    for treatment in treatments:
        try:
            model = CausalModel(
                data                = df,
                treatment           = treatment,
                outcome             = outcome,
                graph               = gml_base,
                logging_level       = "ERROR",
            )
            identified  = model.identify_effect(proceed_when_unidentifiable=True)
            estimate    = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            ate = float(estimate.value)
            results[treatment] = ate
            print(f"         DoWhy ATE ({treatment:<20}) = {ate:+.4f}")
        except Exception as e:
            print(f"         DoWhy failed for {treatment}: {e}")
            results[treatment] = float("nan")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def _regression_ate(df: pd.DataFrame, outcome: str, treatments: list) -> dict:
    """
    Regression-based ATE approximation (linear regression coefficient
    after controlling for age, gender, family_history).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    confounders = ["age", "gender", "family_history"]
    results     = {}

    for treatment in treatments:
        cols = [treatment] + confounders
        sub  = df[[outcome] + cols].dropna()
        X    = StandardScaler().fit_transform(sub[cols])
        y    = sub[outcome].values

        lr  = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr.fit(X, y)
        coef = float(lr.coef_[0, 0])   # coefficient for the treatment variable
        results[treatment] = coef
        print(f"         Regression ATE ({treatment:<20}) = {coef:+.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def _save_causal_report(comp_df: pd.DataFrame, causal_effects: dict):
    lines = [
        "CAUSAL ANALYSIS REPORT",
        "======================",
        "",
        "Method: DoWhy backdoor.linear_regression (or logistic-regression ATE fallback)",
        "",
        "Causal ATE represents the *direct* effect of each treatment variable",
        "on disease_risk after controlling for confounders (age, gender, family_history).",
        "",
        "Interpretation:",
        "  Positive ATE → treatment increases disease risk",
        "  Negative ATE → treatment decreases disease risk",
        "",
        comp_df.to_string(),
        "",
        "─" * 60,
        "Key Causal Insights:",
    ]

    for feat, ate in sorted(causal_effects.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True):
        if np.isnan(ate):
            continue
        direction = "↑ increases" if ate > 0 else "↓ decreases"
        lines.append(f"  • {feat:<25} ATE={ate:+.4f}  → {direction} disease risk")

    lines += [
        "",
        "─" * 60,
        "DISCLAIMER: Causal estimates are approximate and depend on the assumed",
        "causal graph. Consult domain experts before making clinical decisions.",
    ]

    with open("outputs/causal_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("       Causal report saved → outputs/causal_report.txt")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic_data
    df  = generate_synthetic_data(1000)
    res = run_causal_analysis(df)
