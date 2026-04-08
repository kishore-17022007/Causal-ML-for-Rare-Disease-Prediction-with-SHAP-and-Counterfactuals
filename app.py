"""
app.py  ─  Streamlit UI for Patient Risk Dashboard
────────────────────────────────────────────────────
Run with:
    streamlit run app.py

DISCLAIMER: This UI is for demonstration / decision-support purposes only.
            It is NOT a medical diagnostic tool.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ── Try importing Streamlit; give clear error if missing ──────────────────────
try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Run:  pip install streamlit")
    sys.exit(1)

from predictor       import predict_patient, _apply_feature_engineering
from explainability  import get_top_shap_features


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rare Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#2c3e50; }
    .risk-high   { background:#fdedec; border-left:6px solid #e74c3c;
                   padding:1rem; border-radius:4px; color:#c0392b; font-size:1.3rem; font-weight:700; }
    .risk-low    { background:#eafaf1; border-left:6px solid #2ecc71;
                   padding:1rem; border-radius:4px; color:#1e8449; font-size:1.3rem; font-weight:700; }
    .metric-card { background:#f8f9fa; padding:1rem; border-radius:8px;
                   text-align:center; border:1px solid #dee2e6; }
    .disclaimer  { background:#fef9e7; border:1px solid #f39c12;
                   padding:.8rem; border-radius:4px; font-size:.85rem; color:#7d6608; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts (with caching)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    """Load saved models, preprocessor, SHAP objects, and threshold."""
    artefacts = {}

    try:
        with open("outputs/trained_models.pkl", "rb") as f:
            artefacts["models"] = pickle.load(f)
        models = artefacts["models"]

        # Prefer the model with the highest saved ROC-AUC when available.
        best_model_name = None
        try:
            summary = pd.read_csv("outputs/evaluation_summary.csv", index_col=0)
            if "roc_auc" in summary.columns and not summary.empty:
                best_model_name = summary["roc_auc"].astype(float).idxmax()
        except Exception:
            best_model_name = None

        # Fallback order if no summary exists or names differ.
        if best_model_name not in models:
            for candidate in ("XGBoost", "Random Forest", "Logistic Regression"):
                if candidate in models:
                    best_model_name = candidate
                    break

        if best_model_name is None and len(models) > 0:
            best_model_name = next(iter(models.keys()))

        artefacts["best_model"] = models.get(best_model_name)
        artefacts["best_model_name"] = best_model_name
    except FileNotFoundError:
        artefacts["models"]     = None
        artefacts["best_model"] = None
        artefacts["best_model_name"] = None

    try:
        with open("outputs/shap_explainer.pkl", "rb") as f:
            artefacts["explainer"] = pickle.load(f)
    except FileNotFoundError:
        artefacts["explainer"] = None

    try:
        with open("outputs/shap_values.pkl", "rb") as f:
            artefacts["shap_values"] = pickle.load(f)
    except FileNotFoundError:
        artefacts["shap_values"] = None

    # Read optimal threshold from report
    threshold = 0.35      # sensible default
    try:
        df_sweep = pd.read_csv("outputs/threshold_sweep.csv")
        threshold = float(df_sweep.loc[df_sweep["fbeta"].idxmax(), "threshold"])
    except Exception:
        pass
    artefacts["threshold"] = threshold

    return artefacts


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Patient Input Form
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## 🩺 Patient Input")
    st.sidebar.markdown("Enter the patient's clinical information below.")

    st.sidebar.markdown("**Demographics**")
    age    = st.sidebar.slider("Age",    18, 90, 55)
    gender = st.sidebar.radio("Gender", ["Female", "Male"], index=1)

    st.sidebar.markdown("**Vital Signs**")
    sbp = st.sidebar.number_input("Systolic BP (mmHg)",  80,  200, 145, step=1)
    dbp = st.sidebar.number_input("Diastolic BP (mmHg)", 50,  120,  90, step=1)

    st.sidebar.markdown("**Lab Values**")
    chol    = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 350, 240, step=5)
    glucose = st.sidebar.number_input("Glucose (mg/dL)",      60, 350, 130, step=5)
    bmi     = st.sidebar.number_input("BMI",                  15.0, 50.0, 29.5, step=0.5)

    st.sidebar.markdown("**Symptoms**")
    fatigue = st.sidebar.slider("Fatigue Score (0–10)",  0, 10, 6)
    pain    = st.sidebar.slider("Pain Score (0–10)",     0, 10, 4)

    st.sidebar.markdown("**Risk Factors**")
    fam_hist = st.sidebar.checkbox("Family History of Disease", value=True)
    smoking  = st.sidebar.checkbox("Current Smoker",            value=False)
    activity_label = st.sidebar.select_slider(
        "Physical Activity Level",
        options=["None", "Low", "Moderate", "High"],
        value="Low"
    )
    activity_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}

    patient = {
        "age"              : age,
        "gender"           : 1 if gender == "Male" else 0,
        "systolic_bp"      : sbp,
        "diastolic_bp"     : dbp,
        "cholesterol"      : chol,
        "glucose"          : glucose,
        "bmi"              : bmi,
        "fatigue_score"    : fatigue,
        "pain_score"       : pain,
        "family_history"   : int(fam_hist),
        "smoking"          : int(smoking),
        "physical_activity": activity_map[activity_label],
    }
    return patient


def _predict_probability_only(patient_dict, model, preprocessor, feature_names):
    """Fast probability prediction used by simulation views."""
    raw = pd.DataFrame([patient_dict])
    raw = _apply_feature_engineering(raw)
    X_proc = preprocessor.transform(raw[feature_names])
    return float(model.predict_proba(X_proc)[0, 1])


def _render_prediction_cards(result):
    """Reusable KPI cards for prediction output."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if "HIGH RISK" in result["decision"]:
            st.markdown(f'<div class="risk-high">{result["decision"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">{result["decision"]}</div>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:.9rem;color:#7f8c8d">Risk Probability</div>
          <div style="font-size:2rem;font-weight:700;color:#2c3e50">{result["probability"]:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:.9rem;color:#7f8c8d">Decision Threshold</div>
          <div style="font-size:2rem;font-weight:700;color:#2c3e50">{result["threshold"]:.3f}</div>
          <div style="font-size:.75rem;color:#7f8c8d">(adaptive, recall-optimised)</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Title
    st.markdown('<p class="main-title">🏥 Rare Disease Risk Prediction Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("**Causal-Aware ML Framework** | Adaptive Threshold | SHAP Explainability")
    st.markdown('<div class="disclaimer">⚠️ <b>DISCLAIMER:</b> This tool is for research and '
                'clinical decision-support ONLY. It does NOT constitute a medical diagnosis. '
                'All decisions must be reviewed by a qualified healthcare professional.</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar input
    patient = render_sidebar()
    run_btn  = st.sidebar.button("🔍  Run Prediction", type="primary", use_container_width=True)

    # Load artefacts
    artefacts = load_artefacts()

    if artefacts["best_model"] is None:
        st.warning("⚠️  No trained model found. Please run `python main.py` first "
                   "to train and save the models.")
        st.stop()

    # ── Feature names (must match data_generator) ────────────────────────
    feature_names = [
        "age", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi",
        "fatigue_score", "pain_score",
        "metabolic_risk_index", "cv_stress", "symptom_burden",
        "gender", "family_history", "smoking", "physical_activity",
        "genetic_lifestyle_risk",
    ]

    # Preprocessor – try to load, otherwise guide user
    try:
        from data_generator import load_kaggle_data, generate_synthetic_data, preprocess_data

        if os.path.exists("data/kaggle_data.csv"):
            df_tmp = load_kaggle_data("data/kaggle_data.csv")
        else:
            df_tmp = generate_synthetic_data(n_samples=300, random_state=42)

        _, _, _, _, preprocessor, _ = preprocess_data(
            df_tmp,
            apply_smote=False,
            measurement_noise_std=0.0,
        )
    except Exception as e:
        st.error(f"Could not load preprocessor: {e}")
        st.stop()

    pred_tab, sim_tab = st.tabs(["🔍 Single Prediction", "🧪 Simulation Lab"])

    with pred_tab:
        if run_btn:
            with st.spinner("Analysing patient data …"):
                result = predict_patient(
                    patient_dict      = patient,
                    model             = artefacts["best_model"],
                    preprocessor      = preprocessor,
                    feature_names     = feature_names,
                    shap_values_test  = artefacts["shap_values"],
                    explainer         = artefacts["explainer"],
                    threshold         = artefacts["threshold"],
                    X_test            = np.zeros((1, len(feature_names))),
                )

            _render_prediction_cards(result)
            st.markdown("---")

            # ── SHAP Feature Contributions ─────────────────────────────
            st.subheader("🔍 Top Contributing Factors (SHAP)")
            top_feats  = result["top_features"]
            feat_names = [f[0] for f in top_feats]
            feat_vals  = [f[1] for f in top_feats]

            fig, ax = plt.subplots(figsize=(10, 4))
            colors  = ["#e74c3c" if v > 0 else "#3498db" for v in feat_vals]
            ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], edgecolor="none")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP value", fontsize=11)
            ax.set_title("Feature Contributions to This Prediction", fontsize=12)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Risk Gauge ─────────────────────────────────────────────
            st.subheader("📊 Risk Probability Gauge")
            fig2, ax2 = plt.subplots(figsize=(6, 1.2))
            prob = result["probability"]
            ax2.barh(["Risk"], [prob],        color="#e74c3c", height=0.5)
            ax2.barh(["Risk"], [1 - prob], left=[prob], color="#eaecee", height=0.5)
            ax2.axvline(result["threshold"], color="black", lw=2, linestyle="--",
                        label=f"Threshold ({result['threshold']:.3f})")
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Probability")
            ax2.legend(fontsize=9)
            ax2.set_title("Disease Risk Probability", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # ── Saved plots from training ──────────────────────────────
            st.markdown("---")
            st.subheader("📈 Training Analysis")
            plot_tabs = st.tabs(["ROC Curves", "Threshold Sweep", "Feature Importance"])
            plot_files = [
                "outputs/roc_curves.png",
                "outputs/threshold_sweep.png",
                "outputs/shap_feature_importance.png",
            ]
            for tab, fpath in zip(plot_tabs, plot_files):
                with tab:
                    if os.path.exists(fpath):
                        st.image(fpath, use_column_width=True)
                    else:
                        st.info("Plot not found. Run main.py to generate.")

            # ── Raw report ────────────────────────────────────────────
            with st.expander("📋 Full Clinical Report"):
                st.code(result["clinical_summary"], language="text")

        else:
            st.info("👈  Fill in patient details in the sidebar, then click **Run Prediction**.")

            if os.path.exists("outputs/roc_curves.png"):
                st.subheader("📈 Model Performance Overview")
                c1, c2 = st.columns(2)
                with c1:
                    st.image("outputs/roc_curves.png", caption="ROC Curves")
                with c2:
                    st.image("outputs/shap_feature_importance.png",
                             caption="SHAP Feature Importance")

    with sim_tab:
        st.subheader("What-if Simulation")
        st.write(
            "Create a modified scenario and compare it with the current sidebar inputs "
            "to see how prediction probability changes."
        )

        scenario = deepcopy(patient)
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("**Scenario Demographics & Vitals**")
            scenario["age"] = st.slider("Age (Scenario)", 18, 90, int(patient["age"]), key="sim_age")
            scenario["systolic_bp"] = st.slider(
                "Systolic BP (Scenario)", 80, 200, int(patient["systolic_bp"]), key="sim_sbp"
            )
            scenario["diastolic_bp"] = st.slider(
                "Diastolic BP (Scenario)", 50, 120, int(patient["diastolic_bp"]), key="sim_dbp"
            )
            scenario["bmi"] = st.slider("BMI (Scenario)", 15.0, 50.0, float(patient["bmi"]), 0.1, key="sim_bmi")

        with c_right:
            st.markdown("**Scenario Labs & Lifestyle**")
            scenario["cholesterol"] = st.slider(
                "Cholesterol (Scenario)", 100, 350, int(patient["cholesterol"]), key="sim_chol"
            )
            scenario["glucose"] = st.slider(
                "Glucose (Scenario)", 60, 350, int(patient["glucose"]), key="sim_glu"
            )
            scenario["fatigue_score"] = st.slider(
                "Fatigue Score (Scenario)", 0, 10, int(patient["fatigue_score"]), key="sim_fatigue"
            )
            scenario["pain_score"] = st.slider(
                "Pain Score (Scenario)", 0, 10, int(patient["pain_score"]), key="sim_pain"
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            scenario["family_history"] = int(
                st.checkbox("Family History (Scenario)", value=bool(patient["family_history"]), key="sim_fh")
            )
        with c4:
            scenario["smoking"] = int(
                st.checkbox("Smoking (Scenario)", value=bool(patient["smoking"]), key="sim_smoking")
            )
        with c5:
            activity_label = st.select_slider(
                "Physical Activity (Scenario)",
                options=["None", "Low", "Moderate", "High"],
                value={0: "None", 1: "Low", 2: "Moderate", 3: "High"}[patient["physical_activity"]],
                key="sim_activity",
            )
            activity_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
            scenario["physical_activity"] = activity_map[activity_label]

        run_sim = st.button("Run Simulation", type="primary", key="run_sim_btn")

        if run_sim:
            with st.spinner("Running baseline vs scenario simulation …"):
                baseline_prob = _predict_probability_only(
                    patient,
                    artefacts["best_model"],
                    preprocessor,
                    feature_names,
                )
                scenario_prob = _predict_probability_only(
                    scenario,
                    artefacts["best_model"],
                    preprocessor,
                    feature_names,
                )

                baseline_result = predict_patient(
                    patient_dict      = patient,
                    model             = artefacts["best_model"],
                    preprocessor      = preprocessor,
                    feature_names     = feature_names,
                    shap_values_test  = artefacts["shap_values"],
                    explainer         = artefacts["explainer"],
                    threshold         = artefacts["threshold"],
                    X_test            = np.zeros((1, len(feature_names))),
                )
                scenario_result = predict_patient(
                    patient_dict      = scenario,
                    model             = artefacts["best_model"],
                    preprocessor      = preprocessor,
                    feature_names     = feature_names,
                    shap_values_test  = artefacts["shap_values"],
                    explainer         = artefacts["explainer"],
                    threshold         = artefacts["threshold"],
                    X_test            = np.zeros((1, len(feature_names))),
                )

            delta = scenario_prob - baseline_prob
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline Risk", f"{baseline_prob:.1%}")
            m2.metric("Scenario Risk", f"{scenario_prob:.1%}")
            m3.metric("Change", f"{delta:+.1%}")

            fig_cmp, ax_cmp = plt.subplots(figsize=(6, 3.2))
            ax_cmp.bar(["Baseline", "Scenario"], [baseline_prob, scenario_prob],
                       color=["#3498db", "#e67e22"])
            ax_cmp.axhline(artefacts["threshold"], color="black", linestyle="--", linewidth=1.5,
                           label=f"Threshold ({artefacts['threshold']:.3f})")
            ax_cmp.set_ylim(0, 1)
            ax_cmp.set_ylabel("Probability")
            ax_cmp.set_title("Risk Change from Input Modification")
            ax_cmp.legend(fontsize=8)
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

            st.markdown("### Baseline vs Scenario Decision")
            cb, cs = st.columns(2)
            with cb:
                st.markdown("**Baseline**")
                _render_prediction_cards(baseline_result)
            with cs:
                st.markdown("**Scenario**")
                _render_prediction_cards(scenario_result)

            st.markdown("---")
            st.markdown("### Single-Feature Sensitivity Sweep")
            sweep_feature = st.selectbox(
                "Choose one feature to sweep",
                options=[
                    "age", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi",
                    "fatigue_score", "pain_score", "physical_activity"
                ],
                key="sim_sweep_feature",
            )
            sweep_ranges = {
                "age": (18, 90),
                "systolic_bp": (80, 200),
                "diastolic_bp": (50, 120),
                "cholesterol": (100, 350),
                "glucose": (60, 350),
                "bmi": (15.0, 50.0),
                "fatigue_score": (0, 10),
                "pain_score": (0, 10),
                "physical_activity": (0, 3),
            }
            lo, hi = sweep_ranges[sweep_feature]
            grid = np.linspace(lo, hi, 25)
            if sweep_feature in ("fatigue_score", "pain_score", "physical_activity", "age",
                                 "systolic_bp", "diastolic_bp", "cholesterol", "glucose"):
                grid = np.round(grid).astype(int)

            sweep_probs = []
            for v in grid:
                test_patient = deepcopy(scenario)
                test_patient[sweep_feature] = float(v) if sweep_feature == "bmi" else int(v)
                sweep_probs.append(
                    _predict_probability_only(
                        test_patient,
                        artefacts["best_model"],
                        preprocessor,
                        feature_names,
                    )
                )

            fig_sw, ax_sw = plt.subplots(figsize=(8, 3.5))
            ax_sw.plot(grid, sweep_probs, color="#8e44ad", linewidth=2)
            ax_sw.axhline(artefacts["threshold"], color="black", linestyle="--", linewidth=1.2)
            ax_sw.set_xlabel(sweep_feature)
            ax_sw.set_ylabel("Predicted risk probability")
            ax_sw.set_ylim(0, 1)
            ax_sw.set_title(f"Sensitivity: {sweep_feature} vs risk probability")
            ax_sw.grid(alpha=0.25)
            st.pyplot(fig_sw)
            plt.close(fig_sw)

        else:
            st.info("Set a modified scenario and click **Run Simulation**.")


if __name__ == "__main__":
    main()
