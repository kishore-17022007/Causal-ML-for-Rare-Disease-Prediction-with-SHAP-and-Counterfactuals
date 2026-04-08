"""
data_generator.py
─────────────────
Generates a synthetic clinical dataset for rare disease risk prediction
and provides a full sklearn preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


REQUIRED_RAW_COLUMNS = [
    "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi",
    "fatigue_score", "pain_score", "family_history", "smoking", "physical_activity",
    "disease_risk",
]

RAW_ALIASES = {
    "sex": "gender",
    "gender": "gender",
    "systolicbp": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "trestbps": "systolic_bp",
    "blood_pressure": "systolic_bp",
    "diastolicbp": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "dbp": "diastolic_bp",
    "chol": "cholesterol",
    "cholesterol": "cholesterol",
    "glucose_level": "glucose",
    "blood_glucose": "glucose",
    "glucose": "glucose",
    "fatigue": "fatigue_score",
    "fatigue_score": "fatigue_score",
    "pain": "pain_score",
    "pain_score": "pain_score",
    "familyhistory": "family_history",
    "family_history": "family_history",
    "smoker": "smoking",
    "smoking": "smoking",
    "activity": "physical_activity",
    "physical_activity": "physical_activity",
    "target": "disease_risk",
    "label": "disease_risk",
    "outcome": "disease_risk",
    "class": "disease_risk",
    "diagnosis": "disease_risk",
    "has_disease": "disease_risk",
    "disease_risk": "disease_risk",
}


def _to_snake(name: str) -> str:
    return (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _normalize_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce") > 0).astype("Int64")

    mapping = {
        "yes": 1, "y": 1, "true": 1, "positive": 1, "disease": 1, "high": 1,
        "no": 0, "n": 0, "false": 0, "negative": 0, "healthy": 0, "low": 0,
    }
    vals = series.astype(str).str.strip().str.lower().map(mapping)
    return vals.astype("Int64")


def load_kaggle_data(csv_path: str) -> pd.DataFrame:
    """
    Load and normalize a Kaggle dataset into the schema expected by the pipeline.

    The loader maps common alternative column names and creates missing columns
    with NaN/defaults so preprocessing can still run.
    """
    df = pd.read_csv(csv_path).copy()
    df.columns = [_to_snake(c) for c in df.columns]

    rename_map = {}
    for col in df.columns:
        mapped = RAW_ALIASES.get(col)
        if mapped is not None:
            rename_map[col] = mapped
    df = df.rename(columns=rename_map)

    if "disease_risk" not in df.columns:
        raise ValueError(
            "Could not find a target column. Expected one of: "
            "disease_risk, target, label, outcome, class, diagnosis, has_disease."
        )

    # Create missing columns expected by downstream feature engineering.
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize numeric columns.
    numeric_cols = [
        "age", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi",
        "fatigue_score", "pain_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize categorical and target columns.
    for col in ["gender", "family_history", "smoking"]:
        df[col] = _normalize_binary_series(df[col])

    df["physical_activity"] = pd.to_numeric(df["physical_activity"], errors="coerce")
    df["physical_activity"] = df["physical_activity"].clip(lower=0, upper=3)
    df["disease_risk"] = _normalize_binary_series(df["disease_risk"])

    # Conservative defaults for missing categorical values.
    df["gender"] = df["gender"].fillna(0).astype(int)
    df["family_history"] = df["family_history"].fillna(0).astype(int)
    df["smoking"] = df["smoking"].fillna(0).astype(int)
    df["physical_activity"] = df["physical_activity"].fillna(1).round().astype(int)
    df["disease_risk"] = df["disease_risk"].fillna(0).astype(int)

    return df[REQUIRED_RAW_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 3000, random_state: int = 42,
                             imbalance_ratio: float = 0.12) -> pd.DataFrame:
    """
    Generate a realistic synthetic clinical dataset with class imbalance
    (approx. 12 % positive → rare disease scenario).

    Features
    --------
    Demographics   : age, gender
    Clinical       : systolic_bp, diastolic_bp, cholesterol, glucose, bmi
    Symptoms       : fatigue_score (0-10), pain_score (0-10)
    Genetic        : family_history (0/1)
    Lifestyle      : smoking (0/1), physical_activity (0=none, 1=low, 2=moderate, 3=high)
    """
    rng = np.random.default_rng(random_state)

    n_pos = int(n_samples * imbalance_ratio)   # minority (disease)
    n_neg = n_samples - n_pos                  # majority (no disease)

    # ── Helper: draw for one class ──────────────────────────────────────────
    def _draw(n, risk):
        """risk=1 → higher risk phenotype."""
        age     = rng.normal(60 if risk else 42, 12, n).clip(18, 90)
        gender  = rng.integers(0, 2, n)         # 0=F, 1=M

        sbp     = rng.normal(145 if risk else 118, 18, n).clip(80, 200)
        dbp     = rng.normal(92  if risk else 76,  12, n).clip(50, 120)
        chol    = rng.normal(245 if risk else 185, 30, n).clip(100, 350)
        glucose = rng.normal(135 if risk else 92,  25, n).clip(60, 350)
        bmi     = rng.normal(30  if risk else 24,   5, n).clip(15, 50)

        fatigue = rng.integers(5, 11, n) if risk else rng.integers(0, 6, n)
        pain    = rng.integers(4, 11, n) if risk else rng.integers(0, 5, n)

        fam_hist = rng.binomial(1, 0.65 if risk else 0.15, n)
        smoking  = rng.binomial(1, 0.55 if risk else 0.25, n)
        activity = rng.integers(0, 2, n) if risk else rng.integers(1, 4, n)

        return dict(age=age, gender=gender,
                    systolic_bp=sbp, diastolic_bp=dbp,
                    cholesterol=chol, glucose=glucose, bmi=bmi,
                    fatigue_score=fatigue, pain_score=pain,
                    family_history=fam_hist, smoking=smoking,
                    physical_activity=activity)

    pos_data = _draw(n_pos, risk=1)
    neg_data = _draw(n_neg, risk=0)

    rows = []
    for i in range(n_pos):
        rows.append({k: v[i] for k, v in pos_data.items()} | {"disease_risk": 1})
    for i in range(n_neg):
        rows.append({k: v[i] for k, v in neg_data.items()} | {"disease_risk": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ── Introduce 5 % random missingness for realism ─────────────────────
    miss_cols = ["cholesterol", "glucose", "systolic_bp", "bmi"]
    for col in miss_cols:
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, test_size: float = 0.2,
                    random_state: int = 42, apply_smote: bool = True,
                    measurement_noise_std: float = 0.0):
    """
    Full preprocessing pipeline:
      1. Feature engineering (composite risk scores)
      2. Missing-value imputation
      3. Scaling
      4. Train/test split
      5. SMOTE oversampling on training set

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    df = df.copy()

    # ── Feature engineering ───────────────────────────────────────────────
    # Composite metabolic risk index
    df["metabolic_risk_index"] = (
        (df["cholesterol"].fillna(df["cholesterol"].median()) / 200) +
        (df["glucose"].fillna(df["glucose"].median()) / 100) +
        (df["bmi"].fillna(df["bmi"].median()) / 25)
    ) / 3

    # Cardiovascular stress indicator
    df["cv_stress"] = (
        df["systolic_bp"].fillna(df["systolic_bp"].median()) / 120 +
        df["diastolic_bp"].fillna(df["diastolic_bp"].median()) / 80
    ) / 2

    # Overall symptom burden
    df["symptom_burden"] = df["fatigue_score"] + df["pain_score"]

    # Genetic + lifestyle composite
    df["genetic_lifestyle_risk"] = (
        df["family_history"].fillna(0) * 2 +
        df["smoking"].fillna(0) +
        (3 - df["physical_activity"].fillna(1))   # lower activity -> higher risk
    )

    # ── Define feature sets ───────────────────────────────────────────────
    numerical_features = [
        "age", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi",
        "fatigue_score", "pain_score",
        "metabolic_risk_index", "cv_stress", "symptom_burden",
    ]
    categorical_features = [
        "gender", "family_history", "smoking", "physical_activity",
        "genetic_lifestyle_risk",
    ]

    # Cast categoricals to int (safe)
    for col in categorical_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)

    all_features = numerical_features + categorical_features
    target       = "disease_risk"

    X = df[all_features]
    y = df[target]

    # ── sklearn pipeline ──────────────────────────────────────────────────
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler",  StandardScaler()),          # keep numeric encoding
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ])

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # ── Fit & transform ───────────────────────────────────────────────────
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # ── Optional measurement noise for more realistic generalization ─────
    if measurement_noise_std > 0:
        rng = np.random.default_rng(random_state)
        X_train_proc = X_train_proc + rng.normal(
            0.0, measurement_noise_std, size=X_train_proc.shape
        )
        X_test_proc = X_test_proc + rng.normal(
            0.0, measurement_noise_std, size=X_test_proc.shape
        )

    # ── SMOTE ─────────────────────────────────────────────────────────────
    if apply_smote:
        smote = SMOTE(sampling_strategy=0.4, random_state=random_state)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        print(f"       After SMOTE  → train size : {X_train_proc.shape[0]} "
              f"| class balance: {pd.Series(y_train).value_counts().to_dict()}")

    feature_names = numerical_features + categorical_features

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=500)
    print(df.head())
    print(df["disease_risk"].value_counts())
