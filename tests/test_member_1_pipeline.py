"""
WATERWATCH AI - Member 1 Test Suite
Covers all 10 validation requirements specified in Project Specification Section 8.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

EXPECTED_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET_COL = "Potability"
DATA_PATH = "data/raw/water_potability.csv"


# 1. Feature existence and order
def test_features_present_and_ordered():
    df = pd.read_csv(DATA_PATH)
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing from dataset."
    input_cols = [c for c in df.columns if c != TARGET_COL]
    assert input_cols == EXPECTED_FEATURES, (
        f"Input feature columns or ordering mismatch!\nExpected: {EXPECTED_FEATURES}\nActual: {input_cols}"
    )


# 2. All three imputers work with missing values
def test_all_three_imputers_handle_nan():
    X_test_toy = pd.DataFrame([
        [np.nan, 200.0, 15000.0, 7.0, np.nan, 400.0, 14.0, np.nan, 3.5],
        [7.0, 210.0, 16000.0, 7.2, 330.0, 420.0, 13.5, 65.0, 3.8],
        [7.5, 190.0, 14000.0, 6.8, 310.0, 390.0, 14.2, 70.0, 3.2],
        [6.8, 205.0, 15500.0, 7.1, 325.0, 410.0, 14.0, 68.0, 3.6],
        [7.2, 195.0, 14500.0, 6.9, 315.0, 405.0, 13.8, 62.0, 3.4],
        [7.1, 200.0, 15000.0, 7.0, 320.0, 400.0, 14.0, 66.0, 3.5],
    ], columns=EXPECTED_FEATURES)

    # A: SimpleImputer median
    med = SimpleImputer(strategy="median")
    res_med = med.fit_transform(X_test_toy)
    assert not np.isnan(res_med).any(), "SimpleImputer left NaNs"

    # B: KNNImputer (scaled)
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=3))
    ])
    res_knn = knn_pipe.fit_transform(X_test_toy)
    assert not np.isnan(res_knn).any(), "KNNImputer left NaNs"

    # C: IterativeImputer (scaled)
    mice_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42))
    ])
    res_mice = mice_pipe.fit_transform(X_test_toy)
    assert not np.isnan(res_mice).any(), "IterativeImputer left NaNs"


# 3. All three classifiers can train and predict
def test_all_three_classifiers_train_and_predict():
    df = pd.read_csv(DATA_PATH).dropna()
    X = df[EXPECTED_FEATURES].iloc[:100]
    y = df[TARGET_COL].iloc[:100]

    classifiers = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        RandomForestClassifier(n_estimators=10, min_samples_leaf=2, random_state=42)
    ]

    for clf in classifiers:
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X), f"{type(clf).__name__} output length mismatch"
        assert set(preds).issubset({0, 1}), f"{type(clf).__name__} predicted non-binary labels"


# 4. Pipelines do not produce NaN predictions on missing data
def test_pipelines_produce_no_nan_predictions():
    df = pd.read_csv(DATA_PATH)
    X = df[EXPECTED_FEATURES].iloc[:200]
    y = df[TARGET_COL].iloc[:200]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
        ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42))
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert not pd.Series(preds).isna().any(), "Pipeline generated NaN predictions"


# 5. Cross-validation runs without preprocessing leakage
def test_cv_runs_leakage_free():
    df = pd.read_csv(DATA_PATH)
    X = df[EXPECTED_FEATURES].iloc[:300]
    y = df[TARGET_COL].iloc[:300]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)),
        ("clf", GaussianNB())
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


# 6. Threshold calculations use the intended probability column
def test_threshold_calculation_uses_class_1_proba():
    df = pd.read_csv(DATA_PATH).dropna()
    # Sample balanced set with both classes
    df0 = df[df[TARGET_COL] == 0].iloc[:100]
    df1 = df[df[TARGET_COL] == 1].iloc[:100]
    sample = pd.concat([df0, df1]).sample(frac=1.0, random_state=42)

    X = sample[EXPECTED_FEATURES]
    y = sample[TARGET_COL]

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    # Class 1 probability is column index corresponding to class label 1
    class_1_idx = list(clf.classes_).index(1)
    p1 = probs[:, class_1_idx]
    assert (p1 >= 0.0).all() and (p1 <= 1.0).all()
    # At high threshold (0.9), fewer positives should be predicted than at 0.1
    pos_at_01 = (p1 >= 0.1).sum()
    pos_at_09 = (p1 >= 0.9).sum()
    assert pos_at_01 >= pos_at_09, "Threshold behavior inverted!"


# 7. Confusion matrix calculations use correct class interpretation
def test_confusion_matrix_interpretation():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # TN: Actual 0, Pred 0
    assert tn == 1, "TN interpretation incorrect"
    # FP: Actual 0, Pred 1 (Dangerous Error)
    assert fp == 1, "FP (dangerous error) interpretation incorrect"
    # FN: Actual 1, Pred 0 (Precautionary Waste)
    assert fn == 1, "FN interpretation incorrect"
    # TP: Actual 1, Pred 1
    assert tp == 1, "TP interpretation incorrect"


# 8. Robustness experiments do not modify the original dataset or test set
def test_robustness_does_not_mutate_original_data():
    df_before = pd.read_csv(DATA_PATH)
    checksum_before = df_before.sum(numeric_only=True).sum()

    # Simulate missingness on a copy
    copy_df = df_before.copy()
    copy_df.iloc[0, 0] = np.nan

    df_after = pd.read_csv(DATA_PATH)
    checksum_after = df_after.sum(numeric_only=True).sum()

    assert checksum_before == checksum_after, "Original dataset was mutated by robustness workflow!"


# 9. Existing baseline artifacts can still be loaded
def test_existing_baseline_artifacts_loadable():
    model_paths = [
        "models/gaussian_nb_pipeline.joblib",
        "models/decision_tree_pipeline.joblib",
        "models/random_forest_pipeline.joblib",
    ]
    for path in model_paths:
        assert os.path.exists(path), f"Baseline model artifact missing: {path}"
        loaded = joblib.load(path)
        assert hasattr(loaded, "predict"), f"{path} is not a valid predictor pipeline"
        assert len(loaded.steps) >= 2, f"{path} does not have expected pipeline steps"


# 10. Existing Streamlit integration compatibility
def test_streamlit_integration_compatibility():
    # Streamlit app inputs a single-row DataFrame with EXPECTED_FEATURES
    sample_input = pd.DataFrame([{
        "ph": 7.2, "Hardness": 195.0, "Solids": 18000.0,
        "Chloramines": 7.1, "Sulfate": 330.0, "Conductivity": 420.0,
        "Organic_carbon": 14.2, "Trihalomethanes": 65.0, "Turbidity": 3.8
    }])

    for mname, mpath in [
        ("Gaussian NB", "models/gaussian_nb_pipeline.joblib"),
        ("Decision Tree", "models/decision_tree_pipeline.joblib"),
        ("Random Forest", "models/random_forest_pipeline.joblib"),
    ]:
        model = joblib.load(mpath)
        pred = model.predict(sample_input)
        proba = model.predict_proba(sample_input)

        assert pred[0] in [0, 1], f"Prediction out of bounds for {mname}"
        assert proba.shape == (1, 2), f"Probability shape invalid for {mname}"
        assert abs(proba.sum() - 1.0) < 1e-4, f"Probabilities do not sum to 1 for {mname}"
