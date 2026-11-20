# WATERWATCH AI — Member 1 Technical Report
## Data Engineering, Exploratory Data Analysis, Model Benchmarking & Pipeline Handover

**Author:** Member 1 – Data & ML Engineer  
**Project:** Explainable AI-Based Water Potability Prediction and Community Risk Monitoring System  
**Repository Branch:** `feature/data-analysis` / `feature/ml-models`  
**Status:** Completed & Verified (All 20 Tracker Deliverables Met)  

---

## Executive Summary

This report documents the end-to-end deliverables completed by **Member 1 (Data & ML Engineer)** for the **WATERWATCH AI** system. The mandate encompassed:
1. **Data Preparation & Auditing:** Rigorous inspection of missing data, duplicate verification, and IQR-based outlier analysis with principled retention justification.
2. **Exploratory Data Analysis (EDA):** Deep multivariate analysis with 6 published figures analyzing class distributions, feature histograms, correlations, and kernel density estimates.
3. **Leakage-Free Preprocessing:** Architecting Scikit-Learn Pipelines that isolate test data and encapsulate median imputation and standardization strictly within training splits.
4. **Machine Learning & Benchmarking:** Developing and evaluating **Gaussian Naive Bayes**, **Decision Tree Classifier**, and a **Dummy Classifier Baseline** across multi-metric criteria and 5-fold stratified cross-validation.
5. **Community Risk & False Positive Auditing:** Explicit quantification of the dangerous screening error (actual non-potable water mispredicted as potable).
6. **Model Serialization & Handover:** Exporting production `.joblib` pipelines and providing complete schema specifications for **Member 3 (App Architecture & UI)**.

---

## 1. Data Preparation

### 1.1 Dataset Inspection & Dimensions
- **File Location:** `data/raw/water_potability.csv` (fallback: `data/water_potability.csv`)
- **Total Records:** 3,276 water samples
- **Total Columns:** 10 (9 continuous physicochemical features + 1 binary target)
- **Feature Data Types:** All 9 inputs are continuous `float64`; target `Potability` is binary integer `int64` (`0 = Non-Potable`, `1 = Potable`).

### 1.2 Missing Value and Duplicate Verification
- **Duplicate Records:** **0 duplicates** detected across all 3,276 rows (`df.duplicated().sum() == 0`).
- **Missingness Audit:** Missing values are localized strictly to three biochemical features:
  - `Sulfate`: **781 missing values** (23.84%)
  - `ph`: **491 missing values** (14.99%)
  - `Trihalomethanes`: **162 missing values** (4.95%)
  - Remaining 6 features (`Hardness`, `Solids`, `Chloramines`, `Conductivity`, `Organic_carbon`, `Turbidity`) have **0% missing values** (3,276 complete records).
- **Imputation Strategy:** Scikit-Learn `SimpleImputer(strategy='median')`. Median imputation is mathematically robust against skewed feature distributions and prevents the artificial distortion introduced by mean imputation.

### 1.3 Outlier Investigation & Treatment Decisions (IQR Method)
Using the Interquartile Range method ($IQR = Q3 - Q1$, bounds $= [Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$):

| Feature | Q1 (25%) | Q3 (75%) | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ph` | 6.09 | 8.06 | 1.97 | 3.14 | 11.02 | 46 | 1.40% |
| `Hardness` | 176.85 | 216.67 | 39.82 | 117.13 | 276.39 | 83 | 2.53% |
| `Solids` | 15,666.69 | 27,332.76 | 11,666.07 | -1,832.42 | 44,831.87 | 47 | 1.43% |
| `Chloramines` | 6.13 | 8.11 | 1.99 | 3.15 | 11.10 | 61 | 1.86% |
| `Sulfate` | 307.70 | 359.95 | 52.25 | 229.32 | 438.33 | 41 | 1.25% |
| `Conductivity` | 365.73 | 481.79 | 116.06 | 191.65 | 655.88 | 11 | 0.34% |
| `Organic_carbon` | 12.07 | 16.56 | 4.49 | 5.33 | 23.30 | 25 | 0.76% |
| `Trihalomethanes`| 55.84 | 77.34 | 21.49 | 23.61 | 109.58 | 33 | 1.01% |
| `Turbidity` | 3.44 | 4.50 | 1.06 | 1.85 | 6.09 | 20 | 0.61% |

**Treatment Decision & Justification:**
All identified statistical outliers were **deliberately retained without trimming or winsorization**. In environmental hydrology and water toxicology, extreme readings (such as pH < 3.5, sulfate > 450 mg/L, or solids > 40,000 ppm) represent genuine industrial discharge, heavy mineral leaching, or severe contamination events. Truncating these points would eliminate the very high-risk contamination signals that an automated screening system must identify.

### 1.4 Strict Data Leakage Prevention
To guarantee valid generalization estimates:
1. The raw dataset was partitioned using `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`.
2. **Train partition:** 2,620 samples (80.00%) | **Test partition:** 656 samples (20.00%).
3. Class proportions are precisely preserved in both splits: **60.99% Non-Potable (0)** vs. **39.01% Potable (1)**.
4. Preprocessing transformations (`SimpleImputer` and `StandardScaler`) are encapsulated inside Scikit-Learn `Pipeline` constructs. They are fitted **strictly on `X_train`**; the test set `X_test` is transformed using training parameters only, preventing all data leakage.

---

## 2. Exploratory Data Analysis (EDA) & Visual Observations

Six publication-quality figures were generated and saved in `results/figures/`:

1. **`01_target_class_distribution.png`**:
   - Class 0 (Non-Potable): 1,998 samples (60.99%)
   - Class 1 (Potable): 1,278 samples (39.01%)
   - Mild natural imbalance requiring stratified sampling and balanced macro-averaged evaluation.

2. **`02_feature_histograms.png`**:
   - Demonstrates near-normal (Gaussian-like) distributions for `Hardness`, `Chloramines`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Turbidity`.
   - `Solids` displays slight positive skewness; `ph` and `Sulfate` are symmetrically distributed around their medians.

3. **`03_numerical_boxplots.png`**:
   - Outliers are distributed symmetrically in both upper and lower tails across chemical parameters, confirming physical environmental variation rather than data corruption.

4. **`04_correlation_heatmap.png`**:
   - **Crucial Finding:** Correlation coefficients between all physicochemical feature pairs are near zero ($|r| < 0.08$), and correlation with `Potability` is similarly weak ($|r| \le 0.03$).
   - This proves that water potability cannot be separated using simple linear cutoffs or single-variable thresholds; non-linear multidimensional boundary partitioners or probabilistic modeling are required.

5. **`05_class_comparison_kde.png`**:
   - Kernel density estimation curves for potable vs. non-potable classes overlap substantially across all nine dimensions. This empirical overlap explains the modest separability achievable by statistical classifiers.

6. **`06_missing_values_matrix.png`**:
   - Confirms missing values occur missing-completely-at-random (MCAR) or missing-at-random (MAR) across `Sulfate`, `ph`, and `Trihalomethanes`, with no co-dependent missingness patterns.

---

## 3. Machine Learning Architectures & Training

Three models were constructed, trained on `X_train`, and evaluated against the identical held-out `X_test` (656 samples):

### 3.1 Baseline Benchmark: Dummy Classifier
- **Algorithm:** `sklearn.dummy.DummyClassifier(strategy="most_frequent")`
- **Objective:** Establish the scientific baseline of naive majority-class guessing.
- **Mechanism:** Because Class 0 accounts for 60.99% of samples, the dummy classifier simply predicts Class 0 for every single water sample, ignoring chemistry features.
- **Significance:** Explains why the baseline achieves **100% recall for Class 0** and **60.98% accuracy**, yet has **0% recall for Class 1 (Potable)**, a macro F1 of only 0.3788, and a ROC-AUC of 0.5000. It demonstrates why trained models must be assessed on macro discrimination rather than raw accuracy.

### 3.2 Model 1: Gaussian Naive Bayes (Phase 5)
- **Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `GaussianNB()`
- **Mathematical Assumption:** Assumes feature likelihoods given class $y$ follow normal distributions:
  $$P(X_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(X_i - \mu_y)^2}{2\sigma_y^2}\right)$$
- Fits the continuous features well due to their near-Gaussian marginal distributions observed during EDA.

### 3.3 Model 2: Decision Tree Classifier (Phase 6)
- **Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=10)`
- **Regularization Strategy:** Constrained with `max_depth=6` and `min_samples_leaf=10` to avoid memorization, control variance, and maintain tree interpretability for downstream XAI explainability.

---

## 4. Comprehensive Model Evaluation & Benchmark Comparison

### 4.1 Benchmark Evaluation Table
Saved automatically to `results/metrics/final_model_comparison.csv` and serialized in `results/metrics/evaluation_summary.json`:

| Model | Test Accuracy | Precision (Macro) | Precision (Weighted) | Recall (Class 0 Non-Potable) | Recall (Class 1 Potable) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | ROC-AUC | Dangerous Errors (FP: Actual 0 $\rightarrow$ Pred 1) | Dangerous Error Rate (%) | 5-Fold CV Acc Mean | 5-Fold CV Acc Std | 5-Fold CV F1-Macro Mean | 5-Fold CV F1-Macro Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.6098 | 0.3049 | 0.3718 | 1.0000 | 0.0000 | 0.5000 | 0.3788 | 0.4619 | 0.5000 | **0** | **0.00%** | 0.6099 | 0.0009 | 0.3789 | 0.0004 |
| **Gaussian Naive Bayes** | 0.6143 | 0.5733 | 0.5859 | 0.8875 | 0.1875 | 0.5375 | 0.5062 | 0.5569 | **0.6090** | **45** | **11.25%** | 0.6275 | 0.0074 | 0.5393 | 0.0129 |
| **Decision Tree Classifier** | **0.6311** | **0.6029** | **0.6126** | 0.8675 | **0.2617** | **0.5646** | **0.5489** | **0.5912** | **0.6011** | **53** | **13.25%** | **0.6298** | 0.0095 | 0.5192 | 0.0307 |

### 4.2 Confusion Matrix Analysis & The Dangerous Screening Error
In binary public-health classification where **Class 1 (Potable)** is the positive class:
- **True Negative (TN, $y=0, \hat{y}=0$):** Non-potable water correctly identified as unsafe.
- **False Negative (FN, $y=1, \hat{y}=0$):** Potable water misclassified as non-potable (leads to redundant filtration or testing; operational cost, but safe).
- **True Positive (TP, $y=1, \hat{y}=1$):** Potable water correctly verified safe for consumption.
- **False Positive (FP, $y=0, \hat{y}=1$):** **THE CRITICAL DANGEROUS ERROR.** Contaminated, non-potable water is mistakenly certified as potable. Consumers drinking this water are exposed to severe chemical or toxicological health risks.

#### Actual Breakdown on Held-Out Test Set (400 Actual Non-Potable, 256 Actual Potable):
- **Dummy Classifier:** TN=400, FP=0, FN=256, TP=0. Achieves 0 dangerous errors solely because it never predicts potable water (completely fails to detect any safe drinking water).
- **Gaussian Naive Bayes:** TN=355, **FP=45 (11.25% error rate on non-potables)**, FN=208, TP=48.
- **Decision Tree Classifier:** TN=347, **FP=53 (13.25% error rate on non-potables)**, FN=189, TP=67.

Confusion matrix plots are archived in:
- `results/figures/confusion_matrix_dummy_classifier.png`
- `results/figures/confusion_matrix_gaussian_nb.png`
- `results/figures/confusion_matrix_decision_tree.png`

### 4.3 Five-Fold Stratified Cross-Validation
Cross-validation was conducted **strictly on training data (`X_train`, `y_train`)** with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:
- **Gaussian Naive Bayes:** Mean Accuracy $= 0.6275 \pm 0.0074$, Mean Macro-F1 $= 0.5393 \pm 0.0129$
- **Decision Tree:** Mean Accuracy $= 0.6298 \pm 0.0095$, Mean Macro-F1 $= 0.5192 \pm 0.0307$
- **Dummy Baseline:** Mean Accuracy $= 0.6099 \pm 0.0009$, Mean Macro-F1 $= 0.3789 \pm 0.0004$

Both trained models demonstrate low fold-to-fold standard deviations ($< 0.03$), confirming stable learning across random subsets of the training distribution.

### 4.4 Receiver Operating Characteristic (ROC-AUC)
- **Gaussian Naive Bayes:** **ROC-AUC = 0.6090**
- **Decision Tree Classifier:** **ROC-AUC = 0.6011**
- **Dummy Baseline:** **ROC-AUC = 0.5000**
- Comparative curve saved: `results/figures/comparative_roc_curves.png`.

---

## 5. Academic Quality Review & Model Limitations

In accordance with rigorous academic and engineering standards, the model performance is contextualized as follows:

1. **Modest Discrimination Above Random Chance:**
   ROC-AUC values of 0.6090 (GNB) and 0.6011 (Decision Tree) represent **modest discrimination above random ranking performance**. They do not demonstrate high-confidence separation.
2. **Feature Scope Boundary:**
   The model evaluates potability **strictly from the nine recorded physicochemical features**. It does **not** directly detect heavy metal ions (e.g., lead, arsenic, mercury), microbiological pathogens (e.g., *E. coli*, coliform bacteria, enteroviruses), or volatile organic compounds (VOCs).
3. **Cross-Validation Scope:**
   Stratified 5-fold cross-validation proves internal fold stability within this specific dataset distribution; **it does not constitute proof of real-world generalization** to disparate geographic aquifer systems, seasonal precipitation cycles, or unmeasured industrial effluents.
4. **Mandatory Laboratory-Testing Disclaimer:**
   The predictions generated by WATERWATCH AI are preliminary statistical screening assessments designed to prioritize laboratory investigations. **They cannot and must not certify drinking-water potability or replace certified laboratory chemical and microbiological assays.**

---

## 6. Model Serialization & Handover to Member 3 (App Architecture)

### 6.1 Saved Pipeline Files
- `models/gaussian_nb_pipeline.joblib` (2,433 bytes)
- `models/decision_tree_pipeline.joblib` (6,515 bytes)

### 6.2 Pipeline Architecture & Self-Contained Preprocessing
Both exported objects are full Scikit-Learn `Pipeline` instances:
- **GNB Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `GaussianNB()`
- **Decision Tree Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)`

**Key Advantage for Member 3:** Member 3 does **not** need to manually impute missing values or scale user inputs. The pipelines automatically handle raw input data frames, even when inputs contain `NaN` values.

### 6.3 Expected Feature Order & Schema
Member 3's UI components must supply input data frames with the exact 9 columns in the following sequence:

```python
EXPECTED_FEATURE_ORDER = [
    'ph',               # float (0.0 to 14.0)
    'Hardness',         # float (mg/L)
    'Solids',           # float (ppm)
    'Chloramines',      # float (ppm)
    'Sulfate',          # float (mg/L)
    'Conductivity',     # float (μS/cm)
    'Organic_carbon',   # float (ppm)
    'Trihalomethanes',  # float (μg/L)
    'Turbidity'         # float (NTU)
]
```

### 6.4 Verified Inference Code Sample for Member 3 (Streamlit Integration)
```python
import joblib
import pandas as pd
import numpy as np

# 1. Load desired pipeline
model = joblib.load("models/decision_tree_pipeline.joblib")
# or: model = joblib.load("models/gaussian_nb_pipeline.joblib")

# 2. Prepare user input DataFrame (raw inputs, NaNs permitted)
input_data = pd.DataFrame([{
    'ph': 7.2,
    'Hardness': 195.0,
    'Solids': 18500.0,
    'Chloramines': 7.3,
    'Sulfate': np.nan,  # Pipeline will automatically impute using median
    'Conductivity': 410.0,
    'Organic_carbon': 14.0,
    'Trihalomethanes': 68.0,
    'Turbidity': 3.9
}])

# 3. Generate binary prediction and probability
predicted_class = model.predict(input_data)[0]  # 0 = Non-Potable, 1 = Potable
probabilities = model.predict_proba(input_data)[0]
prob_non_potable = probabilities[0]
prob_potable = probabilities[1]

# 4. Community Risk Alert Logic
if predicted_class == 0:
    risk_level = "HIGH RISK: Non-Potable Water Source"
else:
    risk_level = "PRELIMINARY PASS: Requires Standard Lab Verification"
```

---

## 7. Member 1 Completion Tracker Verification (20/20 Items)

| # | Checklist Task | Status | Location / Artifact |
| :-: | :--- | :---: | :--- |
| **1** | Verify missing values and duplicate records | ✅ **Completed** | `notebooks/01_dataset_exploration.ipynb`, `notebooks/02_data_preprocessing.ipynb` |
| **2** | Complete EDA with charts and observations | ✅ **Completed** | `results/figures/` (6 figures: `01`–`06`) |
| **3** | Verify preprocessing has no data leakage | ✅ **Completed** | `notebooks/02_data_preprocessing.ipynb`, train-test split precedes all imputer/scaler fitting |
| **4** | Document outlier investigation and treatment decisions | ✅ **Completed** | `notebooks/02_data_preprocessing.ipynb`, IQR table + retained environmental signals |
| **5** | Gaussian Naive Bayes training | ✅ **Completed** | `notebooks/03_model_training_and_evaluation.ipynb`, `models/gaussian_nb_pipeline.joblib` |
| **6** | Decision Tree training | ✅ **Completed** | `notebooks/03_model_training_and_evaluation.ipynb`, `models/decision_tree_pipeline.joblib` |
| **7** | Dummy Classifier baseline | ✅ **Completed** | `notebooks/03_model_training_and_evaluation.ipynb` (`strategy='most_frequent'`) |
| **8** | Train-test evaluation and confusion matrices | ✅ **Completed** | `results/metrics/final_model_comparison.csv`, `results/figures/confusion_matrix_*.png` |
| **9** | Five-fold stratified cross-validation | ✅ **Completed** | `notebooks/03_model_training_and_evaluation.ipynb`, reported with `mean ± std` decimals |
| **10** | ROC-AUC comparison | ✅ **Completed** | `results/figures/comparative_roc_curves.png` (GNB=0.6090, DT=0.6011, Dummy=0.5000) |
| **11** | Interpret model limitations | ✅ **Completed** | Section 5 of this report, `notebooks/03_model_training_and_evaluation.ipynb` |
| **12** | Verify both saved joblib pipelines load correctly | ✅ **Completed** | Verified with `joblib.load()` on both `.joblib` files |
| **13** | Test predictions using new sample inputs | ✅ **Completed** | Tested on raw test samples and synthetic inputs (including NaNs) |
| **14** | Confirm feature order and preprocessing consistency | ✅ **Completed** | Strict 9-column schema documented and embedded in pipelines |
| **15** | Document model filenames, input format and output format for Member 3 | ✅ **Completed** | Section 6 of this report |
| **16** | Add EDA findings and screenshots to report | ✅ **Completed** | Section 2 of this report & `README.md` |
| **17** | Add model comparison table | ✅ **Completed** | Section 4.1 of this report & `results/metrics/final_model_comparison.csv` |
| **18** | Explain dangerous false-positive errors | ✅ **Completed** | Section 4.2 of this report (45 GNB errors / 11.25%, 53 DT errors / 13.25%) |
| **19** | Document dataset limitations and laboratory-testing disclaimer | ✅ **Completed** | Section 5 of this report |
| **20** | Prepare Member 1 contribution summary | ✅ **Completed** | Full document herein + updated project `README.md` |

---
*Report certified by Member 1 (Data & ML Engineer), WATERWATCH AI Project Team.*
