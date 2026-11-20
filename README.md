# WATERWATCH AI 💧🤖

> **Explainable AI-Based Water Potability Prediction and Community Risk Monitoring System**

---

## 📌 Project Overview

**WATERWATCH AI** is an explainable machine learning system designed for:
- **Water Potability Prediction:** Classifying whether water sources are safe for human consumption based on biochemical water quality parameters.
- **Model Comparison:** Evaluating and benchmarking multiple supervised machine learning architectures.
- **Explainability (XAI):** Interpreting model predictions using feature attribution techniques (e.g., SHAP / LIME) to help local communities and authorities understand *why* a water sample is flagged as non-potable.
- **Community Risk Monitoring:** Providing actionable risk scoring and screening prioritization for public health and water source monitoring.

---

## 👥 Team Structure & Responsibilities

| Role | Member | Assigned Branch | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **Data & ML Engineer** | **Member 1** | `feature/data-analysis` / `feature/ml-models` | Exploratory Data Analysis (EDA), missing value imputation, preprocessing pipelines, baseline benchmarking, Gaussian Naive Bayes & Decision Tree models, 5-fold CV, confusion matrix audits, model export & handover. |
| **ML Modeling** | Member 2 | `feature/ml-models` | Advanced model development, hyperparameter tuning, model benchmarking, and performance evaluation metrics. |
| **App Architecture** | Member 3 *(Lead)* | `feature/app-architecture` | Application architecture, interactive UI/dashboard layout, components, and explainability visualizations. |
| **Database & Docs** | Member 4 | `feature/database-docs` | Database schema design, source monitoring logging, system documentation, and reporting. |

---

## 🌳 Git Branching Strategy & Workflow

This project adheres to a structured, feature-branch collaboration model:

```text
main (Production / Final Release)
  └── dev (Integration Branch)
        ├── feature/data-analysis     (Member 1)
        ├── feature/ml-models         (Member 2)
        ├── feature/app-architecture  (Member 3)
        └── feature/database-docs     (Member 4)
```

### Branch Rules
1. **`main`**: Protected branch reserved strictly for final, evaluated, and stable deliverables. No direct commits allowed.
2. **`dev`**: Central integration branch where all team features are consolidated and verified.
3. **`feature/*`**: Individual workspace branches branched from `dev`. All development work must occur on these feature branches.

---

## 📁 Repository Structure

```text
WATERWATCH-AI/
├── data/
│   ├── raw/                 # Original, immutable datasets (water_potability.csv)
│   └── processed/           # Processed datasets
├── notebooks/               # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb            # Phase 1 & 3: EDA & Visualizations
│   ├── 02_data_preprocessing.ipynb             # Phase 2 & 4: Cleaning, Imputation, Splits
│   └── 03_model_training_and_evaluation.ipynb  # Phase 5, 6, 7 & 8: Models, CV, Benchmarks, Export
├── src/
│   ├── data/                # Data loading, cleaning, and preprocessing modules
│   ├── models/              # Model training, inference, and serialization logic
│   ├── explainability/      # Feature importance and XAI (SHAP, LIME) modules
│   ├── risk_engine/         # Community risk scoring and alert calculation logic
│   ├── database/            # Database schemas, connections, and migrations
│   └── reports/             # Automated summary and reporting utilities
├── app/
│   ├── pages/               # Multi-page dashboard views
│   ├── components/          # Reusable UI widgets and layout elements
│   └── assets/              # Static styling, logos, and reference images
├── models/                  # Serialized trained Scikit-Learn pipelines (.joblib)
│   ├── gaussian_nb_pipeline.joblib             # Imputer + Scaler + GaussianNB
│   └── decision_tree_pipeline.joblib           # Imputer + DecisionTreeClassifier
├── results/
│   ├── figures/             # 10 generated visualizations (EDA, ROC, confusion matrices)
│   └── metrics/             # Benchmark metrics, CSV & JSON reports
├── tests/                   # Unit and integration test suites
├── docs/                    # Architecture diagrams, specifications, and project reports
│   └── MEMBER_1_TECHNICAL_REPORT.md            # Comprehensive Member 1 Report
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation and team guide
```

---

## 🏆 Member 1 Completion Tracker (20/20 Completed)

All tasks assigned to **Member 1 (Data & ML Engineer)** have been completed and verified across the project notebooks, serialized models, metrics, figures, and technical reports:

### 1. Data Preparation (4/4 Completed)
- [x] **Verify missing values and duplicate records:** Confirmed 0 duplicate records across 3,276 rows. Missing values audited: Sulfate (23.84%), pH (14.99%), Trihalomethanes (4.95%).
- [x] **Complete EDA with charts and observations:** 6 visual figures produced and saved to `results/figures/` analyzing distributions, boxplots, class balances, and correlation matrices.
- [x] **Verify preprocessing has no data leakage:** Stratified 80/20 train-test split executed before imputer and scaler fitting. Pipelines fit strictly on `X_train`.
- [x] **Document outlier investigation and treatment decisions:** IQR outlier analysis conducted across all 9 features. Outliers deliberately retained as genuine environmental contamination signals.

### 2. Machine Learning (7/7 Completed)
- [x] **Gaussian Naive Bayes training:** Fitted via Scikit-Learn pipeline with median imputer and standard scaler.
- [x] **Decision Tree training:** Fitted with regularization (`max_depth=6`, `min_samples_leaf=10`) to control variance.
- [x] **Dummy Classifier baseline:** Implemented `DummyClassifier(strategy="most_frequent")` as majority-class baseline.
- [x] **Train-test evaluation and confusion matrices:** Accuracy, precision, non-potable recall, potable recall, F1-macro, F1-weighted, and confusion matrices evaluated on 656 held-out test samples.
- [x] **Five-fold stratified cross-validation:** Executed strictly on training data using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and reported as `mean ± std` decimal values.
- [x] **ROC-AUC comparison:** Evaluated for all models: GNB (0.6090), Decision Tree (0.6011), Baseline (0.5000); comparative curve saved to `results/figures/comparative_roc_curves.png`.
- [x] **Interpret model limitations:** Modest discrimination above random ranking documented; non-guarantee of real-world generalization outside the training distribution explained.

### 3. Model Delivery (4/4 Completed)
- [x] **Verify both saved joblib pipelines load correctly:** Both `models/gaussian_nb_pipeline.joblib` and `models/decision_tree_pipeline.joblib` verified via `joblib.load()`.
- [x] **Test predictions using new sample inputs:** Verified with test records and new synthetic inputs containing missing (`NaN`) values.
- [x] **Confirm feature order and preprocessing consistency:** Exact 9-feature order documented and encapsulated within pipeline imputers.
- [x] **Document model filenames, input format and output format for Member 3:** Integration guide and inference code provided for Streamlit UI development.

### 4. Project Documentation (5/5 Completed)
- [x] **Add EDA findings and screenshots to report:** Documented in `docs/MEMBER_1_TECHNICAL_REPORT.md` and `results/figures/`.
- [x] **Add model comparison table:** Exported to `results/metrics/final_model_comparison.csv` and documented below.
- [x] **Explain dangerous false-positive errors:** Detailed audit of actual non-potable samples falsely predicted as potable (45 GNB errors / 11.25%, 53 Decision Tree errors / 13.25%).
- [x] **Document dataset limitations and laboratory-testing disclaimer:** Explicit disclaimers regarding the 9 parameters, absence of microbiological assays, and mandatory lab testing.
- [x] **Prepare Member 1 contribution summary:** Comprehensive report finalized in `docs/MEMBER_1_TECHNICAL_REPORT.md`.

---

## 📊 Model Comparison Benchmark Table

The actual benchmark results for all three models on the held-out test set (656 samples) and 5-fold cross-validation:

| Model | Test Accuracy | Precision (Macro) | Precision (Weighted) | Recall (Non-Potable / 0) | Recall (Potable / 1) | F1-Score (Macro) | F1-Score (Weighted) | ROC-AUC | Dangerous Errors (Actual 0 $\rightarrow$ Pred 1) | Dangerous Error Rate (%) | 5-Fold CV Acc Mean | 5-Fold CV Acc Std | 5-Fold CV F1-Macro Mean | 5-Fold CV F1-Macro Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.6098 | 0.3049 | 0.3718 | 1.0000 | 0.0000 | 0.3788 | 0.4619 | 0.5000 | **0** | **0.00%** | 0.6099 | 0.0009 | 0.3789 | 0.0004 |
| **Gaussian Naive Bayes** | 0.6143 | 0.5733 | 0.5859 | 0.8875 | 0.1875 | 0.5062 | 0.5569 | **0.6090** | **45** | **11.25%** | 0.6275 | 0.0074 | **0.5393** | 0.0129 |
| **Decision Tree Classifier** | **0.6311** | **0.6029** | **0.6126** | 0.8675 | **0.2617** | **0.5489** | **0.5912** | **0.6011** | **53** | **13.25%** | **0.6298** | 0.0095 | 0.5192 | 0.0307 |

*CSV Export:* `results/metrics/final_model_comparison.csv`  
*JSON Export:* `results/metrics/evaluation_summary.json`

---

## ⚠️ Public Health Screening: The Dangerous False-Positive Error

In water potability screening, **Class 1 (Potable)** is treated as the positive class:
- **False Positive (Actual Class 0 $\rightarrow$ Predicted Class 1):** This is the **critical dangerous error**. A contaminated water source is mislabeled as potable, potentially exposing human populations to gastrointestinal disease, toxic chemical exposure, or systemic poisoning.
- **False Negative (Actual Class 1 $\rightarrow$ Predicted Class 0):** Safe water is flagged as non-potable, prompting redundant filtration or laboratory re-testing (economic/operational cost, but no human toxicity risk).

### Test Set Breakdown (400 Actual Non-Potable Samples):
- **Gaussian Naive Bayes:** Correctly flags 355 unsafe sources (88.75% recall); makes **45 dangerous false-positive errors (11.25%)**.
- **Decision Tree:** Correctly flags 347 unsafe sources (86.75% recall); makes **53 dangerous false-positive errors (13.25%)**.
- **Dummy Classifier:** Produces 0 false positives solely because it predicts 100% of samples as Class 0, completely failing to detect any potable water (0% sensitivity).

---

## 📦 Model Handover Specifications for Member 3 (App Architecture)

Member 3 can integrate the serialized pipelines into the Streamlit web dashboard using the following contract:

### 1. Serialized Model Files
- `models/gaussian_nb_pipeline.joblib`
- `models/decision_tree_pipeline.joblib`

### 2. Feature Schema & Required Order
Input data frames passed to `model.predict()` or `model.predict_proba()` must contain the following 9 columns:
1. `ph`: pH value (0.0 to 14.0)
2. `Hardness`: Water hardness in mg/L
3. `Solids`: Total dissolved solids (TDS) in ppm
4. `Chloramines`: Chloramines concentration in ppm
5. `Sulfate`: Sulfate concentration in mg/L
6. `Conductivity`: Electrical conductivity in μS/cm
7. `Organic_carbon`: Total organic carbon in ppm
8. `Trihalomethanes`: Trihalomethanes concentration in μg/L
9. `Turbidity`: Water turbidity in NTU

*Note:* Both pipelines include internal median imputers; raw inputs with `NaN` / missing values are automatically imputed.

### 3. Streamlit Inference Code Example
```python
import joblib
import pandas as pd
import numpy as np

# Load model pipeline
pipeline = joblib.load("models/decision_tree_pipeline.joblib")

# Prepare single-sample input
sample = pd.DataFrame([{
    'ph': 7.2,
    'Hardness': 200.0,
    'Solids': 18000.0,
    'Chloramines': 7.1,
    'Sulfate': 330.0,
    'Conductivity': 420.0,
    'Organic_carbon': 14.0,
    'Trihalomethanes': 65.0,
    'Turbidity': 3.8
}])

prediction = pipeline.predict(sample)[0]            # 0 = Non-Potable, 1 = Potable
probabilities = pipeline.predict_proba(sample)[0]  # [P(Non-Potable), P(Potable)]
```

---

## 🔬 Dataset Limitations & Laboratory-Testing Disclaimer

1. **Nine Parameters Only:** The model evaluates water safety strictly against the 9 continuous physicochemical parameters available in the dataset.
2. **Unmeasured Contaminants:** The system does **not** test for or detect heavy metal toxins (lead, mercury, cadmium, arsenic), pathogenic microbiological organisms (*E. coli*, Giardia, cholera, viral agents), or volatile synthetic organic pollutants (VOCs/pesticides).
3. **Screening Purpose Only:** Output classifications and probability scores generated by WATERWATCH AI are preliminary decision-support screening metrics. **They do not constitute certified water quality verification and cannot replace ISO/EPA/WHO accredited laboratory analysis.**

---

## ⚙️ Environment Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/WATERWATCH-AI.git
cd WATERWATCH-AI
```

### 2. Set Up a Virtual Environment (Python 3.11 recommended)
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📜 Academic Integrity & Collaboration Notice
This repository is configured for an academic team project. All contributors must follow standard Git etiquette, maintain clean commit histories, and document analytical assumptions. For detailed technical methodology, see [`docs/MEMBER_1_TECHNICAL_REPORT.md`](docs/MEMBER_1_TECHNICAL_REPORT.md).
