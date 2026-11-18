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
| **Data Analysis** | Member 1 | `feature/data-analysis` | Exploratory Data Analysis (EDA), missing value imputation, feature distribution analysis, and data cleaning pipelines. |
| **ML Modeling** | Member 2 | `feature/ml-models` | Model development, hyperparameter tuning, model benchmarking, and performance evaluation metrics. |
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

### Team Collaboration Workflow
1. **Switch to your assigned feature branch:**
   ```bash
   git checkout <your-assigned-feature-branch>
   ```
2. **Keep your branch synchronized with `dev`:**
   ```bash
   git pull origin dev
   ```
3. **Stage and commit your changes:**
   ```bash
   git add .
   git commit -m "feat(<scope>): descriptive summary of work done"
   ```
4. **Push your branch to GitHub:**
   ```bash
   git push origin <your-assigned-feature-branch>
   ```
5. **Open a Pull Request (PR):**
   - Create a Pull Request targeting **`dev`** (never directly into `main`).
   - Review and merge after team verification.

---

## 📁 Repository Structure

```text
WATERWATCH-AI/
├── data/
│   ├── raw/                 # Original, immutable datasets
│   └── processed/           # Cleaned, imputed, and scaled datasets
├── notebooks/               # Jupyter notebooks for EDA and experimentation
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
├── models/                  # Serialized trained model weights and encoders (.pkl, .joblib)
├── results/
│   ├── figures/             # Saved visual plots, ROC curves, and confusion matrices
│   └── metrics/             # Model evaluation summaries, JSON/CSV reports
├── tests/                   # Unit and integration test suites
├── docs/                    # Architecture diagrams, specifications, and project reports
├── .gitignore               # Ignored files (virtual environments, models, databases)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation and team guide
```

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
This repository is configured for an academic team project. All contributors must follow standard Git etiquette, maintain clean commit histories, and document analytical assumptions.
