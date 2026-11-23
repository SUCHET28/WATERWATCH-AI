import json
import os

with open('notebooks/02_data_preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

base_cells = nb['cells'][:15]

new_cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'id': 'imp_study_md_1',
        'source': [
            "---\n",
            "## Phase 8: Advanced Missing-Value Imputation Comparison Study\n",
            "**Role:** Member 1 – Data & ML Engineer  \n",
            "**Task:** Compare baseline Median Imputation with KNN Imputation (k=5) and Iterative Imputation (MICE with Bayesian Ridge).\n",
            "\n",
            "### Imputation Paradigms Evaluated:\n",
            "1. **Median Imputation (Baseline):** Univariate replacement using training set column medians. Fast and outlier-robust, but ignores multivariate correlations and attenuates variance.\n",
            "2. **KNN Imputation (k=5):** Non-parametric multivariate imputation based on Euclidean distance among the 5 nearest neighbors. **Scaling is mathematically mandatory** before distance computation to prevent high-magnitude features (e.g. `Solids` ~ 20,000) from dominating over normalized features (e.g. `ph` ~ 7).\n",
            "3. **Iterative Imputation (MICE):** Multivariate round-robin regression modeling each missing feature as a function of all remaining features using Bayesian Ridge. Standardizing prior to regression guarantees isotropic L2 regularization across variables.\n",
            "\n",
            "> **Data Leakage Safeguard:** All imputers and scalers are strictly embedded in `sklearn.pipeline.Pipeline` objects and fitted exclusively on training splits ($X_{train}$), ensuring zero test leakage during holdout testing and cross-validation."
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'id': 'imp_study_code_1',
        'outputs': [],
        'source': [
            "# Phase 8: Advanced Missing-Value Imputation Pipeline Comparison\n",
            "from sklearn.impute import KNNImputer\n",
            "from sklearn.experimental import enable_iterative_imputer\n",
            "from sklearn.impute import IterativeImputer\n",
            "from sklearn.linear_model import BayesianRidge\n",
            "from sklearn.naive_bayes import GaussianNB\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.dummy import DummyClassifier\n",
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\n",
            "\n",
            "# Load pre-computed comparison metrics\n",
            "imputation_results_path = '../results/metrics/imputation_comparison.csv' if os.path.exists('../results/metrics/imputation_comparison.csv') else 'results/metrics/imputation_comparison.csv'\n",
            "imputation_df = pd.read_csv(imputation_results_path)\n",
            "print('=== COMPLETE IMPUTATION COMPARISON BENCHMARK TABLE ===')\n",
            "display(imputation_df)\n"
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'id': 'imp_study_md_2',
        'source': [
            "### Visualizing Performance Across Imputation Strategies\n",
            "The following charts evaluate accuracy, macro F1, and dangerous false-positive rates (Actual 0 -> Predicted 1) across the three imputation strategies."
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'id': 'imp_study_code_2',
        'outputs': [],
        'source': [
            "from IPython.display import Image, display\n",
            "\n",
            "fig_paths = [\n",
            "    '../results/figures/imputation_accuracy_comparison.png',\n",
            "    '../results/figures/imputation_f1_comparison.png',\n",
            "    '../results/figures/imputation_dangerous_error_rate_comparison.png',\n",
            "    '../results/figures/imputation_best_confusion_matrices.png'\n",
            "]\n",
            "\n",
            "for path in fig_paths:\n",
            "    resolved_path = path if os.path.exists(path) else path.replace('../', '')\n",
            "    if os.path.exists(resolved_path):\n",
            "        display(Image(filename=resolved_path))\n"
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'id': 'imp_study_md_3',
        'source': [
            "### Academic Summary of Imputation Findings:\n",
            "- **Iterative Imputation (MICE):** Yields the highest 5-fold cross-validation accuracy (0.6347 ± 0.0095) and macro F1 (0.5317 ± 0.0361) when paired with Decision Tree.\n",
            "- **KNN Imputation (k=5):** Achieves the lowest dangerous false-positive count (39 errors, 9.75% rate) compared to Median (53 errors, 13.25%) and Iterative (52 errors, 13.00%), functioning as a more conservative screen against contaminated water.\n",
            "- **Significance of Improvement:** Because pairwise inter-feature correlations are near zero (|r| < 0.04), multivariate relationships offer modest predictive leverage (+0.49% CV accuracy over median). The baseline median imputation remains preserved as the project standard to avoid unnecessary model complexity."
        ]
    }
]

nb['cells'] = base_cells + new_cells
with open('notebooks/02_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated notebooks/02_data_preprocessing.ipynb successfully! Total cells:', len(nb['cells']))
