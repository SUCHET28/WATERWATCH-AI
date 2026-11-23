"""
Update the academic summary markdown cell (imp_study_md_3) in
notebooks/02_data_preprocessing.ipynb to include Random Forest results.
"""
import json

nb_path = 'notebooks/02_data_preprocessing.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_SUMMARY = [
    "## Phase 8 Summary: Imputation Comparison Findings\n",
    "\n",
    "### Complete 3x3 Benchmark Results (Holdout Test Set, n=656)\n",
    "\n",
    "| Imputation | Classifier | Accuracy | F1-Macro | ROC-AUC | Dangerous FP | FP Rate |\n",
    "|---|---|---|---|---|---|---|\n",
    "| Median (Baseline) | Dummy | 0.6098 | 0.3788 | 0.5000 | 0 | 0.00% |\n",
    "| Median (Baseline) | Gaussian NB | 0.6143 | 0.5062 | 0.6090 | 45 | 11.25% |\n",
    "| Median (Baseline) | Decision Tree | 0.6311 | 0.5489 | 0.6011 | 53 | 13.25% |\n",
    "| **Median (Baseline)** | **Random Forest** | **0.6662** | **0.5999** | **0.6492** | 48 | 12.00% |\n",
    "| KNN (k=5) | Gaussian NB | 0.6159 | 0.5090 | 0.6083 | 45 | 11.25% |\n",
    "| KNN (k=5) | Decision Tree | 0.6311 | 0.5268 | 0.5852 | **39** | **9.75%** |\n",
    "| KNN (k=5) | Random Forest | 0.6524 | 0.5803 | **0.6630** | 50 | 12.50% |\n",
    "| Iterative MICE | Gaussian NB | 0.6143 | 0.5062 | 0.6082 | 45 | 11.25% |\n",
    "| Iterative MICE | Decision Tree | 0.6326 | 0.5501 | 0.5917 | 52 | 13.00% |\n",
    "| **Iterative MICE** | **Random Forest** | **0.6784** | **0.6098** | 0.6492 | **40** | **10.00%** |\n",
    "\n",
    "### Key Findings\n",
    "\n",
    "**Best Overall:** Iterative MICE + Random Forest — highest accuracy (0.6784), highest F1-Macro (0.6098),  \n",
    "and lowest Random Forest dangerous FP rate (10.00%).\n",
    "\n",
    "**Safest Lightweight Option:** KNN Imputation + Decision Tree — lowest dangerous FP rate of any single-tree  \n",
    "configuration (9.75%, 39 errors; 26.4% reduction vs. Median+DT baseline).\n",
    "\n",
    "**Best ROC-AUC:** KNN + Random Forest (0.6630) — best probability calibration across all configurations.\n",
    "\n",
    "**GNB Invariance:** Gaussian Naive Bayes scores are virtually identical regardless of imputation method  \n",
    "(accuracy range: 0.6143-0.6159), confirming that its conditional independence assumption eliminates any  \n",
    "benefit from multivariate imputation on this near-zero-correlation dataset.\n",
    "\n",
    "**Inter-feature correlations |r| < 0.04** limit the leverage of KNN and MICE over Median imputation.  \n",
    "The modest gains are consistent with the dataset's physicochemical structure.\n",
    "\n",
    "> Full report: `docs/IMPUTATION_METHOD_COMPARISON.md`  \n",
    "> Full metrics: `results/metrics/imputation_comparison.csv` (10 rows)  \n",
    "> Full metrics: `results/metrics/imputation_comparison.json` (10 experiments)\n"
]

updated = False
for i, cell in enumerate(nb['cells']):
    src = cell.get('source', [])
    src_text = ''.join(src)
    # Match the old summary cell by looking for the Phase 8 Summary heading
    if cell['cell_type'] == 'markdown' and 'Phase 8 Summary' in src_text:
        nb['cells'][i]['source'] = NEW_SUMMARY
        updated = True
        print('Updated cell ' + str(i) + ' (Phase 8 Summary markdown)')
        break

if not updated:
    print('WARNING: Phase 8 Summary cell not found — appending as new cell.')
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {'tags': ['imp_study_md_3']},
        'source': NEW_SUMMARY
    })

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook saved. Total cells: ' + str(len(nb['cells'])))
