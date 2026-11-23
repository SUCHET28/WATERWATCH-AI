"""
WATERWATCH AI - Advanced Missing-Value Imputation Comparison Study
Member 1 - Data & ML Engineer Deliverable

Comprehensive 3x3 Imputation Benchmark (+ Baseline):
Imputation Methods:
1. Median Imputation (Existing Baseline)
2. KNN Imputation (k=5, distance-scaled)
3. Iterative Imputation (MICE, BayesianRidge, max_iter=10, random_state=42)

Classifiers:
- Model A: Gaussian Naive Bayes
- Model B: Decision Tree (max_depth=6, min_samples_leaf=10, random_state=42)
- Model C: Random Forest (n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
- Baseline: Dummy Classifier (strategy='most_frequent')
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Set visual styling
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

def main():
    print("=" * 80)
    print("WATERWATCH AI: COMPREHENSIVE 3x3 IMPUTATION & CLASSIFIER BENCHMARK")
    print("=" * 80)

    # 1. Load Dataset
    data_path = 'data/raw/water_potability.csv'
    if not os.path.exists(data_path):
        data_path = '../data/raw/water_potability.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [c for c in df.columns if c != 'Potability']
    target_col = 'Potability'
    X = df[feature_cols]
    y = df[target_col]

    # 2. Stratified 80/20 train-test split (identical across all experiments)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Test Class 0 (Non-Potable): {(y_test == 0).sum()} | Test Class 1 (Potable): {(y_test == 1).sum()}")

    # 5-fold Stratified CV for training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Define the Experimental Configurations
    # 3 Imputation Methods x 3 Classifiers + Dummy Baseline
    pipeline_configs = [
        # --- Baseline Model ---
        {
            'imputation_method': 'Median Imputation (Baseline)',
            'classifier_name': 'Dummy Classifier',
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', DummyClassifier(strategy='most_frequent'))
            ])
        },

        # --- Method A: Median Imputation ---
        {
            'imputation_method': 'Median Imputation (Baseline)',
            'classifier_name': 'Gaussian Naive Bayes',
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', GaussianNB())
            ])
        },
        {
            'imputation_method': 'Median Imputation (Baseline)',
            'classifier_name': 'Decision Tree',
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42))
            ])
        },
        {
            'imputation_method': 'Median Imputation (Baseline)',
            'classifier_name': 'Random Forest',
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
            ])
        },

        # --- Method B: KNN Imputation ---
        {
            'imputation_method': 'KNN Imputation (k=5)',
            'classifier_name': 'Gaussian Naive Bayes',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('classifier', GaussianNB())
            ])
        },
        {
            'imputation_method': 'KNN Imputation (k=5)',
            'classifier_name': 'Decision Tree',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42))
            ])
        },
        {
            'imputation_method': 'KNN Imputation (k=5)',
            'classifier_name': 'Random Forest',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('classifier', RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
            ])
        },

        # --- Method C: Iterative Imputation ---
        {
            'imputation_method': 'Iterative Imputation (MICE)',
            'classifier_name': 'Gaussian Naive Bayes',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)),
                ('classifier', GaussianNB())
            ])
        },
        {
            'imputation_method': 'Iterative Imputation (MICE)',
            'classifier_name': 'Decision Tree',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)),
                ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42))
            ])
        },
        {
            'imputation_method': 'Iterative Imputation (MICE)',
            'classifier_name': 'Random Forest',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1))
            ])
        }
    ]

    results_table = []
    json_summary = {
        'study_title': 'WATERWATCH AI Comprehensive 3x3 Imputation & Classifier Benchmark',
        'dataset_shape': list(df.shape),
        'missing_values_profile': {
            'Sulfate': int(df['Sulfate'].isna().sum()),
            'ph': int(df['ph'].isna().sum()),
            'Trihalomethanes': int(df['Trihalomethanes'].isna().sum())
        },
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'experiments': []
    }

    fitted_models = {}

    print("\nRunning experimental evaluations (fitting pipelines and 5-fold cross-validation)...")
    for cfg in pipeline_configs:
        imp_method = cfg['imputation_method']
        clf_name = cfg['classifier_name']
        pipe = cfg['pipeline']
        print(f"  -> Evaluating: {imp_method} + {clf_name}...")

        # Fit on training set only (zero data leakage)
        pipe.fit(X_train, y_train)
        fitted_models[(imp_method, clf_name)] = pipe

        # Predictions on holdout test set
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            y_prob = np.full(len(y_test), 0.5)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Holdout Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        prec_wt = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_0 = recall_score(y_test, y_pred, pos_label=0) # Non-Potable recall
        rec_1 = recall_score(y_test, y_pred, pos_label=1) # Potable recall
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5000
        except ValueError:
            auc = 0.5000

        # Dangerous Screening Error Audit: Actual Class 0 (Non-Potable) predicted as Class 1 (Potable)
        dangerous_fp = int(fp)
        dangerous_fp_rate = round((fp / (tn + fp)) * 100, 2)

        # 5-Fold Stratified Cross-Validation strictly on training data
        acc_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        f1_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

        row = {
            'Imputation Method': imp_method,
            'Classifier': clf_name,
            'Accuracy': round(acc, 4),
            'Precision (Macro)': round(prec_macro, 4),
            'Precision (Weighted)': round(prec_wt, 4),
            'Recall (Non-Potable / 0)': round(rec_0, 4),
            'Recall (Potable / 1)': round(rec_1, 4),
            'Recall (Macro)': round(rec_macro, 4),
            'F1-Score (Macro)': round(f1_macro, 4),
            'F1-Score (Weighted)': round(f1_wt, 4),
            'ROC-AUC': round(auc, 4),
            'Dangerous Errors (Actual 0 -> Pred 1)': dangerous_fp,
            'Dangerous Error Rate (%)': dangerous_fp_rate,
            '5-Fold CV Accuracy Mean': round(acc_cv.mean(), 4),
            '5-Fold CV Accuracy Std': round(acc_cv.std(), 4),
            '5-Fold CV F1-Macro Mean': round(f1_cv.mean(), 4),
            '5-Fold CV F1-Macro Std': round(f1_cv.std(), 4)
        }
        results_table.append(row)

        json_summary['experiments'].append({
            'imputation_method': imp_method,
            'classifier': clf_name,
            'test_metrics': {
                'accuracy': round(acc, 4),
                'precision_macro': round(prec_macro, 4),
                'precision_weighted': round(prec_wt, 4),
                'recall_non_potable_0': round(rec_0, 4),
                'recall_potable_1': round(rec_1, 4),
                'recall_macro': round(rec_macro, 4),
                'f1_macro': round(f1_macro, 4),
                'f1_weighted': round(f1_wt, 4),
                'roc_auc': round(auc, 4),
                'dangerous_fp_count': dangerous_fp,
                'dangerous_fp_rate_pct': dangerous_fp_rate
            },
            'confusion_matrix': {
                'true_negative_safe_rejection': int(tn),
                'false_positive_dangerous_error': int(fp),
                'false_negative_waste_error': int(fn),
                'true_positive_safe_detection': int(tp)
            },
            'cross_validation_5fold': {
                'accuracy_mean': round(acc_cv.mean(), 4),
                'accuracy_std': round(acc_cv.std(), 4),
                'f1_macro_mean': round(f1_cv.mean(), 4),
                'f1_macro_std': round(f1_cv.std(), 4)
            }
        })

    # Convert to DataFrame
    res_df = pd.DataFrame(results_table)
    print("\n" + "=" * 80)
    print("ALL 10 COMBINATIONS RESULTS TABLE:")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # 4. Save CSV and JSON Artifacts
    os.makedirs('results/metrics', exist_ok=True)
    csv_path = 'results/metrics/imputation_comparison_matrix.csv'
    json_path = 'results/metrics/imputation_comparison.json'

    res_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV results to: {csv_path}")

    # Also try writing imputation_comparison.csv if unlocked
    try:
        res_df.to_csv('results/metrics/imputation_comparison.csv', index=False)
        print("✓ Also updated results/metrics/imputation_comparison.csv")
    except Exception as e:
        print(f"! Notice: results/metrics/imputation_comparison.csv is currently open in another process ({e}). Saved to {csv_path}.")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"✓ Saved JSON summary to: {json_path}")

    # 5. Generate Figures
    os.makedirs('results/figures', exist_ok=True)
    generate_all_visualizations(res_df, fitted_models, X_test, y_test)
    print("\n✓ All imputation study tasks completed successfully!")

def generate_all_visualizations(res_df, fitted_models, X_test, y_test):
    print("\nGenerating publication-quality comparison charts in results/figures/...")

    palette = {
        'Median Imputation (Baseline)': '#2b5c8f',
        'KNN Imputation (k=5)': '#2a9d8f',
        'Iterative Imputation (MICE)': '#e76f51'
    }

    # Filter out Dummy Classifier for bar comparisons across the 3 ML models
    filtered_df = res_df[res_df['Classifier'] != 'Dummy Classifier'].copy()

    # --- Figure 1: Imputation Method vs Accuracy ---
    plt.figure(figsize=(10.5, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='Classifier',
        y='Accuracy',
        hue='Imputation Method',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Figure 1: Test Accuracy Across Imputation Methods and ML Classifiers', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Test Accuracy', fontsize=11, fontweight='semibold')
    plt.xlabel('Machine Learning Classifier', fontsize=11, fontweight='semibold')
    plt.ylim(0.55, 0.72)
    plt.axhline(0.6098, color='crimson', linestyle='--', linewidth=1.2, label='Dummy Baseline (0.6098)')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')

    plt.legend(title='Imputation Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig1_path = 'results/figures/imputation_accuracy_comparison.png'
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"✓ Saved: {fig1_path}")

    # --- Figure 2: Imputation Method vs Macro F1 ---
    plt.figure(figsize=(10.5, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='Classifier',
        y='F1-Score (Macro)',
        hue='Imputation Method',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Figure 2: Macro F1-Score Across Imputation Methods and ML Classifiers', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Macro F1-Score', fontsize=11, fontweight='semibold')
    plt.xlabel('Machine Learning Classifier', fontsize=11, fontweight='semibold')
    plt.ylim(0.45, 0.68)
    plt.axhline(0.3788, color='crimson', linestyle='--', linewidth=1.2, label='Dummy Baseline (0.3788)')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')

    plt.legend(title='Imputation Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig2_path = 'results/figures/imputation_f1_comparison.png'
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"✓ Saved: {fig2_path}")

    # --- Figure 3: Imputation Method vs ROC-AUC ---
    plt.figure(figsize=(10.5, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='Classifier',
        y='ROC-AUC',
        hue='Imputation Method',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Figure 3: ROC-AUC Across Imputation Methods and ML Classifiers', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('ROC-AUC Score', fontsize=11, fontweight='semibold')
    plt.xlabel('Machine Learning Classifier', fontsize=11, fontweight='semibold')
    plt.ylim(0.50, 0.72)
    plt.axhline(0.5000, color='crimson', linestyle='--', linewidth=1.2, label='Random Ranking (0.5000)')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')

    plt.legend(title='Imputation Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig3_path = 'results/figures/imputation_roc_auc_comparison.png'
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"✓ Saved: {fig3_path}")

    # --- Figure 4: Comparison of Dangerous False-Positive Rates ---
    plt.figure(figsize=(10.5, 6))
    ax = sns.barplot(
        data=filtered_df,
        x='Classifier',
        y='Dangerous Error Rate (%)',
        hue='Imputation Method',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Figure 4: Dangerous False-Positive Rate (% Unsafe Water Flagged as Potable)', fontsize=12.5, fontweight='bold', pad=12)
    plt.ylabel('Dangerous Error Rate (%) [Actual 0 -> Predicted 1]', fontsize=10.5, fontweight='semibold')
    plt.xlabel('Machine Learning Classifier', fontsize=11, fontweight='semibold')
    plt.ylim(0, 18)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}%",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')

    plt.legend(title='Imputation Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig4_path = 'results/figures/imputation_dangerous_error_rate_comparison.png'
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"✓ Saved: {fig4_path}")

    # --- Figure 5: Multi-Model Confusion Matrices Across Imputation Strategies ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    classifiers = ['Gaussian Naive Bayes', 'Decision Tree', 'Random Forest']
    imputers = ['Median Imputation (Baseline)', 'KNN Imputation (k=5)', 'Iterative Imputation (MICE)']

    for row_idx, imp in enumerate(imputers):
        for col_idx, clf in enumerate(classifiers):
            ax = axes[row_idx, col_idx]
            model = fitted_models[(imp, clf)]
            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()

            labels = np.array([
                [f"TN (Safe Rej)\n{tn}\n({tn/400:.1%})", f"FP (DANGER)\n{fp}\n({fp/400:.1%})"],
                [f"FN (Alarm)\n{fn}\n({fn/256:.1%})", f"TP (Safe Det)\n{tp}\n({tp/256:.1%})"]
            ])

            sns.heatmap(
                cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
                annot_kws={'fontsize': 9, 'fontweight': 'semibold'},
                xticklabels=['0: Non-Pot', '1: Potable'],
                yticklabels=['0: Non-Pot', '1: Potable']
            )
            acc = accuracy_score(y_test, preds)
            ax.set_title(f"{clf}\n[{imp.split(' ')[0]}]\nAcc: {acc:.4f} | FP: {fp}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=9, fontweight='semibold')
            ax.set_ylabel('Actual Label', fontsize=9, fontweight='semibold')

    plt.suptitle('Figure 5: 3x3 Confusion Matrix Grid Across Classifiers & Imputation Methods', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig5_path = 'results/figures/imputation_best_confusion_matrices.png'
    plt.savefig(fig5_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {fig5_path}")

if __name__ == '__main__':
    main()
