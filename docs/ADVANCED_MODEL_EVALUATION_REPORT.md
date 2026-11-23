# WATERWATCH AI: Advanced Model Evaluation Report

**Document Version:** 2.0.0  
**Role:** Member 1 – Data & ML Engineer  
**Dataset:** `data/raw/water_potability.csv` (N=3,276, 9 features, binary target)  
**Split:** Stratified 80:20, random_state=42 (Train=2,620, Test=656)

---

## 1. Evaluation Methodology

All models were trained exclusively on the 80% training partition and evaluated on the held-out 20% test partition. Cross-validation was performed on training data only using 5-fold Stratified K-Fold (shuffle=True, random_state=42). Preprocessing steps (imputation, scaling) were always encapsulated inside Scikit-learn `Pipeline` objects to prevent data leakage.

---

## 2. Metric Definitions

| Metric | Definition |
|---|---|
| **Accuracy** | Proportion of correctly classified samples |
| **Balanced Accuracy** | Mean of class-specific recalls; robust to imbalance |
| **Macro Precision** | Unweighted mean precision across both classes |
| **Macro Recall** | Unweighted mean recall across both classes |
| **Macro F1** | Harmonic mean of macro precision and recall |
| **Weighted F1** | F1 weighted by class support |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Average Precision** | Area under the Precision-Recall curve |
| **MCC** | Matthews Correlation Coefficient: single balanced metric, range −1 to +1 |
| **Class 0 Recall** | Proportion of non-potable samples correctly classified |
| **Class 1 Recall** | Proportion of potable samples correctly classified |
| **Dangerous FP** | Actual non-potable (class 0) predicted as potable (class 1) |

---

## 3. Confusion Matrix Interpretation

```
                  Predicted 0           Predicted 1
Actual 0 (Non-Potable)  TN (Safe Rejection)   FP ⚠️ DANGEROUS ERROR
Actual 1 (Potable)      FN (Precautionary)    TP (Safe Detection)
```

A **False Positive (FP)** in this project means non-potable water was incorrectly classified as potable. This is the critical screening error: communities may consume contaminated water.

---

## 4. Full Model Comparison (Test Set)

| Imputation | Classifier | Accuracy | Bal. Acc | F1-Macro | ROC-AUC | MCC | Dangerous FP | FP Rate |
|---|---|---|---|---|---|---|---|---|
| Median | Gaussian Naive Bayes | 0.6143 | 0.5375 | 0.5062 | 0.609 | 0.1049 | 45 | 11.25% |
| Median | Decision Tree | 0.6311 | 0.5646 | 0.5489 | 0.6011 | 0.163 | 53 | 13.25% |
| Median | Random Forest | 0.6662 | 0.606 | 0.5999 | 0.6492 | 0.2573 | 48 | 12.0% |
| KNN (k=5) | Gaussian Naive Bayes | 0.6159 | 0.5395 | 0.509 | 0.6083 | 0.1099 | 45 | 11.25% |
| KNN (k=5) | Decision Tree | 0.6311 | 0.5548 | 0.5268 | 0.5852 | 0.1539 | 39 | 9.75% |
| KNN (k=5) | Random Forest | 0.6524 | 0.5898 | 0.5803 | 0.6629 | 0.2212 | 50 | 12.5% |
| Iterative MICE | Gaussian Naive Bayes | 0.6143 | 0.5375 | 0.5062 | 0.6082 | 0.1049 | 45 | 11.25% |
| Iterative MICE | Decision Tree | 0.6326 | 0.5659 | 0.5501 | 0.5917 | 0.1667 | 52 | 13.0% |
| Iterative MICE | Random Forest | 0.6784 | 0.616 | 0.6098 | 0.6492 | 0.2882 | 40 | 10.0% |

---

## 5. Best Configuration by Classifier

- **Best for Gaussian NB:** KNN (k=5) (F1-Macro 0.509, MCC 0.1099)
- **Best for Decision Tree:** Iterative MICE (F1-Macro 0.5501, MCC 0.1667)
- **Best for Random Forest:** Iterative MICE (F1-Macro 0.6098, MCC 0.2882)
- **Best Overall:** Iterative MICE + Random Forest (Accuracy 0.6784, F1-Macro 0.6098, MCC 0.2882)

---

## 6. Threshold Comparison

The default probability threshold of 0.5 is not necessarily optimal for community water screening.
Thresholds evaluated: 0.10 through 0.90 in steps of 0.10.

Key trade-off: lowering the threshold increases Class 0 recall (fewer dangerous FPs) but decreases Class 1 recall (more false alarms). No threshold was selected by evaluating on the test set during development.

Results saved in: `results/metrics/threshold_comparison.csv`  
Visualization: `results/figures/threshold_comparison.png`

---

## 7. Precision-Recall Analysis

Precision-Recall curves were generated for all 9 ML configurations (excluding Dummy baseline).

- **Class 1 (Potable) PR curves:** `results/figures/precision_recall_comparison.png`
- **Class 0 (Non-Potable) PR curves:** `results/figures/class_specific_precision_recall.png`

Average Precision (AP) summarizes PR curve area, insensitive to threshold choice.

---

## 8. Dangerous False-Positive Analysis

Non-potable water incorrectly classified as potable represents the highest-risk error in this system.

Minimum dangerous FP rates observed:
- Decision Tree: 9.75% (KNN imputation, 39/400 non-potable samples)
- Random Forest: 10.00% (Iterative MICE, 40/400 non-potable samples)

**Important disclaimer:** These are model classification errors on a held-out test set. They do not constitute real-world water safety guarantees. All predictions are preliminary screening results only.

---

## 9. Cross-Validation Findings

5-fold stratified CV was run on training data for every configuration. CV accuracy mean and standard deviation confirm that results are internally stable (all standard deviations ≤ 0.016).

---

## 10. Limitations and Recommendations

1. All 9 features are bulk physicochemical parameters. Microbial contamination, heavy metals and pesticides are not measured and cannot be predicted.
2. MCC and balanced accuracy are preferred over raw accuracy given mild class imbalance.
3. The Iterative MICE + Random Forest configuration provides the highest MCC, balanced accuracy, and lowest dangerous FP rate among RF models. It is recommended as the upgraded production pipeline subject to validation on independent data.
4. Threshold selection must be done using out-of-fold predictions, never the test set.
5. No model configuration should certify water potability without laboratory assay confirmation.

---
*Report generated by Member 1 – Data & ML Engineer, WATERWATCH AI.*
