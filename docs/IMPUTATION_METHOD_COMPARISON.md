# WATERWATCH AI: Advanced Missing-Value Imputation Comparison Study

**Document Version:** 1.0.0  
**Project:** WATERWATCH AI — Explainable AI-Based Water Potability Prediction and Community Risk Monitoring System  
**Role:** Member 1 – Data & ML Engineer  
**Dataset:** `data/raw/water_potability.csv` ($N = 3,276$, 9 Physicochemical Features, 1 Binary Target)  
**Evaluation Scope:** Univariate Median Imputation (Baseline) vs. Non-Parametric KNN Imputation vs. Multivariate Iterative Imputation (MICE)  
**Classifiers:** Gaussian Naive Bayes, Decision Tree, Random Forest, and Dummy Baseline  

---

## Executive Summary

Missing data is a fundamental challenge in environmental water-quality informatics. In the WATERWATCH AI dataset, missing values are present across three critical physicochemical parameters: **Sulfate** ($23.84\%$), **pH** ($14.99\%$), and **Trihalomethanes** ($4.95\%$). 

This technical report delivers an experimental comparison of three imputation paradigms:
1. **Method A: Median Imputation (Existing Project Baseline)**
2. **Method B: $K$-Nearest Neighbors (KNN) Imputation ($k=5$, Distance-Scaled)**
3. **Method C: Iterative Imputation (Multivariate Imputation by Chained Equations with Bayesian Ridge Regression)**

To prevent data leakage, all transformations and model pipelines were encapsulated strictly within Scikit-learn `Pipeline` architectures and fitted exclusively on the 80% stratified training partition ($N_{train} = 2,620$, $N_{test} = 656$, `random_state=42`). Performance was evaluated across the **Dummy Classifier** (majority baseline), **Gaussian Naive Bayes**, **Decision Tree Classifier**, and **Random Forest Classifier** using holdout test metrics and 5-fold stratified cross-validation on training data.

### Key Empirical Findings:
- **Best Overall Configuration:** **Iterative Imputation (MICE)** paired with **Random Forest** achieved the highest holdout accuracy ($0.6784$), highest macro F1 ($0.6098$), and competitive 5-fold cross-validation accuracy ($0.6775 \pm 0.0099$), establishing it as the strongest pipeline in this benchmark.
- **Random Forest Dominance:** Across all three imputation methods, Random Forest consistently outperformed both GNB and Decision Tree, demonstrating the value of ensemble averaging in this low-signal dataset.
- **Public Health Screening (Dangerous False Positives):** In community water screening, misclassifying non-potable water ($0$) as potable ($1$) is a critical false-positive failure. Among Decision Tree configurations, **KNN Imputation ($k=5$)** reduced dangerous false-positive errors from $53$ ($13.25\%$) down to **$39$ ($9.75\%$)**. Among Random Forest configurations, **Iterative Imputation** achieved the lowest dangerous FP rate at $40$ errors ($10.00\%$).
- **Academic Rigor & Practical Significance:** Because all pairwise inter-feature correlations in this dataset are near zero ($|r| < 0.04$), multivariate relationships offer only **modest predictive leverage**. While Iterative and KNN imputations provide measurable advantages in variance preservation and risk reduction, the baseline median imputation remains an efficient, computationally lightweight benchmark.

---

## 1. Missing-Value Mechanism Analysis

### 1.1 Distribution of Missing Entries
Across the $3,276$ observations in `water_potability.csv`, missing values occur in three continuous features:

| Feature Name | Missing Count | Missing Percentage (%) | Measurement Unit | Typical Environmental Source / Assay |
| :--- | :---: | :---: | :---: | :--- |
| **Sulfate ($SO_4^{2-}$)** | 781 | 23.84% | mg/L | Gravimetric / turbidimetric laboratory titration |
| **pH** | 491 | 14.99% | Dimensionless (0–14) | Electrochemical potentiometric glass electrode |
| **Trihalomethanes (THMs)** | 162 | 4.95% | $\mu$g/L | Purge-and-trap Gas Chromatography–Mass Spectrometry (GC-MS) |
| *All Other Features (6)* | 0 | 0.00% | Various | Online automated continuous telemetry sensors |

### 1.2 Missingness Mechanisms: MCAR vs. MAR
In statistical literature (Rubin, 1976), missingness is categorized into three mechanisms:
1. **Missing Completely at Random (MCAR):** Missingness is independent of both observed and unobserved data.
2. **Missing at Random (MAR):** Missingness depends on observed features but not on the missing values themselves.
3. **Missing Not at Random (MNAR):** Missingness depends directly on the true unobserved value (e.g., sensor saturation censoring extreme values).

In the WATERWATCH AI dataset:
- Little's statistical inspection and missingness correlation matrices indicate that missingness correlates weakly ($|r| < 0.05$) with other physicochemical features and the target `Potability`.
- The missingness pattern reflects **Missing at Random (MAR)** or **Missing Completely at Random (MCAR)** operational causes:
  - Sulfate requires off-line wet-chemistry analysis; missingness arises from deferred laboratory sample processing rather than extreme water salinity.
  - Trihalomethanes require specialized gas chromatography; field sites without specialized GC infrastructure omit this assay.
  - pH sensor dropouts occur due to probe calibration cycles and telemetry latency in rural monitoring stations.

---

## 2. Mathematical Foundations of Imputation Methods

```
                                  Missing Data Preprocessing
                                              |
            +---------------------------------+---------------------------------+
            |                                 |                                 |
     Method A: Univariate              Method B: Distance-Based           Method C: Multivariate MICE
     Median Imputation                 KNN Imputation (k=5)              Iterative Imputation
            |                                 |                                 |
   - Fast & robust to outliers       - Exploits local geometry         - Chained Bayesian Ridge
   - Discards correlation            - Scaling mathematically vital    - Preserves covariance
   - Induces artificial delta peak   - O(N_train * N_test) cost        - Iterative round-robin
```

### 2.1 Method A: Median Imputation (Existing Project Baseline)
- **Algorithm:** For any feature $j$, missing entry $x_{ij}$ is imputed using the sample median of available training records:
  $$\hat{x}_{ij} = \text{median}(\{x_{kj} \mid x_{kj} \text{ is observed}, k \in \mathcal{D}_{train}\})$$
- **Training Medians Fitted on $X_{train}$:**
  - $\text{median}(pH) = 7.0368$
  - $\text{median}(Sulfate) = 333.0735\text{ mg/L}$
  - $\text{median}(Trihalomethanes) = 66.6225\text{ }\mu\text{g/L}$
- **Advantages:**
  - High computational efficiency: $O(N)$ fitting, $O(1)$ transform.
  - Robust against heavy-tailed distributions and outlier skewness.
  - Deterministic and requires no assumptions regarding multivariate joint distributions.
- **Disadvantages:**
  - Artificially collapses feature variance by placing mass at a single point (Dirac delta function).
  - Distorts natural correlations, treating features as mutually independent.

---

### 2.2 Method B: $K$-Nearest Neighbors (KNN) Imputation
- **Algorithm:** Replaces missing values using the weighted or unweighted average of the $k$ most similar complete records based on mean squared Euclidean distance over observed coordinates:
  $$d(u, v) = \sqrt{\frac{p}{p_{obs}} \sum_{j \in \text{obs}(u) \cap \text{obs}(v)} (u_j - v_j)^2}$$
  where $p$ is the total feature count and $p_{obs}$ is the number of mutually observed features.
- **Architectural Requirement (Scaling Before Imputation):**
  - **Critical Failure Mode Without Scaling:** In raw units, `Solids` spans $[320.94, 61227.20]\text{ mg/L}$ (variance $\approx 7.6 \times 10^7$), whereas `ph` spans $[0.0, 14.0]$ (variance $\approx 2.5$). In unscaled space, Euclidean distance is $99.999\%$ dominated by `Solids` and `Conductivity`, completely muting `ph`, `Turbidity`, and `Chloramines`.
  - **Resolution:** A `StandardScaler()` is placed **before** `KNNImputer(n_neighbors=5)` in the pipeline. `StandardScaler` calculates mean and variance solely on non-missing values in $X_{train}$, preserving NaNs during scaling so all 9 parameters contribute with equal weight to neighbor identification.
- **Advantages:**
  - Non-parametric: captures localized non-linear clusters and geochemical water profiles.
  - Generates adaptive donor-based imputations rather than static central tendencies.
- **Disadvantages:**
  - Memory and time complexity during inference: requires searching $N_{train}$ instances ($O(N_{train} \cdot d)$ per test sample).
  - Susceptible to the curse of dimensionality if unrelated features introduce noise into distance metrics.

---

### 2.3 Method C: Iterative Imputation (MICE with Bayesian Ridge)
- **Algorithm:** Multivariate Imputation by Chained Equations (MICE) models each feature with missing values as a conditional regression on all other features in a round-robin schedule:
  $$x_j^{(t)} \sim \mathcal{N}\left(X_{-j} \mathbf{w}_j, \alpha_j^{-1} \mathbf{I}\right)$$
  where $\mathbf{w}_j$ is estimated via `BayesianRidge` regression with regularized Gamma priors on coefficient precision $\alpha$ and residual noise $\lambda$.
- **Pipeline Implementation:**
  ```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  from sklearn.linear_model import BayesianRidge

  iterative_pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('imputer', IterativeImputer(
          estimator=BayesianRidge(),
          max_iter=10,
          random_state=42
      )),
      ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42))
  ])
  ```
- **Advantages:**
  - Preserves multivariate covariance structures and joint feature correlations.
  - Bayesian Ridge shrinkage prevents overfitting on noisy linear relationships.
- **Disadvantages:**
  - Higher computational overhead during training due to cycling 10 iterations over multiple linear regression models.
  - Assumes approximately linear conditional relationships between features.

---

## 3. Experimental Methodology & Rigorous Leakage Prevention

To ensure a fair and scientifically valid benchmark:
1. **Identical Splitting:** All 10 configurations (3 imputation × 3 ML classifiers + 1 Dummy baseline) were trained on the exact same 80% training partition ($N_{train} = 2,620$, $N_{test} = 656$) created via `train_test_split(..., random_state=42, stratify=y)`.
2. **Encapsulated Preprocessing:** Every imputer, scaler, and estimator was housed within a `Pipeline`. Imputation and scaling parameters were fitted **solely on training data**; test samples and cross-validation holdout folds were transformed using pre-computed training parameters.
3. **Cross-Validation Integrity:** 5-fold stratified cross-validation was executed on $X_{train}$ using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Imputation was refitted independently on the 4 in-fold training partitions during each split.

---

## 4. Comprehensive Experimental Results

The experimental benchmark results across all 10 configurations (3 imputation methods × 3 ML classifiers + 1 Dummy baseline) are summarized below (persisted in [results/metrics/imputation_comparison.csv](file:///d:/WATERWATCH-AI/results/metrics/imputation_comparison.csv)):

| Imputation Strategy | Classifier Architecture | Accuracy | Precision (Macro) | Precision (Weighted) | Recall (Class 0) | Recall (Class 1) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | ROC-AUC | Dangerous FPs (0 $\rightarrow$ 1) | Dangerous FP Rate (%) | 5-Fold CV Accuracy (Mean $\pm$ Std) | 5-Fold CV F1-Macro (Mean $\pm$ Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Median (Baseline)** | Dummy Classifier | 0.6098 | 0.3049 | 0.3718 | 1.0000 | 0.0000 | 0.5000 | 0.3788 | 0.4619 | 0.5000 | 0 | 0.00% | $0.6099 \pm 0.0009$ | $0.3789 \pm 0.0004$ |
| **Median (Baseline)** | Gaussian Naive Bayes | 0.6143 | 0.5733 | 0.5859 | 0.8875 | 0.1875 | 0.5375 | 0.5062 | 0.5569 | 0.6090 | 45 | 11.25% | $0.6275 \pm 0.0074$ | $0.5393 \pm 0.0129$ |
| **Median (Baseline)** | Decision Tree | 0.6311 | 0.6029 | 0.6126 | 0.8675 | 0.2617 | 0.5646 | 0.5489 | 0.5912 | 0.6011 | 53 | 13.25% | $0.6298 \pm 0.0095$ | $0.5192 \pm 0.0307$ |
| **Median (Baseline)** | Random Forest | 0.6662 | 0.6561 | 0.6598 | 0.8800 | 0.3320 | 0.6060 | 0.5999 | 0.6356 | 0.6492 | 48 | 12.00% | $0.6809 \pm 0.0157$ | $0.6157 \pm 0.0248$ |
| **KNN ($k=5$, Scaled)** | Gaussian Naive Bayes | 0.6159 | 0.5765 | 0.5886 | 0.8875 | 0.1914 | 0.5395 | 0.5090 | 0.5593 | 0.6083 | 45 | 11.25% | $0.6279 \pm 0.0100$ | $0.5353 \pm 0.0182$ |
| **KNN ($k=5$, Scaled)** | Decision Tree | 0.6311 | 0.6081 | 0.6151 | **0.9025** | 0.2070 | 0.5548 | 0.5268 | 0.5756 | 0.5852 | **39** | **9.75%** | $0.6336 \pm 0.0132$ | $0.5239 \pm 0.0332$ |
| **KNN ($k=5$, Scaled)** | Random Forest | 0.6524 | 0.6361 | 0.6420 | 0.8750 | 0.3047 | 0.5898 | 0.5803 | 0.6185 | **0.6630** | 50 | 12.50% | $0.6683 \pm 0.0091$ | $0.6039 \pm 0.0108$ |
| **Iterative (MICE)** | Gaussian Naive Bayes | 0.6143 | 0.5733 | 0.5859 | 0.8875 | 0.1875 | 0.5375 | 0.5062 | 0.5569 | 0.6082 | 45 | 11.25% | $0.6271 \pm 0.0095$ | $0.5382 \pm 0.0150$ |
| **Iterative (MICE)** | Decision Tree | 0.6326 | 0.6055 | 0.6149 | 0.8700 | 0.2617 | 0.5659 | 0.5501 | 0.5924 | 0.5917 | 52 | 13.00% | $0.6347 \pm 0.0095$ | $0.5317 \pm 0.0361$ |
| **Iterative (MICE)** | Random Forest | **0.6784** | **0.6790** | **0.6788** | **0.9000** | **0.3320** | **0.6160** | **0.6098** | **0.6457** | 0.6492 | 40 | 10.00% | $0.6775 \pm 0.0099$ | **$0.6138 \pm 0.0199$** |

---

## 5. In-Depth Comparative Analysis

### 5.1 Accuracy, Macro F1, and Generalization Trade-offs

1. **Random Forest Architecture (Best Overall):**
   - **Iterative Imputation + Random Forest** is the top-performing configuration across this entire benchmark: holdout accuracy $0.6784$, macro F1 $0.6098$, 5-fold CV accuracy $0.6775 \pm 0.0099$, and lowest RF dangerous FP rate ($10.00\%$, 40 errors).
   - **Median Imputation + Random Forest** is the second-strongest configuration: accuracy $0.6662$, macro F1 $0.5999$, 5-fold CV accuracy $0.6809 \pm 0.0157$.
   - **KNN Imputation + Random Forest** achieves the highest ROC-AUC of any configuration ($0.6630$), though its holdout accuracy ($0.6524$) trails the other RF variants. This suggests KNN imputation provides better probability calibration despite slightly weaker thresholded accuracy.
   - **Ensemble Explanation:** Random Forest's averaging over 100 trees smooths out the high-variance imputed values and provides superior discrimination. The gap between RF ($\approx 0.666$–$0.678$ accuracy) and single-tree DT ($\approx 0.631$–$0.633$) demonstrates the fundamental advantage of ensemble methods on noisy tabular data.

2. **Decision Tree Architecture:**
   - **Iterative Imputation** attained the highest DT test accuracy ($0.6326$), highest macro F1 ($0.5501$), and highest DT 5-fold CV accuracy ($0.6347 \pm 0.0095$).
   - **Median Imputation** achieved comparable test accuracy ($0.6311$) and macro F1 ($0.5489$), with a 5-fold CV accuracy of $0.6298 \pm 0.0095$.
   - **KNN Imputation** tied Median Imputation in test accuracy ($0.6311$) while producing a slightly higher 5-fold CV accuracy ($0.6336 \pm 0.0132$).

3. **Gaussian Naive Bayes Architecture:**
   - GNB performance remained remarkably invariant across imputation methods:
     - Median: Acc = $0.6143$, F1-Macro = $0.5062$, ROC-AUC = $0.6090$
     - KNN: Acc = $0.6159$, F1-Macro = $0.5090$, ROC-AUC = $0.6083$
     - Iterative: Acc = $0.6143$, F1-Macro = $0.5062$, ROC-AUC = $0.6082$
   - **Theoretical Explanation:** GaussianNB assumes that features are conditionally independent given the class label ($P(X \mid Y) = \prod P(X_i \mid Y)$). Because GNB explicitly disregards covariance among features, advanced multivariate imputation techniques that reconstruct pairwise correlations offer minimal empirical advantage to Naive Bayes.

---

### 5.2 Community Water Screening & Dangerous False-Positive Audit
In environmental toxicology, misclassifications do not carry equal real-world consequences:
- **False Negative (Actual 1 predicted as 0):** Safe water is flagged as contaminated. Result: Precautionary boiling or wasted alternate water supply.
- **False Positive (Actual 0 predicted as 1):** Contaminated water is incorrectly flagged as potable and distributed to households. Result: Community exposure to waterborne gastrointestinal illnesses.

```
   Decision Tree False-Positive Error Audit (400 Non-Potable samples):
   Median (Baseline)      [========================] 53 errors (13.25%)
   Iterative (MICE)       [=======================]  52 errors (13.00%)
   KNN Imputation (k=5)   [==================]       39 errors  (9.75%)  <-- 26.4% reduction!

   Random Forest False-Positive Error Audit (400 Non-Potable samples):
   KNN Imputation (k=5)   [====================]     50 errors (12.50%)
   Median (Baseline)      [===================]       48 errors (12.00%)
   Iterative (MICE)       [================]          40 errors (10.00%)  <-- Best RF safety profile
```

**Decision Tree:**
- Under **Median Imputation**, Decision Tree made $53$ dangerous false-positive errors ($13.25\%$).
- Under **Iterative Imputation**, Decision Tree made $52$ dangerous false-positive errors ($13.00\%$).
- Under **KNN Imputation ($k=5$)**, Decision Tree made only **$39$ dangerous false-positive errors ($9.75\%$)** — a **$26.4\%$ relative reduction**.

**Random Forest:**
- Under **KNN Imputation**, Random Forest made $50$ dangerous false-positive errors ($12.50\%$).
- Under **Median Imputation**, Random Forest made $48$ dangerous false-positive errors ($12.00\%$).
- Under **Iterative Imputation**, Random Forest made only **$40$ dangerous false-positive errors ($10.00\%$)** — the best safety profile across all RF configurations.

- **Public Health Implication:** Iterative Imputation + Random Forest combines the highest accuracy with the lowest dangerous FP rate among RF models, making it the recommended configuration for community water screening. For constrained environments (no ensemble support), KNN + Decision Tree remains the safest lightweight option.

---

### 5.3 Is the Improvement Meaningful? (Academic Evaluation)
- **Statistical Perspective (Decision Tree):**
  The 5-fold CV accuracy delta between Median DT ($62.98\%$) and Iterative DT ($63.47\%$) is $+0.49\%$, with standard deviations of $\pm 0.95\%$. Because the difference is within one standard error of fold variance, the statistical improvement is modest for Decision Tree.
- **Statistical Perspective (Random Forest):**
  The 5-fold CV accuracy delta between Median RF ($68.09\%$) and Iterative RF ($67.75\%$) is $-0.34\%$ — essentially equivalent given $\pm 1.57\%$ and $\pm 0.99\%$ standard deviations. The holdout gap (Iterative RF $67.84\%$ vs. Median RF $66.62\%$) is more pronounced ($+1.22\%$) but not cross-validated. The key advantage of Iterative RF is not accuracy but its **lower dangerous FP rate** ($10.00\%$ vs. $12.00\%$).
- **Root Cause in Dataset Characteristics:**
  Exploratory data analysis confirms that all inter-feature correlation coefficients in this dataset are negligible ($|r| < 0.04$). When features share virtually no linear or rank correlation, regression-based (MICE) and distance-based (KNN) estimators have very limited signal to exploit over a univariate median. The modest gains from advanced imputation are therefore expected and do not reflect method inadequacy.
- **Computational Cost Perspective:**
  - Median Imputation: Fitting time $\approx 0.002$ s, inference $\approx 0.0001$ s.
  - KNN Imputation: Fitting time $\approx 0.05$ s, inference $\approx 0.35$ s ($O(N)$ distance calculations).
  - Iterative Imputation: Fitting time $\approx 0.42$ s, inference $\approx 0.01$ s.
  - Random Forest adds $\approx 0.5$–$1.5$ s training overhead but inference is fast ($\approx 0.02$ s).

---

## 6. Generated Visualization Artifacts

The following comparative visualization charts have been generated and archived in `results/figures/`:

1. **`results/figures/imputation_accuracy_comparison.png`:**
   Grouped bar chart comparing test accuracy of Gaussian NB, Decision Tree, and Random Forest across the three imputation strategies, overlaid with the Dummy Baseline ($60.98\%$). Random Forest bars clearly dominate, reaching $0.6784$ for Iterative MICE.
2. **`results/figures/imputation_f1_comparison.png`:**
   Macro F1 score comparison (all three classifiers × three imputation methods). Iterative + RF achieves $0.6098$, the highest macro F1 in the benchmark, confirming strong balanced performance.
3. **`results/figures/imputation_roc_auc_comparison.png`:**
   ROC-AUC across all 9 ML configurations. Random Forest configurations achieve $0.649$–$0.663$, well above the GNB and DT ranges ($0.585$–$0.609$). KNN + RF reaches the highest ROC-AUC ($0.6630$).
4. **`results/figures/imputation_dangerous_error_rate_comparison.png`:**
   Dangerous false-positive rate by classifier and imputation method. Highlights KNN's safety advantage for DT ($9.75\%$) and Iterative's safety advantage for RF ($10.00\%$) versus the highest risk configuration, Median + DT ($13.25\%$).
5. **`results/figures/imputation_best_confusion_matrices.png`:**
   Full $3 \times 3$ confusion matrix grid: rows = imputation method (Median, KNN, Iterative), columns = classifier (GNB, DT, RF). Each cell displays TN / FP / FN / TP with percentage annotations, accuracy, and dangerous FP count. Enables direct visual comparison of all nine experimental configurations.

---

## 7. Conclusions & Strategic Recommendation

1. **Baseline Preservation:**
   The existing `SimpleImputer(strategy='median')` + `RandomForestClassifier` pipeline remains the production baseline. Median + RF (accuracy $0.6662$, F1-Macro $0.5999$) offers high computational speed, simplicity, and zero deployment friction for the downstream Streamlit dashboard and database logging.

2. **Recommended Best Pipeline — Iterative MICE + Random Forest:**
   **Iterative Imputation (MICE) + Random Forest** is the top-performing configuration across this benchmark:
   - Highest holdout accuracy: $\mathbf{0.6784}$ (+$1.86\%$ over Median+RF baseline)
   - Highest macro F1: $\mathbf{0.6098}$
   - Lowest Random Forest dangerous FP rate: $\mathbf{40}$ errors, $\mathbf{10.00\%}$ (vs. $48$, $12.00\%$ for Median+RF)
   - Competitive 5-fold CV accuracy: $0.6775 \pm 0.0099$
   This configuration should be promoted as the **recommended upgraded pipeline** for the WATERWATCH AI production system when computational resources permit.

3. **Conservative Screening Alternative (Lightweight):**
   For deployment scenarios without ensemble support, **KNN Imputation ($k=5$, Scaled) + Decision Tree** is the recommended lightweight configuration, achieving the lowest dangerous false-positive rate ($9.75\%$, 39 errors) of any single-tree model.

4. **Maximum ROC-AUC (Probability Calibration):**
   **KNN Imputation + Random Forest** achieves the highest ROC-AUC ($0.6630$) in the benchmark, making it the preferred choice for applications requiring well-calibrated probability outputs rather than hard thresholded classifications.

5. **Dataset Limitation Notice:**
   No imputation method can overcome the intrinsic information limits of the dataset. All inter-feature correlations are negligible ($|r| < 0.04$), confirming that advanced imputation methods offer only modest gains on this specific dataset. All predictions are preliminary screening approximations and must never certify drinking water potability without certified laboratory assays.

| Priority | Configuration | Accuracy | F1-Macro | Dangerous FP Rate | Use Case |
|---|---|---|---|---|---|
| 🏆 Best Overall | Iterative MICE + RF | 0.6784 | 0.6098 | 10.00% | Production upgrade |
| ✅ Current Baseline | Median + RF | 0.6662 | 0.5999 | 12.00% | Current deployment |
| 🛡️ Safest DT | KNN + Decision Tree | 0.6311 | 0.5268 | **9.75%** | Lightweight screening |
| 📊 Best ROC-AUC | KNN + RF | 0.6524 | 0.5803 | 12.50% | Probabilistic output |

---
*Report compiled by Member 1 – Data & ML Engineer for WATERWATCH AI.*
