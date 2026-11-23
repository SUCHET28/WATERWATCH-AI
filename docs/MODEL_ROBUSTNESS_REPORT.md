# WATERWATCH AI: Model Robustness Testing Report

**Document Version:** 1.0.0  
**Role:** Member 1 – Data & ML Engineer  
**Dataset:** `data/raw/water_potability.csv` (N=3,276, 9 features, binary target)

> **DISCLAIMER:** These experiments use simulated data perturbations and do not establish  
> real-world generalization, laboratory accuracy, or certification of water safety.

---

## 1. Test Methodology

Robustness testing evaluates how baseline model performance degrades under controlled, simulated deviations from the original test distribution. All existing fitted baseline models are evaluated on modified copies of the held-out test set without retraining.

**Models tested:**
- Gaussian Naive Bayes (models/gaussian_nb_pipeline.joblib)
- Decision Tree (models/decision_tree_pipeline.joblib)
- Random Forest (models/random_forest_pipeline.joblib)

**Experiment categories:**
1. Original baseline (no perturbation)
2. Additional missingness simulation (10%, 20%)
3. Measurement noise simulation (1%, 5% of training standard deviation)
4. Distribution shift simulation (+0.5, +1.0 training standard deviations)
5. Individual feature perturbation (±5% of training standard deviation per feature)

---

## 2. Simulation Assumptions

- Original test set (n=656) is never modified; perturbations are applied to independent copies.
- Random seeds are fixed for reproducibility (numpy.random.default_rng with seeds 42, 99, 7).
- Missingness is introduced only into previously observed (non-NaN) test values.
- Noise magnitude and shift magnitude are scaled relative to per-feature training standard deviations.
- These simulations model idealized perturbation scenarios, not real-world field conditions.

---

## 3. Baseline Performance (Original Untouched Test Set)

| Model | Accuracy | Bal. Acc | F1-Macro | Class 0 Recall | Class 1 Recall | ROC-AUC | MCC | Dangerous FP | Dangerous FP Rate |
|---|---|---|---|---|---|---|---|---|---|
| Gaussian NB | 0.6143 | 0.5375 | 0.5062 | 0.8875 | 0.1875 | 0.609 | 0.1049 | 45 | 11.25% |
| Decision Tree | 0.6311 | 0.5646 | 0.5489 | 0.8675 | 0.2617 | 0.6011 | 0.163 | 53 | 13.25% |
| Random Forest | 0.6631 | 0.5986 | 0.5887 | 0.8925 | 0.3047 | 0.6411 | 0.248 | 43 | 10.75% |

---

## 4. Performance Under Additional Missingness

Additional NaN values were randomly introduced into observed test-set entries at 10% and 20% rates.
Existing pipeline imputers handle the additional missing values using their training-fitted parameters (no retraining).

Results in: `results/metrics/robustness_test_results.csv` (Scenario = Missingness)  
Visualization: `results/figures/missingness_robustness.png`

---

## 5. Performance Under Measurement Noise

Gaussian noise was applied independently to each feature, scaled at 1% and 5% of that feature's training standard deviation.

Results in: `results/metrics/robustness_test_results.csv` (Scenario = Measurement Noise)  
Visualization: `results/figures/noise_robustness.png`

---

## 6. Performance Under Distribution Shift

All features were simultaneously shifted by +0.5 and +1.0 times their respective training standard deviations. Individual feature shifts (+1.0 std) were also tested for ph, Sulfate, and Chloramines.

Results in: `results/metrics/robustness_test_results.csv` (Scenario = Distribution Shift)  
Visualization: `results/figures/distribution_shift_robustness.png`

---

## 7. Individual Feature Perturbation

Each of the 9 input features was perturbed individually by ±5% of its training standard deviation. This reveals which features, when slightly changed, cause comparatively larger changes in model predictions.

**Most sensitive features (by accuracy delta, +5% perturbation):**
- Gaussian NB: most sensitive to perturbation of **ph** (accuracy delta 0.0015)
- Decision Tree: most sensitive to perturbation of **ph** (accuracy delta 0.0061)
- Random Forest: most sensitive to perturbation of **Chloramines** (accuracy delta 0.0091)

Visualization: `results/figures/feature_perturbation_robustness.png`

Do not interpret feature sensitivity as evidence of causal influence on water potability.

---

## 8. Important Observations

1. Tree-based models (Decision Tree, Random Forest) tend to be more robust to small perturbations because their decision structure is piece-wise constant.
2. Gaussian Naive Bayes may be more sensitive to distribution shifts because it models Gaussian class-conditional distributions.
3. All models degrade under heavy distribution shift (+1.0 std), which is expected for any ML model applied outside its training distribution.
4. The imputation step in each pipeline adapts gracefully to additional missingness by applying training-fitted medians to new NaN positions.

---

## 9. Limitations

1. Perturbation experiments use simplified synthetic models. Real-world sensor drift, seasonal variation and measurement error patterns are more complex.
2. Only three baseline models were evaluated. Advanced ensemble configurations (Iterative+RF) are not included as they are not saved baseline artifacts.
3. No external validation dataset was used.
4. Feature sensitivity results are specific to this dataset and cannot be generalized to other water quality monitoring systems.

---
*Report generated by Member 1 – Data & ML Engineer, WATERWATCH AI.*
