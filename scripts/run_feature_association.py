"""
WATERWATCH AI - Advanced Feature Association Analysis (Section 5)
Member 1 – Data & ML Engineer

Outputs:
  results/figures/spearman_correlation_heatmap.png
  results/metrics/mutual_information_scores.csv
  results/figures/mutual_information_scores.png
  results/metrics/point_biserial_results.csv
  results/metrics/feature_association_comparison.csv
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"

FEATURES = ["ph","Hardness","Solids","Chloramines","Sulfate",
            "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
TARGET = "Potability"

# ── 1. Load data ──────────────────────────────────────────────────────────────
data_path = "data/raw/water_potability.csv"
if not os.path.exists(data_path):
    data_path = "../data/raw/water_potability.csv"
df = pd.read_csv(data_path)
print("Dataset loaded:", df.shape)

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# ── 2. Spearman correlation heatmap ──────────────────────────────────────────
spearman_corr = df[FEATURES + [TARGET]].corr(method="spearman", numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
sns.heatmap(
    spearman_corr, annot=True, fmt=".3f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5,
    annot_kws={"fontsize": 9}, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Spearman Rank Correlation Matrix — WATERWATCH AI Features",
             fontsize=13, fontweight="bold", pad=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/spearman_correlation_heatmap.png", dpi=200)
plt.close()
print("✓ Saved results/figures/spearman_correlation_heatmap.png")

# ── 3. Mutual Information ─────────────────────────────────────────────────────
# Impute missing values before MI (pairwise complete doesn't apply to MI)
imp = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imp.fit_transform(df[FEATURES]), columns=FEATURES)
y = df[TARGET].values

mi_scores = mutual_info_classif(X_imputed, y, random_state=42)
mi_df = pd.DataFrame({"Feature": FEATURES, "Mutual Information Score": mi_scores})
mi_df = mi_df.sort_values("Mutual Information Score", ascending=False).reset_index(drop=True)

mi_df.to_csv("results/metrics/mutual_information_scores.csv", index=False)
print("✓ Saved results/metrics/mutual_information_scores.csv")
print(mi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#2b5c8f" if s > mi_df["Mutual Information Score"].median() else "#7fb3d3"
          for s in mi_df["Mutual Information Score"]]
bars = ax.barh(mi_df["Feature"], mi_df["Mutual Information Score"],
               color=colors, edgecolor="black", linewidth=0.6)
ax.set_xlabel("Mutual Information Score (with Potability)", fontsize=11)
ax.set_title("Mutual Information: Feature Association with Water Potability",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, mi_df["Mutual Information Score"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("results/figures/mutual_information_scores.png", dpi=200)
plt.close()
print("✓ Saved results/figures/mutual_information_scores.png")

# ── 4. Point-biserial correlation ─────────────────────────────────────────────
# Use pairwise complete observations (drop NaN per feature pair)
pb_results = []
for feat in FEATURES:
    sub = df[[feat, TARGET]].dropna()
    n = len(sub)
    r, p = stats.pointbiserialr(sub[TARGET], sub[feat])
    pb_results.append({
        "Feature": feat,
        "PB Coefficient": round(r, 4),
        "P-Value": round(p, 6),
        "N (pairwise complete)": n,
    })

pb_df = pd.DataFrame(pb_results)

# Benjamini-Hochberg FDR correction
p_vals = pb_df["P-Value"].values
n_tests = len(p_vals)
rank_order = np.argsort(p_vals)
bh_critical = (np.arange(1, n_tests + 1) / n_tests) * 0.05
adj_p = np.empty(n_tests)
adj_p[rank_order] = [min(p_vals[rank_order[i]] * n_tests / (i + 1), 1.0)
                      for i in range(n_tests)]
# Enforce monotonicity (cumulative min from largest rank)
for i in range(n_tests - 2, -1, -1):
    adj_p[rank_order[i]] = min(adj_p[rank_order[i]], adj_p[rank_order[i+1]])

pb_df["BH Adjusted P-Value"] = np.round(adj_p, 6)
pb_df["Significant (BH, alpha=0.05)"] = adj_p < 0.05

pb_df.to_csv("results/metrics/point_biserial_results.csv", index=False)
print("\n✓ Saved results/metrics/point_biserial_results.csv")
print(pb_df.to_string(index=False))

# ── 5. Association comparison table ──────────────────────────────────────────
# Pearson with target (pairwise complete)
pearson_corr = {}
for feat in FEATURES:
    sub = df[[feat, TARGET]].dropna()
    r, _ = stats.pearsonr(sub[TARGET], sub[feat])
    pearson_corr[feat] = round(r, 4)

# Spearman with target (from full matrix)
spearman_with_target = {
    feat: round(spearman_corr.loc[feat, TARGET], 4) for feat in FEATURES
}

# Merge all into one table
assoc_rows = []
for feat in FEATURES:
    pb_row = pb_df[pb_df["Feature"] == feat].iloc[0]
    mi_row = mi_df[mi_df["Feature"] == feat].iloc[0] if feat in mi_df["Feature"].values else None
    mi_score = round(mi_row["Mutual Information Score"], 4) if mi_row is not None else None
    assoc_rows.append({
        "Feature": feat,
        "Pearson r (with Potability)": pearson_corr[feat],
        "Spearman rho (with Potability)": spearman_with_target[feat],
        "Mutual Information Score": mi_score,
        "Point-Biserial r": pb_row["PB Coefficient"],
        "PB P-Value": pb_row["P-Value"],
        "BH Adjusted P-Value": pb_row["BH Adjusted P-Value"],
        "Significant (BH)": pb_row["Significant (BH, alpha=0.05)"],
    })

assoc_df = pd.DataFrame(assoc_rows)
assoc_df = assoc_df.sort_values("Mutual Information Score", ascending=False).reset_index(drop=True)
assoc_df.to_csv("results/metrics/feature_association_comparison.csv", index=False)
print("\n✓ Saved results/metrics/feature_association_comparison.csv")
print(assoc_df.to_string(index=False))

print("\n✓ Section 5 (Feature Association) complete.")
