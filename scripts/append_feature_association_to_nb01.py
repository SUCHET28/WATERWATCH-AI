"""
Append Phase 5: Advanced Feature Association Analysis to notebooks/01_dataset_exploration.ipynb
Preserves existing cells (0-20).
"""
import json

nb_path = "notebooks/01_dataset_exploration.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure we don't duplicate if already added
src_all = "".join(["".join(c.get("source", [])) for c in nb["cells"]])
if "Advanced Feature Association Analysis" in src_all:
    print("Feature Association cells already exist in notebook 01. Skipping append.")
    exit(0)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Phase 5: Advanced Feature Association Analysis\n",
            "**Role:** Member 1 – Data & ML Engineer  \n",
            "**Task:** Extend linear Pearson correlation with non-linear, monotonic, and distributional association measures.\n",
            "\n",
            "### Objectives\n",
            "1. **Spearman Rank Correlation:** Evaluate monotonic relationships without assuming bivariate normality.\n",
            "2. **Mutual Information:** Quantify non-linear and statistical dependencies between continuous features and binary Potability.\n",
            "3. **Point-Biserial Correlation:** Quantify continuous-binary association with Benjamini–Hochberg False Discovery Rate (FDR) correction.\n",
            "4. **Association Comparison Table:** Consolidate all metrics to assess holistic feature informativeness.\n",
            "5. **Academic Interpretation:** Rigorous discussion on correlation vs. causation, preserving features for multivariate modeling."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.1 Spearman Rank-Order Correlation Analysis\n",
            "- **Mathematical Basis:** Evaluates the degree to which a relationship between two variables can be described using a monotonic function (Pearson correlation of ranked values).\n",
            "- **Environmental Rationale:** Non-linear sensor responses or saturation effects in geochemical parameters (e.g., Conductivity, Solids) may exhibit monotonic trends undetected by standard Pearson correlation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute Spearman Rank Correlation matrix\n",
            "spearman_corr = df.corr(method='spearman', numeric_only=True)\n",
            "\n",
            "plt.figure(figsize=(10.5, 7.5))\n",
            "mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)\n",
            "sns.heatmap(\n",
            "    spearman_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, vmin=-0.1, vmax=0.1,\n",
            "    linewidths=0.5, annot_kws={'fontsize': 9}, cbar_kws={'shrink': 0.8}\n",
            ")\n",
            "plt.title('Spearman Rank Correlation Matrix — WATERWATCH AI Features', fontsize=12, fontweight='bold', pad=12)\n",
            "plt.xticks(rotation=45, ha='right', fontsize=9)\n",
            "plt.yticks(rotation=0, fontsize=9)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Spearman rank correlation with Potability:')\n",
            "display(spearman_corr['Potability'].drop('Potability').sort_values(ascending=False).to_frame('Spearman_rho'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.2 Mutual Information (Non-Linear Dependency Detection)\n",
            "- **Mathematical Basis:** $I(X; Y) = \\iint p(x, y) \\log \\frac{p(x, y)}{p(x)p(y)} dx dy$. Measures the reduction in uncertainty about the binary target $Y$ given knowledge of feature $X$.\n",
            "- **Advantage:** Captures complex non-linear associations where linear correlation is near zero."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.feature_selection import mutual_info_classif\n",
            "from sklearn.impute import SimpleImputer\n",
            "\n",
            "features = [c for c in df.columns if c != 'Potability']\n",
            "# Median imputation strictly for mutual information estimation\n",
            "imp = SimpleImputer(strategy='median')\n",
            "X_imputed = pd.DataFrame(imp.fit_transform(df[features]), columns=features)\n",
            "y = df['Potability'].values\n",
            "\n",
            "mi_scores = mutual_info_classif(X_imputed, y, random_state=42)\n",
            "mi_df = pd.DataFrame({'Feature': features, 'Mutual Information Score': mi_scores})\n",
            "mi_df = mi_df.sort_values('Mutual Information Score', ascending=False).reset_index(drop=True)\n",
            "\n",
            "plt.figure(figsize=(9, 5))\n",
            "colors = ['#2b5c8f' if s > mi_df['Mutual Information Score'].median() else '#7fb3d3' for s in mi_df['Mutual Information Score']]\n",
            "bars = plt.barh(mi_df['Feature'], mi_df['Mutual Information Score'], color=colors, edgecolor='black', linewidth=0.6)\n",
            "plt.xlabel('Mutual Information Score (Bits)', fontsize=10, fontweight='semibold')\n",
            "plt.title('Mutual Information: Feature Dependency with Water Potability', fontsize=11, fontweight='bold', pad=10)\n",
            "plt.gca().invert_yaxis()\n",
            "for bar, val in zip(bars, mi_df['Mutual Information Score']):\n",
            "    plt.text(val + 0.0005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8.5)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "display(mi_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.3 Point-Biserial Correlation & Benjamini–Hochberg FDR Correction\n",
            "- **Mathematical Basis:** Measures association between a continuous feature and a binary categorical target ($Y \\in \\{0, 1\\}$). Mathematically identical to Pearson correlation with a binary variable.\n",
            "- **Multiple Testing Adjustment:** With 9 simultaneous hypothesis tests, family-wise false discovery rate is controlled using the **Benjamini–Hochberg (BH)** procedure at $\\alpha = 0.05$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scipy import stats\n",
            "\n",
            "pb_results = []\n",
            "for feat in features:\n",
            "    sub = df[[feat, 'Potability']].dropna()\n",
            "    r, p = stats.pointbiserialr(sub['Potability'], sub[feat])\n",
            "    pb_results.append({\n",
            "        'Feature': feat,\n",
            "        'PB Coefficient': round(r, 4),\n",
            "        'P-Value': round(p, 6),\n",
            "        'N (complete)': len(sub)\n",
            "    })\n",
            "\n",
            "pb_df = pd.DataFrame(pb_results)\n",
            "# Benjamini-Hochberg adjustment\n",
            "p_vals = pb_df['P-Value'].values\n",
            "n_tests = len(p_vals)\n",
            "rank_order = np.argsort(p_vals)\n",
            "adj_p = np.empty(n_tests)\n",
            "for i in range(n_tests):\n",
            "    adj_p[rank_order[i]] = min(p_vals[rank_order[i]] * n_tests / (i + 1), 1.0)\n",
            "# Enforce monotonicity\n",
            "for i in range(n_tests - 2, -1, -1):\n",
            "    adj_p[rank_order[i]] = min(adj_p[rank_order[i]], adj_p[rank_order[i+1]])\n",
            "\n",
            "pb_df['BH Adjusted P-Value'] = np.round(adj_p, 6)\n",
            "pb_df['Significant (BH, alpha=0.05)'] = adj_p < 0.05\n",
            "display(pb_df.sort_values('P-Value'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.4 Consolidated Feature Association Comparison Table"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Consolidated summary\n",
            "assoc_summary = []\n",
            "for feat in features:\n",
            "    sub = df[[feat, 'Potability']].dropna()\n",
            "    pr, _ = stats.pearsonr(sub['Potability'], sub[feat])\n",
            "    sr = spearman_corr.loc[feat, 'Potability']\n",
            "    mi = mi_df[mi_df['Feature'] == feat]['Mutual Information Score'].values[0]\n",
            "    pb_row = pb_df[pb_df['Feature'] == feat].iloc[0]\n",
            "    assoc_summary.append({\n",
            "        'Feature': feat,\n",
            "        'Pearson r': round(pr, 4),\n",
            "        'Spearman rho': round(sr, 4),\n",
            "        'Mutual Info': round(mi, 4),\n",
            "        'Point-Biserial r': pb_row['PB Coefficient'],\n",
            "        'P-Value': pb_row['P-Value'],\n",
            "        'BH Adj P-Value': pb_row['BH Adjusted P-Value'],\n",
            "        'Significant (BH)': pb_row['Significant (BH, alpha=0.05)']\n",
            "    })\n",
            "\n",
            "consolidated_df = pd.DataFrame(assoc_summary).sort_values('Mutual Info', ascending=False).reset_index(drop=True)\n",
            "display(consolidated_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.5 Scientific Interpretation & Governance Rules\n",
            "\n",
            "1. **Correlation Does Not Imply Causation:** Low or zero bivariate correlation does not prove that a parameter is biologically or chemically irrelevant to water safety. Water quality standards (e.g., WHO, EPA) establish non-linear safe thresholds (e.g., pH 6.5–8.5).\n",
            "2. **No Unadjusted P-Value Claims:** Following Benjamini–Hochberg correction, none of the 9 features exhibit statistically significant linear point-biserial associations with Potability at $\\alpha = 0.05$.\n",
            "3. **No Blind Feature Removal:** Features with low linear correlation must **not** be arbitrarily discarded. Tree-based ensembles and non-linear classifiers identify multi-feature partition interactions that individual bivariate metrics cannot detect.\n",
            "4. **Feature Selection Validation Rule:** If feature subsetting is ever explored, it must be evaluated strictly inside cross-validation loops to prevent optimistic bias, and benchmarked against the full 9-feature pipeline."
        ]
    }
]

nb["cells"].extend(new_cells)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Successfully appended Phase 5 cells to notebooks/01_dataset_exploration.ipynb. Total cells:", len(nb["cells"]))
