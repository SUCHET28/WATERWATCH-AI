"""
WATERWATCH AI — Unified Streamlit Web Application
Explainable AI-Based Water Potability Prediction and Community Risk Monitoring System
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="WATERWATCH AI — Water Potability & Risk System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. CONSTANTS, SCHEMAS & WHO GUIDELINES
FEATURE_NAMES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity"
]

PARAMETER_GUIDELINES = {
    "ph": {"label": "pH Level", "unit": "pH units", "min_safe": 6.5, "max_safe": 8.5, "min_limit": 0.0, "max_limit": 14.0, "step": 0.1, "default": 7.2, "desc": "WHO standard range is 6.5 to 8.5.", "health_impact": "Values <6.5 cause pipe corrosion; values >8.5 produce scaling and bitter taste."},
    "Hardness": {"label": "Hardness", "unit": "mg/L", "min_safe": 60.0, "max_safe": 200.0, "min_limit": 40.0, "max_limit": 350.0, "step": 1.0, "default": 195.0, "desc": "Caused primarily by Calcium and Magnesium ions.", "health_impact": "High hardness (>300 mg/L) causes encrustation in water supplies."},
    "Solids": {"label": "Total Dissolved Solids (TDS)", "unit": "ppm", "min_safe": 100.0, "max_safe": 1000.0, "min_limit": 300.0, "max_limit": 60000.0, "step": 100.0, "default": 18000.0, "desc": "Mineral and organic matter dissolved in water.", "health_impact": "WHO considers TDS <500 ppm desirable and <1000 ppm acceptable."},
    "Chloramines": {"label": "Chloramines", "unit": "ppm", "min_safe": 0.5, "max_safe": 4.0, "min_limit": 0.0, "max_limit": 15.0, "step": 0.1, "default": 7.1, "desc": "Disinfectant compounds created by chlorine and ammonia.", "health_impact": "EPA drinking water guideline limit is 4.0 ppm."},
    "Sulfate": {"label": "Sulfate", "unit": "mg/L", "min_safe": 0.0, "max_safe": 250.0, "min_limit": 100.0, "max_limit": 500.0, "step": 1.0, "default": 330.0, "desc": "Naturally occurring mineral found in soil and rocks.", "health_impact": "Concentrations >500 mg/L induce acute laxative effects."},
    "Conductivity": {"label": "Electrical Conductivity", "unit": "μS/cm", "min_safe": 50.0, "max_safe": 500.0, "min_limit": 100.0, "max_limit": 800.0, "step": 5.0, "default": 420.0, "desc": "Measures electric current conduction from dissolved ions.", "health_impact": "WHO standard generally advises <400-500 μS/cm for potable tap supplies."},
    "Organic_carbon": {"label": "Total Organic Carbon (TOC)", "unit": "ppm", "min_safe": 0.0, "max_safe": 10.0, "min_limit": 2.0, "max_limit": 30.0, "step": 0.1, "default": 14.2, "desc": "Total amount of carbon found in organic compounds.", "health_impact": "EPA limit is <10 ppm for raw water; higher TOC forms carcinogenic disinfection byproducts."},
    "Trihalomethanes": {"label": "Trihalomethanes (THMs)", "unit": "μg/L", "min_safe": 0.0, "max_safe": 80.0, "min_limit": 0.0, "max_limit": 130.0, "step": 0.5, "default": 65.0, "desc": "Byproducts formed when chlorine reacts with organic matter.", "health_impact": "WHO / EPA standard limit is 80 μg/L. Chronic exposure increases cancer risks."},
    "Turbidity": {"label": "Turbidity", "unit": "NTU", "min_safe": 0.0, "max_safe": 5.0, "min_limit": 1.0, "max_limit": 7.0, "step": 0.1, "default": 3.8, "desc": "Measurement of water clarity and cloudiness.", "health_impact": "WHO recommends <1 NTU (acceptable up to 5 NTU). High turbidity shields microbes."}
}

PRESET_SAMPLES = {
    "🌱 Clean Municipal Tap Water (Normal)": {"ph": 7.3, "Hardness": 180.0, "Solids": 8500.0, "Chloramines": 3.2, "Sulfate": 240.0, "Conductivity": 380.0, "Organic_carbon": 6.5, "Trihalomethanes": 45.0, "Turbidity": 2.1},
    "⚠️ High Sulfate & Mineral Contamination": {"ph": 7.9, "Hardness": 245.0, "Solids": 34000.0, "Chloramines": 7.8, "Sulfate": 420.0, "Conductivity": 580.0, "Organic_carbon": 17.5, "Trihalomethanes": 88.0, "Turbidity": 4.6},
    "☢️ Acidic Runoff with High Organic Matter": {"ph": 4.2, "Hardness": 280.0, "Solids": 26000.0, "Chloramines": 9.4, "Sulfate": 390.0, "Conductivity": 620.0, "Organic_carbon": 22.0, "Trihalomethanes": 105.0, "Turbidity": 5.8},
    "🏔️ Untreated Highland Spring (Missing Sulfate)": {"ph": 6.8, "Hardness": 140.0, "Solids": 12000.0, "Chloramines": 4.5, "Sulfate": np.nan, "Conductivity": 320.0, "Organic_carbon": 8.0, "Trihalomethanes": 52.0, "Turbidity": 3.2}
}

PARAMETER_WEIGHTS = {"ph": 0.12, "Hardness": 0.08, "Solids": 0.10, "Chloramines": 0.14, "Sulfate": 0.12, "Conductivity": 0.08, "Organic_carbon": 0.12, "Trihalomethanes": 0.14, "Turbidity": 0.10}

# 2. MODEL & DATA LOADING
@st.cache_resource
def load_model(filename: str):
    candidates = [Path(f"models/{filename}"), Path(f"../models/{filename}"), Path(f"WATERWATCH-AI-main/models/{filename}")]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None

@st.cache_data
def load_csv_dataset():
    candidates = [Path("data/water_potability.csv"), Path("data/raw/water_potability.csv"), Path("WATERWATCH-AI-main/data/water_potability.csv")]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    return None

def predict_sample(sample_dict: dict, model_type: str = "decision_tree"):
    filename = "decision_tree_pipeline.joblib" if model_type == "decision_tree" else "gaussian_nb_pipeline.joblib"
    model = load_model(filename)
    if model is None:
        return None
    df_sample = pd.DataFrame([{f: sample_dict.get(f, np.nan) for f in FEATURE_NAMES}])
    pred = int(model.predict(df_sample)[0])
    probas = model.predict_proba(df_sample)[0]
    return {"pred": pred, "is_potable": bool(pred == 1), "prob_non_potable": float(probas[0]), "prob_potable": float(probas[1]), "model_name": "Decision Tree" if model_type == "decision_tree" else "Gaussian Naive Bayes"}

def audit_parameter(param: str, value: float):
    g = PARAMETER_GUIDELINES.get(param)
    if value is None or np.isnan(value):
        return {"label": g["label"], "unit": g["unit"], "val": "Missing (NaN)", "badge": "⚠️ Imputed", "severity": 1, "msg": "Missing value — median imputed during inference.", "remedy": "Test fresh sample."}
    v = float(value)
    if g["min_safe"] <= v <= g["max_safe"]:
        return {"label": g["label"], "unit": g["unit"], "val": round(v, 2), "badge": "✅ Safe", "severity": 0, "msg": f"Within safe bounds ({g['min_safe']}-{g['max_safe']} {g['unit']}).", "remedy": "Routine monitoring."}
    direction = "Low" if v < g["min_safe"] else "High"
    severity = 3 if (v > g["max_safe"] * 1.5 or v < g["min_safe"] * 0.5) else 2
    badge = "🔴 Critical" if severity == 3 else "🟠 Warning"
    remedy_map = {"ph": "Acidic: Add soda ash/lime; Alkaline: Acid dosing.", "Hardness": "Use water softeners (ion exchange) or RO.", "Solids": "Install multi-stage Reverse Osmosis (RO).", "Chloramines": "Apply activated carbon filtration.", "Sulfate": "Deploy reverse osmosis or ion exchange.", "Conductivity": "Demineralization via deionization/RO.", "Organic_carbon": "Advanced oxidation or Granular Activated Carbon.", "Trihalomethanes": "Granular Activated Carbon (GAC) filtration.", "Turbidity": "Coagulation, flocculation, and rapid sand filtration."}
    return {"label": g["label"], "unit": g["unit"], "val": round(v, 2), "badge": badge, "severity": severity, "msg": f"{direction} ({v:.1f} vs safe max {g['max_safe']}). {g['health_impact']}", "remedy": remedy_map.get(param, "Standard municipal treatment.")}

def calculate_community_risk(sample_dict: dict, prob_potable: float):
    audits = [audit_parameter(f, sample_dict.get(f, np.nan)) for f in FEATURE_NAMES]
    penalty = sum(PARAMETER_WEIGHTS[f] * (100 if a["severity"] == 3 else 50 if a["severity"] == 2 else 0) for f, a in zip(FEATURE_NAMES, audits))
    combined = min(max((0.60 * penalty) + (0.40 * (1.0 - prob_potable) * 100.0), 0.0), 100.0)
    crit = sum(1 for a in audits if a["severity"] == 3)
    if crit >= 2 or combined >= 65:
        tier, color, adv = "CRITICAL RISK", "#e63946", "🚨 DO NOT CONSUME. Severe chemical non-compliance detected. Issue boil/filtration warning and dispatch laboratory testing."
    elif combined >= 40:
        tier, color, adv = "HIGH RISK", "#f77f00", "⚠️ UNSAFE FOR DIRECT DRINKING. One or more parameters exceed permissible thresholds. Requires filtration/RO before use."
    elif combined >= 20:
        tier, color, adv = "MODERATE RISK", "#fcbf49", "⚡ CAUTIONARY MONITORING. Minor guideline variance detected. Secondary filtration recommended."
    else:
        tier, color, adv = "LOW RISK (PRELIMINARY PASS)", "#2a9d8f", "✅ PRELIMINARY PASS. Physicochemical values align with normal guidelines. Laboratory culture assays still required."
    return {"score": round(combined, 1), "tier": tier, "color": color, "advisory": adv, "audits": audits}

def render_metric_card(title: str, value: str, subtext: str = "", delta: str = "", delta_color: str = "#2a9d8f"):
    d_html = f"<div style='color: {delta_color}; font-weight: 600; font-size: 0.85rem; margin-top: 2px;'>{delta}</div>" if delta else ""
    s_html = f"<div style='font-size: 0.8rem; color: #6c757d; margin-top: 4px;'>{subtext}</div>" if subtext else ""
    st.markdown(f'<div style="background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; padding: 16px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); min-height: 120px;"><div style="font-size: 0.82rem; text-transform: uppercase; color: #495057; font-weight: 700; margin-bottom: 6px;">{title}</div><div style="font-size: 1.8rem; font-weight: 800; color: #1d3557; margin-bottom: 2px;">{value}</div>{d_html}{s_html}</div>', unsafe_allow_html=True)

def create_radar_chart(sample_dict: dict):
    categories, sample_ratios, safe_baseline = [], [], []
    for feat in FEATURE_NAMES:
        g = PARAMETER_GUIDELINES[feat]
        val = sample_dict.get(feat, g["default"])
        val = g["default"] if (val is None or np.isnan(val)) else val
        categories.append(g["label"])
        sample_ratios.append(min(max((val / g["max_safe"]) * 100.0, 0.0), 250.0))
        safe_baseline.append(100.0)
    categories.append(categories[0])
    sample_ratios.append(sample_ratios[0])
    safe_baseline.append(safe_baseline[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=safe_baseline, theta=categories, fill='toself', name='WHO Safe Limit (100%)', line=dict(color='rgba(42, 157, 143, 0.8)', dash='dash', width=2), fillcolor='rgba(42, 157, 143, 0.15)'))
    fig.add_trace(go.Scatterpolar(r=sample_ratios, theta=categories, fill='toself', name='Sample Observed (%)', line=dict(color='#e76f51', width=3), fillcolor='rgba(231, 111, 81, 0.25)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 250], ticksuffix="%")), height=350, margin=dict(l=30, r=30, t=20, b=30), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return fig

# 3. HEADER & TABS
st.markdown('<div style="background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%); color: white; padding: 22px 28px; border-radius: 12px; margin-bottom: 20px;"><h1 style="color: #f1faee; margin: 0 0 6px 0; font-size: 2.2rem;">💧 WATERWATCH AI</h1><p style="color: #a8dadc; margin: 0; font-size: 1.05rem;">Explainable AI-Based Water Potability Prediction and Community Risk Monitoring System</p></div>', unsafe_allow_html=True)
st.warning("⚠️ **Public Health Notice:** WATERWATCH AI evaluates 9 physicochemical parameters for preliminary screening only. Does not replace accredited EPA/WHO laboratory assays for pathogens (E. coli) or heavy metals.", icon="🧪")

tab_pred, tab_eda, tab_audit, tab_risk, tab_batch = st.tabs([
    "💧 1. Potability Predictor", "📊 2. Exploratory Data Analysis", "🤖 3. Model Benchmark & Audit", "🛡️ 4. Community Risk Engine", "📁 5. Batch CSV Assessment"
])

# TAB 1: PREDICTOR
with tab_pred:
    col_pre, col_mod = st.columns([2, 1])
    with col_pre:
        preset_choice = st.selectbox("⚡ Quick-Fill with Standard Presets:", ["Custom Input"] + list(PRESET_SAMPLES.keys()))
    with col_mod:
        model_choice = st.selectbox("🤖 ML Model:", ["decision_tree", "gaussian_nb"], format_func=lambda x: "Decision Tree (63.11%)" if x == "decision_tree" else "Gaussian Naive Bayes (ROC 0.6090)")
    
    default_vals = PRESET_SAMPLES.get(preset_choice, {})
    sample_inputs = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Fundamental Chemistry")
        for f in ["ph", "Hardness", "Solids"]:
            g = PARAMETER_GUIDELINES[f]
            v = float(default_vals.get(f, g["default"])) if default_vals.get(f) is not None and not np.isnan(default_vals.get(f, 0)) else g["default"]
            sample_inputs[f] = st.number_input(f"{g['label']} ({g['unit']})", value=v, step=float(g['step']), key=f"p1_{f}")
    with c2:
        st.markdown("#### 2. Disinfection & Minerals")
        for f in ["Chloramines", "Sulfate", "Conductivity"]:
            g = PARAMETER_GUIDELINES[f]
            v = float(default_vals.get(f, g["default"])) if default_vals.get(f) is not None and not np.isnan(default_vals.get(f, 0)) else g["default"]
            sample_inputs[f] = st.number_input(f"{g['label']} ({g['unit']})", value=v, step=float(g['step']), key=f"p1_{f}")
    with c3:
        st.markdown("#### 3. Organic & Physical Quality")
        for f in ["Organic_carbon", "Trihalomethanes", "Turbidity"]:
            g = PARAMETER_GUIDELINES[f]
            v = float(default_vals.get(f, g["default"])) if default_vals.get(f) is not None and not np.isnan(default_vals.get(f, 0)) else g["default"]
            sample_inputs[f] = st.number_input(f"{g['label']} ({g['unit']})", value=v, step=float(g['step']), key=f"p1_{f}")

    if st.button("🚀 Evaluate Potability", type="primary", use_container_width=True) or preset_choice != "Custom Input":
        res = predict_sample(sample_inputs, model_choice)
        if res:
            st.divider()
            cl, cr = st.columns([1, 1])
            with cl:
                if res["is_potable"]:
                    st.success(f"### ✅ POTABLE (Safe)\nConfidence: **{res['prob_potable']*100:.1f}%** ({res['model_name']})")
                else:
                    st.error(f"### 🛑 NON-POTABLE (Unsafe)\nConfidence: **{res['prob_non_potable']*100:.1f}%** ({res['model_name']})")
            with cr:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(y=['Prob'], x=[res['prob_non_potable']*100], name='Unsafe', orientation='h', marker_color='#e63946', text=f"Unsafe: {res['prob_non_potable']*100:.1f}%", textposition='inside'))
                fig_bar.add_trace(go.Bar(y=['Prob'], x=[res['prob_potable']*100], name='Potable', orientation='h', marker_color='#2a9d8f', text=f"Potable: {res['prob_potable']*100:.1f}%", textposition='inside'))
                fig_bar.update_layout(barmode='stack', height=110, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            risk_info = calculate_community_risk(sample_inputs, res["prob_potable"])
            cg, crd = st.columns([1, 1])
            with cg:
                st.plotly_chart(create_risk_gauge(risk_info["score"], risk_info["tier"], risk_info["color"]), use_container_width=True)
                st.info(f"**Advisory:** {risk_info['advisory']}")
            with crd:
                st.plotly_chart(create_radar_chart(sample_inputs), use_container_width=True)

            st.dataframe(pd.DataFrame([{"Parameter": f"{a['label']} ({a['unit']})", "Value": a["val"], "Status": a["badge"], "Diagnosis": a["msg"], "Action": a["remedy"]} for a in risk_info["audits"]]), use_container_width=True, hide_index=True)

# TAB 2: EDA
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")
    df_data = load_csv_dataset()
    if df_data is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1: render_metric_card("Total Samples", f"{len(df_data):,}", "0 duplicate records", "3,276 Cleaned", "#2a9d8f")
        with m2: render_metric_card("Potable (1)", f"{(df_data['Potability']==1).sum():,}", f"{((df_data['Potability']==1).sum()/len(df_data))*100:.1f}%", "Minority Class", "#457b9d")
        with m3: render_metric_card("Non-Potable (0)", f"{(df_data['Potability']==0).sum():,}", f"{((df_data['Potability']==0).sum()/len(df_data))*100:.1f}%", "Majority Class", "#e63946")
        with m4: render_metric_card("Missing Values", f"{df_data.isna().sum().sum():,}", "Sulfate, pH, THMs", "Median Imputed", "#f77f00")
        st.markdown("<br>", unsafe_allow_html=True)
        sel_f = st.selectbox("Select Parameter to Inspect:", FEATURE_NAMES, format_func=lambda x: PARAMETER_GUIDELINES[x]["label"])
        df_p = df_data.copy()
        df_p["Class"] = df_p["Potability"].map({0: "Non-Potable (0)", 1: "Potable (1)"})
        st.plotly_chart(px.histogram(df_p, x=sel_f, color="Class", barmode="overlay", marginal="box", opacity=0.6, color_discrete_map={"Non-Potable (0)": "#e63946", "Potable (1)": "#2a9d8f"}), use_container_width=True)
        st.markdown("#### Correlation Heatmap")
        st.plotly_chart(px.imshow(df_data.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-0.15, zmax=0.15), use_container_width=True)

# TAB 3: AUDIT
with tab_audit:
    st.markdown("### 🤖 Model Benchmark & Screening Audit")
    st.dataframe(pd.DataFrame([
        {"Model": "Dummy Classifier (Baseline)", "Test Accuracy": "60.98%", "ROC-AUC": "0.5000", "Potable Recall": "0.00%", "5-Fold CV Acc": "60.99% ± 0.09%", "Dangerous Errors (FP)": "0 (0.0%)"},
        {"Model": "Gaussian Naive Bayes", "Test Accuracy": "61.43%", "ROC-AUC": "0.6090", "Potable Recall": "18.75%", "5-Fold CV Acc": "62.75% ± 0.74%", "Dangerous Errors (FP)": "45 (11.25%)"},
        {"Model": "Decision Tree Classifier", "Test Accuracy": "63.11%", "ROC-AUC": "0.6011", "Potable Recall": "26.17%", "5-Fold CV Acc": "62.98% ± 0.95%", "Dangerous Errors (FP)": "53 (13.25%)"}
    ]), use_container_width=True, hide_index=True)

# TAB 4: RISK
with tab_risk:
    st.markdown("### 🛡️ Community Risk Monitoring Engine")
    s_name = st.text_input("Station Identifier:", value="Community Well #3")
    pop = st.number_input("Population Served:", value=3500)
    risk_out = calculate_community_risk(sample_inputs, 0.35)
    g_col, rad_col = st.columns([1, 1])
    with g_col:
        st.plotly_chart(create_risk_gauge(risk_out["score"], risk_out["tier"], risk_out["color"]), use_container_width=True)
        st.info(f"**Action Plan:** {risk_out['tier']}\n\n{risk_out['advisory']}")
    with rad_col:
        st.plotly_chart(create_radar_chart(sample_inputs), use_container_width=True)

# TAB 5: BATCH
with tab_batch:
    st.markdown("### 📁 Batch CSV Assessment & Sensor Triage")
    up_file = st.file_uploader("Upload CSV containing the 9 parameters:", type=["csv"])
    batch_df = None
    if up_file:
        batch_df = pd.read_csv(up_file)
    elif st.button("🧪 Run 20-Sample Demo Batch"):
        ds = load_csv_dataset()
        if ds is not None:
            batch_df = ds.sample(n=20, random_state=42)
    if batch_df is not None:
        model = load_model("decision_tree_pipeline.joblib")
        if model:
            inp = batch_df[FEATURE_NAMES]
            preds = model.predict(inp)
            probas = model.predict_proba(inp)
            res_b = batch_df.copy()
            res_b["Potability_Prediction"] = np.where(preds == 1, "Potable", "Non-Potable")
            res_b["Potable_Probability"] = np.round(probas[:, 1], 3)
            res_b["NonPotable_Probability"] = np.round(probas[:, 0], 3)
            b1, b2, b3, b4 = st.columns(4)
            with b1: render_metric_card("Total Samples", f"{len(res_b):,}", "Batch Size", "100% Evaluated", "#457b9d")
            with b2: render_metric_card("Flagged Unsafe", f"{(preds==0).sum():,}", f"{((preds==0).sum()/len(res_b))*100:.1f}%", "Priority Triage", "#e63946")
            with b3: render_metric_card("Passed Potable", f"{(preds==1).sum():,}", f"{((preds==1).sum()/len(res_b))*100:.1f}%", "Screening Pass", "#2a9d8f")
            with b4: render_metric_card("Critical Unsafe", f"{(probas[:,0]>=0.70).sum():,}", "P(Unsafe) ≥ 70%", "Immediate Action", "#f77f00")
            st.dataframe(res_b, use_container_width=True)
