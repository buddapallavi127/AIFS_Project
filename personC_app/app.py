"""
Person C – AI-driven Water Quality Monitoring Dashboard
=======================================================
Streamlit app that loads Person B's trained Soft-Voting Ensemble model
and provides:
  1. Single-sample prediction with live SHAP waterfall
  2. Batch CSV upload with downloadable results
  3. Model performance & comparison dashboard
  4. Feature importance (SHAP + permutation)
"""

import os, sys, json, io, base64, pathlib
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── paths ──────────────────────────────────────────────────────────────
APP_DIR  = pathlib.Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR   = ROOT_DIR / "kaggle_output_personB" / "models"
OUTPUT_DIR  = ROOT_DIR / "kaggle_output_personB" / "outputs_B"

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaGuard AI – Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --accent: #06d6a0;
    --danger: #ef4444;
    --surface: #0f172a;
    --surface-light: #1e293b;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
}

html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark theme overrides */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(14, 165, 233, 0.15);
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(14,165,233,0.15);
}
.metric-card h3 {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card .value {
    color: #f1f5f9;
    font-size: 2.2rem;
    font-weight: 700;
}
.metric-card .value.safe { color: #06d6a0; }
.metric-card .value.unsafe { color: #ef4444; }
.metric-card .value.primary { color: #0ea5e9; }

/* Result banner */
.result-safe {
    background: linear-gradient(135deg, rgba(6,214,160,0.15), rgba(6,214,160,0.05));
    border: 2px solid rgba(6,214,160,0.4);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-unsafe {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 2px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

/* Section header */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(14,165,233,0.3);
}

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #06d6a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* Feature info tooltip */
.feature-info {
    background: rgba(30,41,59,0.9);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.feature-info .name { color: #0ea5e9; font-weight: 600; }
.feature-info .range { color: #94a3b8; font-size: 0.85rem; }

/* Streamlit native overrides */
div[data-testid="stMetricValue"] { font-family: 'Inter', sans-serif !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30,41,59,0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(14,165,233,0.2) !important;
    color: #0ea5e9 !important;
}

/* Slider */
div[data-testid="stSlider"] label { color: #e2e8f0 !important; }

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── load model & metadata ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(str(MODEL_DIR / "best_model.joblib"))
    with open(str(MODEL_DIR / "model_meta.json")) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

FEATURES       = meta["features"]
BASE_FEATURES  = meta["base_features"]
THRESHOLD      = meta["threshold"]
TRAIN_MEDIANS  = meta["train_medians"]
FEATURE_RANGES = meta["feature_ranges"]
TEST_METRICS   = meta["test_metrics"]

# Feature descriptions for the UI
FEATURE_DESC = {
    "ph":               ("pH Level", "Acidity/alkalinity of water (0–14 scale)", "pH units"),
    "Hardness":         ("Hardness", "Calcium & magnesium dissolved in water", "mg/L"),
    "Solids":           ("Total Dissolved Solids", "Minerals, salts, metals dissolved", "ppm"),
    "Chloramines":      ("Chloramines", "Disinfectants used in water treatment", "ppm"),
    "Sulfate":          ("Sulfate", "Naturally occurring ion from minerals", "mg/L"),
    "Conductivity":     ("Conductivity", "Ability to conduct electrical current", "μS/cm"),
    "Organic_carbon":   ("Organic Carbon", "Total organic carbon content", "ppm"),
    "Trihalomethanes":  ("Trihalomethanes", "Chemical byproduct of water chlorination", "μg/L"),
    "Turbidity":        ("Turbidity", "Measure of water clarity / light scattering", "NTU"),
}


# ── prediction function ───────────────────────────────────────────────
def predict_sample(sample: dict) -> dict:
    """Predict potability for a single sample dict."""
    row = {}
    for c in BASE_FEATURES:
        v = sample.get(c)
        missing = v is None or (isinstance(v, float) and np.isnan(v))
        row[c] = TRAIN_MEDIANS[c] if missing else float(v)
        if f"{c}_was_missing" in FEATURES:
            row[f"{c}_was_missing"] = int(missing)
    X = pd.DataFrame([row])[FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "probability_potable": round(prob, 4),
        "potable": prob >= THRESHOLD,
        "threshold": THRESHOLD,
        "X": X,
    }


@st.cache_resource
def get_shap_explainer():
    """Build and cache SHAP TreeExplainer.
    VotingClassifier (ensemble) cannot be directly passed to TreeExplainer.
    We extract the primary tree model (XGBoost, consistent with Person B's explainability analysis).
    """
    import shap
    m = model
    source_name = type(m).__name__

    # If soft-voting ensemble, find a supported tree-based estimator
    if hasattr(m, "named_estimators_"):
        if "XGBoost" in m.named_estimators_:
            target = m.named_estimators_["XGBoost"]
            source_name = "XGBoost (Ensemble Component)"
        elif "LightGBM" in m.named_estimators_:
            target = m.named_estimators_["LightGBM"]
            source_name = "LightGBM (Ensemble Component)"
        elif "Random" in m.named_estimators_:
            target = m.named_estimators_["Random"]
            source_name = "RandomForest (Ensemble Component)"
        else:
            for name, est in m.named_estimators_.items():
                try:
                    return shap.TreeExplainer(est), f"{name} (Ensemble Component)"
                except Exception:
                    continue
            raise ValueError("No tree-based estimator found in ensemble for TreeExplainer.")
        return shap.TreeExplainer(target), source_name
    elif hasattr(m, "estimators_"):
        for est in m.estimators_:
            try:
                return shap.TreeExplainer(est), f"{type(est).__name__} (Ensemble Component)"
            except Exception:
                continue
        raise ValueError("No tree-based estimator found in ensemble for TreeExplainer.")

    return shap.TreeExplainer(m), source_name


def compute_shap_values(X):
    """Compute SHAP values for a single-row DataFrame."""
    explainer, source_name = get_shap_explainer()
    sv = explainer(X)
    return sv, source_name


# ── load CSV results ──────────────────────────────────────────────────
@st.cache_data
def load_csv(name):
    path = OUTPUT_DIR / name
    if path.exists():
        return pd.read_csv(str(path))
    return None


# ── SIDEBAR ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 3rem;">💧</div>
        <div style="font-size: 1.4rem; font-weight: 700;
                    background: linear-gradient(135deg, #0ea5e9, #06d6a0);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            AquaGuard AI
        </div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
            Water Quality Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔬 Predict", "📊 Model Performance", "🧠 Feature Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    # Model info card
    st.markdown("""
    <div class="feature-info">
        <div class="name">Model Info</div>
        <div class="range" style="margin-top:8px;">
            <b>Type:</b> Soft-Voting Ensemble<br>
            <b>Threshold:</b> {:.0%}<br>
            <b>Features:</b> 9 base + 3 flags<br>
            <b>ROC-AUC:</b> {:.1%}
        </div>
    </div>
    """.format(THRESHOLD, TEST_METRICS["roc_auc"]), unsafe_allow_html=True)

    st.caption("AIFS PROJECT")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICT
# ═══════════════════════════════════════════════════════════════════════
if page == "🔬 Predict":
    st.markdown("""
    <div class="hero">
        <h1>💧 AquaGuard AI</h1>
        <p>Enter water sample measurements below to predict whether the water is safe to drink.
        Leave fields blank if a measurement is unavailable — the model handles missing values gracefully.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["🧪 Single Sample", "📁 Batch Upload"])

    # ── Single sample ────────────────────────────────────────────────
    with tab_single:
        st.markdown('<div class="section-header">Water Sample Parameters</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        sample = {}
        for i, feat in enumerate(BASE_FEATURES):
            label, desc, unit = FEATURE_DESC[feat]
            lo, hi = FEATURE_RANGES[feat]
            median = TRAIN_MEDIANS[feat]
            with cols[i % 3]:
                st.markdown(f"""
                <div class="feature-info">
                    <div class="name">{label}</div>
                    <div class="range">{desc}<br>Range: {lo:.1f} – {hi:.1f} {unit}</div>
                </div>
                """, unsafe_allow_html=True)

                val = st.number_input(
                    f"{label} ({unit})",
                    min_value=float(lo) * 0.5,
                    max_value=float(hi) * 1.5,
                    value=float(median),
                    step=(hi - lo) / 100,
                    key=f"input_{feat}",
                    label_visibility="collapsed",
                )
                sample[feat] = val

        st.markdown("")
        predict_btn = st.button("🔍  Analyze Water Sample", use_container_width=True, type="primary")

        if predict_btn:
            with st.spinner("Running prediction & SHAP analysis..."):
                result = predict_sample(sample)
                prob = result["probability_potable"]
                safe = result["potable"]

                # Result banner
                if safe:
                    icon, label, cls = "✅", "POTABLE (Safe to drink)", "result-safe"
                else:
                    icon, label, cls = "⚠️", "NOT POTABLE (Unsafe)", "result-unsafe"

                st.markdown(f"""
                <div class="{cls}">
                    <div style="font-size: 3rem;">{icon}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {'#06d6a0' if safe else '#ef4444'}; margin: 8px 0;">
                        {label}
                    </div>
                    <div style="color: #94a3b8; font-size: 1rem;">
                        Potability probability: <b style="color:#f1f5f9">{prob:.1%}</b> &nbsp;|&nbsp;
                        Threshold: <b style="color:#f1f5f9">{THRESHOLD:.0%}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability gauge
                st.markdown("")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 48, "color": "#f1f5f9"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
                        "bar": {"color": "#0ea5e9"},
                        "bgcolor": "#1e293b",
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, THRESHOLD * 100], "color": "rgba(239,68,68,0.2)"},
                            {"range": [THRESHOLD * 100, 100], "color": "rgba(6,214,160,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "#f59e0b", "width": 3},
                            "thickness": 0.8,
                            "value": THRESHOLD * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=280,
                    margin=dict(t=20, b=20, l=40, r=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#94a3b8"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # SHAP waterfall
                st.markdown('<div class="section-header">🧠 SHAP Explanation</div>', unsafe_allow_html=True)
                st.caption("How each feature pushed the prediction towards Safe (+) or Unsafe (−).")

                try:
                    sv, explainer_source = compute_shap_values(result["X"])
                    shap_vals = sv.values[0]
                    # If 2D (e.g. binary classification [neg, pos]), pick index 1 for positive class
                    if hasattr(shap_vals, "ndim") and shap_vals.ndim == 2:
                        shap_vals = shap_vals[:, 1]
                    vals = [float(v) for v in shap_vals]

                    feat_names = list(result["X"].columns)
                    sorted_idx = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)

                    fig_shap = go.Figure()
                    colors = ["#06d6a0" if v > 0 else "#ef4444" for v in [vals[i] for i in sorted_idx]]
                    fig_shap.add_trace(go.Bar(
                        y=[feat_names[i] for i in sorted_idx],
                        x=[vals[i] for i in sorted_idx],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{vals[i]:+.4f}" for i in sorted_idx],
                        textposition="outside",
                        textfont={"color": "#e2e8f0", "size": 12},
                    ))
                    fig_shap.update_layout(
                        height=400,
                        margin=dict(t=10, b=30, l=150, r=60),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title=f"SHAP value ({explainer_source})",
                                   gridcolor="rgba(51,65,85,0.5)", zerolinecolor="#475569",
                                   tickfont={"color": "#94a3b8"}),
                        yaxis=dict(tickfont={"color": "#e2e8f0"}),
                        font={"color": "#94a3b8"},
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

    # ── Batch upload ─────────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-header">📁 Batch Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        Upload a CSV file with water quality measurements. The file should have columns matching
        the feature names: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`,
        `Organic_carbon`, `Trihalomethanes`, `Turbidity`. Missing values are allowed.
        """)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"**Loaded {len(df)} samples**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀  Run Batch Prediction", use_container_width=True, type="primary"):
                with st.spinner(f"Predicting {len(df)} samples..."):
                    results = []
                    for _, row in df.iterrows():
                        r = predict_sample(row.to_dict())
                        results.append({
                            "probability_potable": r["probability_potable"],
                            "potable": "Yes" if r["potable"] else "No",
                        })
                    res_df = pd.concat([df, pd.DataFrame(results)], axis=1)

                    # Summary
                    n_safe = sum(1 for r in results if r["potable"] == "Yes")
                    n_unsafe = len(results) - n_safe

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Total Samples</h3>
                            <div class="value primary">{len(results)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Safe (Potable)</h3>
                            <div class="value safe">{n_safe}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Unsafe</h3>
                            <div class="value unsafe">{n_unsafe}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")
                    st.dataframe(res_df, use_container_width=True)

                    # Download button
                    csv_buf = res_df.to_csv(index=False)
                    st.download_button(
                        "⬇️  Download Results CSV",
                        csv_buf,
                        "aquaguard_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )

                    # Distribution chart
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=res_df["probability_potable"],
                        nbinsx=30,
                        marker_color="#0ea5e9",
                        opacity=0.8,
                    ))
                    fig_dist.add_vline(x=THRESHOLD, line_dash="dash", line_color="#f59e0b",
                                       annotation_text=f"Threshold ({THRESHOLD})")
                    fig_dist.update_layout(
                        title="Probability Distribution of Batch Predictions",
                        xaxis_title="Probability of Potability",
                        yaxis_title="Count",
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#94a3b8"},
                        xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
                        yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance</h1>
        <p>Comprehensive evaluation of the Soft-Voting Ensemble and comparison with baseline models.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Test metrics cards ───────────────────────────────────────────
    st.markdown('<div class="section-header">Final Test-Set Metrics</div>', unsafe_allow_html=True)

    metric_items = [
        ("ROC-AUC",      TEST_METRICS["roc_auc"],      "primary"),
        ("PR-AUC",       TEST_METRICS["pr_auc"],        "primary"),
        ("F1 Score",     TEST_METRICS["f1"],            "primary"),
        ("Recall",       TEST_METRICS["recall"],        "safe"),
        ("Precision",    TEST_METRICS["precision"],     "unsafe"),
        ("Balanced Acc", TEST_METRICS["balanced_acc"],  "primary"),
    ]
    cols = st.columns(3)
    for i, (name, val, cls) in enumerate(metric_items):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{name}</h3>
                <div class="value {cls}">{val:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

    # ── Radar chart ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Metrics Radar</div>', unsafe_allow_html=True)

    radar_cats = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    radar_vals = [TEST_METRICS[k] for k in ["accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    radar_vals.append(radar_vals[0])  # close the polygon
    radar_cats_closed = radar_cats + [radar_cats[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_cats_closed,
        fill="toself",
        fillcolor="rgba(14,165,233,0.15)",
        line=dict(color="#0ea5e9", width=2),
        marker=dict(size=6, color="#06d6a0"),
        name="Ensemble",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(51,65,85,0.5)",
                            tickfont={"color": "#64748b"}),
            angularaxis=dict(gridcolor="rgba(51,65,85,0.3)", tickfont={"color": "#e2e8f0"}),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        height=420,
        margin=dict(t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Model comparison ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Comparison (Test Set)</div>', unsafe_allow_html=True)
    results_df = load_csv("results_table.csv")
    if results_df is not None:
        results_df = results_df.sort_values("roc_auc", ascending=True)

        fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("ROC-AUC", "F1 Score"),
                                  horizontal_spacing=0.15)

        colors = ["#0ea5e9" if "Ensemble" in m else "#64748b" for m in results_df["model"]]

        fig_comp.add_trace(go.Bar(
            y=results_df["model"], x=results_df["roc_auc"],
            orientation="h", marker_color=colors,
            text=results_df["roc_auc"].apply(lambda v: f"{v:.3f}"),
            textposition="outside", textfont={"color": "#e2e8f0"},
            showlegend=False,
        ), row=1, col=1)

        fig_comp.add_trace(go.Bar(
            y=results_df["model"], x=results_df["f1"],
            orientation="h", marker_color=colors,
            text=results_df["f1"].apply(lambda v: f"{v:.3f}"),
            textposition="outside", textfont={"color": "#e2e8f0"},
            showlegend=False,
        ), row=1, col=2)

        fig_comp.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8"},
            margin=dict(l=160, r=60, t=40, b=30),
        )
        fig_comp.update_xaxes(gridcolor="rgba(51,65,85,0.4)")
        fig_comp.update_yaxes(tickfont={"color": "#e2e8f0"})
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Cross-validation results ─────────────────────────────────────
    st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)

    col_base, col_tuned = st.columns(2)

    baseline_df = load_csv("baseline_cv.csv")
    tuned_df = load_csv("tuned_cv.csv")

    with col_base:
        st.markdown("**Baseline (5-fold × 2 Repeated Stratified)**")
        if baseline_df is not None:
            styled = baseline_df.sort_values("roc_auc", ascending=False)
            st.dataframe(styled, use_container_width=True, hide_index=True)

    with col_tuned:
        st.markdown("**After Hyper-parameter Tuning**")
        if tuned_df is not None:
            styled = tuned_df.sort_values("roc_auc", ascending=False)
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Evaluation plots from Person B ───────────────────────────────
    st.markdown('<div class="section-header">Evaluation Visualisations</div>', unsafe_allow_html=True)

    eval_images = [
        ("test_evaluation.png", "Test Set: Confusion Matrix, ROC & PR Curves"),
        ("threshold.png", "Decision Threshold Optimisation"),
        ("baseline_comparison.png", "Baseline Model Comparison"),
    ]

    for fname, caption in eval_images:
        path = OUTPUT_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
            st.markdown("")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: FEATURE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🧠 Feature Insights":
    st.markdown("""
    <div class="hero">
        <h1>🧠 Feature Insights</h1>
        <p>Understand which water quality parameters matter most for the potability prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP plots from Person B ─────────────────────────────────────
    st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)

    shap_images = [
        ("shap_bar.png", "Mean |SHAP| – Global Feature Importance"),
        ("shap_summary.png", "SHAP Beeswarm – Feature Impact Distribution"),
    ]
    for fname, caption in shap_images:
        path = OUTPUT_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
            st.markdown("")

    # ── Permutation importance ───────────────────────────────────────
    st.markdown('<div class="section-header">Permutation Importance</div>', unsafe_allow_html=True)
    perm_path = OUTPUT_DIR / "permutation_importance.png"
    if perm_path.exists():
        st.image(str(perm_path), caption="Permutation Importance on Test Set", use_container_width=True)

    # ── Feature ranges explorer ──────────────────────────────────────
    st.markdown('<div class="section-header">Feature Range Explorer</div>', unsafe_allow_html=True)
    st.caption("Training-set ranges and median values for each water quality parameter.")

    range_data = []
    for feat in BASE_FEATURES:
        label, desc, unit = FEATURE_DESC[feat]
        lo, hi = FEATURE_RANGES[feat]
        med = TRAIN_MEDIANS[feat]
        range_data.append({
            "Feature": label,
            "Unit": unit,
            "Min": round(lo, 2),
            "Median": round(med, 2),
            "Max": round(hi, 2),
            "Description": desc,
        })

    range_df = pd.DataFrame(range_data)
    st.dataframe(range_df, use_container_width=True, hide_index=True)

    # Interactive range chart
    fig_range = go.Figure()
    for i, row in enumerate(range_data):
        feat = row["Feature"]
        fig_range.add_trace(go.Bar(
            name=feat,
            y=[feat],
            x=[row["Max"] - row["Min"]],
            base=[row["Min"]],
            orientation="h",
            marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
            text=f"Median: {row['Median']}",
            textposition="inside",
            textfont={"color": "#1e293b", "size": 11},
            showlegend=False,
        ))
        fig_range.add_trace(go.Scatter(
            y=[feat], x=[row["Median"]],
            mode="markers",
            marker=dict(color="#f59e0b", size=10, symbol="diamond"),
            name="Median" if i == 0 else None,
            showlegend=(i == 0),
        ))

    fig_range.update_layout(
        height=450,
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Value", gridcolor="rgba(51,65,85,0.4)"),
        yaxis=dict(tickfont={"color": "#e2e8f0"}),
        margin=dict(l=140, r=40, t=20, b=40),
        legend=dict(font={"color": "#e2e8f0"}),
    )
    st.plotly_chart(fig_range, use_container_width=True)
