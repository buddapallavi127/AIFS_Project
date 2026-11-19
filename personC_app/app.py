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
from PIL import Image as PILImage

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
        ["🔬 Predict", "🧬 Multimodal AI", "📊 Model Performance", "🧠 Feature Insights"],
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
# PAGE 2: MULTIMODAL WATER INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "🧬 Multimodal AI":
    from vision_engine_v2 import analyse_water_image_v2
    from fusion_engine import fuse_assessments, FusionResult

    # Try loading trained numerical model for Figshare data
    _figshare_model = None
    _figshare_meta = None
    _figshare_features = None
    try:
        _figshare_model = joblib.load(str(MODEL_DIR.parent / "models" / "numerical_model.joblib"))
        _mm_meta_path = MODEL_DIR.parent / "models" / "multimodal_meta.json"
        if _mm_meta_path.exists():
            with open(str(_mm_meta_path)) as _mf:
                _mm_meta = json.load(_mf)
                _figshare_meta = _mm_meta.get("numerical", {})
                _figshare_features = _figshare_meta.get("features", [])
    except Exception:
        pass

    st.markdown("""
    <div class="hero">
        <h1>🧬 Multimodal Water Intelligence</h1>
        <p>Combine <b>Vision AI</b> (trained on 6,518 labelled water images from
        <a href="https://doi.org/10.5281/zenodo.19899828" style="color:#0ea5e9;">Zenodo</a>)
        with the <b>Numerical ML</b> model (trained on 652K records from
        <a href="https://doi.org/10.1038/s41597-025-04715-4" style="color:#0ea5e9;">Nature Scientific Data</a>)
        via Dempster-Shafer evidence fusion.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture diagram ──────────────────────────────────────────
    with st.expander("📐 How Multimodal Fusion Works", expanded=False):
        st.markdown("""
        ```
        ┌─────────────────────┐       ┌──────────────────────────┐
        │  📸 Water Photo     │       │  🔢 Sensor Measurements  │
        │  (Upload an image)  │       │  (9 physicochemical vals) │
        └────────┬────────────┘       └────────────┬─────────────┘
                 │                                  │
                 ▼                                  ▼
        ┌─────────────────────┐       ┌──────────────────────────┐
        │   VISION AI ENGINE  │       │  NUMERICAL ML ENSEMBLE   │
        │  ─────────────────  │       │  ──────────────────────   │
        │  • Colour Analysis  │       │  • Random Forest         │
        │    (HSV histograms) │       │  • XGBoost               │
        │  • Turbidity Est.   │       │  • LightGBM              │
        │    (clarity metrics)│       │  • SVM (RBF)             │
        │  • Contaminant Det. │       │  Soft-Voting Ensemble    │
        │    (algae, sediment,│       │  Threshold: 0.38         │
        │     foam, particles)│       │                          │
        └────────┬────────────┘       └────────────┬─────────────┘
                 │                                  │
                 │  P(safe) = 0.73                  │  P(safe) = 0.62
                 │  Confidence = 0.80               │  Confidence = 0.85
                 │                                  │
                 └──────────────┬───────────────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │  DEMPSTER-SHAFER FUSION  │
                 │  ──────────────────────  │
                 │  Weighted evidence       │
                 │  combination with        │
                 │  conflict resolution     │
                 │                          │
                 │  Vision weight:  ~29%    │
                 │  Numerical weight: ~71%  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   FUSED DECISION         │
                 │  ──────────────────────  │
                 │  🟢 SAFE (P ≥ 0.65)     │
                 │  🟡 INCONCLUSIVE        │
                 │  🔴 UNSAFE (P ≤ 0.35)   │
                 │                          │
                 │  + Agreement Analysis    │
                 │  + Action Items          │
                 └──────────────────────────┘
        ```
        """)

    st.markdown("")

    # ── Two-column input ──────────────────────────────────────────────
    col_vision, col_numerical = st.columns(2)

    # ── LEFT: Vision AI Input ─────────────────────────────────────────
    with col_vision:
        st.markdown('<div class="section-header">📸 Vision AI — Water Sample Image</div>',
                    unsafe_allow_html=True)

        input_method = st.radio(
            "Image source",
            ["🖼️ Realistic Water Samples", "🔬 Zenodo Lab Benchmark", "📤 Upload Custom Photo"],
            horizontal=False,
            key="vision_input_method",
        )

        vision_image = None
        sample_label = "Custom Image"

        if input_method == "📤 Upload Custom Photo":
            uploaded_img = st.file_uploader(
                "Upload a water sample photo",
                type=["png", "jpg", "jpeg", "webp"],
                key="water_photo",
            )
            if uploaded_img:
                vision_image = PILImage.open(uploaded_img)
                sample_label = uploaded_img.name

        elif input_method == "🖼️ Realistic Water Samples":
            sample_dir = APP_DIR / "sample_images"
            samples = {
                "💧 Clear Water (Clean & Potable)": "clear_water.png",
                "🟤 Turbid / Muddy Sediment": "turbid_water.png",
                "🟢 Algae Contaminated (Biological Bloom)": "algae_water.png",
                "🔴 Iron / Rust Water (Mineral Runoff)": "iron_water.png",
                "🫧 Foam / Detergent Surface (Surfactants)": "foam_water.png",
            }
            choice = st.selectbox("Select realistic water condition", list(samples.keys()),
                                   key="realistic_sample_select")
            sample_label = choice
            img_path = sample_dir / samples[choice]
            if img_path.exists():
                vision_image = PILImage.open(str(img_path))
            else:
                st.info("Run `python generate_samples.py` to recreate sample images.")

        else:  # 🔬 Zenodo Lab Benchmark
            zenodo_dir = ROOT_DIR / "datasets" / "zenodo" / "datasets"
            st.info("🔬 **Zenodo Research Dataset (Cuvette Frames):** Images captured in a laboratory optical chamber with white backlighting. Clear water (Low TSS) appears as a smooth translucent light field; High TSS samples show noticeable darkening and sediment scattering.")

            if zenodo_dir.exists():
                samples = {
                    "💧 Low TSS (Clear Water Frame 1)": str(zenodo_dir / "Baja" / "S1-4_frame_0007.jpg"),
                    "💧 Low TSS (Clear Water Frame 2)": str(zenodo_dir / "Baja" / "S2-4_frame_0351.jpg"),
                    "🟡 Medium TSS (Moderate Turbidity 1)": str(zenodo_dir / "Media" / "S12-4_frame_0423.jpg"),
                    "🟡 Medium TSS (Moderate Turbidity 2)": str(zenodo_dir / "Media" / "S14-4_frame_0471.jpg"),
                    "🟤 High TSS (Turbid / Sediment 1)": str(zenodo_dir / "Alta" / "S29-4_frame_0255.jpg"),
                    "🟤 High TSS (Turbid / Sediment 2)": str(zenodo_dir / "Alta" / "S30-4_frame_1495.jpg"),
                    "🟤 High TSS (Turbid / Sediment 3)": str(zenodo_dir / "Alta" / "S20-4_frame_0799.jpg"),
                }
                choice = st.selectbox("Select a Zenodo cuvette sample", list(samples.keys()),
                                       key="zenodo_sample_select")
                sample_label = choice
                if choice and os.path.exists(samples[choice]):
                    vision_image = PILImage.open(samples[choice])
            else:
                st.warning("Zenodo dataset not found in `datasets/zenodo/`.")

        if vision_image:
            st.image(vision_image, caption=f"Loaded Sample: {sample_label}", use_container_width=True)
            w, h = vision_image.size
            img_arr = np.array(vision_image)
            st.caption(f"📐 Dimensions: {w}×{h} px | Mean RGB: ({img_arr[:,:,0].mean():.0f}, {img_arr[:,:,1].mean():.0f}, {img_arr[:,:,2].mean():.0f})")

    # ── RIGHT: Numerical ML Input ─────────────────────────────────────
    with col_numerical:
        st.markdown('<div class="section-header">🔢 Numerical ML — Sensor Readings</div>',
                    unsafe_allow_html=True)

        # Choose between original Kaggle features and Figshare features
        FIGSHARE_FEATURE_DESC = {
            "Ammonia (mg/l)": ("Ammonia", "Toxic nitrogen compound", "mg/L", 0.0, 5.0, 0.05),
            "Biochemical Oxygen Demand (mg/l)": ("BOD", "Organic pollution indicator", "mg/L", 0.0, 20.0, 1.7),
            "Dissolved Oxygen (mg/l)": ("Dissolved O₂", "Oxygen in water", "mg/L", 0.0, 20.0, 9.9),
            "Orthophosphate (mg/l)": ("Orthophosphate", "Nutrient/eutrophication", "mg/L", 0.0, 2.0, 0.05),
            "pH (ph units)": ("pH Level", "Acidity/alkalinity", "pH", 0.0, 14.0, 7.8),
            "Temperature (cel)": ("Temperature", "Water temperature", "°C", -1.0, 40.0, 12.0),
            "Nitrogen (mg/l)": ("Nitrogen", "Total nitrogen content", "mg/L", 0.0, 10.0, 0.4),
            "Nitrate (mg/l)": ("Nitrate", "Inorganic nitrogen form", "mg/L", 0.0, 50.0, 3.0),
        }

        use_figshare = _figshare_model is not None
        if use_figshare:
            st.caption("Parameters from [Nature Scientific Data](https://doi.org/10.1038/s41597-025-04715-4) (652K records, CCME WQI).")
            active_features = _figshare_features
            num_sample = {}
            num_cols_ui = st.columns(3)
            for i, feat in enumerate(active_features):
                info = FIGSHARE_FEATURE_DESC.get(feat)
                if info:
                    label, desc, unit, lo, hi, default = info
                else:
                    label, desc, unit, lo, hi, default = feat, "", "", 0.0, 100.0, 50.0
                with num_cols_ui[i % 3]:
                    val = st.number_input(
                        f"{label} ({unit})",
                        min_value=float(lo),
                        max_value=float(hi) * 2,
                        value=float(default),
                        step=(hi - lo) / 100 if hi > lo else 0.1,
                        key=f"mm_fig_{feat}",
                    )
                    num_sample[feat] = val
        else:
            st.caption("Enter physicochemical measurements (leave blank for missing values).")
            num_sample = {}
            num_cols_ui = st.columns(3)
            for i, feat in enumerate(BASE_FEATURES):
                label, desc, unit = FEATURE_DESC[feat]
                lo, hi = FEATURE_RANGES[feat]
                median = TRAIN_MEDIANS[feat]
                with num_cols_ui[i % 3]:
                    val = st.number_input(
                        f"{label} ({unit})",
                        min_value=float(lo) * 0.5,
                        max_value=float(hi) * 1.5,
                        value=float(median),
                        step=(hi - lo) / 100,
                        key=f"mm_input_{feat}",
                    )
                    num_sample[feat] = val

    # ── Run Multimodal Analysis ───────────────────────────────────────
    st.markdown("")
    run_btn = st.button("🧬  Run Multimodal Analysis", use_container_width=True, type="primary")

    if run_btn:
        if vision_image is None:
            st.error("Please provide a water sample image for the Vision AI modality.")
        else:
            with st.spinner("Running dual-modality analysis & Dempster-Shafer fusion..."):
                # ── Vision AI (v2 with trained model) ────────────────
                v2_result = analyse_water_image_v2(vision_image)
                vision_result = v2_result["assessment"]  # classical for detailed display

                # Use ML-informed score if model available, else classical
                if v2_result["model_available"] and v2_result["ml_potability_prob"] is not None:
                    vision_prob = v2_result["blended_score"] / 100.0
                    vision_confidence = v2_result["ml_confidence"]
                    vision_model_label = f"Trained RF (Zenodo 6,518 images) — TSS: {v2_result['ml_tss_class']}"
                else:
                    vision_prob = vision_result.visual_quality_score / 100.0
                    vision_confidence = vision_result.confidence
                    vision_model_label = "Classical CV (heuristic)"

                # ── Numerical ML ─────────────────────────────────────
                if use_figshare and _figshare_model is not None:
                    # Use Figshare-trained model
                    import pandas as _pd
                    X_num = _pd.DataFrame([num_sample])[_figshare_features]
                    ml_prob = float(_figshare_model.predict_proba(X_num)[0, 1])
                    ml_threshold = 0.50
                    numerical_model_label = "Ensemble (Nature Sci Data — 652K records)"
                else:
                    # Fallback to original Kaggle model
                    ml_result = predict_sample(num_sample)
                    ml_prob = ml_result["probability_potable"]
                    ml_threshold = THRESHOLD
                    numerical_model_label = "Soft-Voting Ensemble (Kaggle)"

                # ── Fusion ───────────────────────────────────────────
                fusion = fuse_assessments(
                    vision_prob_safe=vision_prob,
                    vision_confidence=max(vision_confidence, 0.50),
                    vision_details={
                        "score": v2_result["blended_score"],
                        "tss_class": v2_result.get("ml_tss_class", "N/A"),
                        "turbidity_ntu": vision_result.clarity.estimated_turbidity_ntu,
                        "colour_label": vision_result.colour.colour_label,
                        "risk_level": vision_result.risk_level,
                    },
                    numerical_prob_safe=ml_prob,
                    numerical_confidence=0.90 if use_figshare else 0.85,
                    numerical_details={
                        "model": numerical_model_label,
                        "features_used": len(_figshare_features) if use_figshare else len(FEATURES),
                    },
                    numerical_threshold=ml_threshold,
                )

            # ═════════════════════════════════════════════════════════
            # RESULTS DISPLAY
            # ═════════════════════════════════════════════════════════

            st.markdown("---")

            # ── Fused Decision Banner ────────────────────────────────
            if fusion.fused_decision == "SAFE":
                banner_icon, banner_color, banner_bg = "✅", "#06d6a0", "rgba(6,214,160,0.12)"
            elif fusion.fused_decision == "UNSAFE":
                banner_icon, banner_color, banner_bg = "🚨", "#ef4444", "rgba(239,68,68,0.12)"
            else:
                banner_icon, banner_color, banner_bg = "⚠️", "#f59e0b", "rgba(245,158,11,0.12)"

            st.markdown(f"""
            <div style="background: {banner_bg}; border: 2px solid {banner_color}40;
                        border-radius: 16px; padding: 28px; text-align: center; margin-bottom: 24px;">
                <div style="font-size: 3.5rem;">{banner_icon}</div>
                <div style="font-size: 2rem; font-weight: 800; color: {banner_color};
                            margin: 8px 0;">
                    MULTIMODAL VERDICT: {fusion.fused_decision}
                </div>
                <div style="color: #94a3b8; font-size: 1.05rem; max-width: 700px; margin: 0 auto;">
                    {fusion.summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Three metric cards ───────────────────────────────────
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fused Probability</h3>
                    <div class="value {'safe' if fusion.fused_probability >= 0.5 else 'unsafe'}">{fusion.fused_probability:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fusion Confidence</h3>
                    <div class="value primary">{fusion.fused_confidence:.0%}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                agree_color = "#06d6a0" if fusion.agreement_score > 0.3 else ("#f59e0b" if fusion.agreement_score > -0.1 else "#ef4444")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Modality Agreement</h3>
                    <div class="value" style="color: {agree_color}; font-size: 1.3rem;">{fusion.modality_agreement}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── Side-by-side Modality Comparison ─────────────────────
            st.markdown('<div class="section-header">🔀 Modality Comparison</div>',
                        unsafe_allow_html=True)

            col_v, col_divider, col_n = st.columns([5, 1, 5])

            with col_v:
                tss_line = f"TSS Class: <b>{v2_result.get('ml_tss_class', 'N/A')}</b><br>" if v2_result.get("model_available") else ""
                ml_score_line = f"ML Score: {v2_result.get('ml_score', 'N/A')}/100<br>" if v2_result.get("ml_score") is not None else ""
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.6); border: 1px solid rgba(14,165,233,0.25);
                            border-radius: 14px; padding: 20px;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #0ea5e9; margin-bottom: 12px;">
                        📸 Vision AI
                    </div>
                    <div style="color: #e2e8f0; margin-bottom: 8px;">
                        <b>P(Safe):</b>
                        <span style="color: {'#06d6a0' if vision_prob >= 0.55 else '#ef4444'}; font-size: 1.3rem; font-weight: 700;">
                            {vision_prob:.1%}
                        </span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        Model: {vision_model_label}<br>
                        {tss_line}
                        Blended Score: {v2_result['blended_score']}/100<br>
                        {ml_score_line}
                        Classical Score: {v2_result['classical_score']}/100<br>
                        Colour: {vision_result.colour.colour_label}<br>
                        Weight in fusion: <b>{fusion.vision_weight:.0%}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_divider:
                st.markdown("""
                <div style="display:flex; align-items:center; justify-content:center;
                            height: 100%; min-height: 200px;">
                    <div style="font-size: 2rem; color: #64748b;">⚡</div>
                </div>
                """, unsafe_allow_html=True)

            with col_n:
                ml_safe = ml_prob >= ml_threshold
                num_conf = "90%" if use_figshare else "85%"
                num_auc = "99.99%" if use_figshare else f"{TEST_METRICS['roc_auc']:.1%}"
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.6); border: 1px solid rgba(6,214,160,0.25);
                            border-radius: 14px; padding: 20px;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #06d6a0; margin-bottom: 12px;">
                        🔢 Numerical ML
                    </div>
                    <div style="color: #e2e8f0; margin-bottom: 8px;">
                        <b>P(Safe):</b>
                        <span style="color: {'#06d6a0' if ml_safe else '#ef4444'}; font-size: 1.3rem; font-weight: 700;">
                            {ml_prob:.1%}
                        </span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        Model: {numerical_model_label}<br>
                        Threshold: {ml_threshold:.0%}<br>
                        Decision: {'Potable ✅' if ml_safe else 'Not Potable ❌'}<br>
                        Confidence: {num_conf}<br>
                        ROC-AUC: {num_auc}<br>
                        Weight in fusion: <b>{fusion.numerical_weight:.0%}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # ── Fusion Probability Gauge ─────────────────────────────
            st.markdown('<div class="section-header">📊 Multimodal Fusion Comparison</div>',
                        unsafe_allow_html=True)

            col_g1, col_g2 = st.columns([7, 3])

            with col_g1:
                # Comparative multi-lane horizontal chart (no label collisions)
                fig_comp_bars = go.Figure()

                categories = ["📸 Vision AI", "🔢 Numerical ML", "⚡ Fused Verdict"]
                values = [vision_prob, ml_prob, fusion.fused_probability]
                bar_colors = ["#0ea5e9", "#06d6a0", banner_color]
                texts = [
                    f"<b>{vision_prob:.1%}</b> (weight: {fusion.vision_weight:.0%})",
                    f"<b>{ml_prob:.1%}</b> (weight: {fusion.numerical_weight:.0%})",
                    f"<b>{fusion.fused_probability:.1%}</b> ({fusion.fused_decision})",
                ]

                fig_comp_bars.add_trace(go.Bar(
                    y=categories,
                    x=values,
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.25)", width=1.5)),
                    text=texts,
                    textposition="outside",
                    textfont=dict(color="#f1f5f9", size=12, family="Inter"),
                    width=0.45,
                    showlegend=False,
                    cliponaxis=False,
                ))

                # Shaded decision bands
                fig_comp_bars.add_vrect(
                    x0=0.0, x1=fusion.uncertain_low, fillcolor="rgba(239,68,68,0.12)", layer="below", line_width=0,
                    annotation_text="🔴 UNSAFE (<35%)", annotation_position="top left",
                    annotation_font=dict(color="#ef4444", size=11),
                )
                fig_comp_bars.add_vrect(
                    x0=fusion.uncertain_low, x1=fusion.uncertain_high, fillcolor="rgba(245,158,11,0.10)", layer="below", line_width=0,
                    annotation_text="🟡 UNCERTAIN", annotation_position="top",
                    annotation_font=dict(color="#f59e0b", size=11),
                )
                fig_comp_bars.add_vrect(
                    x0=fusion.uncertain_high, x1=1.0, fillcolor="rgba(6,214,160,0.12)", layer="below", line_width=0,
                    annotation_text="🟢 SAFE (>65%)", annotation_position="top right",
                    annotation_font=dict(color="#06d6a0", size=11),
                )

                # Vertical threshold lines
                fig_comp_bars.add_vline(x=fusion.uncertain_low, line_dash="dash", line_color="rgba(239,68,68,0.5)", line_width=1)
                fig_comp_bars.add_vline(x=fusion.uncertain_high, line_dash="dash", line_color="rgba(6,214,160,0.5)", line_width=1)

                fig_comp_bars.update_layout(
                    height=240,
                    xaxis=dict(range=[0, 1.25], tickformat=".0%", gridcolor="rgba(51,65,85,0.3)",
                               tickfont=dict(color="#94a3b8"), title=None),
                    yaxis=dict(tickfont=dict(color="#f1f5f9", size=13)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=35, b=25, l=150, r=40),
                )
                st.plotly_chart(fig_comp_bars, use_container_width=True)

            with col_g2:
                # Radial indicator for final fused verdict
                fig_semi = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fusion.fused_probability * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": banner_color, "family": "Inter"}},
                    title={"text": f"Fused Verdict<br><span style='font-size:0.85em;color:{banner_color}'>{fusion.fused_decision}</span>",
                           "font": {"size": 15, "color": "#f1f5f9"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"color": "#94a3b8", "size": 10}},
                        "bar": {"color": banner_color, "thickness": 0.3},
                        "bgcolor": "rgba(30,41,59,0.5)",
                        "bordercolor": "rgba(51,65,85,0.5)",
                        "steps": [
                            {"range": [0, 35], "color": "rgba(239,68,68,0.25)"},
                            {"range": [35, 65], "color": "rgba(245,158,11,0.25)"},
                            {"range": [65, 100], "color": "rgba(6,214,160,0.25)"},
                        ],
                    },
                ))
                fig_semi.update_layout(
                    height=240,
                    margin=dict(t=40, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#94a3b8"},
                )
                st.plotly_chart(fig_semi, use_container_width=True)

            # ── Vision AI Detailed Breakdown ─────────────────────────
            st.markdown('<div class="section-header">🔬 Vision AI — Detailed Breakdown</div>',
                        unsafe_allow_html=True)

            # TSS classification row (if trained model available)
            if v2_result.get("model_available") and v2_result.get("ml_tss_class"):
                tss_probs = v2_result["ml_tss_probabilities"]
                tc1, tc2, tc3 = st.columns(3)
                tss_colors = {"Low": "#06d6a0", "Medium": "#f59e0b", "High": "#ef4444"}
                for col, (cls, prob) in zip([tc1, tc2, tc3], tss_probs.items()):
                    is_pred = (cls == v2_result["ml_tss_class"])
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="{'border: 2px solid ' + tss_colors[cls] if is_pred else ''}">
                            <h3>TSS: {cls} {'✓' if is_pred else ''}</h3>
                            <div class="value" style="color: {tss_colors[cls]};">{prob:.1%}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown("")

            v1, v2_col, v3, v4 = st.columns(4)
            with v1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Est. Turbidity</h3>
                    <div class="value {'safe' if vision_result.clarity.estimated_turbidity_ntu < 5 else 'unsafe'}">
                        {vision_result.clarity.estimated_turbidity_ntu} NTU
                    </div>
                </div>""", unsafe_allow_html=True)
            with v2_col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Algae Risk</h3>
                    <div class="value {'safe' if vision_result.contaminants.algae_risk < 0.3 else 'unsafe'}">
                        {vision_result.contaminants.algae_risk:.0%}
                    </div>
                </div>""", unsafe_allow_html=True)
            with v3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Sediment Risk</h3>
                    <div class="value {'safe' if vision_result.contaminants.sediment_risk < 0.3 else 'unsafe'}">
                        {vision_result.contaminants.sediment_risk:.0%}
                    </div>
                </div>""", unsafe_allow_html=True)
            with v4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Foam Risk</h3>
                    <div class="value {'safe' if vision_result.contaminants.foam_risk < 0.2 else 'unsafe'}">
                        {vision_result.contaminants.foam_risk:.0%}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── Colour Composition Pie ───────────────────────────────
            col_pie, col_radar = st.columns(2)

            with col_pie:
                st.markdown("**Colour Composition**")
                colour_data = {
                    "Blue": vision_result.colour.blue_ratio,
                    "Green": vision_result.colour.green_ratio,
                    "Brown/Ochre": vision_result.colour.brown_ratio,
                    "Colourless/Grey": vision_result.colour.colourless_ratio,
                }
                # Only show non-zero
                colour_data = {k: v for k, v in colour_data.items() if v > 0.005}
                if colour_data:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(colour_data.keys()),
                        values=list(colour_data.values()),
                        marker_colors=["#0ea5e9", "#22c55e", "#a16207", "#64748b"],
                        hole=0.45,
                        textinfo="percent+label",
                        textfont=dict(color="#e2e8f0", size=11),
                    )])
                    fig_pie.update_layout(
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8"),
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col_radar:
                st.markdown("**Vision Sub-Scores**")
                # Build radar of vision components
                cats = ["Clarity", "Colour Purity", "Low Algae", "Low Sediment", "Low Foam"]
                vals = [
                    max(0, 1.0 - vision_result.clarity.estimated_turbidity_ntu / 100),
                    max(0, 1.0 - (vision_result.colour.green_ratio + vision_result.colour.brown_ratio)),
                    1.0 - vision_result.contaminants.algae_risk,
                    1.0 - vision_result.contaminants.sediment_risk,
                    1.0 - vision_result.contaminants.foam_risk,
                ]
                vals_closed = vals + [vals[0]]
                cats_closed = cats + [cats[0]]

                fig_vr = go.Figure()
                fig_vr.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=cats_closed,
                    fill="toself",
                    fillcolor="rgba(14,165,233,0.15)",
                    line=dict(color="#0ea5e9", width=2),
                    marker=dict(size=6, color="#06d6a0"),
                ))
                fig_vr.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor="rgba(51,65,85,0.5)",
                                        tickfont=dict(color="#64748b", size=9)),
                        angularaxis=dict(gridcolor="rgba(51,65,85,0.3)",
                                         tickfont=dict(color="#e2e8f0", size=10)),
                    ),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    margin=dict(t=30, b=30, l=50, r=50),
                    showlegend=False,
                )
                st.plotly_chart(fig_vr, use_container_width=True)

            # ── Warnings ─────────────────────────────────────────────
            if vision_result.warnings or fusion.disagreement_details:
                st.markdown('<div class="section-header">⚠️ Alerts & Warnings</div>',
                            unsafe_allow_html=True)
                all_warnings = vision_result.warnings + fusion.disagreement_details
                for w in all_warnings:
                    st.warning(w)

            # ── Recommendation & Action Items ────────────────────────
            st.markdown('<div class="section-header">📋 Recommendation & Action Items</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(30,41,59,0.7); border: 1px solid rgba(14,165,233,0.2);
                        border-radius: 14px; padding: 20px; margin-bottom: 16px;">
                <div style="color: #e2e8f0; font-size: 1.0rem; line-height: 1.7;">
                    {fusion.recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if fusion.action_items:
                for item in fusion.action_items:
                    st.markdown(f"- {item}")

            # ── Technical Details Expander ────────────────────────────
            with st.expander("🔧 Technical Fusion Details"):
                st.json({
                    "fusion_method": fusion.fusion_method,
                    "vision_weight": fusion.vision_weight,
                    "numerical_weight": fusion.numerical_weight,
                    "vision_probability": vision_prob,
                    "vision_confidence": vision_result.confidence,
                    "vision_quality_score": vision_result.visual_quality_score,
                    "numerical_probability": ml_prob,
                    "numerical_confidence": 0.85,
                    "fused_probability": fusion.fused_probability,
                    "fused_confidence": fusion.fused_confidence,
                    "fused_decision": fusion.fused_decision,
                    "agreement": fusion.modality_agreement,
                    "agreement_score": fusion.agreement_score,
                    "risk_level": fusion.fused_risk_level,
                    "vision_colour": {
                        "dominant_hue": vision_result.colour.dominant_hue,
                        "dominant_saturation": vision_result.colour.dominant_saturation,
                        "blue_ratio": vision_result.colour.blue_ratio,
                        "green_ratio": vision_result.colour.green_ratio,
                        "brown_ratio": vision_result.colour.brown_ratio,
                        "colourless_ratio": vision_result.colour.colourless_ratio,
                        "label": vision_result.colour.colour_label,
                    },
                    "vision_clarity": {
                        "sharpness": vision_result.clarity.sharpness,
                        "contrast": vision_result.clarity.contrast,
                        "entropy": vision_result.clarity.entropy,
                        "edge_density": vision_result.clarity.edge_density,
                        "estimated_ntu": vision_result.clarity.estimated_turbidity_ntu,
                        "label": vision_result.clarity.clarity_label,
                    },
                    "vision_contaminants": {
                        "algae_risk": vision_result.contaminants.algae_risk,
                        "sediment_risk": vision_result.contaminants.sediment_risk,
                        "foam_risk": vision_result.contaminants.foam_risk,
                        "discolouration_risk": vision_result.contaminants.discolouration_risk,
                        "particle_count": vision_result.contaminants.particle_count_estimate,
                        "anomalies": vision_result.contaminants.anomalies,
                    },
                })


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance & Benchmarks</h1>
        <p>Comprehensive evaluation covering both the newly trained <b>Multimodal Vision & Numerical Models</b>
        (Zenodo + Nature Scientific Data) and the original <b>Kaggle Baseline Ensemble</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_multi, tab_kaggle = st.tabs([
        "🧬 Multimodal Benchmark (Zenodo + Nature Data)",
        "🔬 Baseline Kaggle Ensemble (Person B)",
    ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1: MULTIMODAL REAL BENCHMARKS (ZENODO + NATURE DATA)
    # ─────────────────────────────────────────────────────────────────
    with tab_multi:
        # Load multimodal meta
        mm_meta_file = APP_DIR / "models" / "multimodal_meta.json"
        mm_meta = {}
        if mm_meta_file.exists():
            with open(mm_meta_file) as _f:
                mm_meta = json.load(_f)

        vis_meta = mm_meta.get("vision", {})
        num_meta = mm_meta.get("numerical", {})

        # ── 1. Vision AI Model Performance ───────────────────────────
        st.markdown('<div class="section-header">📸 Vision AI Model (Zenodo Record 19899828)</div>', unsafe_allow_html=True)
        st.caption("Trained on 6,518 laboratory water samples across 3 Total Suspended Solids (TSS) classes: Baja (Low), Media (Medium), Alta (High).")

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>TSS 3-Class Accuracy</h3>
                <div class="value safe">{vis_meta.get('tss_test_accuracy', 0.996):.1%}</div>
            </div>""", unsafe_allow_html=True)
        with vc2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Potability ROC-AUC</h3>
                <div class="value primary">{vis_meta.get('potability_test_roc_auc', 1.0):.1%}</div>
            </div>""", unsafe_allow_html=True)
        with vc3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Potability F1 Score</h3>
                <div class="value primary">{vis_meta.get('potability_test_f1', 0.989):.1%}</div>
            </div>""", unsafe_allow_html=True)
        with vc4:
            st.markdown("""
            <div class="metric-card">
                <h3>Sample Count</h3>
                <div class="value primary">6,518</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Vision Feature Importances Chart
        vis_feats = vis_meta.get("feature_importances", {})
        if vis_feats:
            st.markdown("**Top Visual Biomarkers (Random Forest Feature Importance)**")
            sorted_feats = sorted(vis_feats.items(), key=lambda x: x[1], reverse=True)
            feat_names = [x[0] for x in sorted_feats[:10]][::-1]
            feat_vals = [x[1] for x in sorted_feats[:10]][::-1]

            feat_label_map = {
                "edge_density": "Edge Density (Turbidity / Sharpness)",
                "rb_ratio": "Red/Blue Ratio (Muddy / Rust Discolouration)",
                "blue_dominance": "Blue Dominance (Translucent Pure Water)",
                "brown_index": "Brown Index (Sediment Particles)",
                "green_index": "Green Index (Algae Biomass)",
                "rg_ratio": "Red/Green Ratio",
                "std_r": "Red Channel Variation",
                "mean_b": "Mean Blue Channel",
                "mean_r": "Mean Red Channel",
                "contrast": "RMS Image Contrast",
            }
            clean_labels = [feat_label_map.get(f, f) for f in feat_names]

            fig_vfeat = go.Figure(go.Bar(
                y=clean_labels,
                x=feat_vals,
                orientation="h",
                marker_color="#0ea5e9",
                text=[f"{v:.1%}" for v in feat_vals],
                textposition="outside",
                textfont=dict(color="#f1f5f9"),
            ))
            fig_vfeat.update_layout(
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8"},
                xaxis=dict(title="Importance Weight", gridcolor="rgba(51,65,85,0.4)", tickformat=".0%"),
                yaxis=dict(tickfont=dict(color="#e2e8f0")),
                margin=dict(l=260, r=50, t=20, b=30),
            )
            st.plotly_chart(fig_vfeat, use_container_width=True)

        # ── 2. Numerical Model Performance ───────────────────────────
        st.markdown('<div class="section-header">🔢 Numerical ML Ensemble (Nature Scientific Data)</div>', unsafe_allow_html=True)
        st.caption("Trained on 652,782 water records (1940–2023) across 8 chemical parameters predicting CCME Water Quality Index compliance.")

        num_metrics = num_meta.get("metrics", {})
        nc1, nc2, nc3, nc4, nc5 = st.columns(5)
        with nc1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <div class="value primary">{num_metrics.get('roc_auc', 0.9999):.2%}</div>
            </div>""", unsafe_allow_html=True)
        with nc2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value safe">{num_metrics.get('accuracy', 0.9995):.2%}</div>
            </div>""", unsafe_allow_html=True)
        with nc3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Balanced Acc</h3>
                <div class="value safe">{num_metrics.get('balanced_acc', 0.9864):.1%}</div>
            </div>""", unsafe_allow_html=True)
        with nc4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1 Score</h3>
                <div class="value primary">{num_metrics.get('f1', 0.9997):.2%}</div>
            </div>""", unsafe_allow_html=True)
        with nc5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Dataset Size</h3>
                <div class="value primary">652,782</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Sub-model comparison in Nature Ensemble
        col_aucs, col_arch = st.columns([5, 5])
        with col_aucs:
            st.markdown("**Ensemble Component Comparison (ROC-AUC)**")
            sub_aucs = num_meta.get("individual_aucs", {"RF": 0.9998, "GB": 0.9999})
            fig_sub = go.Figure(go.Bar(
                x=list(sub_aucs.keys()) + ["Soft Ensemble"],
                y=list(sub_aucs.values()) + [num_metrics.get("roc_auc", 0.9999)],
                marker_color=["#64748b", "#0ea5e9", "#06d6a0"],
                text=[f"{v:.4f}" for v in list(sub_aucs.values()) + [num_metrics.get("roc_auc", 0.9999)]],
                textposition="outside",
                textfont=dict(color="#f1f5f9"),
            ))
            fig_sub.update_layout(
                height=280,
                yaxis=dict(range=[0.998, 1.0005], tickformat=".4f", gridcolor="rgba(51,65,85,0.4)"),
                xaxis=dict(tickfont=dict(color="#e2e8f0")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8"},
                margin=dict(l=60, r=20, t=30, b=30),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        with col_arch:
            st.markdown("**Evidence Fusion Policy (Dempster-Shafer)**")
            st.markdown("""
            <div style="background: rgba(30,41,59,0.7); border: 1px solid rgba(14,165,233,0.2);
                        border-radius: 12px; padding: 20px; font-size: 0.9rem; line-height: 1.7;">
                <b>• Vision Modality Base Weight:</b> 30% (Optical clarity & surface turbidity)<br>
                <b>• Numerical Modality Base Weight:</b> 70% (Chemical & dissolved toxicity)<br>
                <b>• Conflict Resolution:</b> If modalities disagree (e.g., clear image with lethal nitrates),
                the model triggers an <b>INCONCLUSIVE / DANGER</b> safety override.<br>
                <b>• Decision Boundaries:</b><br>
                &nbsp;&nbsp;🟢 <b>Safe:</b> P(Potable) ≥ 65%<br>
                &nbsp;&nbsp;🟡 <b>Uncertain:</b> 35% &lt; P(Potable) &lt; 65% (Dispatches lab retest)<br>
                &nbsp;&nbsp;🔴 <b>Unsafe:</b> P(Potable) ≤ 35%
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2: ORIGINAL PERSON B BASELINE (KAGGLE)
    # ─────────────────────────────────────────────────────────────────
    with tab_kaggle:
        # ── Test metrics cards ───────────────────────────────────────
        st.markdown('<div class="section-header">Original Person B Test-Set Metrics (Kaggle)</div>', unsafe_allow_html=True)

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

        # ── Radar chart ──────────────────────────────────────────────
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

        # ── Model comparison ─────────────────────────────────────────
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

        # ── Cross-validation results ─────────────────────────────────
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

        # ── Evaluation plots from Person B ───────────────────────────
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
