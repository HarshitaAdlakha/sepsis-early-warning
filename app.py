"""
Sepsis Early Warning System — Streamlit App
Author: Harshita Adlakha
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.generate_demo_data import generate_dataset, ALL_FEATURES, NORMAL_PARAMS, SEPSIS_DELTA
from src.data.preprocessing import (
    encode_time_series_xgboost, fit_xgboost_scaler,
    apply_xgboost_scaler, compute_class_weights,
)
from src.models.xgboost_model import SepsisXGBModel
from src.evaluation.metrics import compute_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2a3a;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    border: 1px solid #2e3f55;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
.metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
.risk-high   { color: #ef5350; font-size: 1.6rem; font-weight: 700; }
.risk-med    { color: #ffa726; font-size: 1.6rem; font-weight: 700; }
.risk-low    { color: #66bb6a; font-size: 1.6rem; font-weight: 700; }
.section-header { font-size: 1.1rem; font-weight: 600;
                  border-left: 4px solid #4fc3f7;
                  padding-left: 10px; margin: 18px 0 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 Sepsis Early Warning")
st.sidebar.caption("Amazon ML Summer Program · Harshita Adlakha")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "⚙️ Train Model", "🩺 Patient Simulator", "📊 Model Insights"],
    index=0,
)

# ── Shared model cache ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training XGBoost model on synthetic data...")
def get_trained_model():
    dataset = generate_dataset(n_sepsis=570, n_control=2000, seed=42,
                               output_dir="data/processed")
    train_s, y_train = dataset["train"]
    val_s,   y_val   = dataset["val"]
    test_s,  y_test  = dataset["test"]

    X_train = encode_time_series_xgboost(train_s)
    X_val   = encode_time_series_xgboost(val_s)
    X_test  = encode_time_series_xgboost(test_s)

    scaler  = fit_xgboost_scaler(X_train)
    X_train = apply_xgboost_scaler(X_train, scaler)
    X_val   = apply_xgboost_scaler(X_val,   scaler)
    X_test  = apply_xgboost_scaler(X_test,  scaler)

    cw    = compute_class_weights(y_train)
    model = SepsisXGBModel(params={"n_estimators": 300, "learning_rate": 0.05,
                                    "max_depth": 6})
    model.fit(X_train, y_train, X_val, y_val, scale_pos_weight=cw[1] / cw[0])

    y_proba  = model.predict_proba(X_test)
    metrics  = compute_metrics(y_test, y_proba)
    return model, scaler, metrics, y_test, y_proba


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏥 Sepsis Early Warning System")
    st.markdown(
        "**Early prediction of sepsis onset in ICU patients** using gradient boosting "
        "on clinical time-series data — with a **7-hour prediction horizon** before onset."
    )

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("44", "Clinical Variables"),
        ("6,188", "Synthetic Patients"),
        ("7 hrs", "Prediction Horizon"),
        ("Sepsis-3", "Label Definition"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">What is Sepsis?</div>',
                    unsafe_allow_html=True)
        st.markdown("""
Sepsis is a **life-threatening dysregulation** of the host response to infection,
responsible for **11 million deaths per year** globally.

- Every hour of delayed antibiotics increases mortality by **7–8%**
- Affects **1.7 million** Americans annually
- Costs the US healthcare system **$62 billion** per year
- Early automated detection from routine ICU data can **directly save lives**
        """)

    with col2:
        st.markdown('<div class="section-header">How This System Works</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**1. Data Collection** — 44 clinical variables measured hourly in the ICU

**2. Feature Engineering** — Each variable encoded as statistical features
(mean, std, slope, min, max, quartiles) → 441 features total

**3. XGBoost Classifier** — Trained on Sepsis-3 labels with class-weight balancing

**4. Prediction** — Risk score output 7 hours before clinical sepsis onset
        """)

    st.markdown("---")
    st.markdown('<div class="section-header">Clinical Variables Used</div>',
                unsafe_allow_html=True)
    vitals_df = pd.DataFrame({
        "Category": ["Vital Signs"] * 15 + ["Laboratory Values"] * 29,
        "Variable": ALL_FEATURES,
    })
    c1, c2 = st.columns(2)
    c1.dataframe(vitals_df[vitals_df["Category"] == "Vital Signs"]["Variable"]
                 .reset_index(drop=True).rename("Vital Signs (15)"),
                 width="stretch", height=300)
    c2.dataframe(vitals_df[vitals_df["Category"] == "Laboratory Values"]["Variable"]
                 .reset_index(drop=True).rename("Laboratory Values (29)"),
                 width="stretch", height=300)

    st.markdown("---")
    st.markdown(
        "**References:** Singer et al. (2016) Sepsis-3 · Johnson et al. (2016) MIMIC-III · "
        "Chen & Guestrin (2016) XGBoost"
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Train Model":
    st.title("⚙️ Train & Evaluate Model")
    st.markdown(
        "Click **Train** to generate synthetic ICU data and train the XGBoost classifier. "
        "Results are cached — subsequent visits are instant."
    )

    if st.button("🚀 Train Model", type="primary", width="stretch"):
        with st.spinner("Training... (first run takes ~30 seconds)"):
            model, scaler, metrics, y_test, y_proba = get_trained_model()
        st.success("Model trained successfully!")
    else:
        model, scaler, metrics, y_test, y_proba = get_trained_model()

    st.markdown("---")
    st.markdown('<div class="section-header">Test Set Performance</div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    key_metrics = [
        ("AUROC",       "auroc",       "Area under ROC curve"),
        ("AUPRC",       "auprc",       "Area under PR curve"),
        ("Sensitivity", "sensitivity", "True positive rate"),
        ("Specificity", "specificity", "True negative rate"),
        ("F1 Score",    "f1",          "Harmonic mean of precision & recall"),
    ]
    for col, (label, key, _) in zip(cols, key_metrics):
        col.metric(label, f"{metrics[key]:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">ROC Curve</div>',
                    unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="gray"),
                                 name="Random (0.50)"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 line=dict(color="#4fc3f7", width=2.5),
                                 name=f"XGBoost (AUROC={metrics['auroc']:.3f})"))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>',
                    unsafe_allow_html=True)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        prevalence = float(y_test.mean())
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[prevalence, prevalence], mode="lines",
            line=dict(dash="dash", color="gray"),
            name=f"No-skill ({prevalence:.2f})",
        ))
        fig2.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            line=dict(color="#66bb6a", width=2.5),
            name=f"XGBoost (AUPRC={metrics['auprc']:.3f})",
        ))
        fig2.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.35, y=0.95),
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    cm_data = [[metrics["tn"], metrics["fp"]],
               [metrics["fn"], metrics["tp"]]]
    cm_labels = ["Predicted: No Sepsis", "Predicted: Sepsis"]
    fig3 = px.imshow(
        cm_data,
        text_auto=True,
        x=cm_labels, y=["Actual: No Sepsis", "Actual: Sepsis"],
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig3.update_layout(template="plotly_dark", height=320,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PATIENT SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Patient Simulator":
    st.title("🩺 Interactive Patient Simulator")
    st.markdown(
        "Adjust the patient's **vital signs and lab values** below. "
        "The model will predict the **sepsis risk score** in real time."
    )

    model, scaler, metrics, _, _ = get_trained_model()

    VITALS_DISPLAY = {
        "heart_rate":       ("Heart Rate",        "bpm",   40,  180, 75),
        "systolic_bp":      ("Systolic BP",        "mmHg",  60,  200, 120),
        "diastolic_bp":     ("Diastolic BP",       "mmHg",  30,  130, 75),
        "mean_arterial_bp": ("Mean Arterial BP",   "mmHg",  40,  150, 90),
        "resp_rate":        ("Respiratory Rate",   "/min",  6,   45,  16),
        "spo2":             ("SpO₂",               "%",     70,  100, 97),
        "temperature":      ("Temperature",        "°C",    34.0, 42.0, 37.0),
        "gcs_total":        ("GCS Total",          "",      3,   15,  14),
        "urine_output":     ("Urine Output",       "mL/hr", 0,   400, 60),
    }

    LABS_DISPLAY = {
        "lactate":       ("Lactate",       "mmol/L", 0.1, 15.0, 1.2),
        "wbc":           ("WBC",           "10⁹/L",  0.5, 30.0, 8.0),
        "creatinine":    ("Creatinine",    "mg/dL",  0.2, 8.0,  0.9),
        "bilirubin":     ("Bilirubin",     "mg/dL",  0.1, 15.0, 0.8),
        "platelets":     ("Platelets",     "10⁹/L",  20,  600,  220),
        "procalcitonin": ("Procalcitonin", "ng/mL",  0.0, 50.0, 0.05),
        "crp":           ("CRP",           "mg/L",   0.1, 200.0,5.0),
        "ph":            ("pH",            "",       6.9, 7.7,  7.40),
        "bicarbonate":   ("Bicarbonate",   "mEq/L",  5,   45,   24),
    }

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">Vital Signs</div>',
                    unsafe_allow_html=True)
        vitals_vals = {}
        vc1, vc2, vc3 = st.columns(3)
        for i, (key, (label, unit, lo, hi, default)) in enumerate(VITALS_DISPLAY.items()):
            col = [vc1, vc2, vc3][i % 3]
            step = 0.1 if isinstance(default, float) else 1
            vitals_vals[key] = col.number_input(
                f"{label} ({unit})" if unit else label,
                min_value=float(lo), max_value=float(hi),
                value=float(default), step=float(step),
            )

        st.markdown('<div class="section-header">Key Lab Values</div>',
                    unsafe_allow_html=True)
        labs_vals = {}
        lc1, lc2, lc3 = st.columns(3)
        for i, (key, (label, unit, lo, hi, default)) in enumerate(LABS_DISPLAY.items()):
            col = [lc1, lc2, lc3][i % 3]
            step = 0.01 if isinstance(default, float) and default < 5 else (
                   0.1  if isinstance(default, float) else 1)
            labs_vals[key] = col.number_input(
                f"{label} ({unit})" if unit else label,
                min_value=float(lo), max_value=float(hi),
                value=float(default), step=float(step),
            )

    # Build a synthetic 24-hour series using the entered values
    def build_patient_series(vitals_vals, labs_vals, n_hours=24):
        rng = np.random.default_rng(0)
        rows = []
        for h in range(n_hours):
            row = {"hour": h}
            for feat in ALL_FEATURES:
                mu, sigma = NORMAL_PARAMS[feat]
                if feat in vitals_vals:
                    mu = vitals_vals[feat]
                    sigma = sigma * 0.3
                elif feat in labs_vals:
                    mu = labs_vals[feat]
                    sigma = sigma * 0.3
                val = rng.normal(mu, max(sigma, 0.01))
                # labs measured every 4h
                if feat not in vitals_vals and h % 4 != 0:
                    row[feat] = np.nan
                else:
                    row[feat] = round(float(val), 3)
            rows.append(row)
        return pd.DataFrame(rows)

    patient_df = build_patient_series(vitals_vals, labs_vals)
    X = encode_time_series_xgboost([patient_df])
    X = apply_xgboost_scaler(X, scaler)
    risk_score = float(model.predict_proba(X)[0])

    with col_right:
        st.markdown('<div class="section-header">Sepsis Risk Score</div>',
                    unsafe_allow_html=True)

        if risk_score >= 0.6:
            level, colour, icon = "HIGH RISK", "risk-high", "🔴"
        elif risk_score >= 0.35:
            level, colour, icon = "MODERATE RISK", "risk-med", "🟡"
        else:
            level, colour, icon = "LOW RISK", "risk-low", "🟢"

        st.markdown(
            f'<div style="background:#1e2a3a;border-radius:12px;padding:30px;'
            f'text-align:center;border:1px solid #2e3f55;">'
            f'<div style="font-size:3.5rem;">{icon}</div>'
            f'<div class="{colour}">{risk_score:.1%}</div>'
            f'<div style="color:#90a4ae;font-size:0.9rem;margin-top:6px;">'
            f'Predicted Sepsis Risk</div>'
            f'<hr style="border-color:#2e3f55;margin:16px 0">'
            f'<div class="{colour}" style="font-size:1.1rem;">{level}</div>'
            f'</div>', unsafe_allow_html=True
        )

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ef5350" if risk_score >= 0.6 else
                                  "#ffa726" if risk_score >= 0.35 else "#66bb6a"},
                "steps": [
                    {"range": [0,  35], "color": "#1b3a2d"},
                    {"range": [35, 60], "color": "#3a2e10"},
                    {"range": [60, 100],"color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "value": 50},
            },
        ))
        fig_gauge.update_layout(
            height=220, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig_gauge, width="stretch")

        st.info(
            "**Interpretation**\n\n"
            "- < 35% → Low risk\n"
            "- 35–60% → Moderate: monitor closely\n"
            "- > 60% → High: consider immediate assessment\n\n"
            "*For research purposes only — not clinical advice.*"
        )

    # Trend chart of entered vitals
    st.markdown("---")
    st.markdown('<div class="section-header">Simulated 24-Hour Trends</div>',
                unsafe_allow_html=True)
    trend_feat = st.selectbox("Select variable to plot", list(VITALS_DISPLAY.keys()),
                              format_func=lambda k: VITALS_DISPLAY[k][0])
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=patient_df["hour"], y=patient_df[trend_feat],
        mode="lines+markers", line=dict(color="#4fc3f7"),
        name=VITALS_DISPLAY.get(trend_feat, (trend_feat,))[0],
    ))
    normal_mu = NORMAL_PARAMS[trend_feat][0]
    fig_trend.add_hline(y=normal_mu, line_dash="dash", line_color="gray",
                        annotation_text="Normal", annotation_position="right")
    fig_trend.update_layout(
        template="plotly_dark", height=280,
        xaxis_title="Hour in ICU",
        yaxis_title=VITALS_DISPLAY.get(trend_feat, ("", ""))[1],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_trend, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")

    model, scaler, metrics, y_test, y_proba = get_trained_model()

    st.markdown('<div class="section-header">Top 20 Predictive Features</div>',
                unsafe_allow_html=True)

    from src.data.preprocessing import STAT_FUNCTIONS
    feat_names = ["series_length"] + [
        f"{feat}_{stat}" for feat in ALL_FEATURES for stat in STAT_FUNCTIONS
    ]
    importance = model.feature_importance(feature_names=feat_names)
    top20 = dict(list(importance.items())[:20])

    fig_imp = go.Figure(go.Bar(
        x=list(top20.values())[::-1],
        y=list(top20.keys())[::-1],
        orientation="h",
        marker_color="#4fc3f7",
    ))
    fig_imp.update_layout(
        template="plotly_dark", height=500,
        xaxis_title="Importance Score",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_imp, width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-header">Risk Score Distribution</div>',
                unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y_test == 0], name="Control (No Sepsis)",
        opacity=0.7, nbinsx=40, marker_color="#4fc3f7",
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_proba[y_test == 1], name="Sepsis",
        opacity=0.7, nbinsx=40, marker_color="#ef5350",
    ))
    fig_dist.update_layout(
        barmode="overlay", template="plotly_dark",
        xaxis_title="Predicted Risk Score",
        yaxis_title="Count",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-header">Full Metrics Summary</div>',
                unsafe_allow_html=True)
    metrics_df = pd.DataFrame([{
        "Metric": k.replace("_", " ").title(),
        "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
    } for k, v in metrics.items() if k not in ("tp", "tn", "fp", "fn")])
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown(
        "**About the model:** XGBoost classifier trained on synthetic ICU time-series "
        "data. Each patient's variable-length time series is encoded into 441 statistical "
        "features (count, mean, std, min, max, Q25/50/75, last value, slope per variable). "
        "Class imbalance handled via `scale_pos_weight`. "
        "Hyperparameters tunable via Optuna."
    )
