"""
Semiconductor Yield Debug Dashboard

Interactive diagnostic tool for inspecting wafer samples, understanding
model predictions, and investigating potential failure root causes.

Run:  streamlit run app/streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.diagnostics import DiagnosticsPipeline, generate_root_cause_summary

# ─── Page config ──────────────────────────────────────────

st.set_page_config(
    page_title="Yield Debug Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Pipeline loading (cached) ───────────────────────────

@st.cache_resource
def load_pipeline():
    pipeline = DiagnosticsPipeline(model_dir="models", data_dir="data/processed")
    pipeline.load()
    return pipeline


def main():
    # ── Header ────────────────────────────────────────────
    st.title("Semiconductor Yield Debug Dashboard")
    st.markdown(
        "Inspect wafer samples, understand model-driven pass/fail predictions, "
        "and investigate sensor-level root causes against the healthy production baseline."
    )

    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(
            f"**Failed to load pipeline artifacts.** Make sure you have run "
            f"`python src/train.py` and that `data/processed/` contains "
            f"`X_clean.csv` and `y.csv`.\n\nError: {e}"
        )
        return

    # ── Sidebar: Sample selection ─────────────────────────
    st.sidebar.header("Sample Selection")

    mode = st.sidebar.radio(
        "Input mode",
        ["Select by index", "Manual sensor override"],
        index=0,
    )

    if mode == "Select by index":
        filter_type = st.sidebar.selectbox(
            "Filter samples",
            ["All samples", "Fail only", "Pass only"],
        )

        if filter_type == "Fail only":
            valid_indices = pipeline.fail_indices
        elif filter_type == "Pass only":
            valid_indices = pipeline.pass_indices
        else:
            valid_indices = list(range(pipeline.n_samples))

        sample_idx = st.sidebar.selectbox(
            f"Sample index ({len(valid_indices)} available)",
            valid_indices,
            index=0,
        )

        raw_features = pipeline.get_sample(sample_idx)
        actual_label = pipeline.get_label(sample_idx)
        actual_label_str = "FAIL" if actual_label == 1 else "PASS"

        st.sidebar.markdown(f"**Actual label:** {actual_label_str}")
    else:
        st.sidebar.markdown("Start from a base sample, then override sensor values below.")
        sample_idx = st.sidebar.number_input(
            "Base sample index", min_value=0, max_value=pipeline.n_samples - 1, value=0
        )
        raw_features = pipeline.get_sample(sample_idx).copy()
        actual_label = pipeline.get_label(sample_idx)
        actual_label_str = "FAIL" if actual_label == 1 else "PASS"

    # ── Run diagnostics ──────────────────────────────────
    top_k = st.sidebar.slider("Top sensors to analyze", min_value=3, max_value=20, value=10)

    diag = pipeline.diagnose_sample(raw_features, sample_index=sample_idx, top_k=top_k)

    # ── Manual override UI (after initial diagnosis to show feature names) ──
    if mode == "Manual sensor override":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Override Sensor Values")
        overrides = {}
        for feat in diag.top_shap_features[:5]:
            current_val = float(raw_features.get(feat, 0.0))
            new_val = st.sidebar.number_input(
                f"{feat}", value=current_val, format="%.4f", key=f"override_{feat}"
            )
            if new_val != current_val:
                overrides[feat] = new_val

        if overrides:
            for feat_name in overrides:
                if feat_name in raw_features.index:
                    raw_features[feat_name] = overrides[feat_name]
            diag = pipeline.diagnose_sample(raw_features, sample_index=sample_idx, top_k=top_k)
            st.sidebar.success(f"Applied {len(overrides)} override(s). Results updated.")

    # ── Layout: Main panels ──────────────────────────────

    # ── Panel 1: Sample Diagnostic Overview ──
    st.header("Sample Diagnostic Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pred_color = "🔴" if diag.prediction == "FAIL" else "🟢"
        st.metric("Prediction", f"{pred_color} {diag.prediction}")
    with col2:
        st.metric("Failure Probability", f"{diag.probability:.1%}")
    with col3:
        st.metric("Actual Label", actual_label_str)
    with col4:
        match = "✓ Correct" if (
            (diag.prediction == "FAIL" and actual_label == 1) or
            (diag.prediction == "PASS" and actual_label == 0)
        ) else "✗ Mismatch"
        st.metric("Model vs Actual", match)

    # ── Panel 2: Top Sensor Drivers (SHAP) ──
    st.header("Top Sensor Drivers")
    st.markdown(
        "SHAP values show each sensor's contribution to the failure prediction. "
        "Positive values push toward FAIL; negative toward PASS."
    )

    top_features = diag.top_shap_features[:top_k]
    top_shap_vals = [float(diag.shap_values[f]) for f in top_features]

    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in top_shap_vals]

    fig_shap = go.Figure(go.Bar(
        x=top_shap_vals[::-1],
        y=top_features[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in top_shap_vals[::-1]],
        textposition="auto",
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP Value (impact on failure prediction)",
        yaxis_title="Sensor",
        height=max(300, top_k * 35),
        margin=dict(l=20, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Panel 3: Deviation from Healthy Baseline ──
    st.header("Deviation from Healthy Baseline")
    st.markdown(
        f"Z-scores measure how far this sample's sensor readings deviate from the "
        f"healthy production baseline (computed from {pipeline.baseline.n_samples} "
        f"pass-only samples). Values beyond ±2σ are flagged."
    )

    comparison_df = pipeline.get_baseline_comparison_df(diag, features=top_features)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        z_vals = comparison_df["Z-Score"].values
        z_colors = ["#d32f2f" if abs(z) > 2 else "#ff9800" if abs(z) > 1 else "#4caf50"
                     for z in z_vals]

        fig_dev = go.Figure(go.Bar(
            x=z_vals,
            y=comparison_df["Sensor"].values,
            orientation="h",
            marker_color=z_colors,
            text=[f"{z:+.1f}σ" for z in z_vals],
            textposition="auto",
        ))
        fig_dev.add_vline(x=-2, line_dash="dash", line_color="gray", opacity=0.5)
        fig_dev.add_vline(x=2, line_dash="dash", line_color="gray", opacity=0.5)
        fig_dev.update_layout(
            xaxis_title="Z-Score (σ from baseline mean)",
            yaxis_title="Sensor",
            height=max(300, top_k * 35),
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_table:
        def highlight_deviation(row):
            z = abs(row["Z-Score"])
            if z > 3:
                return ["background-color: #ffcdd2"] * len(row)
            elif z > 2:
                return ["background-color: #ffe0b2"] * len(row)
            return [""] * len(row)

        styled = comparison_df.style.apply(highlight_deviation, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Panel 4: Sample vs Baseline Overlay ──
    st.header("Sample vs Baseline Comparison")
    st.markdown("Direct comparison of this sample's values against the healthy population range.")

    overlay_features = top_features[:min(10, len(top_features))]
    sample_vals = [float(diag.feature_values[f]) for f in overlay_features]
    baseline_means = [float(pipeline.baseline.mean[f]) for f in overlay_features]
    baseline_stds = [float(pipeline.baseline.std[f]) for f in overlay_features]

    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Bar(
        name="Baseline Mean ± 1σ",
        x=overlay_features,
        y=baseline_means,
        error_y=dict(type="data", array=baseline_stds, visible=True),
        marker_color="#90caf9",
        opacity=0.7,
    ))
    fig_overlay.add_trace(go.Scatter(
        name="This Sample",
        x=overlay_features,
        y=sample_vals,
        mode="markers+lines",
        marker=dict(size=10, color="#d32f2f", symbol="diamond"),
        line=dict(color="#d32f2f", width=2),
    ))
    fig_overlay.update_layout(
        yaxis_title="Scaled Sensor Value",
        xaxis_title="Sensor",
        barmode="group",
        height=400,
        margin=dict(l=20, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    # ── Panel 5: Preliminary Failure Pattern Summary ──
    st.header("Preliminary Failure Pattern Summary")

    summary_text = generate_root_cause_summary(diag, pipeline.baseline, top_k=5)
    st.markdown(summary_text)

    # ── Panel 6: Raw Feature Inspector ──
    with st.expander("Raw Feature Inspector (all 50 model features)"):
        full_df = pd.DataFrame({
            "Sensor": diag.feature_values.index,
            "Value": diag.feature_values.values.round(4),
            "Baseline Mean": [round(float(pipeline.baseline.mean.get(f, 0)), 4)
                              for f in diag.feature_values.index],
            "Z-Score": diag.deviations_z.values.round(2),
            "SHAP": diag.shap_values.values.round(6),
        })
        full_df = full_df.sort_values("SHAP", key=lambda x: x.abs(), ascending=False)
        st.dataframe(full_df, use_container_width=True, hide_index=True, height=400)

    # ── Sidebar: Pipeline info ────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline Info")
    st.sidebar.markdown(f"""
    - **Dataset:** {pipeline.n_samples} samples
    - **Pass samples:** {len(pipeline.pass_indices)}
    - **Fail samples:** {len(pipeline.fail_indices)}
    - **Model features:** {len(pipeline.mi_selected_cols)}
    - **Model:** Random Forest ({pipeline.model.n_estimators} trees)
    """)


if __name__ == "__main__":
    main()
