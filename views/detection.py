"""
Real-time Network Flow Detection & Testing View.
Simulates live enterprise intrusion detection and alert triage on real CIC-IDS2017 traffic.
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.model_trainer import load_saved_model
from src.evaluator import run_batch_detection
from src.config import RISK_LEVELS, BENIGN_LABEL


def render_detection_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🚨 Live Network Flow Anomaly Detection")
    st.caption("Evaluate incoming traffic flows against the trained Random Forest detection engine.")

    # Model loader
    artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")
    if artifact is None:
        artifact = st.session_state.get("model_binary") or load_saved_model("binary")

    if artifact is None:
        st.warning("Please train or load a model from the ML Evaluation view first.")
        return

    # Simulation batch configuration
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider("Traffic Batch Size", 50, 1000, 200, step=50)
    with c2:
        traffic_mix = st.selectbox(
            "Traffic Profile",
            ["Natural Mix (Real IDS Ratios)", "High-Threat Stress Test (Attacks Heavy)", "Pure Benign Baseline"],
        )
    with c3:
        run_detection_btn = st.button("⚡ Inspect Traffic Batch", type="primary")

    # Sample generator
    if "detection_batch" not in st.session_state or run_detection_btn:
        if traffic_mix == "High-Threat Stress Test (Attacks Heavy)":
            attack_pool = df_sample[df_sample["Label"] != BENIGN_LABEL]
            benign_pool = df_sample[df_sample["Label"] == BENIGN_LABEL]
            n_attacks = min(int(batch_size * 0.7), len(attack_pool))
            n_benign = batch_size - n_attacks
            sampled = pd.concat([
                attack_pool.sample(n=n_attacks, replace=True, random_state=None),
                benign_pool.sample(n=n_benign, replace=True, random_state=None),
            ]).sample(frac=1.0)
        elif traffic_mix == "Pure Benign Baseline":
            benign_pool = df_sample[df_sample["Label"] == BENIGN_LABEL]
            sampled = benign_pool.sample(n=batch_size, replace=True, random_state=None)
        else:
            sampled = df_sample.sample(n=batch_size, replace=True, random_state=None)

        results_df = run_batch_detection(artifact, sampled)
        st.session_state["detection_batch"] = results_df

    results_df = st.session_state["detection_batch"]

    # Telemetry KPIs
    total_pkts = len(results_df)
    detected_attacks = int(results_df["Is_Anomaly"].sum())
    benign_count = total_pkts - detected_attacks
    crit_count = int((results_df["Risk_Level"] == "CRITICAL").sum())
    high_count = int((results_df["Risk_Level"] == "HIGH").sum())

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Flows Inspected", f"{total_pkts:,}")
    with k2:
        st.metric("Anomalies Blocked", f"{detected_attacks:,}", f"{(detected_attacks/total_pkts)*100:.1f}%")
    with k3:
        st.metric("Critical Alerts", f"{crit_count:,}", "Immediate Action", delta_color="inverse")
    with k4:
        st.metric("Benign Verified", f"{benign_count:,}", f"{(benign_count/total_pkts)*100:.1f}%")

    st.markdown("---")

    # Live Alert Feed Table
    st.markdown("#### 🛰️ Live Detection Event Log")
    
    # Format table for display
    display_cols = [
        "Risk_Level",
        "Predicted_Label",
        "Confidence",
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Flow Packets/s",
    ]
    if "Label" in results_df.columns:
        display_cols.insert(2, "Label")

    preview_df = results_df[display_cols].copy()
    preview_df["Confidence"] = preview_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(
        preview_df.head(100),
        use_container_width=True,
        height=400,
    )

    st.caption("Click any column header to sort by Risk Level, Confidence, or Protocol Port.")
