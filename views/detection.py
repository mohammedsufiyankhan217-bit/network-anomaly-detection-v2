"""
Real-time Network Flow Detection & Batch Replay View for Network Anomaly Detection V2.
Simulates enterprise intrusion triage and dual-classifier consensus analysis
on verified CIC-IDS2017 network traffic batches.
"""
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.model_trainer import load_saved_model
from src.evaluator import (
    run_batch_detection,
    run_dual_detection,
    plot_dual_classifier_comparison,
)
from src.config import RISK_LEVELS, BENIGN_LABEL, CATEGORY_COLORS
from src.ui_components import render_chart_header, apply_cyber_theme, CYBER_PALETTE, TRAFFIC_COLOR_MAP


def render_detection_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🚨 Network Flow Anomaly Detection & Incident Triage")
    st.caption("Batch-based forensic inspection and dual-classifier consensus analysis on real CIC-IDS2017 traffic flows.")

    # Load both models for dual inspection
    bin_artifact = st.session_state.get("model_binary") or load_saved_model("binary")
    multi_artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")

    active_artifact = multi_artifact if multi_artifact is not None else bin_artifact
    if active_artifact is None:
        st.warning("⚠️ No trained models found. Please train a model in the ML Training & Evaluation view first.")
        return

    # Simulation batch configuration controls
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider("Traffic Batch Size", 50, 1000, 200, step=50, key="det_batch_size")
    with c2:
        traffic_mix = st.selectbox(
            "Traffic Profile",
            ["Natural Mix (Real IDS Ratios)", "High-Threat Stress Test (Attacks Heavy)", "Pure Benign Baseline"],
            key="det_traffic_mix",
        )
    with c3:
        run_detection_btn = st.button("⚡ Inspect Traffic Batch", type="primary", key="det_run_btn")

    # Sample and process batch
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

        # Run dual detection if both models are present, else single
        if bin_artifact is not None and multi_artifact is not None:
            results_df = run_dual_detection(bin_artifact, multi_artifact, sampled)
            # Reconcile primary columns for compatibility
            results_df["Predicted_Label"] = results_df["Multi_Pred"]
            results_df["Confidence"] = results_df["Multi_Conf"]
            results_df["Anomaly_Score"] = results_df["Multi_Anomaly_Score"]
            results_df["Risk_Level"] = results_df["Multi_Risk"]
            results_df["Is_Anomaly"] = results_df["Binary_Pred"] == "ATTACK"
        else:
            results_df = run_batch_detection(active_artifact, sampled)

        st.session_state["detection_batch"] = results_df

    results_df = st.session_state["detection_batch"]

    # ==========================================
    # Detection Summary & Telemetry Cards
    # ==========================================
    total_pkts = len(results_df)
    detected_attacks = int(results_df["Is_Anomaly"].sum())
    benign_count = total_pkts - detected_attacks
    crit_count = int((results_df["Risk_Level"] == "CRITICAL").sum())
    high_count = int((results_df["Risk_Level"] == "HIGH").sum())
    avg_conf = float(results_df["Confidence"].mean()) * 100

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Total Flows Analyzed",
            f"{total_pkts:,}",
            help="Total network flow records processed in current inspection batch.",
        )
    with k2:
        st.metric(
            "Benign Verified Flows",
            f"{benign_count:,}",
            f"{(benign_count / total_pkts) * 100:.1f}% clean",
            delta_color="normal",
        )
    with k3:
        st.metric(
            "Malicious Flows Flagged",
            f"{detected_attacks:,}",
            f"{(detected_attacks / total_pkts) * 100:.1f}% threat rate",
            delta_color="inverse",
            help="Network flows flagged as anomalous/malicious by the detection engine.",
        )
    with k4:
        st.metric(
            "Critical Security Alerts",
            f"{crit_count:,}",
            f"High Alert: {high_count}",
            delta_color="inverse",
            help="Flows exceeding critical anomaly confidence threshold (>= 85%).",
        )

    st.markdown("---")

    # ==========================================
    # ROW 1: Batch Prediction Distributions
    # ==========================================
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        bm_chart_type = render_chart_header(
            title="🛡️ Benign vs Malicious Predictions",
            chart_types=["🍩 Donut Chart", "🥧 Pie Chart", "📊 Bar Chart", "📶 Horizontal Bar"],
            key="det_bm_chart_type",
            default="🍩 Donut Chart",
            help_text="Classification distribution of analyzed batch records.",
        )

        batch_bm_df = pd.DataFrame([
            {"Classification": "Benign Verified", "Flows": benign_count},
            {"Classification": "Malicious Anomaly", "Flows": detected_attacks},
        ])

        if bm_chart_type in ["🍩 Donut Chart", "🥧 Pie Chart"]:
            hole = 0.55 if "Donut" in bm_chart_type else 0.0
            fig_bm = px.pie(
                batch_bm_df,
                names="Classification",
                values="Flows",
                hole=hole,
                color="Classification",
                color_discrete_map={"Benign Verified": "#10b981", "Malicious Anomaly": "#ef4444"},
            )
            fig_bm.update_traces(textposition="inside", textinfo="percent+label")
            fig_bm = apply_cyber_theme(fig_bm, height=380, legend_orientation="v")
        elif bm_chart_type == "📊 Bar Chart":
            fig_bm = px.bar(
                batch_bm_df,
                x="Classification",
                y="Flows",
                color="Classification",
                color_discrete_map={"Benign Verified": "#10b981", "Malicious Anomaly": "#ef4444"},
                text="Flows",
            )
            fig_bm.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bm.update_layout(yaxis=dict(range=[0, max(batch_bm_df["Flows"]) * 1.2]))
            fig_bm = apply_cyber_theme(fig_bm, height=380, show_legend=False)
        else:  # 📶 Horizontal Bar
            fig_bm = px.bar(
                batch_bm_df,
                x="Flows",
                y="Classification",
                orientation="h",
                color="Classification",
                color_discrete_map={"Benign Verified": "#10b981", "Malicious Anomaly": "#ef4444"},
                text="Flows",
            )
            fig_bm.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bm.update_layout(xaxis=dict(range=[0, max(batch_bm_df["Flows"]) * 1.25]))
            fig_bm = apply_cyber_theme(fig_bm, height=380, show_legend=False)

        st.plotly_chart(fig_bm, use_container_width=True)

    with row1_c2:
        pred_cat_chart_type = render_chart_header(
            title="🧬 Predicted Threat Category Distribution",
            chart_types=["📶 Horizontal Bar", "📊 Vertical Bar", "🍩 Donut Chart", "🥧 Pie Chart"],
            key="det_cat_chart_type",
            default="📶 Horizontal Bar",
            help_text="Breakdown of specific attack categories identified in the batch.",
        )

        cat_col = "Multi_Category" if "Multi_Category" in results_df.columns else "Predicted_Threat_Category"
        if cat_col in results_df.columns:
            # Exclude benign from threat breakdown if any threats exist
            threat_df = results_df[results_df["Is_Anomaly"]]
            if len(threat_df) == 0:
                threat_counts = results_df[cat_col].value_counts().reset_index()
            else:
                threat_counts = threat_df[cat_col].value_counts().reset_index()
            threat_counts.columns = ["Threat Category", "Flows"]

            if pred_cat_chart_type == "📶 Horizontal Bar":
                threat_sorted = threat_counts.sort_values("Flows", ascending=True)
                fig_cat = px.bar(
                    threat_sorted,
                    x="Flows",
                    y="Threat Category",
                    orientation="h",
                    color="Threat Category",
                    color_discrete_map=CATEGORY_COLORS,
                    text="Flows",
                )
                fig_cat.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
                fig_cat.update_layout(yaxis=dict(title=""), xaxis=dict(range=[0, max(threat_sorted["Flows"]) * 1.25]))
                fig_cat = apply_cyber_theme(fig_cat, height=380, show_legend=False)
            elif pred_cat_chart_type == "📊 Vertical Bar":
                fig_cat = px.bar(
                    threat_counts,
                    x="Threat Category",
                    y="Flows",
                    color="Threat Category",
                    color_discrete_map=CATEGORY_COLORS,
                    text="Flows",
                )
                fig_cat.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
                fig_cat.update_layout(xaxis=dict(tickangle=-25), yaxis=dict(range=[0, max(threat_counts["Flows"]) * 1.2]))
                fig_cat = apply_cyber_theme(fig_cat, height=380, show_legend=False)
            else:  # 🍩 Donut / 🥧 Pie
                hole = 0.55 if "Donut" in pred_cat_chart_type else 0.0
                fig_cat = px.pie(
                    threat_counts,
                    names="Threat Category",
                    values="Flows",
                    hole=hole,
                    color="Threat Category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig_cat.update_traces(textposition="inside", textinfo="percent+label")
                fig_cat = apply_cyber_theme(fig_cat, height=380, legend_orientation="v")

            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 2: Chronological Sequence & Dual Classifier Comparison
    # ==========================================
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        time_chart_type = render_chart_header(
            title="⏱️ Detection Sequence & Timeline Replay",
            chart_types=["📈 Anomaly Score Timeline", "🌊 Cumulative Attacks Area", "📊 Sequence Anomaly Bar", "📦 Scatter Detection Feed"],
            key="det_time_chart_type",
            default="📈 Anomaly Score Timeline",
            help_text="Chronological progression of inspected flow records across the batch.",
        )

        timeline_df = results_df.copy().reset_index(drop=True)
        timeline_df["Flow_Sequence"] = timeline_df.index + 1
        timeline_df["Cumulative_Anomalies"] = timeline_df["Is_Anomaly"].cumsum()

        if time_chart_type == "📈 Anomaly Score Timeline":
            fig_time = px.line(
                timeline_df,
                x="Flow_Sequence",
                y="Anomaly_Score",
                color="Risk_Level",
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#10b981"},
                markers=True,
                labels={"Flow_Sequence": "Analyzed Flow Index (Chronological)", "Anomaly_Score": "Anomaly Probability"},
            )
            # Add threshold reference line at 0.85 (Critical)
            fig_time.add_hline(y=0.85, line_dash="dash", line_color="#ef4444", annotation_text="Critical Threshold (0.85)")
            fig_time = apply_cyber_theme(fig_time, height=400)

        elif time_chart_type == "🌊 Cumulative Attacks Area":
            fig_time = px.area(
                timeline_df,
                x="Flow_Sequence",
                y="Cumulative_Anomalies",
                labels={"Flow_Sequence": "Analyzed Flow Index", "Cumulative_Anomalies": "Cumulative Intrusions Flagged"},
            )
            fig_time.update_traces(line=dict(color="#ef4444", width=3), fillcolor="rgba(239, 68, 68, 0.25)")
            fig_time = apply_cyber_theme(fig_time, height=400, show_legend=False)

        elif time_chart_type == "📊 Sequence Anomaly Bar":
            fig_time = px.bar(
                timeline_df,
                x="Flow_Sequence",
                y="Anomaly_Score",
                color="Is_Anomaly",
                color_discrete_map={False: "#10b981", True: "#ef4444"},
                labels={"Flow_Sequence": "Flow Index", "Anomaly_Score": "Anomaly Score"},
            )
            fig_time = apply_cyber_theme(fig_time, height=400)

        else:  # 📦 Scatter Detection Feed
            fig_time = px.scatter(
                timeline_df,
                x="Flow_Sequence",
                y="Confidence",
                color="Risk_Level",
                size="Anomaly_Score",
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#10b981"},
                hover_data=["Predicted_Label", "Destination Port"],
            )
            fig_time = apply_cyber_theme(fig_time, height=400)

        st.plotly_chart(fig_time, use_container_width=True)

    with row2_c2:
        dual_chart_type = render_chart_header(
            title="⚖️ Dual-Classifier Consensus & Comparison",
            chart_types=["🗺️ Agreement Heatmap", "📊 Consensus Bar", "🍩 Agreement Share"],
            key="det_dual_chart_type",
            default="🗺️ Agreement Heatmap",
            help_text="Consensus agreement between Binary Anomaly Detector and Multiclass Classifier on the analyzed batch.",
        )

        if "Binary_Pred" in results_df.columns and "Multi_Pred" in results_df.columns:
            fig_dual = plot_dual_classifier_comparison(results_df, chart_type=dual_chart_type)
        else:
            # Fallback if only single model loaded
            risk_dist = results_df["Risk_Level"].value_counts().reset_index()
            risk_dist.columns = ["Risk Tier", "Flows"]
            fig_dual = px.bar(
                risk_dist,
                x="Risk Tier",
                y="Flows",
                color="Risk Tier",
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#10b981"},
            )
            fig_dual = apply_cyber_theme(fig_dual, height=400, show_legend=False)

        st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 3: Live Detection Event Log Table
    # ==========================================
    st.markdown("#### 🛰️ Live Detection Forensic Event Feed")
    st.caption("Inspected flow records with classification verdicts, confidence levels, and port signatures:")

    display_cols = ["Risk_Level"]
    if "Binary_Pred" in results_df.columns:
        display_cols.append("Binary_Pred")
    display_cols.extend(["Predicted_Label", "Confidence", "Anomaly_Score", "Destination Port", "Flow Duration", "Flow Packets/s"])
    if "Label" in results_df.columns:
        display_cols.insert(1, "Label")

    preview_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()
    preview_df["Confidence"] = preview_df["Confidence"].apply(lambda x: f"{x * 100:.1f}%")
    preview_df["Anomaly_Score"] = preview_df["Anomaly_Score"].apply(lambda x: f"{x:.4f}")

    # Risk badge formatting
    st.dataframe(
        preview_df.sort_values("Anomaly_Score", ascending=False).head(100),
        use_container_width=True,
        height=380,
    )

    st.caption("Click any column header to sort by Risk Level, Confidence, or Port Signature.")
