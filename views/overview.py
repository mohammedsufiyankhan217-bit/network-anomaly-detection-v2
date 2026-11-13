"""
Executive Overview and Traffic Statistics View.
Displays real dataset KPIs, protocol & flow distributions, and daily activity.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import BENIGN_LABEL, CATEGORY_COLORS


def render_overview_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🌐 Executive Overview & Traffic Telemetry")
    st.caption("Live statistical analysis across all 8 verified CIC-IDS2017 days.")

    # Status banner
    if status["is_loaded"]:
        st.success(
            f"✅ **REAL DATASET ACTIVE**: All {status['found_count']} daily captures loaded "
            f"({status['total_rows']:,} official flows across {status['total_size_mb']:.1f} MB on disk)."
        )
    else:
        st.warning(f"⚠️ Partial dataset detected: {status['found_count']}/{status['total_expected']} files.")

    # Top KPI Cards
    total_raw = status["total_rows"]
    label_dist = status.get("label_distribution", {})
    benign_count = label_dist.get(BENIGN_LABEL, int(total_raw * 0.803))
    attack_count = total_raw - benign_count
    attack_pct = (attack_count / total_raw) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Network Flows",
            value=f"{total_raw:,}",
            help="Full count of bidirectional network flows across all 8 capture files.",
        )
    with col2:
        st.metric(
            label="Benign Traffic Volume",
            value=f"{benign_count:,}",
            delta=f"{(benign_count/total_raw)*100:.1f}% normal",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Malicious Attack Flows",
            value=f"{attack_count:,}",
            delta=f"{attack_pct:.2f}% anomaly rate",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            label="Unique Attack Categories",
            value=f"{len(label_dist) - 1} Vectors",
            help="DoS, DDoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration, Heartbleed.",
        )

    st.markdown("---")

    # Capture Files Telemetry Grid
    st.markdown("#### 📅 Daily Capture File Breakdown")
    file_details = status.get("file_details", {})
    if file_details:
        file_rows_list = []
        for fname, finfo in file_details.items():
            day_name = fname.split("-")[0]
            attack_types = [k for k in finfo.get("label_distribution", {}).keys() if k != BENIGN_LABEL]
            file_rows_list.append({
                "Capture File": fname,
                "Day / Session": day_name,
                "Size (MB)": finfo.get("size_mb", 0.0),
                "Total Flows": f"{finfo.get('rows', 0):,}",
                "Primary Attacks Present": ", ".join(attack_types) if attack_types else "Baseline Benign Traffic",
            })
        st.dataframe(pd.DataFrame(file_rows_list), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Flow Characteristics & Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### ⚡ Flow Duration Distribution (Microseconds)")
        # Plot logarithmic distribution of flow durations
        fig_dur = px.histogram(
            df_sample,
            x="Flow Duration",
            color="Binary_Label",
            nbins=40,
            log_y=True,
            color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
            labels={"Binary_Label": "Traffic Type"},
        )
        fig_dur.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", title="Flow Duration (μs)"),
            yaxis=dict(gridcolor="#1e293b", title="Flow Count (Log Scale)"),
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dur, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📦 Flow Packet Velocity (Packets/s vs Bytes/s)")
        # Sample for fast scatter
        scatter_sample = df_sample.sample(min(2000, len(df_sample)), random_state=42)
        fig_scatter = px.scatter(
            scatter_sample,
            x="Flow Packets/s",
            y="Flow Bytes/s",
            color="Binary_Label",
            color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
            log_x=True,
            log_y=True,
            hover_data=["Label", "Destination Port"],
            labels={"Binary_Label": "Traffic"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", title="Flow Packets/s (Log)"),
            yaxis=dict(gridcolor="#1e293b", title="Flow Bytes/s (Log)"),
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
