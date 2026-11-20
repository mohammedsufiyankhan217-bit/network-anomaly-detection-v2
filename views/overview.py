"""
Executive Overview and Traffic Statistics View for Network Anomaly Detection V2.
Displays real dataset KPIs, protocol & flow distributions, daily telemetry,
top destination ports, and network traffic volume comparisons.
"""
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    BENIGN_LABEL,
    ATTACK_CATEGORIES,
    CATEGORY_COLORS,
    resolve_port_service,
    resolve_protocol_group,
)
from src.ui_components import render_chart_header, apply_cyber_theme, CYBER_PALETTE, TRAFFIC_COLOR_MAP

DATASET_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def render_overview_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🌐 Executive Overview & Traffic Telemetry")
    st.caption(
        "Network telemetry across all 8 verified CIC-IDS2017 daily sessions. "
        "Global baseline ground-truth KPIs paired with active stratified sample distributions."
    )

    # Status banner
    if status["is_loaded"]:
        st.success(
            f"✅ **OFFICIAL CIC-IDS2017 DATASET ACTIVE**: All {status['found_count']} daily captures verified "
            f"({status['total_rows']:,} total flows across {status['total_size_mb']:.1f} MB on disk)."
        )
    else:
        st.warning(f"⚠️ Partial dataset detected: {status['found_count']}/{status['total_expected']} files.")

    # Top KPI Cards (Full Dataset Ground Truth)
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
            help="Full count of bidirectional network flows across all 8 official CIC-IDS2017 capture files.",
        )
    with col2:
        st.metric(
            label="Benign Traffic Volume",
            value=f"{benign_count:,}",
            delta=f"{(benign_count/total_raw)*100:.1f}% normal",
            delta_color="normal",
            help="Authorized production background traffic across the enterprise testbed.",
        )
    with col3:
        st.metric(
            label="Malicious Attack Flows",
            value=f"{attack_count:,}",
            delta=f"{attack_pct:.2f}% anomaly rate",
            delta_color="inverse",
            help="Confirmed malicious network attack flows across all capture sessions.",
        )
    with col4:
        st.metric(
            label="Unique Attack Vectors",
            value=f"{max(1, len(label_dist) - 1)} Threat Vectors",
            help="DoS, DDoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration, Heartbleed.",
        )

    st.markdown("---")

    # ==========================================
    # ROW 1: High-Level Distribution & Attack Breakdown
    # ==========================================
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        bm_chart_type = render_chart_header(
            title="🛡️ Benign vs Malicious Traffic Split",
            chart_types=["🍩 Donut Chart", "🥧 Pie Chart", "📊 Bar Chart", "📶 Horizontal Bar"],
            key="ov_bm_chart_type",
            default="🍩 Donut Chart",
            help_text="Full-dataset proportion of legitimate baseline traffic vs confirmed cyber attack flows (2,830,743 flows).",
        )

        df_bm = pd.DataFrame([
            {"Traffic": "Benign Traffic", "Count": benign_count, "Share": (benign_count / total_raw) * 100},
            {"Traffic": "Malicious Attacks", "Count": attack_count, "Share": (attack_count / total_raw) * 100},
        ])

        if bm_chart_type in ["🍩 Donut Chart", "🥧 Pie Chart"]:
            hole = 0.55 if "Donut" in bm_chart_type else 0.0
            fig_bm = px.pie(
                df_bm,
                names="Traffic",
                values="Count",
                hole=hole,
                color="Traffic",
                color_discrete_map={"Benign Traffic": "#10b981", "Malicious Attacks": "#ef4444"},
            )
            fig_bm.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Flows: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            fig_bm = apply_cyber_theme(fig_bm, height=360, legend_orientation="v")
        elif bm_chart_type == "📊 Bar Chart":
            fig_bm = px.bar(
                df_bm,
                x="Traffic",
                y="Count",
                color="Traffic",
                color_discrete_map={"Benign Traffic": "#10b981", "Malicious Attacks": "#ef4444"},
                text="Count",
                labels={"Count": "Flow Count", "Traffic": "Traffic Class"},
            )
            fig_bm.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bm.update_layout(yaxis=dict(range=[0, max(df_bm["Count"]) * 1.18]))
            fig_bm = apply_cyber_theme(fig_bm, height=360, show_legend=False)
        else:  # 📶 Horizontal Bar
            fig_bm = px.bar(
                df_bm,
                x="Count",
                y="Traffic",
                orientation="h",
                color="Traffic",
                color_discrete_map={"Benign Traffic": "#10b981", "Malicious Attacks": "#ef4444"},
                text="Count",
                labels={"Count": "Flow Count", "Traffic": ""},
            )
            fig_bm.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bm.update_layout(xaxis=dict(range=[0, max(df_bm["Count"]) * 1.25]))
            fig_bm = apply_cyber_theme(fig_bm, height=360, show_legend=False)

        st.plotly_chart(fig_bm, use_container_width=True)

    with row1_c2:
        cat_chart_type = render_chart_header(
            title="🧬 Attack Categories by Volume",
            chart_types=["📶 Ranked Horizontal Bar", "📊 Vertical Bar", "🍩 Donut Share (Attacks Only)", "🥧 Pie Chart"],
            key="ov_cat_chart_type",
            default="📶 Ranked Horizontal Bar",
            help_text="Volume ranking of the 7 malicious threat families across all 557,646 attack flows (full dataset).",
        )

        # Aggregate threat categories strictly across attacks from label_dist to avoid redundant benign slice
        category_attack_sums = {}
        for attack, count in label_dist.items():
            if attack != BENIGN_LABEL:
                cat = ATTACK_CATEGORIES.get(attack, "Other Attacks")
                category_attack_sums[cat] = category_attack_sums.get(cat, 0) + count

        df_cat_attacks = pd.DataFrame([
            {"Threat Category": k, "Flows": v, "Share": (v / attack_count) * 100 if attack_count > 0 else 0}
            for k, v in category_attack_sums.items()
        ]).sort_values("Flows", ascending=True)

        if cat_chart_type == "📶 Ranked Horizontal Bar":
            fig_cat = px.bar(
                df_cat_attacks,
                x="Flows",
                y="Threat Category",
                orientation="h",
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
                log_x=True,
                text="Flows",
                labels={"Flows": "Occurrences (Log Scale)", "Threat Category": ""},
            )
            fig_cat.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            max_log_val = np.log10(max(df_cat_attacks["Flows"])) if len(df_cat_attacks) > 0 else 5
            fig_cat.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(title="Flows (Log Scale)", range=[0, max_log_val + 0.45]),
                yaxis=dict(title=""),
                margin=dict(l=140, r=60, t=20, b=40),
            )
            fig_cat = apply_cyber_theme(fig_cat, height=360, show_legend=False)

        elif cat_chart_type == "📊 Vertical Bar":
            df_desc = df_cat_attacks.sort_values("Flows", ascending=False)
            fig_cat = px.bar(
                df_desc,
                x="Threat Category",
                y="Flows",
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
                log_y=True,
                text="Flows",
            )
            fig_cat.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_cat.update_layout(xaxis=dict(tickangle=-25), yaxis=dict(title="Flows (Log Scale)"))
            fig_cat = apply_cyber_theme(fig_cat, height=360, show_legend=False)

        else:  # 🍩 Donut Share (Attacks Only) or 🥧 Pie Chart
            hole = 0.55 if "Donut" in cat_chart_type else 0.0
            fig_cat = px.pie(
                df_cat_attacks,
                names="Threat Category",
                values="Flows",
                hole=hole,
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig_cat.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Flows: %{value:,}<br>Share of Attacks: %{percent}<extra></extra>",
            )
            fig_cat = apply_cyber_theme(fig_cat, height=360, legend_orientation="v")

        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 2: Temporal Days & Inferred Service Breakdown
    # ==========================================
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        day_chart_type = render_chart_header(
            title="📅 Traffic Distribution Across Dataset Days",
            chart_types=["📊 Stacked Bar", "📊 Grouped Bar", "📈 Trend Line", "🌊 Area Chart", "🍩 Donut Share"],
            key="ov_day_chart_type",
            default="📊 Stacked Bar",
            help_text="Volume and threat breakdown strictly ordered across Monday through Friday official captures (2,830,743 flows).",
        )

        day_data_map = {d: {"Total": 0, "Benign": 0, "Attack": 0} for d in DATASET_DAYS}
        file_details = status.get("file_details", {})

        for fname, finfo in file_details.items():
            matched_day = None
            for d in DATASET_DAYS:
                if d.lower() in fname.lower():
                    matched_day = d
                    break
            if not matched_day:
                matched_day = "Friday"

            rows = finfo.get("rows", 0)
            ld = finfo.get("label_distribution", {})
            b = ld.get(BENIGN_LABEL, rows if matched_day == "Monday" else int(rows * 0.75))
            a = rows - b
            day_data_map[matched_day]["Total"] += rows
            day_data_map[matched_day]["Benign"] += b
            day_data_map[matched_day]["Attack"] += a

        df_days = []
        for d in DATASET_DAYS:
            df_days.append({"Day": d, "Traffic Type": "Benign", "Flows": day_data_map[d]["Benign"]})
            df_days.append({"Day": d, "Traffic Type": "Malicious", "Flows": day_data_map[d]["Attack"]})
        df_days_df = pd.DataFrame(df_days)

        if day_chart_type in ["📊 Stacked Bar", "📊 Grouped Bar"]:
            barmode = "stack" if "Stacked" in day_chart_type else "group"
            fig_day = px.bar(
                df_days_df,
                x="Day",
                y="Flows",
                color="Traffic Type",
                barmode=barmode,
                color_discrete_map={"Benign": "#10b981", "Malicious": "#ef4444"},
                labels={"Flows": "Total Daily Flows", "Day": "Capture Day"},
            )
            fig_day.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:,} flows<extra></extra>")
            fig_day.update_xaxes(categoryorder="array", categoryarray=DATASET_DAYS)
            fig_day = apply_cyber_theme(fig_day, height=360)

        elif day_chart_type == "📈 Trend Line":
            fig_day = px.line(
                df_days_df,
                x="Day",
                y="Flows",
                color="Traffic Type",
                markers=True,
                color_discrete_map={"Benign": "#10b981", "Malicious": "#ef4444"},
            )
            fig_day.update_traces(line=dict(width=3), marker=dict(size=8))
            fig_day.update_xaxes(categoryorder="array", categoryarray=DATASET_DAYS)
            fig_day = apply_cyber_theme(fig_day, height=360)

        elif day_chart_type == "🌊 Area Chart":
            fig_day = px.area(
                df_days_df,
                x="Day",
                y="Flows",
                color="Traffic Type",
                color_discrete_map={"Benign": "#10b981", "Malicious": "#ef4444"},
            )
            fig_day.update_xaxes(categoryorder="array", categoryarray=DATASET_DAYS)
            fig_day = apply_cyber_theme(fig_day, height=360)

        else:  # 🍩 Donut Share
            day_totals = pd.DataFrame([{"Day": d, "Total Flows": day_data_map[d]["Total"]} for d in DATASET_DAYS])
            fig_day = px.pie(
                day_totals,
                names="Day",
                values="Total Flows",
                hole=0.55,
                color="Day",
                color_discrete_sequence=CYBER_PALETTE,
            )
            fig_day.update_traces(textposition="inside", textinfo="percent+label")
            fig_day = apply_cyber_theme(fig_day, height=360, legend_orientation="v")

        st.plotly_chart(fig_day, use_container_width=True)

    with row2_c2:
        proto_chart_type = render_chart_header(
            title="📡 Inferred Service & Port Distribution",
            chart_types=["🍩 Donut Chart", "🥧 Pie Chart", "📶 Horizontal Bar", "📊 Vertical Bar"],
            key="ov_proto_chart_type",
            default="🍩 Donut Chart",
            help_text="Inferred application service category based on destination port conventions (e.g. 80 HTTP, 443 HTTPS, 53 DNS) from active sample.",
        )

        proto_counts = df_sample["Destination Port"].apply(resolve_protocol_group).value_counts().reset_index()
        proto_counts.columns = ["Inferred Service Category", "Flows"]

        if proto_chart_type in ["🍩 Donut Chart", "🥧 Pie Chart"]:
            hole = 0.55 if "Donut" in proto_chart_type else 0.0
            fig_proto = px.pie(
                proto_counts,
                names="Inferred Service Category",
                values="Flows",
                hole=hole,
                color_discrete_sequence=CYBER_PALETTE,
            )
            fig_proto.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Sample Flows: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            fig_proto = apply_cyber_theme(fig_proto, height=360, legend_orientation="v")
        elif proto_chart_type == "📶 Horizontal Bar":
            proto_sorted = proto_counts.sort_values("Flows", ascending=True)
            fig_proto = px.bar(
                proto_sorted,
                x="Flows",
                y="Inferred Service Category",
                orientation="h",
                color="Flows",
                color_continuous_scale="Blues",
                text="Flows",
                log_x=True,
            )
            fig_proto.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            max_log_p = np.log10(max(proto_sorted["Flows"])) if len(proto_sorted) > 0 else 4
            fig_proto.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(title="Flows (Log Scale)", range=[0, max_log_p + 0.45]),
                yaxis=dict(title=""),
                margin=dict(l=150, r=60, t=20, b=40),
            )
            fig_proto = apply_cyber_theme(fig_proto, height=360, show_legend=False)
        else:  # 📊 Vertical Bar
            fig_proto = px.bar(
                proto_counts,
                x="Inferred Service Category",
                y="Flows",
                color="Inferred Service Category",
                color_discrete_sequence=CYBER_PALETTE,
                log_y=True,
                text="Flows",
            )
            fig_proto.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_proto.update_layout(xaxis=dict(tickangle=-30), yaxis=dict(title="Flows (Log Scale)"))
            fig_proto = apply_cyber_theme(fig_proto, height=360, show_legend=False)

        st.plotly_chart(fig_proto, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 3: Destination Ports & Volume Analysis
    # ==========================================
    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        port_chart_type = render_chart_header(
            title="🎯 Top Destination Ports by Volume",
            chart_types=["📶 Horizontal Bar", "📊 Vertical Bar", "🍩 Donut Chart", "🥧 Pie Chart"],
            key="ov_port_chart_type",
            default="📶 Horizontal Bar",
            help_text="Most frequently targeted network service ports in active sample with inferred protocol annotations.",
        )

        top_ports_series = df_sample["Destination Port"].value_counts().head(12).reset_index()
        top_ports_series.columns = ["Port", "Flows"]
        top_ports_series["Service"] = top_ports_series["Port"].apply(resolve_port_service)

        if port_chart_type == "📶 Horizontal Bar":
            top_ports_sorted = top_ports_series.sort_values("Flows", ascending=True)
            fig_port = px.bar(
                top_ports_sorted,
                x="Flows",
                y="Service",
                orientation="h",
                color="Flows",
                color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#67e8f9"]],
                text="Flows",
                log_x=True,
            )
            fig_port.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            max_log_port = np.log10(max(top_ports_sorted["Flows"])) if len(top_ports_sorted) > 0 else 5
            fig_port.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(title=""),
                xaxis=dict(title="Sample Flows (Log Scale)", range=[0, max_log_port + 0.45]),
                margin=dict(l=140, r=60, t=20, b=40),
            )
            fig_port = apply_cyber_theme(fig_port, height=380, show_legend=False)
        elif port_chart_type == "📊 Vertical Bar":
            fig_port = px.bar(
                top_ports_series,
                x="Service",
                y="Flows",
                color="Flows",
                color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#67e8f9"]],
                log_y=True,
                text="Flows",
            )
            fig_port.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_port.update_layout(coloraxis_showscale=False, xaxis=dict(tickangle=-35), yaxis=dict(title="Flows (Log Scale)"))
            fig_port = apply_cyber_theme(fig_port, height=380, show_legend=False)
        else:  # 🍩 Donut / 🥧 Pie
            hole = 0.55 if "Donut" in port_chart_type else 0.0
            fig_port = px.pie(
                top_ports_series,
                names="Service",
                values="Flows",
                hole=hole,
                color_discrete_sequence=CYBER_PALETTE,
            )
            fig_port.update_traces(textposition="inside", textinfo="percent")
            fig_port = apply_cyber_theme(fig_port, height=380, legend_orientation="v")

        st.plotly_chart(fig_port, use_container_width=True)

    with row3_c2:
        vol_chart_type = render_chart_header(
            title="📊 Network Payload Volume Comparison",
            chart_types=["📊 Grouped Bar", "📊 Stacked Bar", "🍩 Donut Share", "📶 Horizontal Bar"],
            key="ov_vol_chart_type",
            default="📊 Grouped Bar",
            help_text="Forward vs Backward transmitted megabytes by threat family in active sample.",
        )

        vol_agg = (
            df_sample.groupby("Threat_Category")[
                ["Total Length of Fwd Packets", "Total Length of Bwd Packets", "Total Fwd Packets", "Total Backward Packets"]
            ]
            .sum()
            .reset_index()
        )
        vol_agg["Fwd MB"] = (vol_agg["Total Length of Fwd Packets"] / (1024 * 1024)).round(2)
        vol_agg["Bwd MB"] = (vol_agg["Total Length of Bwd Packets"] / (1024 * 1024)).round(2)
        vol_agg["Total MB"] = vol_agg["Fwd MB"] + vol_agg["Bwd MB"]

        if vol_chart_type in ["📊 Grouped Bar", "📊 Stacked Bar"]:
            barmode = "group" if "Grouped" in vol_chart_type else "stack"
            vol_melt = pd.melt(
                vol_agg,
                id_vars=["Threat_Category"],
                value_vars=["Fwd MB", "Bwd MB"],
                var_name="Direction",
                value_name="Megabytes",
            )
            fig_vol = px.bar(
                vol_melt,
                x="Threat_Category",
                y="Megabytes",
                color="Direction",
                barmode=barmode,
                log_y=True,
                color_discrete_map={"Fwd MB": "#38bdf8", "Bwd MB": "#f59e0b"},
                labels={"Threat_Category": "", "Megabytes": "Volume (MB, Log Scale)"},
            )
            fig_vol.update_layout(xaxis=dict(tickangle=-25))
            fig_vol = apply_cyber_theme(fig_vol, height=380)
        elif vol_chart_type == "🍩 Donut Share":
            fig_vol = px.pie(
                vol_agg,
                names="Threat_Category",
                values="Total MB",
                hole=0.55,
                color="Threat_Category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig_vol.update_traces(textposition="inside", textinfo="percent")
            fig_vol = apply_cyber_theme(fig_vol, height=380, legend_orientation="v")
        else:  # 📶 Horizontal Bar
            vol_sorted = vol_agg.sort_values("Total MB", ascending=True)
            fig_vol = px.bar(
                vol_sorted,
                x="Total MB",
                y="Threat_Category",
                orientation="h",
                color="Threat_Category",
                color_discrete_map=CATEGORY_COLORS,
                log_x=True,
                text="Total MB",
                labels={"Total MB": "Total Volume (MB, Log Scale)", "Threat_Category": ""},
            )
            fig_vol.update_traces(texttemplate="%{text:.1f} MB", textposition="outside", cliponaxis=False)
            max_log_v = np.log10(max(vol_sorted["Total MB"])) if len(vol_sorted) > 0 and max(vol_sorted["Total MB"]) > 0 else 3
            fig_vol.update_layout(
                xaxis=dict(title="Total Volume (MB, Log Scale)", range=[0, max_log_v + 0.45]),
                yaxis=dict(title=""),
                margin=dict(l=140, r=60, t=20, b=40),
            )
            fig_vol = apply_cyber_theme(fig_vol, height=380, show_legend=False)

        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 4: Network Flow Characteristics & Dynamics
    # ==========================================
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        dur_chart_type = render_chart_header(
            title="⚡ Flow Duration Characteristics",
            chart_types=["📊 Histogram", "📦 Box Plot", "📈 Line Distribution", "🍩 Duration Tiers"],
            key="ov_dur_chart_type",
            default="📊 Histogram",
            help_text="Microsecond session duration comparison between benign and malicious flows in active sample.",
        )

        if dur_chart_type == "📊 Histogram":
            fig_dur = px.histogram(
                df_sample,
                x="Flow Duration",
                color="Binary_Label",
                nbins=40,
                log_y=True,
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
                labels={"Binary_Label": "Traffic Type", "Flow Duration": "Flow Duration (μs)"},
            )
            fig_dur.update_layout(
                xaxis=dict(title="Flow Duration (μs)"),
                yaxis=dict(title="Flow Count (Log Scale)"),
            )
            fig_dur = apply_cyber_theme(fig_dur, height=380)

        elif dur_chart_type == "📦 Box Plot":
            fig_dur = px.box(
                df_sample,
                x="Binary_Label",
                y="Flow Duration",
                color="Binary_Label",
                log_y=True,
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
                labels={"Binary_Label": "Traffic Type", "Flow Duration": "Flow Duration (μs, Log)"},
            )
            fig_dur = apply_cyber_theme(fig_dur, height=380, show_legend=False)

        elif dur_chart_type == "📈 Line Distribution":
            benign_dur = df_sample[df_sample["Binary_Label"] == "BENIGN"]["Flow Duration"]
            attack_dur = df_sample[df_sample["Binary_Label"] == "ATTACK"]["Flow Duration"]
            min_val = max(0, df_sample["Flow Duration"].min())
            max_val = df_sample["Flow Duration"].max()
            bins = np.linspace(min_val, max_val, 41)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            b_counts, _ = np.histogram(benign_dur, bins=bins)
            a_counts, _ = np.histogram(attack_dur, bins=bins)

            df_dist = pd.DataFrame({
                "Flow Duration (μs)": np.concatenate([bin_centers, bin_centers]),
                "Flow Count": np.concatenate([b_counts, a_counts]),
                "Traffic Type": ["BENIGN"] * len(bin_centers) + ["ATTACK"] * len(bin_centers),
            })
            fig_dur = px.line(
                df_dist,
                x="Flow Duration (μs)",
                y="Flow Count",
                color="Traffic Type",
                log_y=True,
                markers=True,
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
            )
            fig_dur = apply_cyber_theme(fig_dur, height=380)

        else:  # 🍩 Duration Tiers
            def get_dur_tier(d):
                if d < 1000:
                    return "< 1 ms (Micro-flows)"
                elif d < 100000:
                    return "1 ms - 100 ms (Fast)"
                elif d < 1000000:
                    return "100 ms - 1 s (Interactive)"
                elif d < 10000000:
                    return "1 s - 10 s (Extended)"
                return "> 10 s (Persistent Session)"

            tier_df = df_sample["Flow Duration"].apply(get_dur_tier).value_counts().reset_index()
            tier_df.columns = ["Duration Tier", "Count"]
            fig_dur = px.pie(
                tier_df,
                names="Duration Tier",
                values="Count",
                hole=0.55,
                color_discrete_sequence=["#38bdf8", "#0284c7", "#10b981", "#f59e0b", "#ef4444"],
            )
            fig_dur.update_traces(textposition="inside", textinfo="percent+label")
            fig_dur = apply_cyber_theme(fig_dur, height=380, legend_orientation="v")

        st.plotly_chart(fig_dur, use_container_width=True)

    with col_chart2:
        vel_chart_type = render_chart_header(
            title="📦 Flow Velocity (Packets/s vs Bytes/s)",
            chart_types=["📦 Scatter Plot", "📊 Bar (Mean Velocity)", "📈 Percentile Line", "🌊 Velocity Area"],
            key="ov_vel_chart_type",
            default="📦 Scatter Plot",
            help_text="Transmission velocity highlighting volumetric attacks and stealth probes in active sample.",
        )

        if vel_chart_type == "📦 Scatter Plot":
            scatter_sample = df_sample.sample(min(2000, len(df_sample)), random_state=42)
            fig_vel = px.scatter(
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
            fig_vel.update_layout(
                xaxis=dict(title="Flow Packets/s (Log Scale)"),
                yaxis=dict(title="Flow Bytes/s (Log Scale)"),
            )
            fig_vel = apply_cyber_theme(fig_vel, height=380)

        elif vel_chart_type == "📊 Bar (Mean Velocity)":
            vel_summary = df_sample.groupby("Binary_Label")[["Flow Packets/s", "Flow Bytes/s"]].mean().reset_index()
            vel_melted = pd.melt(
                vel_summary,
                id_vars=["Binary_Label"],
                value_vars=["Flow Packets/s", "Flow Bytes/s"],
                var_name="Metric",
                value_name="Average Velocity",
            )
            fig_vel = px.bar(
                vel_melted,
                x="Binary_Label",
                y="Average Velocity",
                color="Metric",
                barmode="group",
                log_y=True,
                color_discrete_map={"Flow Packets/s": "#38bdf8", "Flow Bytes/s": "#f59e0b"},
                labels={"Binary_Label": "Traffic Type", "Average Velocity": "Mean Rate (Log Scale)"},
            )
            fig_vel = apply_cyber_theme(fig_vel, height=380)

        elif vel_chart_type == "📈 Percentile Line":
            quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            q_rows = []
            for q in quantiles:
                for lbl in ["BENIGN", "ATTACK"]:
                    sub = df_sample[df_sample["Binary_Label"] == lbl]
                    q_rows.append({
                        "Percentile": f"{int(q*100)}th",
                        "Flow Packets/s": max(1e-1, sub["Flow Packets/s"].quantile(q)),
                        "Traffic": lbl,
                    })
            q_df = pd.DataFrame(q_rows)
            fig_vel = px.line(
                q_df,
                x="Percentile",
                y="Flow Packets/s",
                color="Traffic",
                markers=True,
                log_y=True,
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
                labels={"Flow Packets/s": "Flow Packets/s (Log Scale)"},
            )
            fig_vel = apply_cyber_theme(fig_vel, height=380)

        else:  # 🌊 Velocity Area
            quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            q_rows = []
            for q in quantiles:
                for lbl in ["BENIGN", "ATTACK"]:
                    sub = df_sample[df_sample["Binary_Label"] == lbl]
                    q_rows.append({
                        "Percentile": f"{int(q*100)}th",
                        "Flow Bytes/s": max(1e-1, sub["Flow Bytes/s"].quantile(q)),
                        "Traffic": lbl,
                    })
            q_df = pd.DataFrame(q_rows)
            fig_vel = px.area(
                q_df,
                x="Percentile",
                y="Flow Bytes/s",
                color="Traffic",
                log_y=True,
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
                labels={"Flow Bytes/s": "Flow Bytes/s (Log Scale)"},
            )
            fig_vel = apply_cyber_theme(fig_vel, height=380)

        st.plotly_chart(fig_vel, use_container_width=True)

    # Capture Files Telemetry Grid in Expander
    with st.expander("📁 Detailed Daily Capture Telemetry Grid (All 8 Verified Files)", expanded=False):
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
