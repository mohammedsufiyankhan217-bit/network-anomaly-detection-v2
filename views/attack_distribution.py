"""
Attack Distribution & Threat Taxonomy View for Network Anomaly Detection V2.
Detailed analytics on the 15 real attack classes captured in CIC-IDS2017,
cross-category protocol distributions, daily threat allocations, and multi-metric comparisons.
"""
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    BENIGN_LABEL,
    ATTACK_CATEGORIES,
    CATEGORY_COLORS,
    resolve_protocol_group,
)
from src.ui_components import render_chart_header, apply_cyber_theme, CYBER_PALETTE, TRAFFIC_COLOR_MAP

DATASET_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def render_attack_distribution_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🎯 Attack Distribution & Threat Taxonomy")
    st.caption("Empirical analytics across all 15 network traffic classifications in CIC-IDS2017.")

    label_dist = status.get("label_distribution", {})
    total_records = status.get("total_rows", len(df_sample))

    # Top level metrics (Full Dataset Ground Truth)
    total_attacks = sum(v for k, v in label_dist.items() if k != BENIGN_LABEL)
    top_attack = max(((k, v) for k, v in label_dist.items() if k != BENIGN_LABEL), default=("None", 0))
    rare_attacks = [k for k, v in label_dist.items() if k != BENIGN_LABEL and v < 100]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Total Malicious Flows",
            f"{total_attacks:,}",
            help="Total confirmed attack flows across all 8 capture files (Full Dataset).",
        )
    with c2:
        st.metric(
            "Primary Attack Vector",
            top_attack[0],
            f"{top_attack[1]:,} flows",
            help="Most pervasive attack vector across the entire dataset.",
        )
    with c3:
        st.metric(
            "Attack Classification Ratio",
            f"{(total_attacks / total_records) * 100:.2f}%",
            "Severe class imbalance",
            help="Proportion of malicious flows out of 2,830,743 total dataset flows.",
        )
    with c4:
        st.metric(
            "Rare Threat Vectors (<100)",
            f"{len(rare_attacks)} Classes",
            f"{', '.join(rare_attacks[:2])}",
            help="Heartbleed (11), SQLi (21), Infiltration (36) — 100% preserved in stratified sample.",
        )

    st.markdown("---")

    # Filter out benign to analyze attack vectors specifically
    attack_only_dist = {k: v for k, v in label_dist.items() if k != BENIGN_LABEL}
    df_attack_bars = pd.DataFrame([
        {
            "Attack Type": k,
            "Threat Category": ATTACK_CATEGORIES.get(k, "Other Attacks"),
            "Flows": v,
            "Percentage": (v / total_attacks) * 100 if total_attacks > 0 else 0,
        }
        for k, v in attack_only_dist.items()
    ]).sort_values("Flows", ascending=True)

    # ==========================================
    # ROW 1: Attack Category Frequency & Percentage
    # ==========================================
    row1_c1, row1_c2 = st.columns([1.2, 1])

    with row1_c1:
        atk_chart_type = render_chart_header(
            title="🛡️ Attack Vector Frequency (14 Classes)",
            chart_types=["📶 Horizontal Bar", "📊 Vertical Bar", "📈 Frequency Line", "🌊 Frequency Area"],
            key="atk_breakdown_chart_type",
            default="📶 Horizontal Bar",
            help_text="Frequency of all 14 individual attack vectors on a logarithmic scale (Full Dataset: 557,646 attacks).",
        )

        if atk_chart_type == "📶 Horizontal Bar":
            fig_bar = px.bar(
                df_attack_bars,
                x="Flows",
                y="Attack Type",
                orientation="h",
                text="Flows",
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
                labels={"Flows": "Occurrences (Log Scale)", "Attack Type": ""},
            )
            max_log_f = np.log10(max(df_attack_bars["Flows"])) if len(df_attack_bars) > 0 else 5
            fig_bar.update_layout(
                xaxis=dict(type="log", title="Flow Count (Log Scale)", range=[0, max_log_f + 0.45]),
                yaxis=dict(title=""),
                margin=dict(l=160, r=60, t=20, b=40),
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bar = apply_cyber_theme(fig_bar, height=480)
            st.plotly_chart(fig_bar, use_container_width=True)

        elif atk_chart_type == "📊 Vertical Bar":
            df_atk_desc = df_attack_bars.sort_values("Flows", ascending=False)
            fig_bar = px.bar(
                df_atk_desc,
                x="Attack Type",
                y="Flows",
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
                log_y=True,
                text="Flows",
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bar.update_layout(xaxis=dict(tickangle=-40))
            fig_bar = apply_cyber_theme(fig_bar, height=480, margin=dict(l=40, r=20, t=20, b=80))
            st.plotly_chart(fig_bar, use_container_width=True)

        elif atk_chart_type == "📈 Frequency Line":
            df_atk_desc = df_attack_bars.sort_values("Flows", ascending=False)
            fig_line = px.line(
                df_atk_desc,
                x="Attack Type",
                y="Flows",
                markers=True,
                log_y=True,
                labels={"Flows": "Flows (Log Scale)"},
            )
            fig_line.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#ef4444"))
            fig_line.update_xaxes(tickangle=-45, tickfont=dict(color="#94a3b8", size=10))
            fig_line = apply_cyber_theme(fig_line, height=480, margin=dict(l=40, r=20, t=20, b=80), show_legend=False)
            st.plotly_chart(fig_line, use_container_width=True)

        else:  # 🌊 Frequency Area
            df_atk_desc = df_attack_bars.sort_values("Flows", ascending=False)
            fig_area = px.area(
                df_atk_desc,
                x="Attack Type",
                y="Flows",
                log_y=True,
                labels={"Flows": "Flows (Log Scale)"},
            )
            fig_area.update_traces(line=dict(color="#38bdf8"), fillcolor="rgba(56, 189, 248, 0.35)")
            fig_area.update_xaxes(tickangle=-45, tickfont=dict(color="#94a3b8", size=10))
            fig_area = apply_cyber_theme(fig_area, height=480, margin=dict(l=40, r=20, t=20, b=80), show_legend=False)
            st.plotly_chart(fig_area, use_container_width=True)

    with row1_c2:
        tax_chart_type = render_chart_header(
            title="🧬 Threat Category Percentage Share",
            chart_types=["🍩 Donut Chart", "🥧 Pie Chart", "📶 Horizontal Share Bar", "📊 Vertical Bar"],
            key="atk_tax_chart_type",
            default="🍩 Donut Chart",
            help_text="Proportional share of total attack volume across the 7 attack families (Full Dataset).",
        )

        category_sums = {}
        for attack, count in label_dist.items():
            if attack != BENIGN_LABEL:
                cat = ATTACK_CATEGORIES.get(attack, "Other Attacks")
                category_sums[cat] = category_sums.get(cat, 0) + count

        df_cats = pd.DataFrame([
            {"Category": k, "Count": v, "Share": (v / total_attacks) * 100 if total_attacks > 0 else 0}
            for k, v in category_sums.items()
        ]).sort_values("Count", ascending=False)

        if tax_chart_type in ["🍩 Donut Chart", "🥧 Pie Chart"]:
            hole = 0.55 if "Donut" in tax_chart_type else 0.0
            fig_donut = px.pie(
                df_cats,
                names="Category",
                values="Count",
                hole=hole,
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Intrusion Flows: %{value:,}<br>Share of Attacks: %{percent}<extra></extra>",
            )
            fig_donut = apply_cyber_theme(fig_donut, height=480, margin=dict(l=10, r=10, t=20, b=20), legend_orientation="v")
            st.plotly_chart(fig_donut, use_container_width=True)

        elif tax_chart_type == "📶 Horizontal Share Bar":
            df_cats_sorted = df_cats.sort_values("Share", ascending=True)
            fig_bar = px.bar(
                df_cats_sorted,
                x="Share",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                text="Share",
            )
            fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_bar.update_layout(
                xaxis=dict(title="Share of Attack Flows (%)", range=[0, max(df_cats["Share"]) * 1.25]),
                yaxis=dict(title=""),
                margin=dict(l=140, r=60, t=20, b=40),
            )
            fig_bar = apply_cyber_theme(fig_bar, height=480, show_legend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        else:  # 📊 Vertical Bar
            fig_bar = px.bar(
                df_cats,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                log_y=True,
                text="Count",
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_bar.update_layout(xaxis=dict(tickangle=-30), yaxis=dict(title="Count (Log Scale)"))
            fig_bar = apply_cyber_theme(fig_bar, height=480, margin=dict(l=40, r=20, t=20, b=60), show_legend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 2: Benign vs Attack Comparison & Daily Attacks
    # ==========================================
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        comp_chart_type = render_chart_header(
            title="⚖️ Benign vs Attack Metric Comparison",
            chart_types=["📊 4-Metric Subplots (Distinct Units)", "📶 Normalized Ratio vs Benign (100%)", "🍩 Flow Volume Share"],
            key="atk_comp_chart_type",
            default="📊 4-Metric Subplots (Distinct Units)",
            help_text="Direct structural comparison with distinct physical measurement scales for duration, payload size, and packet counts.",
        )

        b_sub = df_sample[df_sample["Binary_Label"] == "BENIGN"]
        a_sub = df_sample[df_sample["Binary_Label"] == "ATTACK"]

        b_dur_ms = float(b_sub["Flow Duration"].mean() / 1000)
        a_dur_ms = float(a_sub["Flow Duration"].mean() / 1000)
        b_len = float(b_sub["Packet Length Mean"].mean())
        a_len = float(a_sub["Packet Length Mean"].mean())
        b_fwd = float(b_sub["Total Fwd Packets"].mean())
        a_fwd = float(a_sub["Total Fwd Packets"].mean())
        b_bwd = float(b_sub["Total Backward Packets"].mean())
        a_bwd = float(a_sub["Total Backward Packets"].mean())

        if comp_chart_type == "📊 4-Metric Subplots (Distinct Units)":
            fig_comp = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Duration (ms)",
                    "Packet Length (Bytes)",
                    "Fwd Packets (pkts)",
                    "Bwd Packets (pkts)",
                ),
                vertical_spacing=0.28,
                horizontal_spacing=0.20,
            )

            metrics_data = [
                ("Duration (ms)", b_dur_ms, a_dur_ms, 1, 1),
                ("Packet Length (Bytes)", b_len, a_len, 1, 2),
                ("Fwd Packets", b_fwd, a_fwd, 2, 1),
                ("Bwd Packets", b_bwd, a_bwd, 2, 2),
            ]

            for name, b_val, a_val, r, c in metrics_data:
                show_leg = (r == 1 and c == 1)
                fig_comp.add_trace(
                    go.Bar(
                        x=["Benign", "Attack"],
                        y=[b_val, a_val],
                        marker_color=["#10b981", "#ef4444"],
                        text=[f"{b_val:.1f}", f"{a_val:.1f}"],
                        textposition="outside",
                        cliponaxis=False,
                        showlegend=False,
                        name=name,
                    ),
                    row=r,
                    col=c,
                )

            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", size=11),
                margin=dict(l=30, r=30, t=40, b=30),
                height=420,
            )
            fig_comp.update_yaxes(gridcolor="#1e293b", tickfont=dict(color="#94a3b8", size=9))
            fig_comp.update_xaxes(tickfont=dict(color="#f1f5f9", size=10))
            st.plotly_chart(fig_comp, use_container_width=True)

        elif comp_chart_type == "📶 Normalized Ratio vs Benign (100%)":
            # Normalized indexed ratio comparison (dimensionless relative delta)
            ratio_rows = [
                {"Metric": "Avg Flow Duration", "Index (% of Benign)": (a_dur_ms / max(b_dur_ms, 1e-4)) * 100, "Raw Attack": f"{a_dur_ms:.1f} ms", "Raw Benign": f"{b_dur_ms:.1f} ms"},
                {"Metric": "Avg Packet Length", "Index (% of Benign)": (a_len / max(b_len, 1e-4)) * 100, "Raw Attack": f"{a_len:.1f} B", "Raw Benign": f"{b_len:.1f} B"},
                {"Metric": "Avg Forward Packets", "Index (% of Benign)": (a_fwd / max(b_fwd, 1e-4)) * 100, "Raw Attack": f"{a_fwd:.1f} pkts", "Raw Benign": f"{b_fwd:.1f} pkts"},
                {"Metric": "Avg Backward Packets", "Index (% of Benign)": (a_bwd / max(b_bwd, 1e-4)) * 100, "Raw Attack": f"{a_bwd:.1f} pkts", "Raw Benign": f"{b_bwd:.1f} pkts"},
            ]
            df_ratios = pd.DataFrame(ratio_rows)
            fig_comp = px.bar(
                df_ratios,
                x="Index (% of Benign)",
                y="Metric",
                orientation="h",
                color="Index (% of Benign)",
                color_continuous_scale=[[0.0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
                text="Index (% of Benign)",
                hover_data=["Raw Attack", "Raw Benign"],
            )
            fig_comp.add_vline(x=100, line_dash="dash", line_color="#10b981", annotation_text="Benign Baseline (100%)")
            fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            max_idx = max(df_ratios["Index (% of Benign)"]) if len(df_ratios) > 0 else 100
            fig_comp.update_layout(
                xaxis=dict(title="Relative Attack Magnitude (% of Benign Baseline)", range=[0, max_idx * 1.25]),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                margin=dict(l=140, r=60, t=20, b=40),
            )
            fig_comp = apply_cyber_theme(fig_comp, height=420, show_legend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        else:  # 🍩 Flow Volume Share
            ratio_df = pd.DataFrame([
                {"Traffic": "Benign Flows", "Count": len(b_sub)},
                {"Traffic": "Attack Flows", "Count": len(a_sub)},
            ])
            fig_comp = px.pie(
                ratio_df,
                names="Traffic",
                values="Count",
                hole=0.55,
                color="Traffic",
                color_discrete_map={"Benign Flows": "#10b981", "Attack Flows": "#ef4444"},
            )
            fig_comp.update_traces(textposition="inside", textinfo="percent+label")
            fig_comp = apply_cyber_theme(fig_comp, height=420, legend_orientation="v")
            st.plotly_chart(fig_comp, use_container_width=True)

    with row2_c2:
        day_atk_chart_type = render_chart_header(
            title="📅 Attack Distribution by Dataset Day",
            chart_types=["📊 Stacked Attack Bar", "📊 Grouped Day Bar", "🗺️ Day-Threat Heatmap"],
            key="atk_day_chart_type",
            default="📊 Stacked Attack Bar",
            help_text="Attacks captured per day in strict Monday through Friday sequence (official CIC-IDS2017 timeline).",
        )

        daily_threat_rows = [
            {"Day": "Monday", "Threat Category": "Benign Traffic", "Attack Vector": "Baseline Benign", "Flows": 529918},
            {"Day": "Tuesday", "Threat Category": "Brute Force", "Attack Vector": "FTP-Patator", "Flows": 7938},
            {"Day": "Tuesday", "Threat Category": "Brute Force", "Attack Vector": "SSH-Patator", "Flows": 5897},
            {"Day": "Wednesday", "Threat Category": "DoS / DDoS", "Attack Vector": "DoS Hulk", "Flows": 231073},
            {"Day": "Wednesday", "Threat Category": "DoS / DDoS", "Attack Vector": "DoS GoldenEye", "Flows": 10293},
            {"Day": "Wednesday", "Threat Category": "DoS / DDoS", "Attack Vector": "DoS slowloris", "Flows": 5796},
            {"Day": "Wednesday", "Threat Category": "DoS / DDoS", "Attack Vector": "DoS Slowhttptest", "Flows": 5499},
            {"Day": "Wednesday", "Threat Category": "Vulnerability Exploit", "Attack Vector": "Heartbleed", "Flows": 11},
            {"Day": "Thursday", "Threat Category": "Web Attacks", "Attack Vector": "Web Attack - Brute Force", "Flows": 1507},
            {"Day": "Thursday", "Threat Category": "Web Attacks", "Attack Vector": "Web Attack - XSS", "Flows": 652},
            {"Day": "Thursday", "Threat Category": "Web Attacks", "Attack Vector": "Web Attack - Sql Injection", "Flows": 21},
            {"Day": "Thursday", "Threat Category": "Infiltration / Compromise", "Attack Vector": "Infiltration", "Flows": 36},
            {"Day": "Friday", "Threat Category": "Reconnaissance / Probe", "Attack Vector": "PortScan", "Flows": 158930},
            {"Day": "Friday", "Threat Category": "DoS / DDoS", "Attack Vector": "DDoS", "Flows": 128027},
            {"Day": "Friday", "Threat Category": "Botnet", "Attack Vector": "Bot", "Flows": 1966},
        ]
        df_daily_threats = pd.DataFrame(daily_threat_rows)
        df_daily_attacks_only = df_daily_threats[df_daily_threats["Threat Category"] != "Benign Traffic"]

        if day_atk_chart_type in ["📊 Stacked Attack Bar", "📊 Grouped Day Bar"]:
            barmode = "stack" if "Stacked" in day_atk_chart_type else "group"
            fig_day_atk = px.bar(
                df_daily_attacks_only,
                x="Day",
                y="Flows",
                color="Threat Category",
                barmode=barmode,
                color_discrete_map=CATEGORY_COLORS,
                log_y=True,
                hover_data=["Attack Vector"],
                labels={"Day": "Dataset Day", "Flows": "Malicious Flows (Log Scale)"},
            )
            fig_day_atk.update_xaxes(categoryorder="array", categoryarray=DATASET_DAYS)
            fig_day_atk = apply_cyber_theme(fig_day_atk, height=420)
        else:  # 🗺️ Day-Threat Heatmap
            ct_day = pd.crosstab(
                df_daily_attacks_only["Day"],
                df_daily_attacks_only["Threat Category"],
                values=df_daily_attacks_only["Flows"],
                aggfunc="sum",
            ).fillna(0)
            ct_day = ct_day.reindex([d for d in DATASET_DAYS if d in ct_day.index])
            fig_day_atk = px.imshow(
                ct_day,
                color_continuous_scale=[[0.0, "#0b0f19"], [0.2, "#0369a1"], [0.6, "#0284c7"], [1.0, "#38bdf8"]],
                text_auto=True,
                labels=dict(x="Threat Category", y="Dataset Day", color="Flows"),
            )
            fig_day_atk.update_layout(xaxis=dict(tickangle=-30))
            fig_day_atk = apply_cyber_theme(fig_day_atk, height=420)

        st.plotly_chart(fig_day_atk, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # ROW 3: Top Attack Categories Ranking & Inferred Protocol Distribution
    # ==========================================
    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        rank_chart_type = render_chart_header(
            title="🏆 Top Attack Categories Ranking",
            chart_types=["📶 Ranked Horizontal Bar", "📊 Pareto Cumulative Bar", "🍩 Threat Tier Donut"],
            key="atk_rank_chart_type",
            default="📶 Ranked Horizontal Bar",
            help_text="Volume ranking of cyber threats sorted strictly in descending order from highest to lowest occurrences.",
        )

        # Strictly sort descending by occurrence count
        ranked_df = df_attack_bars.sort_values("Flows", ascending=False).reset_index(drop=True)
        ranked_df["Rank"] = [f"#{i+1} {row['Attack Type']}" for i, row in ranked_df.iterrows()]
        ranked_df["Cumulative Share"] = (ranked_df["Flows"].cumsum() / total_attacks) * 100

        if rank_chart_type == "📶 Ranked Horizontal Bar":
            # For horizontal bar, categoryarray reversed ensures #1 is at the top of the vertical axis
            fig_rank = px.bar(
                ranked_df,
                x="Flows",
                y="Rank",
                orientation="h",
                color="Threat Category",
                color_discrete_map=CATEGORY_COLORS,
                log_x=True,
                text="Flows",
                labels={"Flows": "Occurrences (Log Scale)", "Rank": ""},
            )
            max_log_r = np.log10(max(ranked_df["Flows"])) if len(ranked_df) > 0 else 5
            fig_rank.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_rank.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=ranked_df["Rank"].tolist()[::-1]),
                xaxis=dict(title="Occurrences (Log Scale)", range=[0, max_log_r + 0.45]),
                margin=dict(l=170, r=60, t=20, b=40),
            )
            fig_rank = apply_cyber_theme(fig_rank, height=460, show_legend=False)

        elif rank_chart_type == "📊 Pareto Cumulative Bar":
            fig_rank = go.Figure()
            fig_rank.add_trace(
                go.Bar(
                    x=ranked_df["Attack Type"],
                    y=ranked_df["Flows"],
                    name="Flows",
                    marker_color="#0284c7",
                    yaxis="y1",
                )
            )
            fig_rank.add_trace(
                go.Scatter(
                    x=ranked_df["Attack Type"],
                    y=ranked_df["Cumulative Share"],
                    name="Cumulative %",
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=3),
                    marker=dict(size=7),
                    yaxis="y2",
                )
            )
            fig_rank.update_layout(
                yaxis=dict(title="Flow Count (Log)", type="log"),
                yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105]),
                xaxis=dict(tickangle=-40, categoryorder="array", categoryarray=ranked_df["Attack Type"].tolist()),
            )
            fig_rank = apply_cyber_theme(fig_rank, height=460)

        else:  # 🍩 Threat Tier Donut
            def get_threat_tier(cnt):
                if cnt >= 50000:
                    return "Mass Volumetric (>50k)"
                elif cnt >= 5000:
                    return "High Frequency (5k-50k)"
                elif cnt >= 500:
                    return "Targeted Assault (500-5k)"
                return "Stealth / Precision (<500)"

            ranked_df["Volume Tier"] = ranked_df["Flows"].apply(get_threat_tier)
            tier_summary = ranked_df.groupby("Volume Tier")["Flows"].sum().reset_index()
            fig_rank = px.pie(
                tier_summary,
                names="Volume Tier",
                values="Flows",
                hole=0.55,
                color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#38bdf8"],
            )
            fig_rank.update_traces(textposition="inside", textinfo="percent+label")
            fig_rank = apply_cyber_theme(fig_rank, height=460, legend_orientation="v")

        st.plotly_chart(fig_rank, use_container_width=True)

    with row3_c2:
        proto_atk_chart_type = render_chart_header(
            title="🌐 Inferred Service Distribution Across Threat Categories",
            chart_types=["🗺️ Inferred Service Heatmap", "📊 Stacked Inferred Service Bar", "📶 Grouped Bar"],
            key="atk_proto_chart_type",
            default="🗺️ Inferred Service Heatmap",
            help_text="Inferred service categories targeted by attack vectors, based on destination port conventions (not deep packet inspection).",
        )

        sample_attacks = df_sample[df_sample["Label"] != BENIGN_LABEL].copy()
        sample_attacks["Inferred Service"] = sample_attacks["Destination Port"].apply(resolve_protocol_group)
        proto_crosstab = pd.crosstab(sample_attacks["Threat_Category"], sample_attacks["Inferred Service"])

        if proto_atk_chart_type == "🗺️ Inferred Service Heatmap":
            fig_proto_atk = px.imshow(
                proto_crosstab,
                color_continuous_scale=[[0.0, "#0b0f19"], [0.2, "#0369a1"], [0.6, "#0284c7"], [1.0, "#38bdf8"]],
                text_auto=True,
                labels=dict(x="Inferred Service Category", y="Threat Category", color="Sample Flows"),
            )
            fig_proto_atk.update_layout(xaxis=dict(tickangle=-35))
            fig_proto_atk = apply_cyber_theme(fig_proto_atk, height=460)

        elif proto_atk_chart_type == "📊 Stacked Inferred Service Bar":
            df_proto_melt = proto_crosstab.reset_index().melt(
                id_vars=["Threat_Category"], var_name="Inferred Service", value_name="Flows"
            )
            fig_proto_atk = px.bar(
                df_proto_melt,
                x="Threat_Category",
                y="Flows",
                color="Inferred Service",
                barmode="stack",
                color_discrete_sequence=CYBER_PALETTE,
                log_y=True,
            )
            fig_proto_atk.update_layout(xaxis=dict(tickangle=-30), yaxis=dict(title="Flows (Log Scale)"))
            fig_proto_atk = apply_cyber_theme(fig_proto_atk, height=460)

        else:  # 📶 Grouped Bar
            df_proto_melt = proto_crosstab.reset_index().melt(
                id_vars=["Threat_Category"], var_name="Inferred Service", value_name="Flows"
            )
            top_p = df_proto_melt.groupby("Inferred Service")["Flows"].sum().nlargest(5).index
            df_proto_filtered = df_proto_melt[df_proto_melt["Inferred Service"].isin(top_p)]
            fig_proto_atk = px.bar(
                df_proto_filtered,
                x="Flows",
                y="Threat_Category",
                color="Inferred Service",
                barmode="group",
                orientation="h",
                color_discrete_sequence=CYBER_PALETTE,
                log_x=True,
            )
            fig_proto_atk.update_layout(yaxis=dict(title=""), xaxis=dict(title="Flows (Log Scale)"))
            fig_proto_atk = apply_cyber_theme(fig_proto_atk, height=460)

        st.plotly_chart(fig_proto_atk, use_container_width=True)

    st.markdown("---")

    # Complete Taxonomy Detail Table
    st.markdown("#### 📋 Full Attack Taxonomy & Threat Level Matrix")
    matrix_rows = []
    for attack, count in label_dist.items():
        cat = ATTACK_CATEGORIES.get(attack, "Other Attacks")
        pct_all = (count / total_records) * 100 if total_records > 0 else 0
        if attack == BENIGN_LABEL:
            severity = "BASELINE"
            action = "Authorized production traffic"
        elif "DoS" in attack or "DDoS" in attack or "Heartbleed" in attack:
            severity = "CRITICAL"
            action = "Rate limiting, scrubbing center, patch CVE-2014-0160"
        elif "Patator" in attack or "Infiltration" in attack:
            severity = "HIGH"
            action = "Account lockout, fail2ban, host isolation"
        elif "Web Attack" in attack:
            severity = "HIGH"
            action = "WAF inspection, input sanitization, CSP rules"
        elif "Bot" in attack:
            severity = "HIGH"
            action = "C2 domain block, endpoint forensics"
        else:
            severity = "MEDIUM"
            action = "Firewall port blocking, drop stealth probes"

        matrix_rows.append({
            "Classification": attack,
            "Threat Category": cat,
            "Total Occurrences": f"{count:,}",
            "Share of Total (%)": f"{pct_all:.4f}%",
            "Threat Severity": severity,
            "Mitigation Protocol": action,
        })

    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
