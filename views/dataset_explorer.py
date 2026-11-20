"""
Dataset Explorer View for Network Anomaly Detection V2.
Interactive multi-dimensional data inspection, feature distributions,
hygiene verification, correlation heatmaps, and category box plot comparisons.
"""
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_single_file
from src.config import EXPECTED_FILES, BENIGN_LABEL, KEY_PREDICTIVE_FEATURES, CATEGORY_COLORS
from src.ui_components import render_chart_header, apply_cyber_theme, CYBER_PALETTE, TRAFFIC_COLOR_MAP


def render_dataset_explorer_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🔍 Dataset Explorer & Forensic Telemetry")
    st.caption("Inspect raw and processed network flow vectors, feature distributions, and hygiene.")

    tab1, tab2, tab3 = st.tabs(["📊 Sample Data & Distributions", "📈 Statistical Profiles & Heatmaps", "📁 Daily Capture Inspector"])

    with tab1:
        st.markdown("#### Interactive Flow Inspection & Filter Controls")
        c1, c2, c3 = st.columns([1, 1.2, 0.8])
        with c1:
            all_classes = ["ALL"] + sorted(list(df_sample["Label"].unique()))
            selected_class = st.selectbox("Filter by Attack Class", all_classes, key="exp_class_filter")
        with c2:
            default_cols = [
                "Label",
                "Destination Port",
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
                "Flow Packets/s",
                "Flow Bytes/s",
                "Packet Length Mean",
                "Init_Win_bytes_forward",
            ]
            selected_cols = st.multiselect(
                "Select Feature Columns to Inspect",
                options=list(df_sample.columns),
                default=[c for c in default_cols if c in df_sample.columns],
                key="exp_col_filter",
            )
        with c3:
            row_limit = st.slider("Display Limit (Rows)", min_value=10, max_value=200, value=50, step=10, key="exp_row_limit")

        # Apply filter reactively
        display_df = df_sample if selected_class == "ALL" else df_sample[df_sample["Label"] == selected_class]

        if not selected_cols:
            selected_cols = list(df_sample.columns)[:10]

        st.dataframe(
            display_df[selected_cols].head(row_limit),
            use_container_width=True,
            height=320,
        )

        st.info(
            f"Displaying **{min(row_limit, len(display_df)):,}** of **{len(display_df):,}** matching flows "
            f"({'Filtered: ' + selected_class if selected_class != 'ALL' else 'All Traffic Classes'}) "
            f"from active sample of {len(df_sample):,} flows."
        )

        st.markdown("---")

        # Visualizations responsive to the filter
        col_d1, col_d2 = st.columns(2)

        # 1. Flow Duration Distribution
        with col_d1:
            dur_chart = render_chart_header(
                title="⚡ Flow Duration Distribution",
                chart_types=["📊 Histogram", "📦 Box Plot", "📈 Line Distribution"],
                key="exp_dur_chart_type",
                default="📊 Histogram",
                help_text="Distribution of flow elapsed time in microseconds for filtered flows.",
            )

            if dur_chart == "📊 Histogram":
                fig_dur = px.histogram(
                    display_df,
                    x="Flow Duration",
                    color="Binary_Label",
                    nbins=35,
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Duration": "Flow Duration (μs)", "Binary_Label": "Traffic"},
                )
                fig_dur.update_layout(yaxis=dict(title="Flows (Log Scale)"))
                fig_dur = apply_cyber_theme(fig_dur, height=360)
            elif dur_chart == "📦 Box Plot":
                fig_dur = px.box(
                    display_df,
                    x="Binary_Label",
                    y="Flow Duration",
                    color="Binary_Label",
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Duration": "Flow Duration (μs, Log)", "Binary_Label": ""},
                )
                fig_dur = apply_cyber_theme(fig_dur, height=360, show_legend=False)
            else:  # 📈 Line Distribution
                vals = display_df["Flow Duration"].dropna()
                counts, bin_edges = np.histogram(vals, bins=35)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                df_line = pd.DataFrame({"Duration (μs)": bin_centers, "Count": counts})
                fig_dur = px.line(df_line, x="Duration (μs)", y="Count", log_y=True, markers=True)
                fig_dur.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=7, color="#10b981"))
                fig_dur = apply_cyber_theme(fig_dur, height=360, show_legend=False)

            st.plotly_chart(fig_dur, use_container_width=True)

        # 2. Packet Length Distribution
        with col_d2:
            pkt_chart = render_chart_header(
                title="📏 Packet Length Distribution",
                chart_types=["📊 Histogram", "📦 Box Plot", "📈 Line Distribution"],
                key="exp_pkt_chart_type",
                default="📊 Histogram",
                help_text="Mean packet size in bytes for bidirectional communications.",
            )

            pkt_col = "Packet Length Mean" if "Packet Length Mean" in display_df.columns else "Average Packet Size"
            if pkt_chart == "📊 Histogram":
                fig_pkt = px.histogram(
                    display_df,
                    x=pkt_col,
                    color="Binary_Label",
                    nbins=35,
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={pkt_col: "Packet Length (Bytes)", "Binary_Label": "Traffic"},
                )
                fig_pkt.update_layout(yaxis=dict(title="Flows (Log Scale)"))
                fig_pkt = apply_cyber_theme(fig_pkt, height=360)
            elif pkt_chart == "📦 Box Plot":
                fig_pkt = px.box(
                    display_df,
                    x="Binary_Label",
                    y=pkt_col,
                    color="Binary_Label",
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={pkt_col: "Packet Length (Bytes, Log)", "Binary_Label": ""},
                )
                fig_pkt = apply_cyber_theme(fig_pkt, height=360, show_legend=False)
            else:  # 📈 Line Distribution
                vals = display_df[pkt_col].dropna()
                counts, bin_edges = np.histogram(vals, bins=35)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                df_line = pd.DataFrame({"Packet Length (Bytes)": bin_centers, "Count": counts})
                fig_pkt = px.line(df_line, x="Packet Length (Bytes)", y="Count", log_y=True, markers=True)
                fig_pkt.update_traces(line=dict(color="#10b981", width=3), marker=dict(size=7, color="#38bdf8"))
                fig_pkt = apply_cyber_theme(fig_pkt, height=360, show_legend=False)

            st.plotly_chart(fig_pkt, use_container_width=True)

        st.markdown("---")

        col_d3, col_d4 = st.columns(2)

        # 3. Flow Bytes/s Distribution
        with col_d3:
            bps_chart = render_chart_header(
                title="💾 Flow Bytes/s Velocity Distribution",
                chart_types=["📊 Histogram", "📦 Box Plot", "📈 Line Distribution"],
                key="exp_bps_chart_type",
                default="📊 Histogram",
                help_text="Transmission bandwidth velocity in bytes per second.",
            )

            if bps_chart == "📊 Histogram":
                fig_bps = px.histogram(
                    display_df,
                    x="Flow Bytes/s",
                    color="Binary_Label",
                    nbins=35,
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Bytes/s": "Flow Bytes/s", "Binary_Label": "Traffic"},
                )
                fig_bps.update_layout(yaxis=dict(title="Flows (Log Scale)"))
                fig_bps = apply_cyber_theme(fig_bps, height=360)
            elif bps_chart == "📦 Box Plot":
                fig_bps = px.box(
                    display_df,
                    x="Binary_Label",
                    y="Flow Bytes/s",
                    color="Binary_Label",
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Bytes/s": "Flow Bytes/s (Log)", "Binary_Label": ""},
                )
                fig_bps = apply_cyber_theme(fig_bps, height=360, show_legend=False)
            else:
                vals = display_df["Flow Bytes/s"].dropna()
                counts, bin_edges = np.histogram(vals, bins=35)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                df_line = pd.DataFrame({"Flow Bytes/s": bin_centers, "Count": counts})
                fig_bps = px.line(df_line, x="Flow Bytes/s", y="Count", log_y=True, markers=True)
                fig_bps.update_traces(line=dict(color="#f59e0b", width=3), marker=dict(size=7, color="#ef4444"))
                fig_bps = apply_cyber_theme(fig_bps, height=360, show_legend=False)

            st.plotly_chart(fig_bps, use_container_width=True)

        # 4. Flow Packets/s Distribution
        with col_d4:
            pps_chart = render_chart_header(
                title="📦 Flow Packets/s Velocity Distribution",
                chart_types=["📊 Histogram", "📦 Box Plot", "📈 Line Distribution"],
                key="exp_pps_chart_type",
                default="📊 Histogram",
                help_text="Packet frequency per second highlighting flood-based volumetric attacks.",
            )

            if pps_chart == "📊 Histogram":
                fig_pps = px.histogram(
                    display_df,
                    x="Flow Packets/s",
                    color="Binary_Label",
                    nbins=35,
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Packets/s": "Flow Packets/s", "Binary_Label": "Traffic"},
                )
                fig_pps.update_layout(yaxis=dict(title="Flows (Log Scale)"))
                fig_pps = apply_cyber_theme(fig_pps, height=360)
            elif pps_chart == "📦 Box Plot":
                fig_pps = px.box(
                    display_df,
                    x="Binary_Label",
                    y="Flow Packets/s",
                    color="Binary_Label",
                    log_y=True,
                    color_discrete_map=TRAFFIC_COLOR_MAP,
                    labels={"Flow Packets/s": "Flow Packets/s (Log)", "Binary_Label": ""},
                )
                fig_pps = apply_cyber_theme(fig_pps, height=360, show_legend=False)
            else:
                vals = display_df["Flow Packets/s"].dropna()
                counts, bin_edges = np.histogram(vals, bins=35)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                df_line = pd.DataFrame({"Flow Packets/s": bin_centers, "Count": counts})
                fig_pps = px.line(df_line, x="Flow Packets/s", y="Count", log_y=True, markers=True)
                fig_pps.update_traces(line=dict(color="#8b5cf6", width=3), marker=dict(size=7, color="#38bdf8"))
                fig_pps = apply_cyber_theme(fig_pps, height=360, show_legend=False)

            st.plotly_chart(fig_pps, use_container_width=True)

        st.markdown("---")

        # 5. Forward vs Backward Packet Comparison
        fwd_bwd_chart = render_chart_header(
            title="🔄 Forward vs Backward Packet Comparison",
            chart_types=["📦 Scatter Plot", "📊 Grouped Direction Bar", "📦 Box Plot"],
            key="exp_fb_chart_type",
            default="📦 Scatter Plot",
            help_text="Bidirectional packet balance showing asymmetry, amplification, and handshake dynamics.",
        )

        if fwd_bwd_chart == "📦 Scatter Plot":
            sample_for_scatter = display_df.sample(min(2500, len(display_df)), random_state=42)
            fig_fb = px.scatter(
                sample_for_scatter,
                x="Total Fwd Packets",
                y="Total Backward Packets",
                color="Binary_Label",
                color_discrete_map=TRAFFIC_COLOR_MAP,
                log_x=True,
                log_y=True,
                hover_data=["Label", "Destination Port"],
                labels={"Binary_Label": "Traffic"},
            )
            fig_fb.update_layout(
                xaxis=dict(title="Forward Packets (Log Scale)"),
                yaxis=dict(title="Backward Packets (Log Scale)"),
            )
            fig_fb = apply_cyber_theme(fig_fb, height=420)

        elif fwd_bwd_chart == "📊 Grouped Direction Bar":
            fb_summary = (
                display_df.groupby("Binary_Label")[["Total Fwd Packets", "Total Backward Packets"]]
                .median()
                .reset_index()
            )
            fb_melted = pd.melt(
                fb_summary,
                id_vars=["Binary_Label"],
                value_vars=["Total Fwd Packets", "Total Backward Packets"],
                var_name="Packet Direction",
                value_name="Median Packet Count",
            )
            fig_fb = px.bar(
                fb_melted,
                x="Binary_Label",
                y="Median Packet Count",
                color="Packet Direction",
                barmode="group",
                color_discrete_map={"Total Fwd Packets": "#38bdf8", "Total Backward Packets": "#f59e0b"},
                text="Median Packet Count",
            )
            fig_fb.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig_fb.update_layout(yaxis=dict(range=[0, max(fb_melted["Median Packet Count"]) * 1.25]))
            fig_fb = apply_cyber_theme(fig_fb, height=420)

        else:  # 📦 Box Plot
            fb_long = pd.melt(
                display_df,
                id_vars=["Binary_Label"],
                value_vars=["Total Fwd Packets", "Total Backward Packets"],
                var_name="Packet Direction",
                value_name="Packet Count",
            )
            fig_fb = px.box(
                fb_long,
                x="Binary_Label",
                y="Packet Count",
                color="Packet Direction",
                log_y=True,
                color_discrete_map={"Total Fwd Packets": "#38bdf8", "Total Backward Packets": "#f59e0b"},
            )
            fig_fb = apply_cyber_theme(fig_fb, height=420)

        st.plotly_chart(fig_fb, use_container_width=True)

    with tab2:
        st.markdown("#### 📐 Statistical Profiles & Data Hygiene")
        col_stat1, col_stat2 = st.columns([2, 1])

        with col_stat1:
            st.markdown("##### Descriptive Summary of Key Network Features")
            num_cols = [c for c in KEY_PREDICTIVE_FEATURES if c in df_sample.columns][:8]
            desc = df_sample[num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
            desc.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
            st.dataframe(desc.style.format("{:,.2f}"), use_container_width=True)

        with col_stat2:
            st.markdown("##### 🛡️ Data Hygiene Verification")
            num_df = df_sample.select_dtypes(include=[np.number])
            na_count = num_df.isna().sum().sum()
            inf_count = np.isinf(num_df).sum().sum()
            dup_cols = len(df_sample.columns) - len(set(df_sample.columns))

            st.metric("Missing (NaN) Feature Cells", f"{na_count:,}")
            st.metric("Infinite (±Inf) Values", f"{inf_count:,}")
            st.metric("Duplicate Column Headers", f"{dup_cols}")
            if na_count == 0 and inf_count == 0 and dup_cols == 0:
                st.success("✅ **Zero Data Anomalies**: All features sanitized and clean.")

        st.markdown("---")

        # Feature Correlation Analysis with configurable subset
        st.markdown("#### 🔗 Numeric Feature Correlation Heatmap")
        st.caption("Manageable subset of relevant predictive features showing multicollinearity and dependencies.")

        available_features = [c for c in KEY_PREDICTIVE_FEATURES if c in df_sample.columns]
        default_corr_feats = available_features[:12]

        corr_feats = st.multiselect(
            "Select Subset of Features for Correlation Matrix:",
            options=available_features,
            default=default_corr_feats,
            key="exp_corr_feats_sel",
        )

        if len(corr_feats) < 2:
            corr_feats = default_corr_feats

        corr_chart_type = render_chart_header(
            title="Correlation Matrix Format",
            chart_types=["🗺️ Heatmap", "📶 Horizontal Collinearity Bar", "📈 Collinearity Trend"],
            key="exp_corr_chart_type",
            default="🗺️ Heatmap",
        )

        corr_matrix = df_sample[corr_feats].corr()

        if corr_chart_type == "🗺️ Heatmap":
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale=[[0.0, "#0b0f19"], [0.5, "#0284c7"], [1.0, "#38bdf8"]],
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                labels=dict(color="Pearson r"),
            )
            fig_corr.update_layout(xaxis=dict(tickangle=-35))
            fig_corr = apply_cyber_theme(fig_corr, height=max(440, len(corr_feats) * 28), margin=dict(l=60, r=40, t=20, b=80))

        elif corr_chart_type == "📶 Horizontal Collinearity Bar":
            mean_corr = corr_matrix.abs().mean().sort_values(ascending=True).reset_index()
            mean_corr.columns = ["Feature", "Mean Collinearity (|r|)"]
            fig_corr = px.bar(
                mean_corr,
                x="Mean Collinearity (|r|)",
                y="Feature",
                orientation="h",
                color="Mean Collinearity (|r|)",
                color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#67e8f9"]],
                text="Mean Collinearity (|r|)",
            )
            fig_corr.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
            fig_corr.update_layout(coloraxis_showscale=False, xaxis=dict(range=[0, 1.25]))
            fig_corr = apply_cyber_theme(fig_corr, height=max(400, len(corr_feats) * 24), show_legend=False)

        else:  # 📈 Collinearity Trend
            mean_corr_desc = corr_matrix.abs().mean().sort_values(ascending=False).reset_index()
            mean_corr_desc.columns = ["Feature", "Mean Collinearity (|r|)"]
            fig_corr = px.line(
                mean_corr_desc,
                x="Feature",
                y="Mean Collinearity (|r|)",
                markers=True,
            )
            fig_corr.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#10b981"))
            fig_corr.update_layout(xaxis=dict(tickangle=-35), yaxis=dict(range=[0, 1.05]))
            fig_corr = apply_cyber_theme(fig_corr, height=420, show_legend=False)

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # Box plots comparing selected numeric features across traffic categories
        st.markdown("#### 📦 Feature Comparison Across Traffic Categories")
        st.caption("Inspect how distinct cyber threat families manifest across individual continuous features.")

        c_box1, c_box2 = st.columns([1.5, 1])
        with c_box1:
            box_feature = st.selectbox(
                "Select Numeric Feature to Compare",
                options=[c for c in KEY_PREDICTIVE_FEATURES if c in df_sample.columns],
                index=0,
                key="exp_box_feature_sel",
            )
        with c_box2:
            box_group = st.selectbox(
                "Group By Categorical Dimension",
                options=["Threat_Category", "Binary_Label"],
                format_func=lambda x: "Threat Category (8 Families)" if x == "Threat_Category" else "Binary Label (Benign vs Attack)",
                key="exp_box_group_sel",
            )

        box_view_type = render_chart_header(
            title=f"Distribution of '{box_feature}'",
            chart_types=["📦 Box Plot", "📊 Bar (Median Value)", "📈 Profile Line (Median)", "🌊 Distribution Area"],
            key="exp_box_view_type",
            default="📦 Box Plot",
        )

        if box_view_type == "📦 Box Plot":
            fig_box_feat = px.box(
                df_sample,
                x=box_group,
                y=box_feature,
                color=box_group,
                log_y=True,
                color_discrete_map=CATEGORY_COLORS if box_group == "Threat_Category" else TRAFFIC_COLOR_MAP,
                title=f"Box Plot of '{box_feature}' by {box_group} (Log Scale)",
            )
            fig_box_feat.update_layout(xaxis=dict(tickangle=-25))
            fig_box_feat = apply_cyber_theme(fig_box_feat, height=440, show_legend=False)

        elif box_view_type == "📊 Bar (Median Value)":
            med_df = df_sample.groupby(box_group)[box_feature].median().reset_index()
            fig_box_feat = px.bar(
                med_df,
                x=box_group,
                y=box_feature,
                color=box_group,
                log_y=True,
                color_discrete_map=CATEGORY_COLORS if box_group == "Threat_Category" else TRAFFIC_COLOR_MAP,
                text=box_feature,
                title=f"Median '{box_feature}' by {box_group} (Log Scale)",
            )
            fig_box_feat.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            fig_box_feat.update_layout(xaxis=dict(tickangle=-25))
            fig_box_feat = apply_cyber_theme(fig_box_feat, height=440, show_legend=False)

        elif box_view_type == "📈 Profile Line (Median)":
            med_df = df_sample.groupby(box_group)[box_feature].median().reset_index()
            fig_box_feat = px.line(
                med_df,
                x=box_group,
                y=box_feature,
                markers=True,
                log_y=True,
                title=f"Median Profile of '{box_feature}' across {box_group}",
            )
            fig_box_feat.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#ef4444"))
            fig_box_feat.update_layout(xaxis=dict(tickangle=-25))
            fig_box_feat = apply_cyber_theme(fig_box_feat, height=440, show_legend=False)

        else:  # 🌊 Distribution Area
            med_df = df_sample.groupby(box_group)[box_feature].median().reset_index()
            fig_box_feat = px.area(
                med_df,
                x=box_group,
                y=box_feature,
                log_y=True,
                title=f"Median Area Envelope of '{box_feature}' across {box_group}",
            )
            fig_box_feat.update_traces(line=dict(color="#38bdf8"), fillcolor="rgba(56, 189, 248, 0.35)")
            fig_box_feat.update_layout(xaxis=dict(tickangle=-25))
            fig_box_feat = apply_cyber_theme(fig_box_feat, height=440, show_legend=False)

        st.plotly_chart(fig_box_feat, use_container_width=True)

    with tab3:
        st.markdown("#### 📁 Direct Daily File Inspection")
        st.caption("Load and inspect an isolated slice directly from any of the 8 raw capture CSVs.")

        chosen_file = st.selectbox("Select Raw Capture File", EXPECTED_FILES, key="exp_daily_file_sel")
        slice_rows = st.slider("Rows to load from file", 1000, 20000, 5000, step=1000, key="exp_slice_rows_sel")

        if st.button("📥 Load File Slice", key="exp_load_slice_btn"):
            with st.spinner(f"Loading {chosen_file}..."):
                try:
                    slice_df = load_single_file(chosen_file, max_rows=slice_rows)
                    st.success(f"Successfully loaded {len(slice_df):,} rows from `{chosen_file}`!")
                    c1, c2 = st.columns(2)
                    with c1:
                        slice_chart_type = render_chart_header(
                            title="Slice Traffic Distribution",
                            chart_types=["📊 Bar Chart", "🍩 Donut Chart", "🥧 Pie Chart", "📈 Line Chart", "📋 Data Table"],
                            key="exp_slice_chart_type",
                            default="📊 Bar Chart",
                        )
                        vc = slice_df["Label"].value_counts().reset_index()
                        vc.columns = ["Label", "Count"]

                        if slice_chart_type == "📊 Bar Chart":
                            fig_sl = px.bar(
                                vc,
                                x="Count",
                                y="Label",
                                orientation="h",
                                text="Count",
                                color="Count",
                                color_continuous_scale="Blues",
                                log_x=True,
                            )
                            fig_sl.update_traces(texttemplate="%{text:,}", textposition="outside")
                            fig_sl.update_layout(coloraxis_showscale=False)
                            fig_sl = apply_cyber_theme(fig_sl, height=320, margin=dict(l=120, r=20, t=20, b=40), show_legend=False)
                            st.plotly_chart(fig_sl, use_container_width=True)

                        elif slice_chart_type in ["🍩 Donut Chart", "🥧 Pie Chart"]:
                            hole = 0.55 if "Donut" in slice_chart_type else 0.0
                            fig_sl = px.pie(
                                vc,
                                names="Label",
                                values="Count",
                                hole=hole,
                                color_discrete_sequence=CYBER_PALETTE,
                            )
                            fig_sl.update_traces(textposition="inside", textinfo="percent")
                            fig_sl = apply_cyber_theme(fig_sl, height=320, margin=dict(l=10, r=10, t=20, b=20), legend_orientation="v")
                            st.plotly_chart(fig_sl, use_container_width=True)

                        elif slice_chart_type == "📈 Line Chart":
                            fig_sl = px.line(
                                vc,
                                x="Label",
                                y="Count",
                                markers=True,
                                log_y=True,
                            )
                            fig_sl.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#ef4444"))
                            fig_sl = apply_cyber_theme(fig_sl, height=320, margin=dict(l=40, r=20, t=20, b=60), show_legend=False)
                            fig_sl.update_xaxes(tickangle=-30, tickfont=dict(color="#94a3b8", size=9))
                            st.plotly_chart(fig_sl, use_container_width=True)

                        else:  # 📋 Data Table
                            st.dataframe(slice_df["Label"].value_counts(), use_container_width=True)
                    with c2:
                        st.markdown("**Feature Snapshot:**")
                        preview_cols = [c for c in ["Label", "Destination Port", "Flow Duration", "Flow Packets/s"] if c in slice_df.columns]
                        st.dataframe(slice_df[preview_cols].head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file slice: {e}")
