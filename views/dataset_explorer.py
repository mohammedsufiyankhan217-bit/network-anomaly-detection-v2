"""
Dataset Explorer View for Network Anomaly Detection V2.
Interactive multi-dimensional data inspection, column stats, and hygiene verification.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_single_file
from src.config import EXPECTED_FILES, BENIGN_LABEL, KEY_PREDICTIVE_FEATURES


def render_dataset_explorer_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🔍 Dataset Explorer & Forensic Telemetry")
    st.caption("Inspect raw and processed network flow vectors, feature distributions, and hygiene.")

    tab1, tab2, tab3 = st.tabs(["📊 Sample Data Explorer", "📈 Feature Statistics", "📁 Daily Capture Inspector"])

    with tab1:
        st.markdown("#### Interactive Flow Inspection")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            all_classes = ["ALL"] + sorted(list(df_sample["Label"].unique()))
            selected_class = st.selectbox("Filter by Attack Class", all_classes)
        with c2:
            default_cols = ["Label", "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets", "Flow Packets/s", "Flow Bytes/s", "Init_Win_bytes_forward"]
            selected_cols = st.multiselect(
                "Select Feature Columns",
                options=list(df_sample.columns),
                default=[c for c in default_cols if c in df_sample.columns],
            )
        with c3:
            row_limit = st.slider("Display Limit", min_value=10, max_value=200, value=50, step=10)

        # Apply filter
        display_df = df_sample
        if selected_class != "ALL":
            display_df = display_df[display_df["Label"] == selected_class]

        if not selected_cols:
            selected_cols = list(df_sample.columns)[:10]

        st.dataframe(
            display_df[selected_cols].head(row_limit),
            use_container_width=True,
            height=350,
        )

        st.info(
            f"Displaying {min(row_limit, len(display_df))} of {len(display_df):,} matching rows "
            f"from the active {len(df_sample):,} stratified sample."
        )

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
        st.markdown("##### 🔗 Key Feature Correlation Heatmap")
        corr_feats = [c for c in KEY_PREDICTIVE_FEATURES if c in df_sample.columns][:10]
        corr_matrix = df_sample[corr_feats].corr()
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="Blues",
            zmin=-1,
            zmax=1,
            labels=dict(color="Pearson Corr"),
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=50, t=20, b=40),
            height=420,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("#### 📁 Direct Daily File Inspection")
        st.caption("Load and inspect an isolated slice directly from any of the 8 raw capture CSVs.")

        chosen_file = st.selectbox("Select Raw Capture File", EXPECTED_FILES)
        slice_rows = st.slider("Rows to load from file", 1000, 20000, 5000, step=1000)

        if st.button("📥 Load File Slice"):
            with st.spinner(f"Loading {chosen_file}..."):
                try:
                    slice_df = load_single_file(chosen_file, max_rows=slice_rows)
                    st.success(f"Successfully loaded {len(slice_df):,} rows from `{chosen_file}`!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Label Distribution in Slice:**")
                        st.dataframe(slice_df["Label"].value_counts(), use_container_width=True)
                    with c2:
                        st.markdown("**Feature Snapshot:**")
                        preview_cols = [c for c in ["Label", "Destination Port", "Flow Duration", "Flow Packets/s"] if c in slice_df.columns]
                        st.dataframe(slice_df[preview_cols].head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file slice: {e}")
