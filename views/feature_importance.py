"""
Feature Importance & Cybersecurity Explainability View for Network Anomaly Detection V2.
Interactive visualization of network decision boundaries, Gini importance rankings,
side-by-side comparison of binary vs multiclass feature importance, and domain explanations.
"""
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.model_trainer import load_saved_model
from src.explainability import plot_feature_importance, get_feature_explanation
from src.config import BENIGN_LABEL, CATEGORY_COLORS
from src.ui_components import render_chart_header, apply_cyber_theme, CYBER_PALETTE, TRAFFIC_COLOR_MAP


def render_feature_importance_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🧬 Feature Importance & Threat Explainability")
    st.caption("Inspect mathematical feature importances and domain mechanics across trained models.")

    # Load both models for comparative explainability
    bin_artifact = st.session_state.get("model_binary") or load_saved_model("binary")
    multi_artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")

    if bin_artifact is None and multi_artifact is None:
        st.warning("⚠️ Please train or load a model from the ML Training & Evaluation view to inspect feature importances.")
        return

    # User Configuration Controls
    col_c1, col_c2 = st.columns([1.5, 1])
    with col_c1:
        top_k = st.slider("Number of Top Features to Display (k)", 10, 30, 15, step=5, key="feat_top_k_slider")
    with col_c2:
        model_view = st.selectbox(
            "Feature Importance Engine",
            ["Multiclass Classifier", "Binary Anomaly Detector", "Direct Comparison (Binary vs Multiclass)"],
            index=0 if multi_artifact is not None else 1,
            key="feat_model_view_sel",
        )

    st.markdown("---")

    # Extract feature importances
    bin_feat_imps = bin_artifact["results"].get("feature_importances", []) if bin_artifact else []
    multi_feat_imps = multi_artifact["results"].get("feature_importances", []) if multi_artifact else []

    active_imps = multi_feat_imps if "Multiclass" in model_view else bin_feat_imps
    if not active_imps:
        active_imps = bin_feat_imps if bin_feat_imps else multi_feat_imps

    # ==========================================
    # SECTION 1: Horizontal Feature Importance Ranking / Comparison
    # ==========================================
    if "Direct Comparison" in model_view and bin_artifact and multi_artifact:
        st.markdown("#### ⚖️ Binary vs Multiclass Feature Importance Comparison")
        st.caption("Analyzes how feature discriminability shifts when distinguishing general anomalies vs specific attack taxonomy.")

        comp_chart_type = render_chart_header(
            title="Importance Comparison Format",
            chart_types=["📶 Grouped Horizontal Bar", "📊 Grouped Vertical Bar", "🗺️ Heatmap", "📈 Profile Line"],
            key="feat_comp_chart_type",
            default="📶 Grouped Horizontal Bar",
        )

        # Build comparison dataframe
        bin_dict = dict(bin_feat_imps)
        multi_dict = dict(multi_feat_imps)
        all_top_names = list(dict.fromkeys([f[0] for f in multi_feat_imps[:top_k]] + [f[0] for f in bin_feat_imps[:top_k]]))[:top_k]

        comp_rows = []
        for name in all_top_names:
            comp_rows.append({
                "Feature": name,
                "Binary Detector": bin_dict.get(name, 0.0),
                "Multiclass Classifier": multi_dict.get(name, 0.0),
            })
        df_comp_feat = pd.DataFrame(comp_rows)

        if comp_chart_type == "📶 Grouped Horizontal Bar":
            df_melt = pd.melt(
                df_comp_feat,
                id_vars=["Feature"],
                value_vars=["Binary Detector", "Multiclass Classifier"],
                var_name="Model Architecture",
                value_name="Gini Importance",
            )
            fig_imp_comp = px.bar(
                df_melt,
                x="Gini Importance",
                y="Feature",
                color="Model Architecture",
                barmode="group",
                orientation="h",
                color_discrete_map={"Binary Detector": "#10b981", "Multiclass Classifier": "#38bdf8"},
            )
            fig_imp_comp.update_layout(yaxis=dict(title=""), xaxis=dict(title="Gini Importance Score"))
            fig_imp_comp = apply_cyber_theme(fig_imp_comp, height=max(440, top_k * 26))
            st.plotly_chart(fig_imp_comp, use_container_width=True)

        elif comp_chart_type == "📊 Grouped Vertical Bar":
            df_melt = pd.melt(
                df_comp_feat,
                id_vars=["Feature"],
                value_vars=["Binary Detector", "Multiclass Classifier"],
                var_name="Model Architecture",
                value_name="Gini Importance",
            )
            fig_imp_comp = px.bar(
                df_melt,
                x="Feature",
                y="Gini Importance",
                color="Model Architecture",
                barmode="group",
                color_discrete_map={"Binary Detector": "#10b981", "Multiclass Classifier": "#38bdf8"},
            )
            fig_imp_comp.update_layout(xaxis=dict(tickangle=-35))
            fig_imp_comp = apply_cyber_theme(fig_imp_comp, height=440)
            st.plotly_chart(fig_imp_comp, use_container_width=True)

        elif comp_chart_type == "🗺️ Heatmap":
            heat_df = df_comp_feat.set_index("Feature")
            fig_imp_comp = px.imshow(
                heat_df,
                color_continuous_scale=[[0.0, "#0b0f19"], [0.5, "#0284c7"], [1.0, "#38bdf8"]],
                text_auto=".4f",
                aspect="auto",
            )
            fig_imp_comp = apply_cyber_theme(fig_imp_comp, height=max(440, top_k * 26))
            st.plotly_chart(fig_imp_comp, use_container_width=True)

        else:  # 📈 Profile Line
            df_melt = pd.melt(
                df_comp_feat,
                id_vars=["Feature"],
                value_vars=["Binary Detector", "Multiclass Classifier"],
                var_name="Model Architecture",
                value_name="Gini Importance",
            )
            fig_imp_comp = px.line(
                df_melt,
                x="Feature",
                y="Gini Importance",
                color="Model Architecture",
                markers=True,
                color_discrete_map={"Binary Detector": "#10b981", "Multiclass Classifier": "#38bdf8"},
            )
            fig_imp_comp.update_layout(xaxis=dict(tickangle=-35))
            fig_imp_comp = apply_cyber_theme(fig_imp_comp, height=440)
            st.plotly_chart(fig_imp_comp, use_container_width=True)

    else:
        # Single Model Ranking View
        col_rank, col_domain = st.columns([1.5, 1])

        with col_rank:
            imp_chart_type = render_chart_header(
                title=f"Top {top_k} Discriminative Features",
                chart_types=["📶 Horizontal Bar", "📊 Vertical Bar", "📈 Profile Line", "🌊 Area Curve", "🍩 Donut Share"],
                key="feat_imp_chart_type",
                default="📶 Horizontal Bar",
                help_text=f"Relative Gini importance calculated from the trained {model_view}.",
            )

            top_feats = active_imps[:top_k]
            df_imp = pd.DataFrame([{"Feature": f[0], "Importance": f[1]} for f in top_feats])

            if imp_chart_type == "📶 Horizontal Bar":
                df_imp_sorted = df_imp.sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    df_imp_sorted,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#67e8f9"]],
                    text="Importance",
                )
                fig_imp.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
                fig_imp.update_layout(
                    coloraxis_showscale=False,
                    yaxis=dict(title=""),
                    xaxis=dict(range=[0, max(df_imp["Importance"]) * 1.3]),
                    margin=dict(l=160, r=60, t=20, b=40),
                )
                fig_imp = apply_cyber_theme(fig_imp, height=max(440, top_k * 25), show_legend=False)

            elif imp_chart_type == "📊 Vertical Bar":
                fig_imp = px.bar(
                    df_imp,
                    x="Feature",
                    y="Importance",
                    color="Importance",
                    color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#67e8f9"]],
                    text="Importance",
                )
                fig_imp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_imp.update_layout(coloraxis_showscale=False, xaxis=dict(tickangle=-35))
                fig_imp = apply_cyber_theme(fig_imp, height=440, show_legend=False)

            elif imp_chart_type == "📈 Profile Line":
                fig_imp = px.line(
                    df_imp,
                    x="Feature",
                    y="Importance",
                    markers=True,
                )
                fig_imp.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=8, color="#10b981"))
                fig_imp.update_layout(xaxis=dict(tickangle=-35))
                fig_imp = apply_cyber_theme(fig_imp, height=440, show_legend=False)

            elif imp_chart_type == "🌊 Area Curve":
                fig_imp = px.area(
                    df_imp,
                    x="Feature",
                    y="Importance",
                )
                fig_imp.update_traces(line=dict(color="#38bdf8"), fillcolor="rgba(56, 189, 248, 0.35)")
                fig_imp.update_layout(xaxis=dict(tickangle=-35))
                fig_imp = apply_cyber_theme(fig_imp, height=440, show_legend=False)

            else:  # 🍩 Donut Share
                fig_imp = px.pie(
                    df_imp,
                    names="Feature",
                    values="Importance",
                    hole=0.55,
                    color_discrete_sequence=CYBER_PALETTE,
                )
                fig_imp.update_traces(textposition="inside", textinfo="percent")
                fig_imp = apply_cyber_theme(fig_imp, height=440, legend_orientation="v")

            st.plotly_chart(fig_imp, use_container_width=True)

        with col_domain:
            st.markdown("#### 🛡️ Cybersecurity Domain Insights")
            st.caption("Physical meaning of top discriminative network features:")
            for name, score in active_imps[:6]:
                with st.expander(f"🔹 **{name}** (Gini: {score:.4f})", expanded=True):
                    st.write(get_feature_explanation(name))

    st.markdown("---")

    # ==========================================
    # SECTION 2: Feature Contrast Across Threat Categories
    # ==========================================
    contrast_chart_type = render_chart_header(
        title="🔬 Feature Contrast Across Threat Categories",
        chart_types=["📦 Box Plot", "📊 Bar (Median Value)", "📈 Profile Line (Median)", "🌊 Area Envelope"],
        key="feat_contrast_chart_type",
        default="📦 Box Plot",
        help_text="Inspect how selected discriminative features vary across the 8 threat taxonomy families.",
    )

    top_feature_names = [f[0] for f in active_imps[:20]]
    selected_feat = st.selectbox("Select Feature to Contrast", top_feature_names, key="feat_contrast_sel")

    if contrast_chart_type == "📦 Box Plot":
        fig_contrast = px.box(
            df_sample,
            x="Threat_Category",
            y=selected_feat,
            color="Threat_Category",
            log_y=True,
            color_discrete_map=CATEGORY_COLORS,
            title=f"Distribution of '{selected_feat}' across Threat Categories (Log Scale)",
        )
        fig_contrast.update_layout(xaxis=dict(tickangle=-25))
        fig_contrast = apply_cyber_theme(fig_contrast, height=420, show_legend=False)

    elif contrast_chart_type == "📊 Bar (Median Value)":
        cat_agg = df_sample.groupby("Threat_Category")[selected_feat].median().reset_index()
        fig_contrast = px.bar(
            cat_agg,
            x="Threat_Category",
            y=selected_feat,
            color="Threat_Category",
            log_y=True,
            color_discrete_map=CATEGORY_COLORS,
            text=selected_feat,
            title=f"Median '{selected_feat}' across Threat Categories (Log Scale)",
        )
        fig_contrast.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig_contrast.update_layout(xaxis=dict(tickangle=-25))
        fig_contrast = apply_cyber_theme(fig_contrast, height=420, show_legend=False)

    elif contrast_chart_type == "📈 Profile Line (Median)":
        cat_agg = df_sample.groupby("Threat_Category")[selected_feat].median().reset_index()
        fig_contrast = px.line(
            cat_agg,
            x="Threat_Category",
            y=selected_feat,
            markers=True,
            log_y=True,
            title=f"Median '{selected_feat}' Profile across Threat Categories (Log Scale)",
        )
        fig_contrast.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=9, color="#ef4444"))
        fig_contrast.update_layout(xaxis=dict(tickangle=-25))
        fig_contrast = apply_cyber_theme(fig_contrast, height=420, show_legend=False)

    else:  # 🌊 Area Envelope
        cat_agg = df_sample.groupby("Threat_Category")[selected_feat].median().reset_index()
        fig_contrast = px.area(
            cat_agg,
            x="Threat_Category",
            y=selected_feat,
            log_y=True,
            title=f"Median Area Envelope for '{selected_feat}'",
        )
        fig_contrast.update_traces(line=dict(color="#38bdf8"), fillcolor="rgba(56, 189, 248, 0.35)")
        fig_contrast.update_layout(xaxis=dict(tickangle=-25))
        fig_contrast = apply_cyber_theme(fig_contrast, height=420, show_legend=False)

    st.plotly_chart(fig_contrast, use_container_width=True)
