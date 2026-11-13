"""
Feature Importance & Cybersecurity Explainability View.
Interactive visualization of network decision boundaries and cyber domain explanations.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from src.model_trainer import load_saved_model
from src.explainability import plot_feature_importance, get_feature_explanation
from src.config import BENIGN_LABEL


def render_feature_importance_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🧬 Feature Importance & Threat Explainability")
    st.caption("Understand what network flow artifacts drive anomaly detection decisions.")

    # Load active model
    artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")
    if artifact is None:
        artifact = st.session_state.get("model_binary") or load_saved_model("binary")

    if artifact is None:
        st.warning("Please train or load a model from the ML Evaluation view to inspect feature importances.")
        return

    results = artifact["results"]
    feat_imps = results.get("feature_importances", [])

    c1, c2 = st.columns([1.5, 1])
    with c1:
        top_k = st.slider("Number of Top Features to Display", 10, 30, 15, step=5)
        fig_imp = plot_feature_importance(feat_imps, top_n=top_k)
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        st.markdown("#### 🛡️ Cybersecurity Domain Insights")
        st.caption("Physical meaning of top discriminative network features:")
        
        # Display domain explanations for top 5 features
        for name, score in feat_imps[:6]:
            with st.expander(f"🔹 **{name}** (Importance: {score:.4f})", expanded=True):
                st.write(get_feature_explanation(name))

    st.markdown("---")

    # Feature Distribution Deep Dive
    st.markdown("#### 🔬 Flow Feature Contrast: Benign vs Attacks")
    top_feature_names = [f[0] for f in feat_imps[:15]]
    selected_feat = st.selectbox("Select Feature to Compare", top_feature_names)

    # Boxplot comparison
    fig_box = px.box(
        df_sample,
        x="Threat_Category",
        y=selected_feat,
        color="Threat_Category",
        log_y=True,
        title=f"Distribution of '{selected_feat}' across Threat Categories (Log Scale)",
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b", title=""),
        yaxis=dict(gridcolor="#1e293b", title=f"{selected_feat} (Log)"),
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True)
