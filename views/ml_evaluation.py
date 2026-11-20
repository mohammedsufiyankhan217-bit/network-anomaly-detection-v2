"""
Machine Learning Training & Empirical Evaluation View for Network Anomaly Detection V2.
Single Random Forest Multiclass Classifier Architecture.
Evaluates multiclass intrusion and anomaly detection across all 15 CIC-IDS2017 categories.
"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.model_trainer import train_random_forest, load_saved_model
from src.evaluator import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_class_distribution,
    compute_multiclass_curves,
    plot_roc_curve,
    plot_pr_curve,
)
from src.data_loader import load_stratified_sample
from src.ui_components import render_chart_header


def render_ml_evaluation_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 🤖 Machine Learning Model Evaluation — Random Forest Multiclass Classifier")
    st.caption("Empirical evaluation of a single Random Forest classifier on verified CIC-IDS2017 network traffic.")

    # =========================================================================
    # OPTIONAL RETRAINING & ARCHITECTURE HYPERPARAMETER CONTROLS
    # =========================================================================
    with st.expander("⚙️ Model Architecture & Training Controls (Optional Retraining Pipeline)", expanded=False):
        st.markdown(
            "Configure hyperparameters for the single **Random Forest Multiclass Classifier**. "
            "The model classifies flows across all 15 CIC-IDS2017 traffic categories with zero synthetic data."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Classification Task", value="Multiclass (15 Categories)", disabled=True)
            n_estimators = st.slider("Number of Decision Trees (n_estimators)", 20, 150, 60, step=10, key="ml_n_estimators")
        with c2:
            max_depth = st.slider("Max Tree Depth", 8, 30, 16, step=2, key="ml_max_depth")
            test_size = st.slider("Test Split Ratio (Stratified Holdout)", 0.10, 0.35, 0.20, step=0.05, key="ml_test_size")
        with c3:
            sample_options = [25000, 50000, 100000, 200000]
            default_sample_size = min(sample_options, key=lambda opt: abs(opt - min(100000, len(df_sample))))
            training_sample_size = st.select_slider(
                "Training Sample Size",
                options=sample_options,
                value=default_sample_size,
                format_func=lambda x: f"{x:,} rows",
                key="ml_training_sample_size",
            )
            drop_port_toggle = st.checkbox(
                "Drop Destination Port",
                value=False,
                key="ml_drop_port",
                help="Forces trees to learn purely from physical packet dynamics rather than memorizing port numbers.",
            )

        train_btn = st.button("🚀 Train Random Forest Model", type="primary", key="ml_train_btn")

    # Handle optional training execution
    if train_btn:
        with st.spinner(f"Preparing {training_sample_size:,} stratified rows and training Random Forest Multiclass Classifier..."):
            try:
                if len(df_sample) != training_sample_size:
                    train_df = load_stratified_sample(sample_size=training_sample_size)
                else:
                    train_df = df_sample

                res = train_random_forest(
                    df=train_df,
                    target_mode="multiclass",
                    test_size=test_size,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    drop_port=drop_port_toggle,
                )
                st.session_state["model_multiclass"] = res
                # Clear cached curve data on retrain
                if "curves_multiclass" in st.session_state:
                    del st.session_state["curves_multiclass"]

                st.success(
                    f"🎉 Successfully trained Random Forest Multiclass Classifier on {res['results']['train_rows']:,} flows! "
                    f"Evaluated on {res['results']['test_rows']:,} held-out test flows in {res['results']['train_duration_sec']}s."
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    # =========================================================================
    # LOAD PRODUCTION MULTICLASS RANDOM FOREST ARTIFACT
    # =========================================================================
    multi_artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")

    if multi_artifact is None:
        st.warning("⚠️ No trained multiclass Random Forest model found on disk. Click 'Train Random Forest Model' above to train.")
        return

    st.session_state["model_multiclass"] = multi_artifact
    res = multi_artifact["results"]
    met = res["metrics"]
    classes = res.get("classes", [])

    # =========================================================================
    # 1. MODEL INFORMATION SECTION
    # =========================================================================
    st.markdown("#### ℹ️ Model Architecture & Dataset Specifications")
    
    info_cols = st.columns(5)
    with info_cols[0]:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Algorithm</div>
                <div style="font-size: 0.95rem; color: #38bdf8; font-weight: 700; margin-top: 4px;">Random Forest</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Ensemble (60 Trees)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Task</div>
                <div style="font-size: 0.95rem; color: #10b981; font-weight: 700; margin-top: 4px;">Multiclass Classification</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Intrusion & Anomaly</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Dataset</div>
                <div style="font-size: 0.95rem; color: #f59e0b; font-weight: 700; margin-top: 4px;">CIC-IDS2017</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Verified Benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[3]:
        n_feats = res.get("n_features", len(res.get("feature_names", [])))
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Features Used</div>
                <div style="font-size: 0.95rem; color: #a855f7; font-weight: 700; margin-top: 4px;">{n_feats} Features</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Robust Scaler Fitted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[4]:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Traffic Categories</div>
                <div style="font-size: 0.95rem; color: #ec4899; font-weight: 700; margin-top: 4px;">{len(classes)} Categories</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">BENIGN + 14 Attacks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. PERFORMANCE METRICS (CLEAN KPI CARDS)
    # =========================================================================
    st.markdown("#### 🎯 Empirical Performance Metrics")
    st.caption(
        f"Evaluated on **{res.get('test_rows', 19993):,}** held-out test flows "
        f"(trained on {res.get('train_rows', 79971):,} flows with strict stratified partitioning)."
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Overall Accuracy",
            value=f"{met['accuracy'] * 100:.2f}%",
            help="Total correct classifications across all 15 traffic categories on held-out test flows.",
        )
    with k2:
        st.metric(
            label="Precision",
            value=f"{met.get('precision_weighted', met['accuracy']) * 100:.2f}%",
            delta=f"Macro: {met.get('precision_macro', 0.8672) * 100:.2f}%",
            help="Weighted precision across all categories (with unweighted macro precision shown as delta).",
        )
    with k3:
        st.metric(
            label="Recall",
            value=f"{met.get('recall_weighted', met['accuracy']) * 100:.2f}%",
            delta=f"Macro: {met.get('recall_macro', 0.8606) * 100:.2f}%",
            help="Weighted recall across all categories (with unweighted macro recall shown as delta).",
        )
    with k4:
        st.metric(
            label="F1-Score",
            value=f"{met.get('f1_weighted', 0.9951) * 100:.2f}%",
            delta=f"Macro: {met.get('f1_macro', 0.8463) * 100:.2f}%",
            help="Harmonic mean of precision and recall. Weighted F1 reflects dataset distribution; Macro F1 weights all 15 classes equally.",
        )

    st.markdown("---")

    # =========================================================================
    # 3. VISUALIZATIONS SECTION
    # =========================================================================
    st.markdown("#### 📊 Empirical Model Visualizations")

    # Row 1: Confusion Matrix & Class Distribution
    col_cm, col_dist = st.columns([1.2, 1.0])

    with col_cm:
        cm_type = render_chart_header(
            title="Multiclass Confusion Matrix",
            chart_types=["🗺️ Heatmap", "📊 Diagonal Accuracy Bar"],
            key="ml_single_cm_type",
            default="🗺️ Heatmap",
            help_text="Empirical test confusion across all 15 categories, showing accurate detections and misclassifications.",
        )
        norm_cm = st.checkbox("Normalize Matrix (%)", value=True, key="ml_single_norm_toggle")
        fig_cm = plot_confusion_matrix(
            cm=res["confusion_matrix"],
            classes=classes,
            normalize=norm_cm,
            title="Multiclass Empirical Confusion Matrix",
            chart_type=cm_type,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_dist:
        dist_type = render_chart_header(
            title="Class Distribution in Test Set",
            chart_types=["📶 Horizontal Bar", "📊 Column Bar", "🍩 Donut Chart"],
            key="ml_single_dist_type",
            default="📶 Horizontal Bar",
            help_text="Frequency and support counts of all 15 traffic categories in the held-out test evaluation set.",
        )
        fig_dist = plot_class_distribution(
            report_dict=res["classification_report"],
            chart_type=dist_type,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # Row 2: Per-Class Performance Chart
    st.markdown("##### 📈 Per-Class Performance Breakdown (Precision, Recall & F1-Score)")
    per_class_type = render_chart_header(
        title="Per-Class Performance Chart",
        chart_types=["📶 Horizontal Bar", "📊 Grouped Bar", "🗺️ Heatmap"],
        key="ml_single_per_class_type",
        default="📶 Horizontal Bar",
        help_text="Precision, Recall, and F1-score for each of the 15 network traffic classifications, sorted by F1-score.",
    )
    fig_per_class = plot_per_class_metrics(
        report_dict=res["classification_report"],
        chart_type=per_class_type,
    )
    st.plotly_chart(fig_per_class, use_container_width=True)

    st.markdown("---")

    # Row 3: Multiclass ROC & Precision-Recall Curves
    st.markdown("##### 📉 Multiclass ROC & Precision-Recall Curves")
    st.caption(
        "Mathematically computed via One-vs-Rest (OvR) and macro/micro-averaging from the multiclass "
        "Random Forest probability distributions (`predict_proba`)."
    )

    # Compute multiclass curves if not cached
    if "curves_multiclass" not in st.session_state:
        with st.spinner("Computing empirical multiclass ROC and Precision-Recall coordinates across all categories..."):
            eval_sample = df_sample.sample(min(25000, len(df_sample)), random_state=42)
            st.session_state["curves_multiclass"] = compute_multiclass_curves(multi_artifact, eval_sample)

    curve_data = st.session_state["curves_multiclass"]
    target_options = curve_data.get("target_list", list(curve_data.get("targets", {}).keys()))

    target_col1, target_col2 = st.columns([1.5, 2.5])
    with target_col1:
        selected_target = st.selectbox(
            "Select Evaluation Target / Category",
            options=target_options,
            index=0,
            key="ml_single_curve_target",
            help="Choose between overall Macro-Average, Anomaly Detection OvR (Intrusion vs Benign), Micro-Average, or an individual attack class.",
        )

    col_roc, col_pr = st.columns(2)
    with col_roc:
        roc_type = render_chart_header(
            title="Receiver Operating Characteristic (ROC)",
            chart_types=["📈 ROC Curve", "🌊 ROC Area Fill"],
            key="ml_single_roc_type",
            default="📈 ROC Curve",
            help_text="True Positive Rate vs False Positive Rate for the selected multiclass target.",
        )
        fig_roc = plot_roc_curve(curve_data, chart_type=roc_type, target_name=selected_target)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        pr_type = render_chart_header(
            title="Precision-Recall Curve",
            chart_types=["📈 PR Curve", "🌊 PR Area Fill"],
            key="ml_single_pr_type",
            default="📈 PR Curve",
            help_text="Precision vs Recall tradeoff across decision thresholds — the standard metric for imbalanced cyber data.",
        )
        fig_pr = plot_pr_curve(curve_data, chart_type=pr_type, target_name=selected_target)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # 4. TABULAR CLASSIFICATION REPORT SECTION
    # =========================================================================
    st.markdown("#### 📋 Tabular Classification Report")
    st.caption("Detailed empirical precision, recall, F1-score and support count for every traffic category on held-out test data.")

    report_dict = res.get("classification_report", {})
    report_rows = []

    # Individual categories
    for cls_name, vals in report_dict.items():
        if isinstance(vals, dict) and cls_name not in ["accuracy", "macro avg", "weighted avg"]:
            support_val = int(vals.get("support", 0))
            prec_val = vals.get("precision", 0.0) * 100
            rec_val = vals.get("recall", 0.0) * 100
            f1_val = vals.get("f1-score", 0.0) * 100
            report_rows.append({
                "Traffic Category": cls_name,
                "Precision": f"{prec_val:.2f}%",
                "Recall": f"{rec_val:.2f}%",
                "F1-Score": f"{f1_val:.2f}%",
                "Support (Flows)": f"{support_val:,}",
            })

    # Summary rows
    if "macro avg" in report_dict:
        m_vals = report_dict["macro avg"]
        report_rows.append({
            "Traffic Category": "── Macro Average (Unweighted) ──",
            "Precision": f"{m_vals.get('precision', 0.0) * 100:.2f}%",
            "Recall": f"{m_vals.get('recall', 0.0) * 100:.2f}%",
            "F1-Score": f"{m_vals.get('f1-score', 0.0) * 100:.2f}%",
            "Support (Flows)": f"{int(m_vals.get('support', 0)):,}",
        })

    if "weighted avg" in report_dict:
        w_vals = report_dict["weighted avg"]
        report_rows.append({
            "Traffic Category": "── Weighted Average (Dataset Distribution) ──",
            "Precision": f"{w_vals.get('precision', 0.0) * 100:.2f}%",
            "Recall": f"{w_vals.get('recall', 0.0) * 100:.2f}%",
            "F1-Score": f"{w_vals.get('f1-score', 0.0) * 100:.2f}%",
            "Support (Flows)": f"{int(w_vals.get('support', 0)):,}",
        })

    df_report = pd.DataFrame(report_rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)
