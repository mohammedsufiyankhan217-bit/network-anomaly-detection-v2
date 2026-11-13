"""
Machine Learning Training & Empirical Evaluation View.
Configurable Random Forest training with zero data leakage and empirical validation.
"""
import streamlit as st
import pandas as pd

from src.model_trainer import train_random_forest, load_saved_model
from src.evaluator import plot_confusion_matrix, plot_metric_kpi_cards
from src.data_loader import load_stratified_sample


def render_ml_evaluation_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🤖 Machine Learning Model Training & Empirical Evaluation")
    st.caption("Train and evaluate Random Forest models on verified CIC-IDS2017 network flows.")

    # Model Configuration & Training Controls
    with st.expander("⚙️ Model Architecture & Training Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            target_mode = st.selectbox(
                "Classification Mode",
                ["multiclass", "binary"],
                format_func=lambda x: "Multi-Class (15 Attack Categories)" if x == "multiclass" else "Binary (Benign vs Attack)",
                index=0,
            )
            n_estimators = st.slider("Number of Trees (n_estimators)", 20, 150, 60, step=10)
        with c2:
            max_depth = st.slider("Max Tree Depth", 8, 30, 16, step=2)
            test_size = st.slider("Test Split Ratio (Holdout)", 0.10, 0.35, 0.20, step=0.05)
        with c3:
            sample_options = [25000, 50000, 100000, 200000]
            # Ensure default value always exists in sample_options and safely handles any dataset size
            default_sample_size = min(sample_options, key=lambda opt: abs(opt - min(100000, len(df_sample))))
            training_sample_size = st.select_slider(
                "Training Sample Size",
                options=sample_options,
                value=default_sample_size,
                format_func=lambda x: f"{x:,} rows",
            )
            drop_port_toggle = st.checkbox("Drop Destination Port (Zero Service Memorization)", value=False)

        train_btn = st.button("🚀 Train Random Forest Model", type="primary")

    # Load existing or trigger training
    if "current_model_mode" not in st.session_state:
        st.session_state["current_model_mode"] = "multiclass"

    # Handle training
    if train_btn:
        with st.spinner(f"Preparing {training_sample_size:,} stratified rows and training Random Forest..."):
            try:
                # Load sample of requested size
                if len(df_sample) != training_sample_size:
                    train_df = load_stratified_sample(sample_size=training_sample_size)
                else:
                    train_df = df_sample

                if len(train_df) < training_sample_size:
                    st.info(f"Available data contains {len(train_df):,} rows. Proceeding with all available rows.")

                res = train_random_forest(
                    df=train_df,
                    target_mode=target_mode,
                    test_size=test_size,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    drop_port=drop_port_toggle,
                )
                st.session_state[f"model_{target_mode}"] = res
                st.session_state["current_model_mode"] = target_mode
                st.success(
                    f"🎉 Successfully trained Random Forest on {res['results']['train_rows']:,} rows! "
                    f"Evaluated on {res['results']['test_rows']:,} unseen test flows in {res['results']['train_duration_sec']}s."
                )
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Mode Selector for viewing results
    view_mode = st.radio(
        "Active Model View:",
        ["multiclass", "binary"],
        format_func=lambda x: "Multi-Class Classifier (15 Vectors)" if x == "multiclass" else "Binary Anomaly Classifier",
        horizontal=True,
        index=0 if st.session_state["current_model_mode"] == "multiclass" else 1,
    )

    # Fetch model artifact
    artifact = st.session_state.get(f"model_{view_mode}")
    if artifact is None:
        artifact = load_saved_model(target_mode=view_mode)
        if artifact is not None:
            st.session_state[f"model_{view_mode}"] = artifact

    if artifact is None:
        st.info("No trained model loaded yet. Click 'Train Random Forest Model' above to train on the real dataset.")
        return

    results = artifact["results"]
    metrics = results["metrics"]

    # Transparency & Disclosure Card
    st.markdown("---")
    st.info(
        f"📊 **Verified Dataset Disclosure**: Model trained on **{results['total_rows_used']:,}** real CIC-IDS2017 flows "
        f"({results['train_rows']:,} train / {results['test_rows']:,} test) across **{results['n_features']}** network features. "
        f"Zero synthetic data. Stratified split with robust feature scaling."
    )

    # Metric KPI Cards
    st.markdown("#### 🏆 Empirical Test Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with c2:
        st.metric("Balanced Accuracy", f"{metrics['balanced_accuracy'] * 100:.2f}%")
    with c3:
        st.metric("Precision (Weighted)", f"{metrics['precision_weighted'] * 100:.2f}%")
    with c4:
        st.metric("F1-Score (Weighted)", f"{metrics['f1_weighted'] * 100:.2f}%")

    st.markdown("---")

    # Confusion Matrix & Classification Report
    col_cm, col_report = st.columns([1.2, 1])

    with col_cm:
        st.markdown("#### 🎯 Confusion Matrix")
        norm_toggle = st.checkbox("Normalize by True Label (%)", value=True)
        cm_fig = plot_confusion_matrix(
            cm=results["confusion_matrix"],
            classes=results["classes"],
            normalize=norm_toggle,
            title=f"Actual vs Predicted ({view_mode.capitalize()} Test Set)",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_report:
        st.markdown("#### 📋 Per-Class Performance Breakdown")
        report_dict = results["classification_report"]
        report_rows = []
        for cls_name, vals in report_dict.items():
            if isinstance(vals, dict):
                report_rows.append({
                    "Class / Label": cls_name,
                    "Precision": f"{vals.get('precision', 0.0) * 100:.1f}%",
                    "Recall": f"{vals.get('recall', 0.0) * 100:.1f}%",
                    "F1-Score": f"{vals.get('f1-score', 0.0) * 100:.1f}%",
                    "Support": f"{int(vals.get('support', 0)):,}",
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
