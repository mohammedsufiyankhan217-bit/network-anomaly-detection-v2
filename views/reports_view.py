"""
Forensic Reports & CSV Export View for Network Anomaly Detection V2.
Generates exportable compliance reports, audit logs, and benchmark summaries.
"""
from typing import Dict, Any
import streamlit as st
import pandas as pd

from src.reporting import generate_detection_report, format_summary_metrics
from src.model_trainer import load_saved_model


def render_reports_view(df_sample: pd.DataFrame, status: Dict[str, Any]):
    st.markdown("### 📄 Security Reports & Compliance Exports")
    st.caption("Generate forensic CSV audit logs and empirical model validation summaries.")

    tab1, tab2 = st.tabs(["📥 Detection Audit Log Export", "📊 Model Validation Summary"])

    with tab1:
        st.markdown("#### Export Forensic Network Flow Audit Log")
        
        # Check if detection batch exists in session
        results_df = st.session_state.get("detection_batch")
        if results_df is None:
            # Fallback to sample predictions
            artifact = st.session_state.get("model_multiclass") or load_saved_model("multiclass")
            if artifact is not None:
                from src.evaluator import run_batch_detection
                results_df = run_batch_detection(artifact, df_sample.head(500))
                st.session_state["detection_batch"] = results_df

        if results_df is not None:
            st.info(f"Ready to export audit log containing **{len(results_df):,}** evaluated flows.")
            
            # CSV Download
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Detection Audit Log (CSV)",
                data=csv_data,
                file_name="cicids2017_detection_audit_log.csv",
                mime="text/csv",
                type="primary",
                key="rep_download_log_btn",
            )

            # Preview
            st.markdown("**Report Snapshot Preview:**")
            st.dataframe(results_df.head(20), use_container_width=True)
        else:
            st.warning("No detection runs found. Run detection in the Live Detection view first.")

    with tab2:
        st.markdown("#### Model Benchmark & Governance Report")
        rep_model_mode = st.radio(
            "Select Model Architecture for Benchmark Export:",
            ["multiclass", "binary"],
            format_func=lambda x: "Multi-Class Threat Classifier (15 Vectors)" if x == "multiclass" else "Binary Anomaly Detector",
            horizontal=True,
            key="rep_bench_model_radio",
        )
        artifact = st.session_state.get(f"model_{rep_model_mode}") or load_saved_model(rep_model_mode)
        if artifact is not None:
            summary_df = format_summary_metrics(artifact["results"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            csv_metrics = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"💾 Download {rep_model_mode.capitalize()} Model Benchmark Summary (CSV)",
                data=csv_metrics,
                file_name=f"model_benchmark_summary_{rep_model_mode}.csv",
                mime="text/csv",
                key="rep_download_bench_btn",
            )
        else:
            st.info("Train a model to generate governance and benchmark reports.")
