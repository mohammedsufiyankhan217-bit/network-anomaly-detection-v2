"""
Reporting and CSV export generator for Network Anomaly Detection V2.
Produces forensic audit logs and model validation reports.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config import REPORTS_DIR


def generate_detection_report(
    results_df: pd.DataFrame,
    report_name: str = "forensic_detection_audit",
) -> Path:
    """
    Format and export detection results to a standardized CSV audit log.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_name}_{timestamp}.csv"
    out_path = REPORTS_DIR / filename

    # Select key forensic columns
    priority_cols = [
        "Predicted_Label",
        "Confidence",
        "Anomaly_Score",
        "Risk_Level",
        "Is_Anomaly",
    ]
    if "Label" in results_df.columns:
        priority_cols.insert(0, "Label")

    network_cols = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
    ]

    export_cols = [c for c in priority_cols if c in results_df.columns]
    for c in network_cols:
        if c in results_df.columns and c not in export_cols:
            export_cols.append(c)

    # Append any remaining columns up to reasonable width
    remaining = [c for c in results_df.columns if c not in export_cols]
    export_cols.extend(remaining[:15])

    export_df = results_df[export_cols].copy()
    export_df.to_csv(out_path, index=False)
    return out_path


def format_summary_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert model results dictionary into tabular DataFrame for export.
    """
    metrics = results.get("metrics", {})
    data = [
        {"Metric": "Model Type", "Value": results.get("model_type")},
        {"Metric": "Target Mode", "Value": results.get("target_mode")},
        {"Metric": "Dataset Used", "Value": f"CIC-IDS2017 ({results.get('total_rows_used', 0):,} rows)"},
        {"Metric": "Training Rows", "Value": f"{results.get('train_rows', 0):,}"},
        {"Metric": "Test Rows", "Value": f"{results.get('test_rows', 0):,}"},
        {"Metric": "Features Used", "Value": str(results.get("n_features", 0))},
        {"Metric": "Accuracy", "Value": f"{metrics.get('accuracy', 0.0) * 100:.2f}%"},
        {"Metric": "Balanced Accuracy", "Value": f"{metrics.get('balanced_accuracy', 0.0) * 100:.2f}%"},
        {"Metric": "Precision (Weighted)", "Value": f"{metrics.get('precision_weighted', 0.0) * 100:.2f}%"},
        {"Metric": "Recall (Weighted)", "Value": f"{metrics.get('recall_weighted', 0.0) * 100:.2f}%"},
        {"Metric": "F1-Score (Weighted)", "Value": f"{metrics.get('f1_weighted', 0.0) * 100:.2f}%"},
        {"Metric": "Precision (Macro)", "Value": f"{metrics.get('precision_macro', 0.0) * 100:.2f}%"},
        {"Metric": "Recall (Macro)", "Value": f"{metrics.get('recall_macro', 0.0) * 100:.2f}%"},
        {"Metric": "F1-Score (Macro)", "Value": f"{metrics.get('f1_macro', 0.0) * 100:.2f}%"},
        {"Metric": "Training Time", "Value": f"{results.get('train_duration_sec', 0)} seconds"},
        {"Metric": "Evaluation Time", "Value": f"{results.get('eval_duration_sec', 0)} seconds"},
    ]
    df = pd.DataFrame(data)
    df["Value"] = df["Value"].astype(str)
    return df
