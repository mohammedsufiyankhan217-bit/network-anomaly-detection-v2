"""
Evaluation, inference, and visualization utilities for Network Anomaly Detection V2.
Produces interactive Plotly cyber-themed visualizations and real detection results.
"""
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.config import RISK_LEVELS, CATEGORY_COLORS, BENIGN_LABEL


def run_batch_detection(
    model_artifact: Dict[str, Any],
    df_flows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run detection on a set of network flows using a trained model artifact.
    Extracts predictions, probability confidence, risk scores, and alert tiers.
    """
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_names = model_artifact["results"]["feature_names"]
    target_mode = model_artifact["results"]["target_mode"]

    # Ensure required features exist in df_flows
    missing_feats = [f for f in feature_names if f not in df_flows.columns]
    if missing_feats:
        raise ValueError(f"Input data is missing {len(missing_feats)} features required by model: {missing_feats[:5]}")

    X = df_flows[feature_names].copy()
    X_scaled = scaler.transform(X)

    # Predictions
    preds = model.predict(X_scaled)
    results_df = df_flows.copy()
    results_df["Predicted_Label"] = preds

    # Confidence scores
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)
        classes = list(model.classes_)
        # Maximum probability
        max_probas = np.max(probas, axis=1)
        results_df["Confidence"] = np.round(max_probas, 4)

        # Anomaly probability
        if target_mode == "binary":
            attack_idx = classes.index("ATTACK") if "ATTACK" in classes else 1
            anomaly_scores = probas[:, attack_idx]
        else:
            if BENIGN_LABEL in classes:
                benign_idx = classes.index(BENIGN_LABEL)
                anomaly_scores = 1.0 - probas[:, benign_idx]
            else:
                anomaly_scores = np.ones(len(preds))
        results_df["Anomaly_Score"] = np.round(anomaly_scores, 4)
    else:
        results_df["Confidence"] = 1.0
        results_df["Anomaly_Score"] = np.where(results_df["Predicted_Label"] == BENIGN_LABEL, 0.0, 1.0)

    # Assign Risk Level
    def assign_risk(score: float, pred: str) -> str:
        if pred == BENIGN_LABEL or pred == "BENIGN":
            return "LOW"
        if score >= RISK_LEVELS["CRITICAL"]["min_score"]:
            return "CRITICAL"
        if score >= RISK_LEVELS["HIGH"]["min_score"]:
            return "HIGH"
        if score >= RISK_LEVELS["MEDIUM"]["min_score"]:
            return "MEDIUM"
        return "LOW"

    results_df["Risk_Level"] = [
        assign_risk(score, pred)
        for score, pred in zip(results_df["Anomaly_Score"], results_df["Predicted_Label"])
    ]
    results_df["Is_Anomaly"] = results_df["Predicted_Label"] != BENIGN_LABEL

    return results_df


def plot_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    normalize: bool = False,
    title: str = "Empirical Test Confusion Matrix",
) -> go.Figure:
    """
    Generate an interactive Plotly confusion matrix in dark cybersecurity aesthetic.
    """
    cm_arr = np.array(cm)
    if normalize:
        with np.errstate(all="ignore"):
            cm_arr = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
            cm_arr = np.nan_to_num(cm_arr)
        z_values = np.round(cm_arr, 3)
        text_values = [[f"{val:.1%}" for val in row] for row in z_values]
    else:
        z_values = cm_arr
        text_values = [[f"{int(val):,}" for val in row] for row in z_values]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=classes,
            y=classes,
            text=text_values,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#0b0f19"],
                [0.15, "#1e293b"],
                [0.4, "#0369a1"],
                [0.7, "#0284c7"],
                [1.0, "#38bdf8"],
            ],
            colorbar=dict(
                title=dict(text="Share" if normalize else "Count", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=15)),
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8", size=10),
            tickangle=-35,
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title=dict(text="Actual True Label", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8", size=10),
            autorange="reversed",
            gridcolor="#1e293b",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=50, b=80),
        height=520,
    )
    return fig


def plot_metric_kpi_cards(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format empirical metrics for UI display.
    """
    return {
        "Accuracy": f"{metrics.get('accuracy', 0.0) * 100:.2f}%",
        "Balanced Accuracy": f"{metrics.get('balanced_accuracy', 0.0) * 100:.2f}%",
        "Precision (Weighted)": f"{metrics.get('precision_weighted', 0.0) * 100:.2f}%",
        "Recall (Weighted)": f"{metrics.get('recall_weighted', 0.0) * 100:.2f}%",
        "F1-Score (Weighted)": f"{metrics.get('f1_weighted', 0.0) * 100:.2f}%",
        "F1-Score (Macro)": f"{metrics.get('f1_macro', 0.0) * 100:.2f}%",
    }
