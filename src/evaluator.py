"""
Evaluation, inference, and visualization utilities for Network Anomaly Detection V2.
Produces interactive Plotly cyber-themed visualizations and real detection results.
"""
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.config import RISK_LEVELS, CATEGORY_COLORS, BENIGN_LABEL, ATTACK_CATEGORIES
from src.ui_components import apply_cyber_theme, CYBER_PALETTE


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
    results_df["Predicted_Threat_Category"] = results_df["Predicted_Label"].map(
        lambda x: ATTACK_CATEGORIES.get(x, "Other Attacks" if x != "BENIGN" else "Benign Traffic")
    )

    return results_df


def run_dual_detection(
    binary_artifact: Dict[str, Any],
    multiclass_artifact: Dict[str, Any],
    df_flows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run detection using BOTH the binary anomaly detector and multiclass threat classifier
    on the exact same network flows to enable comparative forensics and consensus analysis.
    """
    # 1. Binary inference
    df_bin = run_batch_detection(binary_artifact, df_flows)
    
    # 2. Multiclass inference
    df_multi = run_batch_detection(multiclass_artifact, df_flows)

    dual_df = df_flows.copy()
    dual_df["Binary_Pred"] = df_bin["Predicted_Label"]
    dual_df["Binary_Conf"] = df_bin["Confidence"]
    dual_df["Binary_Anomaly_Score"] = df_bin["Anomaly_Score"]
    dual_df["Binary_Risk"] = df_bin["Risk_Level"]

    dual_df["Multi_Pred"] = df_multi["Predicted_Label"]
    dual_df["Multi_Conf"] = df_multi["Confidence"]
    dual_df["Multi_Anomaly_Score"] = df_multi["Anomaly_Score"]
    dual_df["Multi_Risk"] = df_multi["Risk_Level"]
    dual_df["Multi_Category"] = df_multi["Predicted_Threat_Category"]

    # Consensus determination
    def determine_consensus(bin_p: str, mul_p: str) -> str:
        bin_attack = (bin_p == "ATTACK")
        mul_attack = (mul_p != BENIGN_LABEL and mul_p != "BENIGN")

        if not bin_attack and not mul_attack:
            return "Consensus Benign (Both Clean)"
        elif bin_attack and mul_attack:
            return f"Consensus Intrusion ({mul_p})"
        elif bin_attack and not mul_attack:
            return "Divergent: Binary Anomaly Only"
        else:
            return "Divergent: Multiclass Specific Only"

    dual_df["Consensus"] = [
        determine_consensus(b, m)
        for b, m in zip(dual_df["Binary_Pred"], dual_df["Multi_Pred"])
    ]
    dual_df["Agreed"] = (
        (dual_df["Binary_Pred"] == "ATTACK") == (dual_df["Multi_Pred"] != BENIGN_LABEL)
    )

    return dual_df


def compute_binary_curves(
    binary_artifact: Dict[str, Any],
    df_eval: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute genuine ROC and Precision-Recall curve coordinates using the actual
    trained binary Random Forest model and real CIC-IDS2017 stratified flows.
    """
    model = binary_artifact["model"]
    scaler = binary_artifact["scaler"]
    feature_names = binary_artifact["results"]["feature_names"]

    X = df_eval[feature_names].copy()
    X_scaled = scaler.transform(X)

    # Actual binary ground truth (1 for ATTACK, 0 for BENIGN)
    if "Binary_Label" in df_eval.columns:
        y_true = (df_eval["Binary_Label"] == "ATTACK").astype(int)
    else:
        y_true = (df_eval["Label"] != BENIGN_LABEL).astype(int)

    classes = list(model.classes_)
    attack_idx = classes.index("ATTACK") if "ATTACK" in classes else 1
    y_scores = model.predict_proba(X_scaled)[:, attack_idx]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_val = float(auc(fpr, tpr))

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_val = float(average_precision_score(y_true, y_scores))

    # Downsample points for ultra-smooth interactive web rendering if > 1500 points
    if len(fpr) > 1000:
        indices = np.linspace(0, len(fpr) - 1, 600, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
    if len(precision) > 1000:
        indices = np.linspace(0, len(precision) - 1, 600, dtype=int)
        precision = precision[indices]
        recall = recall[indices]

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc_val,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_auc": pr_auc_val,
        "test_flows": len(df_eval),
        "attack_flows": int(y_true.sum()),
        "benign_flows": int((1 - y_true).sum()),
    }


def compute_multiclass_curves(
    multiclass_artifact: Dict[str, Any],
    df_eval: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute genuine One-vs-Rest (OvR) ROC and Precision-Recall curve coordinates
    for all classes, plus Macro-Average, Micro-Average, and Anomaly OvR using
    the actual trained multiclass Random Forest model on verified CIC-IDS2017 flows.
    """
    model = multiclass_artifact["model"]
    scaler = multiclass_artifact["scaler"]
    feature_names = multiclass_artifact["results"]["feature_names"]
    classes = list(model.classes_)

    # Extract & scale features
    X = df_eval[feature_names].copy()
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)

    # Ground truth labels
    y_true = df_eval["Label"].copy()
    y_bin = label_binarize(y_true, classes=classes)

    targets = {}
    target_list = [
        "🌐 Macro-Average (All Categories)",
        "🛡️ Anomaly Detection (Intrusion vs Benign OvR)",
        "🔬 Micro-Average (Overall Test Set)",
    ]

    # 1. Macro-Average calculation (interpolated across common evaluation grids)
    mean_fpr = np.linspace(0, 1, 300)
    tprs = []
    mean_recall = np.linspace(0, 1, 300)
    precisions = []

    for i in range(len(classes)):
        if y_bin[:, i].sum() > 0:
            f, t, _ = roc_curve(y_bin[:, i], probas[:, i])
            tprs.append(np.interp(mean_fpr, f, t))
            p, r, _ = precision_recall_curve(y_bin[:, i], probas[:, i])
            precisions.append(np.interp(mean_recall, r[::-1], p[::-1]))

    mean_tpr = np.mean(tprs, axis=0) if tprs else np.linspace(0, 1, 300)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    macro_roc_auc = float(auc(mean_fpr, mean_tpr))
    mean_prec = np.mean(precisions, axis=0) if precisions else np.zeros(300)

    # Filter columns with non-zero support to avoid sklearn zero-division warnings
    valid_cols = np.where(y_bin.sum(axis=0) > 0)[0]
    macro_pr_auc = (
        float(average_precision_score(y_bin[:, valid_cols], probas[:, valid_cols], average="macro"))
        if len(valid_cols) > 0 else 0.0
    )

    targets["🌐 Macro-Average (All Categories)"] = {
        "fpr": mean_fpr.tolist(),
        "tpr": mean_tpr.tolist(),
        "roc_auc": macro_roc_auc,
        "precision": mean_prec.tolist(),
        "recall": mean_recall.tolist(),
        "pr_auc": macro_pr_auc,
        "base_rate": 1.0 / max(len(classes), 1),
    }

    # 2. Anomaly Detection OvR (Benign vs All Attacks)
    y_anom = (y_true != BENIGN_LABEL).astype(int)
    benign_idx = classes.index(BENIGN_LABEL) if BENIGN_LABEL in classes else 0
    p_anom = 1.0 - probas[:, benign_idx]
    f_anom, t_anom, _ = roc_curve(y_anom, p_anom)
    p_anom_curve, r_anom_curve, _ = precision_recall_curve(y_anom, p_anom)

    step_anom_roc = max(1, len(f_anom) // 300)
    step_anom_pr = max(1, len(p_anom_curve) // 300)

    targets["🛡️ Anomaly Detection (Intrusion vs Benign OvR)"] = {
        "fpr": f_anom[::step_anom_roc].tolist(),
        "tpr": t_anom[::step_anom_roc].tolist(),
        "roc_auc": float(auc(f_anom, t_anom)),
        "precision": p_anom_curve[::step_anom_pr].tolist(),
        "recall": r_anom_curve[::step_anom_pr].tolist(),
        "pr_auc": float(average_precision_score(y_anom, p_anom)),
        "base_rate": float(y_anom.mean()),
    }

    # 3. Micro-Average
    if len(valid_cols) > 0:
        f_micro, t_micro, _ = roc_curve(y_bin[:, valid_cols].ravel(), probas[:, valid_cols].ravel())
        p_micro, r_micro, _ = precision_recall_curve(y_bin[:, valid_cols].ravel(), probas[:, valid_cols].ravel())
        micro_pr_auc = float(average_precision_score(y_bin[:, valid_cols], probas[:, valid_cols], average="micro"))
    else:
        f_micro, t_micro = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        p_micro, r_micro = np.array([0.0, 0.0]), np.array([0.0, 1.0])
        micro_pr_auc = 0.0

    step_micro_roc = max(1, len(f_micro) // 300)
    step_micro_pr = max(1, len(p_micro) // 300)

    targets["🔬 Micro-Average (Overall Test Set)"] = {
        "fpr": f_micro[::step_micro_roc].tolist(),
        "tpr": t_micro[::step_micro_roc].tolist(),
        "roc_auc": float(auc(f_micro, t_micro)),
        "precision": p_micro[::step_micro_pr].tolist(),
        "recall": r_micro[::step_micro_pr].tolist(),
        "pr_auc": micro_pr_auc,
        "base_rate": 1.0 / max(len(classes), 1),
    }

    # 4. Individual classes
    for i, cls in enumerate(classes):
        target_label = f"Class: {cls}"
        target_list.append(target_label)
        if y_bin[:, i].sum() > 0:
            f, t, _ = roc_curve(y_bin[:, i], probas[:, i])
            p, r, _ = precision_recall_curve(y_bin[:, i], probas[:, i])
            s_roc = max(1, len(f) // 300)
            s_pr = max(1, len(p) // 300)
            targets[target_label] = {
                "fpr": f[::s_roc].tolist(),
                "tpr": t[::s_roc].tolist(),
                "roc_auc": float(auc(f, t)),
                "precision": p[::s_pr].tolist(),
                "recall": r[::s_pr].tolist(),
                "pr_auc": float(average_precision_score(y_bin[:, i], probas[:, i])),
                "base_rate": float(y_bin[:, i].mean()),
            }
        else:
            targets[target_label] = {
                "fpr": [0.0, 1.0],
                "tpr": [0.0, 1.0],
                "roc_auc": 0.5,
                "precision": [0.0, 0.0],
                "recall": [0.0, 1.0],
                "pr_auc": 0.0,
                "base_rate": 0.0,
            }

    return {
        "targets": targets,
        "target_list": target_list,
        "test_flows": len(df_eval),
        "num_classes": len(classes),
    }


def plot_roc_curve(
    curve_data: Dict[str, Any],
    chart_type: str = "📈 ROC Curve",
    target_name: Optional[str] = None,
) -> go.Figure:
    """
    Render interactive ROC Curve in cybersecurity dark aesthetic.
    Supports single binary curve data or multiclass target curve data.
    """
    if "targets" in curve_data:
        targets_dict = curve_data["targets"]
        t_name = target_name if (target_name and target_name in targets_dict) else list(targets_dict.keys())[0]
        t_data = targets_dict[t_name]
        fpr = t_data["fpr"]
        tpr = t_data["tpr"]
        roc_auc = t_data["roc_auc"]
        title_suffix = t_name
    else:
        fpr = curve_data["fpr"]
        tpr = curve_data["tpr"]
        roc_auc = curve_data["roc_auc"]
        title_suffix = "Random Forest Anomaly Detector"

    fig = go.Figure()

    # Diagonal baseline (random guess)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#64748b", dash="dash", width=1.5),
            name="Random Baseline (AUC = 0.500)",
            hoverinfo="skip",
        )
    )

    # Empirical ROC
    fill_mode = "tozeroy" if "Area" in chart_type else "none"
    fill_color = "rgba(56, 189, 248, 0.15)" if "Area" in chart_type else None

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color="#38bdf8", width=3),
            fill=fill_mode,
            fillcolor=fill_color,
            name=f"ROC (AUC = {roc_auc:.4f})",
            hovertemplate="False Alarm Rate: %{x:.4f}<br>True Positive Rate: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"ROC Curve: <b>{title_suffix}</b> — Empirical AUC: <b>{roc_auc:.4f}</b>",
            font=dict(color="#f8fafc", size=13),
        ),
        xaxis=dict(
            title="False Alarm Rate (False Positive Rate)",
            range=[-0.01, 1.01],
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title="True Detection Rate (Recall / Sensitivity)",
            range=[-0.01, 1.02],
            gridcolor="#1e293b",
        ),
    )
    return apply_cyber_theme(fig, height=420)


def plot_pr_curve(
    curve_data: Dict[str, Any],
    chart_type: str = "📈 PR Curve",
    target_name: Optional[str] = None,
) -> go.Figure:
    """
    Render interactive Precision-Recall Curve in cybersecurity dark aesthetic.
    Supports single binary curve data or multiclass target curve data.
    """
    if "targets" in curve_data:
        targets_dict = curve_data["targets"]
        t_name = target_name if (target_name and target_name in targets_dict) else list(targets_dict.keys())[0]
        t_data = targets_dict[t_name]
        precision = t_data["precision"]
        recall = t_data["recall"]
        pr_auc = t_data["pr_auc"]
        base_rate = t_data.get("base_rate", 0.1)
        title_suffix = t_name
    else:
        precision = curve_data["precision"]
        recall = curve_data["recall"]
        pr_auc = curve_data["pr_auc"]
        base_rate = curve_data["attack_flows"] / max(curve_data["test_flows"], 1)
        title_suffix = "Random Forest Anomaly Detector"

    fig = go.Figure()

    # Base prevalence baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[base_rate, base_rate],
            mode="lines",
            line=dict(color="#64748b", dash="dash", width=1.5),
            name=f"Prevalence Baseline ({base_rate:.1%})",
            hoverinfo="skip",
        )
    )

    fill_mode = "tozeroy" if "Area" in chart_type else "none"
    fill_color = "rgba(16, 185, 129, 0.15)" if "Area" in chart_type else None

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            line=dict(color="#10b981", width=3),
            fill=fill_mode,
            fillcolor=fill_color,
            name=f"Precision-Recall (Avg Prec = {pr_auc:.4f})",
            hovertemplate="Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"PR Curve: <b>{title_suffix}</b> — Avg Precision: <b>{pr_auc:.4f}</b>",
            font=dict(color="#f8fafc", size=13),
        ),
        xaxis=dict(
            title="Recall (Detection Coverage)",
            range=[-0.01, 1.01],
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title="Precision (Alert Positive Predictive Value)",
            range=[-0.01, 1.02],
            gridcolor="#1e293b",
        ),
    )
    return apply_cyber_theme(fig, height=420)


def plot_class_distribution(
    report_dict: Dict[str, Any],
    chart_type: str = "📶 Horizontal Bar",
) -> go.Figure:
    """
    Render test dataset traffic category distribution.
    Supports Horizontal Bar, Column Bar, and Donut Chart.
    """
    rows = []
    for cls_name, vals in report_dict.items():
        if isinstance(vals, dict) and cls_name not in ["accuracy", "macro avg", "weighted avg"]:
            rows.append({
                "Class": cls_name,
                "Support": int(vals.get("support", 0)),
            })

    df_dist = pd.DataFrame(rows)

    if chart_type == "🍩 Donut Chart":
        fig = px.pie(
            df_dist,
            names="Class",
            values="Support",
            hole=0.45,
            color="Class",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
        fig.update_layout(
            title=dict(text="Evaluation Test Flow Distribution by Class", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=420)

    elif chart_type == "📊 Column Bar":
        df_sorted = df_dist.sort_values("Support", ascending=False)
        fig = px.bar(
            df_sorted,
            x="Class",
            y="Support",
            color="Support",
            color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#10b981"]],
            text="Support",
            labels={"Support": "Test Flows", "Class": ""},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            xaxis=dict(tickangle=-35),
            coloraxis_showscale=False,
            title=dict(text="Evaluation Test Flow Distribution (Ranked)", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=420)

    else:  # 📶 Horizontal Bar (default)
        df_sorted = df_dist.sort_values("Support", ascending=True)
        fig = px.bar(
            df_sorted,
            x="Support",
            y="Class",
            orientation="h",
            color="Support",
            color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#38bdf8"], [1.0, "#10b981"]],
            text="Support",
            labels={"Support": "Test Flows", "Class": ""},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            title=dict(text="Evaluation Test Flow Distribution (Horizontal)", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=max(420, len(df_sorted) * 26))


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    chart_type: str = "📊 Grouped Bar",
) -> go.Figure:
    """
    Render per-class Precision, Recall, and F1-Score comparison.
    """
    rows = []
    for cls_name, vals in report_dict.items():
        if isinstance(vals, dict) and cls_name not in ["accuracy", "macro avg", "weighted avg"]:
            rows.append({
                "Class": cls_name,
                "Precision": vals.get("precision", 0.0),
                "Recall": vals.get("recall", 0.0),
                "F1-Score": vals.get("f1-score", 0.0),
                "Support": vals.get("support", 0),
            })

    df_metrics = pd.DataFrame(rows).sort_values("F1-Score", ascending=True)

    if chart_type == "🗺️ Heatmap":
        heatmap_data = df_metrics.set_index("Class")[["Precision", "Recall", "F1-Score"]]
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=[[0.0, "#0b0f19"], [0.5, "#0284c7"], [1.0, "#38bdf8"]],
            zmin=0,
            zmax=1,
            aspect="auto",
            text_auto=".2%",
            labels=dict(color="Score"),
        )
        fig.update_layout(
            title=dict(text="Per-Class Performance Heatmap (Precision, Recall, F1)", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=max(420, len(df_metrics) * 26))

    elif chart_type == "📶 Horizontal Bar":
        df_melted = pd.melt(
            df_metrics,
            id_vars=["Class"],
            value_vars=["Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            df_melted,
            x="Score",
            y="Class",
            color="Metric",
            barmode="group",
            orientation="h",
            color_discrete_map={"Precision": "#38bdf8", "Recall": "#f59e0b", "F1-Score": "#10b981"},
            labels={"Score": "Empirical Score", "Class": ""},
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1.15], tickformat=".0%"),
            title=dict(text="Per-Class Metric Comparison (Horizontal)", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=max(420, len(df_metrics) * 26))

    else:  # 📊 Grouped Bar (default)
        df_melted = pd.melt(
            df_metrics,
            id_vars=["Class"],
            value_vars=["Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            df_melted,
            x="Class",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_map={"Precision": "#38bdf8", "Recall": "#f59e0b", "F1-Score": "#10b981"},
            labels={"Score": "Score", "Class": ""},
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            xaxis=dict(tickangle=-35),
            title=dict(text="Per-Class Precision, Recall, and F1-Score Breakdown", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=440)


def plot_model_comparison(
    binary_metrics: Dict[str, float],
    multi_metrics: Dict[str, float],
    chart_type: str = "📊 Grouped Bar",
) -> go.Figure:
    """
    Visual comparison of Binary vs Multiclass model performance on held-out test data.
    """
    compare_keys = [
        ("accuracy", "Overall Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision_weighted", "Weighted Precision"),
        ("recall_weighted", "Weighted Recall"),
        ("f1_weighted", "Weighted F1-Score"),
        ("f1_macro", "Macro F1-Score"),
    ]

    rows = []
    for k, label in compare_keys:
        rows.append({
            "Metric": label,
            "Binary Detector": binary_metrics.get(k, 0.0),
            "Multiclass Classifier": multi_metrics.get(k, 0.0),
        })
    df_comp = pd.DataFrame(rows)

    if chart_type == "📶 Horizontal Bar":
        df_melted = pd.melt(
            df_comp,
            id_vars=["Metric"],
            value_vars=["Binary Detector", "Multiclass Classifier"],
            var_name="Model",
            value_name="Score",
        )
        fig = px.bar(
            df_melted,
            x="Score",
            y="Metric",
            color="Model",
            barmode="group",
            orientation="h",
            color_discrete_map={"Binary Detector": "#10b981", "Multiclass Classifier": "#38bdf8"},
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1.15], tickformat=".1%"),
            title=dict(text="Classifier Head-to-Head Performance", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=420)

    elif chart_type == "🗺️ Heatmap":
        heatmap_data = df_comp.set_index("Metric")
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale="Blues",
            zmin=0.5,
            zmax=1.0,
            text_auto=".2%",
        )
        fig.update_layout(
            title=dict(text="Classifier Performance Matrix", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=420)

    else:  # 📊 Grouped Bar (default)
        df_melted = pd.melt(
            df_comp,
            id_vars=["Metric"],
            value_vars=["Binary Detector", "Multiclass Classifier"],
            var_name="Model",
            value_name="Score",
        )
        fig = px.bar(
            df_melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={"Binary Detector": "#10b981", "Multiclass Classifier": "#38bdf8"},
            text="Score",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis=dict(range=[0, 1.22], tickformat=".0%"),
            xaxis=dict(tickangle=-20),
            title=dict(text="Direct Model Architecture Performance Comparison", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=420)


def plot_dual_classifier_comparison(
    dual_df: pd.DataFrame,
    chart_type: str = "🗺️ Agreement Heatmap",
) -> go.Figure:
    """
    Render a visual comparison of how Binary vs Multiclass classifiers evaluated the analyzed batch.
    """
    if chart_type == "🗺️ Agreement Heatmap":
        ct = pd.crosstab(dual_df["Binary_Pred"], dual_df["Multi_Pred"])
        fig = px.imshow(
            ct,
            color_continuous_scale=[[0.0, "#0b0f19"], [0.2, "#0369a1"], [0.6, "#0284c7"], [1.0, "#38bdf8"]],
            text_auto=True,
            labels=dict(x="Multiclass Model Prediction", y="Binary Model Prediction", color="Flows"),
        )
        fig.update_layout(
            title=dict(text="Classifier Consensus Matrix (Batch Agreement)", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=400)

    elif chart_type == "🍩 Agreement Share":
        agree_counts = dual_df["Agreed"].value_counts().reset_index()
        agree_counts.columns = ["Agreement", "Flows"]
        agree_counts["Status"] = agree_counts["Agreement"].map({True: "Full Consensus", False: "Divergent Prediction"})
        fig = px.pie(
            agree_counts,
            names="Status",
            values="Flows",
            hole=0.55,
            color="Status",
            color_discrete_map={"Full Consensus": "#10b981", "Divergent Prediction": "#f59e0b"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            title=dict(text="Model Consensus Ratio", font=dict(color="#f8fafc")),
        )
        return apply_cyber_theme(fig, height=400)

    else:  # 📊 Consensus Bar
        cons_counts = dual_df["Consensus"].value_counts().reset_index()
        cons_counts.columns = ["Consensus Type", "Flows"]
        fig = px.bar(
            cons_counts,
            x="Flows",
            y="Consensus Type",
            orientation="h",
            color="Consensus Type",
            color_discrete_sequence=CYBER_PALETTE,
            text="Flows",
            log_x=True,
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            title=dict(text="Detection Decision Consensus Categories", font=dict(color="#f8fafc")),
            coloraxis_showscale=False,
            yaxis=dict(title=""),
        )
        return apply_cyber_theme(fig, height=400, show_legend=False)


def plot_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    normalize: bool = False,
    title: str = "Empirical Test Confusion Matrix",
    chart_type: str = "🗺️ Heatmap",
) -> go.Figure:
    """
    Generate an interactive Plotly confusion matrix in dark cybersecurity aesthetic.
    Supports Heatmap, Normalized Share Heatmap, and Diagonal Accuracy Bar.
    """
    cm_arr = np.array(cm)
    if normalize:
        with np.errstate(all="ignore"):
            norm_cm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
            norm_cm = np.nan_to_num(norm_cm)
        z_values = np.round(norm_cm, 3)
        text_values = [[f"{val:.1%}" for val in row] for row in z_values]
    else:
        z_values = cm_arr
        text_values = [[f"{int(val):,}" for val in row] for row in z_values]

    if chart_type == "📊 Diagonal Accuracy Bar":
        # Per-class recall (accuracy on diagonal)
        diag = np.diag(cm_arr)
        totals = cm_arr.sum(axis=1)
        recalls = [d / t if t > 0 else 0.0 for d, t in zip(diag, totals)]
        diag_df = pd.DataFrame({"Class": classes, "Recall Accuracy": recalls}).sort_values("Recall Accuracy", ascending=True)
        fig = px.bar(
            diag_df,
            x="Recall Accuracy",
            y="Class",
            orientation="h",
            color="Recall Accuracy",
            color_continuous_scale=[[0.0, "#ef4444"], [0.5, "#f59e0b"], [1.0, "#10b981"]],
            text="Recall Accuracy",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(
            title=dict(text=f"{title} — Class Sensitivity (Diagonal)", font=dict(color="#f1f5f9", size=14)),
            xaxis=dict(range=[0, 1.15], tickformat=".0%"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
        )
        return apply_cyber_theme(fig, height=max(440, len(classes) * 26), show_legend=False)

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
        height=max(480, len(classes) * 28),
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
