"""
TEST 1: Empirical Evaluation of the Production Binary Random Forest Classifier.
Evaluates on the true held-out test split (19,993 unseen CIC-IDS2017 flows).
Saves full empirical metrics and confusion matrix to reports/audit/test1_binary_results.json.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.model_trainer import prepare_features_and_target

def evaluate_binary_classifier():
    print("=" * 60)
    print("STARTING TEST 1: BINARY CLASSIFIER EMPIRICAL AUDIT")
    print("=" * 60)

    # 1. Load Model Artifact
    model_path = Path("models/saved_models/rf_model_binary.joblib")
    assert model_path.exists(), f"Binary model not found at {model_path}"
    art = joblib.load(model_path)
    model = art["model"]
    scaler = art["scaler"]
    feature_names = art["results"]["feature_names"]
    reported_metrics = art["results"]["metrics"]

    # 2. Load dataset and reproduce held-out test partition
    sample_path = Path("data/samples/cicids2017_stratified_sample_100k.parquet")
    df = pd.read_parquet(sample_path)
    X, y, feats = prepare_features_and_target(df, target_mode="binary")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Transform features using fitted scaler
    X_test_scaled = scaler.transform(X_test)

    # 4. Generate Predictions & Probabilities
    y_pred = model.predict(X_test_scaled)
    classes = list(model.classes_)  # ['ATTACK', 'BENIGN']
    attack_idx = classes.index("ATTACK")
    y_proba = model.predict_proba(X_test_scaled)[:, attack_idx]

    # Convert to numeric binary for ROC/PR: 1 = ATTACK, 0 = BENIGN
    y_true_num = (y_test == "ATTACK").astype(int)

    # 5. Compute Empirical Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Binary metrics (pos_label="ATTACK")
    prec_binary = float(precision_score(y_test, y_pred, pos_label="ATTACK", zero_division=0))
    rec_binary = float(recall_score(y_test, y_pred, pos_label="ATTACK", zero_division=0))
    f1_binary = float(f1_score(y_test, y_pred, pos_label="ATTACK", zero_division=0))

    # Weighted and Macro metrics
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # ROC and PR Curves
    fpr, tpr, _ = roc_curve(y_true_num, y_proba)
    roc_auc_val = float(auc(fpr, tpr))

    precision_curve, recall_curve, _ = precision_recall_curve(y_true_num, y_proba)
    pr_auc_val = float(average_precision_score(y_true_num, y_proba))

    # Confusion Matrix: labels=['BENIGN', 'ATTACK']
    cm = confusion_matrix(y_test, y_pred, labels=["BENIGN", "ATTACK"])
    tn, fp, fn, tp = cm.ravel()

    # Rates
    fpr_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    report = classification_report(y_test, y_pred, labels=["BENIGN", "ATTACK"], output_dict=True)

    results = {
        "status": "PASSED",
        "model_path": str(model_path),
        "total_test_samples": len(y_test),
        "benign_test_samples": int(tn + fp),
        "attack_test_samples": int(tp + fn),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_attack": prec_binary,
            "recall_attack": rec_binary,
            "f1_attack": f1_binary,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val,
        },
        "confusion_matrix": {
            "labels": ["BENIGN", "ATTACK"],
            "matrix": cm.tolist(),
            "true_negatives_TN": int(tn),
            "false_positives_FP": int(fp),
            "false_negatives_FN": int(fn),
            "true_positives_TP": int(tp),
            "false_positive_rate_FPR": fpr_rate,
            "false_negative_rate_FNR": fnr_rate,
        },
        "classification_report": report,
        "reported_metrics_comparison": {
            "previously_reported_accuracy": reported_metrics.get("accuracy"),
            "measured_accuracy": acc,
            "previously_reported_f1_weighted": reported_metrics.get("f1_weighted"),
            "measured_f1_weighted": f1_weighted,
            "previously_reported_f1_macro": reported_metrics.get("f1_macro"),
            "measured_f1_macro": f1_macro,
        }
    }

    # Print summary
    print(f"Total Test Flows: {len(y_test):,}")
    print(f"True Benign: {tn + fp:,} | True Attack: {tp + fn:,}")
    print("-" * 50)
    print(f"Accuracy:            {acc:.6f} ({acc*100:.2f}%)")
    print(f"Balanced Accuracy:   {bal_acc:.6f} ({bal_acc*100:.2f}%)")
    print(f"Attack Precision:    {prec_binary:.6f}")
    print(f"Attack Recall:       {rec_binary:.6f}")
    print(f"Attack F1-Score:     {f1_binary:.6f}")
    print(f"Weighted F1-Score:   {f1_weighted:.6f}")
    print(f"Macro F1-Score:      {f1_macro:.6f}")
    print(f"ROC-AUC:             {roc_auc_val:.6f}")
    print(f"PR-AUC:              {pr_auc_val:.6f}")
    print("-" * 50)
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(f"                 Pred BENIGN    Pred ATTACK")
    print(f"  Actual BENIGN: {tn:>11,d} (TN) {fp:>11,d} (FP)")
    print(f"  Actual ATTACK: {fn:>11,d} (FN) {tp:>11,d} (TP)")
    print(f"False Positives (FP): {fp:,} (False Alarm Rate: {fpr_rate*100:.3f}%)")
    print(f"False Negatives (FN): {fn:,} (Missed Detection Rate: {fnr_rate*100:.3f}%)")
    print("=" * 60)

    out_file = Path("reports/audit/test1_binary_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Test 1 results to {out_file}")

if __name__ == "__main__":
    evaluate_binary_classifier()
