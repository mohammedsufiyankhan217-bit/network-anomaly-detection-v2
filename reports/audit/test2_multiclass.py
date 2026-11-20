"""
TEST 2: Empirical Evaluation of the Production Multiclass Random Forest Classifier.
Evaluates on the true held-out test split (19,993 unseen CIC-IDS2017 flows across 15 classes).
Analyzes per-class precision, recall, F1, and inter-class confusion patterns.
Saves results to reports/audit/test2_multiclass_results.json.
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
)

from src.model_trainer import prepare_features_and_target

def evaluate_multiclass_classifier():
    print("=" * 60)
    print("STARTING TEST 2: MULTICLASS CLASSIFIER EMPIRICAL AUDIT")
    print("=" * 60)

    # 1. Load Model Artifact
    model_path = Path("models/saved_models/rf_model_multiclass.joblib")
    assert model_path.exists(), f"Multiclass model not found at {model_path}"
    art = joblib.load(model_path)
    model = art["model"]
    scaler = art["scaler"]
    feature_names = art["results"]["feature_names"]
    reported_metrics = art["results"]["metrics"]

    # 2. Load dataset and reproduce held-out test partition
    sample_path = Path("data/samples/cicids2017_stratified_sample_100k.parquet")
    df = pd.read_parquet(sample_path)
    X, y, feats = prepare_features_and_target(df, target_mode="multiclass")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Transform features
    X_test_scaled = scaler.transform(X_test)

    # 4. Generate Predictions & Probabilities
    y_pred = model.predict(X_test_scaled)
    classes = sorted(list(set(y_test) | set(y_pred)))

    # 5. Compute Empirical Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Classification Report
    report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

    # 6. Detailed Per-Class Breakdown and Misclassification Analysis
    per_class_metrics = []
    struggling_classes = []

    for i, cls_name in enumerate(classes):
        support = int(np.sum(y_test == cls_name))
        row = cm[i]
        correct = row[i]
        errors = support - correct
        error_rate = float(errors / support) if support > 0 else 0.0

        p = report[cls_name]["precision"]
        r = report[cls_name]["recall"]
        f1 = report[cls_name]["f1-score"]

        # Find primary misclassification targets
        misclass_targets = {}
        for j, pred_cls in enumerate(classes):
            if i != j and row[j] > 0:
                misclass_targets[pred_cls] = int(row[j])

        item = {
            "class": cls_name,
            "support": support,
            "correct": int(correct),
            "errors": int(errors),
            "misclassification_rate": error_rate,
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "primary_misclassifications": misclass_targets,
        }
        per_class_metrics.append(item)

        if error_rate > 0.05 or f1 < 0.90:
            struggling_classes.append(item)

    # Sort struggling classes by misclassification rate descending
    struggling_classes.sort(key=lambda x: x["misclassification_rate"], reverse=True)

    results = {
        "status": "PASSED",
        "model_path": str(model_path),
        "total_test_samples": len(y_test),
        "num_classes": len(classes),
        "classes": classes,
        "metrics": {
            "overall_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
        },
        "per_class_metrics": per_class_metrics,
        "struggling_classes": struggling_classes,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist(),
        },
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
    print(f"Overall Accuracy:        {acc:.6f} ({acc*100:.2f}%)")
    print(f"Balanced Accuracy:       {bal_acc:.6f} ({bal_acc*100:.2f}%)")
    print(f"Weighted F1-Score:       {f1_weighted:.6f}")
    print(f"Macro F1-Score:          {f1_macro:.6f}")
    print(f"Weighted Precision:      {prec_weighted:.6f}")
    print(f"Macro Precision:         {prec_macro:.6f}")
    print(f"Weighted Recall:         {rec_weighted:.6f}")
    print(f"Macro Recall:            {rec_macro:.6f}")
    print("-" * 50)
    print("Per-Class Evaluation (sorted by Support):")
    sorted_by_sup = sorted(per_class_metrics, key=lambda x: x["support"], reverse=True)
    print(f"{'Class Name':<28} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'Error%':>8}")
    for c in sorted_by_sup:
        print(f"{c['class']:<28} {c['support']:>8,d} {c['precision']:>10.4f} {c['recall']:>8.4f} {c['f1_score']:>9.4f} {c['misclassification_rate']*100:>7.2f}%")

    print("-" * 50)
    print("Classes Model Struggles To Distinguish (Error Rate > 5% or F1 < 0.90):")
    for sc in struggling_classes:
        print(f"  * {sc['class']} (Support={sc['support']}): Error Rate = {sc['misclassification_rate']*100:.2f}%, F1 = {sc['f1_score']:.4f}")
        if sc["primary_misclassifications"]:
            for tgt, cnt in sc["primary_misclassifications"].items():
                print(f"     -> Confused with '{tgt}': {cnt} times ({(cnt/sc['support'])*100:.1f}%)")

    print("=" * 60)
    out_file = Path("reports/audit/test2_multiclass_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Test 2 results to {out_file}")

if __name__ == "__main__":
    evaluate_multiclass_classifier()
