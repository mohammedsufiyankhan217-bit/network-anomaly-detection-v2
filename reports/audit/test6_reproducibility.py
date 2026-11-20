"""
TEST 6: Reproducibility & Determinism Audit.
Evaluates:
1. Model loading integrity from joblib serialization.
2. Train/Test split reproducibility with fixed random_state=42.
3. Repeated inference determinism (identical predictions and probas across repeated runs).
4. Reproduction of recorded metadata metrics vs empirical re-calculation.
5. Deterministic retraining check using temporary model (production models untouched).
Saves results to reports/audit/test6_reproducibility_results.json.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from src.model_trainer import prepare_features_and_target
from src.config import BENIGN_LABEL

def test_reproducibility():
    print("=" * 60)
    print("STARTING TEST 6: REPRODUCIBILITY & DETERMINISM AUDIT")
    print("=" * 60)

    # 1. Model Loading Integrity
    print("\n[CHECK 1] Model Loading Integrity...")
    bin_path = "models/saved_models/rf_model_binary.joblib"
    multi_path = "models/saved_models/rf_model_multiclass.joblib"
    art_bin = joblib.load(bin_path)
    art_multi = joblib.load(multi_path)
    print(f"  Successfully loaded binary model ({art_bin['results']['model_type']}) from {bin_path}")
    print(f"  Successfully loaded multiclass model ({art_multi['results']['model_type']}) from {multi_path}")

    # 2. Split Reproducibility
    print("\n[CHECK 2] Train/Test Split Reproducibility...")
    df = pd.read_parquet("data/samples/cicids2017_stratified_sample_100k.parquet")
    X_bin, y_bin, feats_bin = prepare_features_and_target(df, target_mode="binary")

    # Run split twice with seed 42
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_bin, y_bin, test_size=0.20, stratify=y_bin, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_bin, y_bin, test_size=0.20, stratify=y_bin, random_state=42)

    indices_tr_match = (X_tr1.index == X_tr2.index).all()
    indices_te_match = (X_te1.index == X_te2.index).all()
    values_match = (X_te1.values == X_te2.values).all()
    labels_match = (y_te1.values == y_te2.values).all()

    split_reproducible = bool(indices_tr_match and indices_te_match and values_match and labels_match)
    print(f"  Train index match: {indices_tr_match}")
    print(f"  Test index match: {indices_te_match}")
    print(f"  Test values bit-for-bit identical: {values_match}")
    print(f"  Test labels bit-for-bit identical: {labels_match}")
    print(f"  -> Split Reproducibility: {'PASSED' if split_reproducible else 'FAILED'}")

    # 3. Repeated Inference Determinism
    print("\n[CHECK 3] Repeated Inference Determinism...")
    scaler_bin = art_bin["scaler"]
    model_bin = art_bin["model"]
    X_te_scaled = scaler_bin.transform(X_te1)

    preds_run1 = model_bin.predict(X_te_scaled)
    probas_run1 = model_bin.predict_proba(X_te_scaled)

    preds_run2 = model_bin.predict(X_te_scaled)
    probas_run2 = model_bin.predict_proba(X_te_scaled)

    preds_match = (preds_run1 == preds_run2).all()
    max_proba_diff = float(np.max(np.abs(probas_run1 - probas_run2)))
    inference_deterministic = bool(preds_match and max_proba_diff < 1e-12)

    print(f"  Predictions identical across runs: {preds_match}")
    print(f"  Max absolute probability difference: {max_proba_diff}")
    print(f"  -> Inference Determinism: {'PASSED' if inference_deterministic else 'FAILED'}")

    # 4. Reproduction of Recorded Metrics
    print("\n[CHECK 4] Reproduction of Recorded Model Metrics...")
    # Binary reproduction
    rec_bin_metrics = art_bin["results"]["metrics"]
    acc_bin = accuracy_score(y_te1, preds_run1)
    b_acc_bin = balanced_accuracy_score(y_te1, preds_run1)
    f1_w_bin = f1_score(y_te1, preds_run1, average="weighted")
    f1_m_bin = f1_score(y_te1, preds_run1, average="macro")

    bin_acc_diff = abs(acc_bin - rec_bin_metrics["accuracy"])
    bin_f1w_diff = abs(f1_w_bin - rec_bin_metrics["f1_weighted"])
    bin_f1m_diff = abs(f1_m_bin - rec_bin_metrics["f1_macro"])

    print(f"  Binary Recorded Acc:  {rec_bin_metrics['accuracy']:.8f} | Re-measured: {acc_bin:.8f} (diff: {bin_acc_diff:.2e})")
    print(f"  Binary Recorded F1-w: {rec_bin_metrics['f1_weighted']:.8f} | Re-measured: {f1_w_bin:.8f} (diff: {bin_f1w_diff:.2e})")
    print(f"  Binary Recorded F1-m: {rec_bin_metrics['f1_macro']:.8f} | Re-measured: {f1_m_bin:.8f} (diff: {bin_f1m_diff:.2e})")

    # Multiclass reproduction
    X_multi, y_multi, feats_multi = prepare_features_and_target(df, target_mode="multiclass")
    class_counts = y_multi.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y_multi.isin(valid_classes)
    X_m_valid = X_multi[mask]
    y_m_valid = y_multi[mask]

    X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
        X_m_valid, y_m_valid, test_size=0.20, stratify=y_m_valid, random_state=42
    )
    scaler_multi = art_multi["scaler"]
    model_multi = art_multi["model"]
    X_m_te_scaled = scaler_multi.transform(X_m_te)
    preds_multi = model_multi.predict(X_m_te_scaled)

    rec_multi_metrics = art_multi["results"]["metrics"]
    acc_multi = accuracy_score(y_m_te, preds_multi)
    b_acc_multi = balanced_accuracy_score(y_m_te, preds_multi)
    f1_w_multi = f1_score(y_m_te, preds_multi, average="weighted")
    f1_m_multi = f1_score(y_m_te, preds_multi, average="macro")

    multi_acc_diff = abs(acc_multi - rec_multi_metrics["accuracy"])
    multi_f1w_diff = abs(f1_w_multi - rec_multi_metrics["f1_weighted"])
    multi_f1m_diff = abs(f1_m_multi - rec_multi_metrics["f1_macro"])

    print(f"  Multiclass Recorded Acc:  {rec_multi_metrics['accuracy']:.8f} | Re-measured: {acc_multi:.8f} (diff: {multi_acc_diff:.2e})")
    print(f"  Multiclass Recorded F1-w: {rec_multi_metrics['f1_weighted']:.8f} | Re-measured: {f1_w_multi:.8f} (diff: {multi_f1w_diff:.2e})")
    print(f"  Multiclass Recorded F1-m: {rec_multi_metrics['f1_macro']:.8f} | Re-measured: {f1_m_multi:.8f} (diff: {multi_f1m_diff:.2e})")

    metrics_reproduced = (bin_acc_diff < 1e-6 and multi_acc_diff < 1e-6)
    print(f"  -> Metrics Reproduction: {'PASSED' if metrics_reproduced else 'FAILED'}")

    # 5. Deterministic Retraining Verification (Temporary model, production model NOT modified)
    print("\n[CHECK 5] Fresh Retraining Determinism (Temporary Model)...")
    temp_scaler = RobustScaler()
    X_tr_sc = temp_scaler.fit_transform(X_tr1)
    X_te_sc = temp_scaler.transform(X_te1)

    temp_rf = RandomForestClassifier(
        n_estimators=art_bin["results"]["hyperparameters"]["n_estimators"],
        max_depth=art_bin["results"]["hyperparameters"]["max_depth"],
        random_state=42,
        n_jobs=-1,
    )
    temp_rf.fit(X_tr_sc, y_tr1)
    temp_preds = temp_rf.predict(X_te_sc)
    temp_acc = accuracy_score(y_te1, temp_preds)
    retrain_acc_diff = abs(temp_acc - acc_bin)
    print(f"  Retrained Model Accuracy: {temp_acc:.8f} vs Production: {acc_bin:.8f} (diff: {retrain_acc_diff:.2e})")
    retrain_deterministic = (retrain_acc_diff < 1e-4)
    print(f"  -> Retraining Determinism: {'PASSED' if retrain_deterministic else 'FAILED'}")

    results = {
        "status": "PASSED" if (split_reproducible and inference_deterministic and metrics_reproduced and retrain_deterministic) else "WARNING",
        "split_reproducible": split_reproducible,
        "inference_deterministic": inference_deterministic,
        "metrics_reproduced": metrics_reproduced,
        "retrain_deterministic": retrain_deterministic,
        "recorded_vs_remeasured": {
            "binary": {
                "accuracy_recorded": float(rec_bin_metrics["accuracy"]),
                "accuracy_remeasured": float(acc_bin),
                "accuracy_diff": float(bin_acc_diff),
                "f1_weighted_recorded": float(rec_bin_metrics["f1_weighted"]),
                "f1_weighted_remeasured": float(f1_w_bin),
                "f1_macro_recorded": float(rec_bin_metrics["f1_macro"]),
                "f1_macro_remeasured": float(f1_m_bin),
            },
            "multiclass": {
                "accuracy_recorded": float(rec_multi_metrics["accuracy"]),
                "accuracy_remeasured": float(acc_multi),
                "accuracy_diff": float(multi_acc_diff),
                "f1_weighted_recorded": float(rec_multi_metrics["f1_weighted"]),
                "f1_weighted_remeasured": float(f1_w_multi),
                "f1_macro_recorded": float(rec_multi_metrics["f1_macro"]),
                "f1_macro_remeasured": float(f1_m_multi),
            }
        },
        "retrain_evaluation": {
            "retrained_binary_accuracy": float(temp_acc),
            "production_binary_accuracy": float(acc_bin),
            "accuracy_difference": float(retrain_acc_diff)
        }
    }

    out_file = Path("reports/audit/test6_reproducibility_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 60)
    print(f"Saved Test 6 results to {out_file}")

if __name__ == "__main__":
    test_reproducibility()
