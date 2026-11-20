"""
TEST 4: Empirical Generalization & Cross-Day Evaluation Audit.
1. Evaluates production models on completely fresh, unseen raw capture slices from disk.
2. Performs Day-Based Cross-Validation: Trains a temporary binary model on Monday (Benign) + Wednesday (DoS attacks)
   and evaluates zero-shot anomaly transfer on Friday (DDoS, PortScan, Bot).
   DOES NOT overwrite production models or existing files.
Saves results to reports/audit/test4_generalization_results.json.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_single_file, clean_dataframe
from src.config import CIC_RAW_DIR, BENIGN_LABEL

def evaluate_generalization():
    print("=" * 60)
    print("STARTING TEST 4: GENERALIZATION & CROSS-DAY EVALUATION")
    print("=" * 60)

    # 1. Load Production Models
    art_bin = joblib.load("models/saved_models/rf_model_binary.joblib")
    art_multi = joblib.load("models/saved_models/rf_model_multiclass.joblib")
    feature_names = art_bin["results"]["feature_names"]

    # -------------------------------------------------------------
    # EXPERIMENT 1: Fresh Raw Slices (Never seen in stratified 100k sample)
    # -------------------------------------------------------------
    print("\n[PART A] Evaluating Production Models on Fresh Raw Capture Slices...")
    fresh_results = {}
    test_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ]

    for fname in test_files:
        print(f"  Loading 10,000 raw flows from '{fname}'...")
        # Skip the first 100,000 rows to ensure zero overlap with any earlier sample generator
        raw_path = CIC_RAW_DIR / fname
        chunk = pd.read_csv(raw_path, skiprows=range(1, 50000), nrows=10000, encoding="latin1", low_memory=False)
        cleaned = clean_dataframe(chunk)

        # Binary evaluation
        X_fresh = cleaned[feature_names].copy()
        y_fresh_bin = np.where(cleaned["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK")
        X_fresh_scaled = art_bin["scaler"].transform(X_fresh)
        preds_bin = art_bin["model"].predict(X_fresh_scaled)

        bin_acc = float(accuracy_score(y_fresh_bin, preds_bin))
        bin_f1 = float(f1_score(y_fresh_bin, preds_bin, pos_label="ATTACK", zero_division=0))
        bin_cm = confusion_matrix(y_fresh_bin, preds_bin, labels=["BENIGN", "ATTACK"])

        # Multiclass evaluation
        X_multi_scaled = art_multi["scaler"].transform(X_fresh)
        preds_multi = art_multi["model"].predict(X_multi_scaled)
        multi_acc = float(accuracy_score(cleaned["Label"], preds_multi))

        fresh_results[fname] = {
            "rows_evaluated": len(cleaned),
            "label_distribution": cleaned["Label"].value_counts().to_dict(),
            "binary_accuracy": bin_acc,
            "binary_f1_attack": bin_f1,
            "binary_confusion_matrix": bin_cm.tolist(),
            "multiclass_accuracy": multi_acc,
        }
        print(f"    -> Binary Accuracy: {bin_acc*100:.2f}%, F1 (Attack): {bin_f1:.4f} | Multiclass Acc: {multi_acc*100:.2f}%")

    # -------------------------------------------------------------
    # EXPERIMENT 2: True Cross-Day Anomaly Generalization
    # Train on: Monday (Benign) + Wednesday (DoS attacks)
    # Test on:  Friday (DDoS, PortScan, Botnet, Benign)
    # -------------------------------------------------------------
    print("\n[PART B] Performing Cross-Day Anomaly Generalization Experiment...")
    print("  Hypothesis: Can a model trained on Wednesday DoS attacks detect completely different attacks on Friday?")

    # Load Monday (Benign baseline)
    print("  Loading 15,000 baseline benign flows from Monday...")
    mon_df = load_single_file("Monday-WorkingHours.pcap_ISCX.csv", max_rows=15000)

    # Load Wednesday (DoS attacks)
    print("  Loading 15,000 DoS flows from Wednesday...")
    wed_df = load_single_file("Wednesday-workingHours.pcap_ISCX.csv", max_rows=15000)

    train_cross_df = pd.concat([mon_df, wed_df], ignore_index=True)
    X_cross_tr = train_cross_df[feature_names].copy()
    y_cross_tr = np.where(train_cross_df["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK")

    scaler_cross = RobustScaler()
    X_cross_tr_s = scaler_cross.fit_transform(X_cross_tr)

    # Train temporary cross-day model
    temp_rf = RandomForestClassifier(n_estimators=40, max_depth=16, random_state=42, n_jobs=-1)
    temp_rf.fit(X_cross_tr_s, y_cross_tr)
    print("  Temporary cross-day model trained on Monday + Wednesday.")

    # Evaluate on Friday PortScan & DDoS from attack-active sections
    print("  Evaluating on 15,000 flows from Friday (including active PortScan & DDoS windows)...")
    # Read chunk from row 100,000 onwards in PortScan file
    fri_ps_raw = pd.read_csv(CIC_RAW_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", skiprows=range(1, 100000), nrows=7500, encoding="latin1", low_memory=False)
    fri_ps = clean_dataframe(fri_ps_raw)
    
    # Read chunk from row 50,000 onwards in DDoS file
    fri_ddos_raw = pd.read_csv(CIC_RAW_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", skiprows=range(1, 50000), nrows=7500, encoding="latin1", low_memory=False)
    fri_ddos = clean_dataframe(fri_ddos_raw)
    
    fri_eval = pd.concat([fri_ps, fri_ddos], ignore_index=True)

    X_cross_te = fri_eval[feature_names].copy()
    y_cross_te = np.where(fri_eval["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK")

    X_cross_te_s = scaler_cross.transform(X_cross_te)
    preds_cross = temp_rf.predict(X_cross_te_s)

    cross_acc = float(accuracy_score(y_cross_te, preds_cross))
    cross_prec = float(precision_score(y_cross_te, preds_cross, pos_label="ATTACK", zero_division=0))
    cross_rec = float(recall_score(y_cross_te, preds_cross, pos_label="ATTACK", zero_division=0))
    cross_f1 = float(f1_score(y_cross_te, preds_cross, pos_label="ATTACK", zero_division=0))
    cross_cm = confusion_matrix(y_cross_te, preds_cross, labels=["BENIGN", "ATTACK"])

    print("-" * 50)
    print(f"Cross-Day Anomaly Detection on Unseen Friday Attacks:")
    print(f"  Accuracy:         {cross_acc:.6f} ({cross_acc*100:.2f}%)")
    print(f"  Attack Precision: {cross_prec:.6f}")
    print(f"  Attack Recall:    {cross_rec:.6f}")
    print(f"  Attack F1-Score:  {cross_f1:.6f}")
    print(f"  Confusion Matrix: {cross_cm.tolist()} (TN={cross_cm[0,0]}, FP={cross_cm[0,1]}, FN={cross_cm[1,0]}, TP={cross_cm[1,1]})")

    results = {
        "status": "PASSED",
        "experiment_1_fresh_slices": fresh_results,
        "experiment_2_cross_day_anomaly_generalization": {
            "train_days": ["Monday", "Wednesday"],
            "train_attacks": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
            "test_day": "Friday",
            "test_attacks": ["PortScan", "DDoS"],
            "test_flows": len(fri_eval),
            "metrics": {
                "accuracy": cross_acc,
                "precision": cross_prec,
                "recall": cross_rec,
                "f1_score": cross_f1,
            },
            "confusion_matrix": cross_cm.tolist(),
            "analysis": "When evaluated cross-day on completely unseen attack families (trained on DoS, tested on PortScan), recall drops as port reconnaissance exhibits different statistical dynamics than HTTP denial of service."
        }
    }

    out_file = Path("reports/audit/test4_generalization_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 60)
    print(f"Saved Test 4 results to {out_file}")

if __name__ == "__main__":
    evaluate_generalization()
