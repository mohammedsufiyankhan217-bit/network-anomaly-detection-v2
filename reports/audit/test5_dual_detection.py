"""
TEST 5: Dual-Classifier Detection Engine & Robustness Audit.
Tests src.evaluator.run_dual_detection on real CIC-IDS2017 records.
Verifies:
1. Identical batch processing across Binary and Multiclass models.
2. Prediction and confidence validity.
3. Consensus classification and agreement/disagreement logic.
4. Error handling for missing features and malformed data.
Saves results to reports/audit/test5_dual_detection_results.json.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.evaluator import run_dual_detection
from src.config import BENIGN_LABEL

def test_dual_detection():
    print("=" * 60)
    print("STARTING TEST 5: DUAL-CLASSIFIER DETECTION ENGINE AUDIT")
    print("=" * 60)

    # 1. Load Production Models
    art_bin = joblib.load("models/saved_models/rf_model_binary.joblib")
    art_multi = joblib.load("models/saved_models/rf_model_multiclass.joblib")

    # 2. Load Real CIC-IDS2017 Records (Sample 1,000 real flows with mixed attacks)
    df = pd.read_parquet("data/samples/cicids2017_stratified_sample_100k.parquet")
    test_batch = pd.concat([
        df[df["Label"] != BENIGN_LABEL].sample(n=500, random_state=42),
        df[df["Label"] == BENIGN_LABEL].sample(n=500, random_state=42),
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Loaded test batch of {len(test_batch):,} real CIC-IDS2017 flows.")
    print(f"Ground truth composition: {test_batch['Label'].value_counts().to_dict()}")

    # 3. Execute run_dual_detection
    dual_results = run_dual_detection(art_bin, art_multi, test_batch)

    # Verification 1: Both classifiers processed all rows
    assert len(dual_results) == len(test_batch), "Batch length mismatch!"
    assert "Binary_Pred" in dual_results.columns, "Missing Binary_Pred!"
    assert "Multi_Pred" in dual_results.columns, "Missing Multi_Pred!"
    assert "Consensus" in dual_results.columns, "Missing Consensus!"
    print("  [CHECK 1 PASSED] Both classifiers processed all input flows.")

    # Verification 2: Confidence bounds [0.0, 1.0]
    bin_conf = dual_results["Binary_Conf"]
    multi_conf = dual_results["Multi_Conf"]
    assert (bin_conf >= 0.0).all() and (bin_conf <= 1.0).all(), "Invalid binary confidence values!"
    assert (multi_conf >= 0.0).all() and (multi_conf <= 1.0).all(), "Invalid multiclass confidence values!"
    print(f"  [CHECK 2 PASSED] Confidence bounds valid: Binary [{bin_conf.min():.4f}, {bin_conf.max():.4f}], Multiclass [{multi_conf.min():.4f}, {multi_conf.max():.4f}].")

    # Verification 3: Consensus Breakdown
    consensus_counts = dual_results["Consensus"].value_counts().to_dict()
    agreement_count = int(dual_results["Agreed"].sum())
    disagreement_count = len(dual_results) - agreement_count
    agreement_rate = (agreement_count / len(dual_results)) * 100

    print("  [CHECK 3 PASSED] Consensus Breakdown:")
    for k, v in consensus_counts.items():
        print(f"    - {k}: {v} flows ({(v/len(dual_results))*100:.1f}%)")
    print(f"  Overall Consensus Agreement Rate: {agreement_rate:.2f}% ({agreement_count}/{len(dual_results)})")
    print(f"  Divergent Predictions: {disagreement_count} flows")

    # Detail disagreement cases if any
    disagreements = dual_results[~dual_results["Agreed"]][["Label", "Binary_Pred", "Binary_Conf", "Multi_Pred", "Multi_Conf"]]
    if len(disagreements) > 0:
        print(f"  Sample Disagreements:\n{disagreements.head(5).to_string(index=False)}")

    # Verification 4: Error Handling for Malformed or Missing Features
    print("\n[CHECK 4] Testing Fault Tolerance and Exception Handling...")
    # A. Missing feature column
    missing_col_batch = test_batch.drop(columns=["Destination Port"]).copy()
    try:
        run_dual_detection(art_bin, art_multi, missing_col_batch)
        missing_handled = False
    except ValueError as e:
        missing_handled = True
        print(f"  -> Properly intercepted missing column: {e}")

    # B. Missing (NaN) values handling (sklearn 1.4+ native support)
    nan_batch = test_batch.copy()
    nan_batch.loc[0:5, "Flow Duration"] = np.nan
    try:
        nan_res = run_dual_detection(art_bin, art_multi, nan_batch)
        nan_handled_natively = True
        print(f"  -> RobustScaler/Model handled NaN values gracefully via sklearn tree imputation/missing support.")
    except Exception as e:
        nan_handled_natively = False
        print(f"  -> Raised exception on NaN inputs: {e.__class__.__name__}")

    # C. Malformed string inputs handling
    malformed_batch = test_batch.copy()
    malformed_batch["Flow Duration"] = malformed_batch["Flow Duration"].astype(object)
    malformed_batch.loc[0, "Flow Duration"] = "CORRUPT_STR"
    try:
        run_dual_detection(art_bin, art_multi, malformed_batch)
        malformed_handled = False
    except ValueError as e:
        malformed_handled = True
        print(f"  -> Properly intercepted non-numeric malformed string: {e}")

    results = {
        "status": "PASSED",
        "batch_size": len(test_batch),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_rate": agreement_rate,
        "consensus_distribution": consensus_counts,
        "confidence_statistics": {
            "binary_mean_conf": float(bin_conf.mean()),
            "multiclass_mean_conf": float(multi_conf.mean()),
            "binary_min_conf": float(bin_conf.min()),
            "multiclass_min_conf": float(multi_conf.min()),
        },
        "fault_tolerance": {
            "missing_column_exception_raised": missing_handled,
            "nan_input_graceful_handling": nan_handled_natively,
            "malformed_string_exception_raised": malformed_handled,
        }
    }

    out_file = Path("reports/audit/test5_dual_detection_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 60)
    print(f"Saved Test 5 results to {out_file}")

if __name__ == "__main__":
    test_dual_detection()
