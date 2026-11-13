"""
Model Training and Evaluation Pipeline for Network Anomaly Detection V2.

Trains Random Forest classifiers on real CIC-IDS2017 network traffic.
Prevents data leakage by strictly separating train and test sets and fitting
scalers only on training data. Computes genuine empirical evaluation metrics.
"""
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import MODELS_DIR, BENIGN_LABEL

logger = logging.getLogger(__name__)


def prepare_features_and_target(
    df: pd.DataFrame,
    target_mode: str = "multiclass",
    drop_port: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract feature matrix X and target y while strictly eliminating data leakage.
    Excludes labels, binary flags, and non-predictive metadata.
    """
    # Columns that must NEVER be in X (labels and metadata)
    leakage_cols = [
        "Label",
        "Binary_Label",
        "Threat_Category",
        "Attack_Type",
        "Timestamp",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Simulated_Time",
    ]

    feature_cols = [c for c in df.columns if c not in leakage_cols]
    if drop_port and "Destination Port" in feature_cols:
        feature_cols.remove("Destination Port")

    X = df[feature_cols].copy()

    if target_mode == "binary":
        if "Binary_Label" in df.columns:
            y = df["Binary_Label"].copy()
        else:
            y = np.where(df["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK")
            y = pd.Series(y, index=df.index)
    else:
        y = df["Label"].copy()

    return X, y, feature_cols


def train_random_forest(
    df: pd.DataFrame,
    target_mode: str = "multiclass",
    test_size: float = 0.20,
    n_estimators: int = 60,
    max_depth: Optional[int] = 16,
    random_state: int = 42,
    drop_port: bool = False,
) -> Dict[str, Any]:
    """
    Train a Random Forest classifier using a stratified train/test split.
    Prevents data leakage by fitting the scaler solely on training features.
    Computes genuine test metrics on unseen test flows.
    """
    start_time = time.time()
    total_samples = len(df)

    # 1. Prepare features and target
    X, y, feature_names = prepare_features_and_target(
        df, target_mode=target_mode, drop_port=drop_port
    )

    # 2. Stratified train/test split
    # For rare classes with < 2 instances, handle gracefully
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    if len(valid_classes) < len(class_counts):
        # Filter out classes with strictly 1 sample from stratify split requirement
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 3. Fit scaler ONLY on X_train to prevent data leakage
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize and fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample" if target_mode == "multiclass" else "balanced",
    )
    rf.fit(X_train_scaled, y_train)
    train_duration = time.time() - start_time

    # 5. Evaluate on strictly held-out test data
    eval_start = time.time()
    y_pred = rf.predict(X_test_scaled)
    eval_duration = time.time() - eval_start

    # Determine unique classes in evaluation
    unique_classes = sorted(list(set(y_test) | set(y_pred)))

    # Compute genuine metrics
    acc = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    report_dict = classification_report(
        y_test, y_pred, labels=unique_classes, output_dict=True, zero_division=0
    )

    # Feature importances
    importances = rf.feature_importances_
    feat_imp = sorted(
        zip(feature_names, [float(x) for x in importances]),
        key=lambda x: x[1],
        reverse=True,
    )

    results = {
        "model_type": "Random Forest Classifier",
        "target_mode": target_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows_used": total_samples,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "classes": unique_classes,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state,
            "drop_port": drop_port,
        },
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "feature_importances": feat_imp,
        "train_duration_sec": round(train_duration, 2),
        "eval_duration_sec": round(eval_duration, 2),
    }

    # Package objects for persistence
    artifact = {
        "model": rf,
        "scaler": scaler,
        "results": results,
    }

    save_path = MODELS_DIR / f"rf_model_{target_mode}.joblib"
    joblib.dump(artifact, save_path)
    logger.info(f"Model and evaluation saved to {save_path}")

    return {
        "model": rf,
        "scaler": scaler,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def load_saved_model(target_mode: str = "multiclass") -> Optional[Dict[str, Any]]:
    """
    Load a pre-trained model and evaluation metrics from disk if available.
    """
    model_path = MODELS_DIR / f"rf_model_{target_mode}.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load saved model from {model_path}: {e}")
            return None
    return None
