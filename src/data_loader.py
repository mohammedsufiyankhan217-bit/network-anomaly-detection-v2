"""
Robust Data Loader for the Official Canadian Institute for Cybersecurity
CIC-IDS2017 MachineLearningCSV Dataset.

Handles chunking, column sanitization, missing & infinite value handling,
duplicate column resolution, label normalization, and stratified sampling.
"""
import os
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.config import (
    CIC_RAW_DIR,
    PROCESSED_DATA_DIR,
    SAMPLES_DATA_DIR,
    EXPECTED_FILES,
    BENIGN_LABEL,
    ATTACK_CATEGORIES,
    TOTAL_DATASET_ROWS,
)

logger = logging.getLogger(__name__)

SUMMARY_FILE = PROCESSED_DATA_DIR / "dataset_summary.json"
DEFAULT_SAMPLE_PATH = SAMPLES_DATA_DIR / "cicids2017_stratified_sample_100k.parquet"


def get_dataset_status() -> Dict[str, Any]:
    """
    Check whether the official CIC-IDS2017 CSV files exist locally,
    which files are present/missing, and total records.
    """
    found_files = []
    missing_files = []
    file_details = {}
    total_size_bytes = 0

    for expected_name in EXPECTED_FILES:
        # Check in CIC_RAW_DIR first, then case-insensitive / wildcard
        file_path = CIC_RAW_DIR / expected_name
        if not file_path.exists():
            matches = list(CIC_RAW_DIR.glob(f"*{expected_name}*"))
            if matches:
                file_path = matches[0]

        if file_path.exists():
            size = file_path.stat().st_size
            total_size_bytes += size
            found_files.append(file_path.name)
            file_details[file_path.name] = {
                "path": str(file_path),
                "size_mb": round(size / (1024 * 1024), 2),
                "status": "Ready",
            }
        else:
            missing_files.append(expected_name)

    is_loaded = len(found_files) == len(EXPECTED_FILES)
    is_partially_loaded = len(found_files) > 0

    # Load precomputed summary if available
    total_rows = 0
    label_dist = {}
    if SUMMARY_FILE.exists():
        try:
            with open(SUMMARY_FILE, "r") as f:
                summary_data = json.load(f)
                total_rows = summary_data.get("total_rows", 0)
                label_dist = summary_data.get("labels", {})
                for fname, f_info in summary_data.get("files", {}).items():
                    if fname in file_details:
                        file_details[fname]["rows"] = f_info.get("rows", 0)
        except Exception as e:
            logger.warning(f"Could not read dataset summary cache: {e}")

    return {
        "is_loaded": is_loaded,
        "is_partially_loaded": is_partially_loaded,
        "found_count": len(found_files),
        "total_expected": len(EXPECTED_FILES),
        "found_files": found_files,
        "missing_files": missing_files,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "total_rows": total_rows if total_rows > 0 else TOTAL_DATASET_ROWS,
        "label_distribution": label_dist,
        "file_details": file_details,
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize and clean a raw CIC-IDS2017 DataFrame:
    1. Strip leading/trailing whitespaces from column names.
    2. Remove duplicate feature columns (e.g., duplicate 'Fwd Header Length.1').
    3. Identify and clean Label column (strip whitespace, fix encoding anomalies).
    4. Convert features to numeric, replace infinity with NaN, drop NaN rows.
    5. Cast float64 to float32 to reduce memory footprint by 50%.
    """
    # 1. Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Drop known duplicate column artifacts
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])
    # Drop any remaining duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Locate Label column
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if not label_cols:
        raise ValueError("No 'Label' column found in dataset!")
    label_col = label_cols[0]

    # Standardize label values (fix encoding artifacts such as 'Web Attack ï¿½ Brute Force')
    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    # 4. Process feature columns
    feature_cols = [c for c in df.columns if c != label_col]
    
    # Coerce to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace positive and negative infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in features
    df = df.dropna(subset=feature_cols)

    # 5. Optimize memory: float64 -> float32
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df


def load_stratified_sample(
    sample_size: int = 100000,
    force_regenerate: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load or generate a representative stratified sample from all 8 CIC-IDS2017 files.
    Preserves 100% of rare attack categories (Heartbleed, Infiltration, SQLi)
    and stratifies major categories to preserve overall attack ratios.
    """
    sample_cache_file = SAMPLES_DATA_DIR / f"cicids2017_stratified_sample_{sample_size}.parquet"

    # Try loading from cache
    if not force_regenerate and sample_cache_file.exists():
        try:
            df = pd.read_parquet(sample_cache_file)
            return df
        except Exception as e:
            logger.warning(f"Error reading sample cache, regenerating: {e}")

    # Fallback to default 100k sample if requested size is 100k and default exists
    if not force_regenerate and sample_size == 100000 and DEFAULT_SAMPLE_PATH.exists():
        try:
            return pd.read_parquet(DEFAULT_SAMPLE_PATH)
        except Exception:
            pass

    # Generate sample from all 8 raw CSVs
    status = get_dataset_status()
    if not status["is_partially_loaded"]:
        raise FileNotFoundError(
            f"CIC-IDS2017 dataset files not found in {CIC_RAW_DIR}. "
            "Please ensure the official MachineLearningCSV files are placed there."
        )

    files = [CIC_RAW_DIR / f for f in status["found_files"]]
    total_raw_rows = status["total_rows"]
    base_sample_rate = sample_size / max(total_raw_rows, 1)

    file_samples = []

    for f_path in files:
        chunks = []
        for chunk in pd.read_csv(f_path, chunksize=50000, encoding="latin1", low_memory=False):
            cleaned_chunk = clean_dataframe(chunk)
            chunks.append(cleaned_chunk)

        if not chunks:
            continue

        file_df = pd.concat(chunks, ignore_index=True)
        label_col = [c for c in file_df.columns if "label" in c.lower()][0]

        # Stratified sampling per label in this file
        file_sampled_parts = []
        for lbl, group in file_df.groupby(label_col):
            n_avail = len(group)
            if n_avail <= 150:
                # Keep 100% of rare attacks
                n_take = n_avail
            else:
                n_take = max(10, int(round(n_avail * base_sample_rate)))
                n_take = min(n_take, n_avail)
            file_sampled_parts.append(group.sample(n=n_take, random_state=random_state))

        if file_sampled_parts:
            file_samples.append(pd.concat(file_sampled_parts, ignore_index=True))

    if not file_samples:
        raise ValueError("Failed to extract any samples from raw dataset.")

    master_sample = pd.concat(file_samples, ignore_index=True)

    # Standardize label column name to 'Label'
    label_cols = [c for c in master_sample.columns if "label" in c.lower()]
    if label_cols[0] != "Label":
        master_sample = master_sample.rename(columns={label_cols[0]: "Label"})

    # Add Binary_Label and Threat_Category
    master_sample["Binary_Label"] = np.where(
        master_sample["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK"
    )
    master_sample["Threat_Category"] = master_sample["Label"].map(
        lambda x: ATTACK_CATEGORIES.get(x, "Other Attacks")
    )

    # Cache sample
    try:
        master_sample.to_parquet(sample_cache_file, index=False)
        if sample_size == 100000 and not DEFAULT_SAMPLE_PATH.exists():
            master_sample.to_parquet(DEFAULT_SAMPLE_PATH, index=False)
    except Exception as e:
        logger.warning(f"Could not cache sample: {e}")

    return master_sample


def load_single_file(filename: str, max_rows: Optional[int] = 50000) -> pd.DataFrame:
    """
    Load a specific daily CSV file with optional row limit.
    """
    file_path = CIC_RAW_DIR / filename
    if not file_path.exists():
        matches = list(CIC_RAW_DIR.glob(f"*{filename}*"))
        if matches:
            file_path = matches[0]
        else:
            raise FileNotFoundError(f"File {filename} not found in {CIC_RAW_DIR}")

    if max_rows:
        df = pd.read_csv(file_path, nrows=max_rows, encoding="latin1", low_memory=False)
    else:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

    df = clean_dataframe(df)
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if label_cols and label_cols[0] != "Label":
        df = df.rename(columns={label_cols[0]: "Label"})

    df["Binary_Label"] = np.where(df["Label"] == BENIGN_LABEL, "BENIGN", "ATTACK")
    df["Threat_Category"] = df["Label"].map(lambda x: ATTACK_CATEGORIES.get(x, "Other Attacks"))
    return df
