"""
data/preprocess.py — Load, clean, and preprocess the CIC-IDS-2017 dataset.

Steps:
  1. Load all CSV files (with optional sampling)
  2. Clean column names
  3. Select the 20 features defined in feature_config.py
  4. Handle missing / infinite values
  5. Encode labels (binary + simplified 5-class)
  6. Scale features with StandardScaler
  7. Save processed train/test splits and the fitted scaler

UPGRADE: Uses simplified 5-category labels instead of 13 fine-grained labels.

Run:
    python -m data.preprocess
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Add project root to path ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    SAMPLE_PER_FILE, TEST_SIZE, RANDOM_STATE,
)
from features.feature_config import (
    DATASET_COLUMNS, FEATURE_NAMES, LABEL_COLUMN,
    SIMPLIFIED_LABELS, SIMPLIFIED_ENCODE, LABEL_BINARY,
)


def load_all_csvs(data_dir: str, sample_n: int | None) -> pd.DataFrame:
    """
    Load all CIC-IDS-2017 CSV files from *data_dir* and concatenate.

    Args:
        data_dir:  path to folder containing the CSV files
        sample_n:  if set, randomly sample this many rows per file

    Returns:
        Combined DataFrame with all files.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        print(f"  Loading {fname} ...", end=" ")
        df = pd.read_csv(fpath, low_memory=False)
        print(f"{len(df):,} rows", end="")

        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=RANDOM_STATE)
            print(f" → sampled {sample_n:,}", end="")
        print()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined):,}")
    return combined


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (CIC dataset has leading spaces)."""
    df.columns = df.columns.str.strip()
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the 20 features + label column from the full dataset."""
    # Strip the leading space from our column list for matching
    cols_stripped = [c.strip() for c in DATASET_COLUMNS]
    label_stripped = LABEL_COLUMN.strip()

    missing = [c for c in cols_stripped if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    selected = df[cols_stripped + [label_stripped]].copy()
    # Rename to our internal names
    rename_map = dict(zip(cols_stripped, FEATURE_NAMES))
    selected.rename(columns=rename_map, inplace=True)
    selected.rename(columns={label_stripped: "label"}, inplace=True)
    return selected


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN, then drop rows with any NaN."""
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    after = len(df)
    dropped = before - after
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing/infinite values")
    else:
        print("  No missing/infinite values found")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add label columns using the SIMPLIFIED 5-category scheme:
      - label_binary:     0 = BENIGN, 1 = attack
      - label_simplified: one of 5 categories (Benign, Port Scan, DoS/DDoS, Brute Force, Web Attack)
      - label_encoded:    integer for each simplified category
    """
    df["label"] = df["label"].str.strip()

    # Binary
    df["label_binary"] = df["label"].map(LABEL_BINARY)

    # Simplified 5-category label
    df["label_simplified"] = df["label"].map(SIMPLIFIED_LABELS)

    # Numeric encoding for the simplified categories
    df["label_encoded"] = df["label_simplified"].map(SIMPLIFIED_ENCODE)

    # Drop rows with unknown labels
    unknown = df["label_encoded"].isna().sum()
    if unknown:
        print(f"  Dropping {unknown} rows with unknown labels")
        df = df.dropna(subset=["label_encoded"])
    df["label_encoded"] = df["label_encoded"].astype(int)
    df["label_binary"] = df["label_binary"].astype(int)

    print("\n  Simplified label distribution:")
    for lbl, cnt in df["label_simplified"].value_counts().items():
        print(f"    {lbl:<20s}  {cnt:>8,}")
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on the 20 features and transform them."""
    scaler = StandardScaler()
    df[FEATURE_NAMES] = scaler.fit_transform(df[FEATURE_NAMES].values)
    return df, scaler


def main():
    print("=" * 60)
    print("  CIC-IDS-2017 Data Preprocessing Pipeline (UPGRADED)")
    print("  Using simplified 5-category labels")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/6] Loading CSV files ...")
    df = load_all_csvs(RAW_DATA_DIR, SAMPLE_PER_FILE)

    # Step 2: Clean column names
    print("\n[2/6] Cleaning column names ...")
    df = clean_columns(df)

    # Step 3: Select features
    print("\n[3/6] Selecting 20 features + label ...")
    df = select_features(df)
    print(f"  Shape after selection: {df.shape}")

    # Step 4: Handle missing values
    print("\n[4/6] Handling missing / infinite values ...")
    df = handle_missing_values(df)

    # Step 5: Encode labels (simplified 5-class)
    print("\n[5/6] Encoding labels (5 simplified categories) ...")
    df = encode_labels(df)

    # Step 6: Scale and split
    print("\n[6/6] Scaling features and splitting train/test ...")
    df, scaler = scale_features(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_encoded"]
    )
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Save outputs
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path  = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"\n  ✅ Saved train data  → {train_path}")
    print(f"  ✅ Saved test data   → {test_path}")
    print(f"  ✅ Saved scaler      → {scaler_path}")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
