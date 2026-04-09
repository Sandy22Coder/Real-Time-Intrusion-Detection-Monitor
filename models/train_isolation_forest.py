"""
models/train_isolation_forest.py — Train an Isolation Forest for anomaly-based
intrusion detection.

Why Isolation Forest?
  - Unsupervised — learns what "normal" traffic looks like
  - Catches previously unseen (zero-day) attacks
  - Very fast at inference time
  - Complementary to the supervised Random Forest

Training approach:
  - Train ONLY on BENIGN traffic
  - At test time, anything scored as an anomaly (-1) is flagged as an attack

Run:
    python -m models.train_isolation_forest
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR,
    IF_N_ESTIMATORS, IF_CONTAMINATION, RANDOM_STATE,
)
from features.feature_config import FEATURE_NAMES


def load_data():
    """Load preprocessed data; separate BENIGN for training."""
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))
    return train, test


def train_model(X_train_normal: np.ndarray) -> IsolationForest:
    """Train Isolation Forest on normal traffic only."""
    print(f"  Training Isolation Forest  (n_estimators={IF_N_ESTIMATORS}, "
          f"contamination={IF_CONTAMINATION}) ...")
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_normal)
    return model


def evaluate(model, X_test, y_test_binary):
    """
    Evaluate anomaly detection performance.
    
    Isolation Forest outputs:
       1  → inlier  (normal)
      -1  → outlier (attack)
    
    We convert to our binary: 0=normal, 1=attack
    """
    raw_pred = model.predict(X_test)
    # Convert:  1 (inlier) → 0 (normal),  -1 (outlier) → 1 (attack)
    y_pred = np.where(raw_pred == -1, 1, 0)

    acc  = accuracy_score(y_test_binary, y_pred)
    prec = precision_score(y_test_binary, y_pred, zero_division=0)
    rec  = recall_score(y_test_binary, y_pred, zero_division=0)
    f1   = f1_score(y_test_binary, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("  Isolation Forest — Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test_binary, y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))


def main():
    print("=" * 60)
    print("  Isolation Forest — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading preprocessed data ...")
    train_df, test_df = load_data()

    # Isolation Forest trains only on BENIGN
    train_normal = train_df[train_df["label_binary"] == 0]
    X_train = train_normal[FEATURE_NAMES].values
    X_test  = test_df[FEATURE_NAMES].values
    y_test  = test_df["label_binary"].values

    print(f"  Training on BENIGN only: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples  "
          f"(normal={np.sum(y_test==0):,}, attack={np.sum(y_test==1):,})")

    # Train
    print("\n[2/3] Training model ...")
    model = train_model(X_train)

    # Evaluate
    print("\n[3/3] Evaluating model ...")
    evaluate(model, X_test, y_test)

    # Save
    model_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    joblib.dump(model, model_path)
    print(f"\n  ✅ Model saved → {model_path}")


if __name__ == "__main__":
    main()
