"""
models/train_random_forest.py — Train a Random Forest classifier for
SIMPLIFIED 5-class intrusion detection.

Categories: Benign, Port Scan, DoS/DDoS, Brute Force, Web Attack

Why Random Forest?
  - Handles high-dimensional data well
  - Provides feature importance scores and predict_proba for confidence
  - Fast inference (important for real-time)
  - Robust to outliers and noisy features

Run:
    python -m models.train_random_forest
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR, RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_STATE
from features.feature_config import FEATURE_NAMES, SIMPLIFIED_DECODE


def load_data():
    """Load preprocessed train and test CSVs."""
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))
    return train, test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest with the configured hyperparameters."""
    print(f"  Training Random Forest  (n_estimators={RF_N_ESTIMATORS}, "
          f"max_depth={RF_MAX_DEPTH}) ...")
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,          # use all CPU cores
        class_weight="balanced",  # handle class imbalance
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Print accuracy, precision, recall, F1, and classification report."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 50)
    print("  Random Forest — Evaluation Results (5-class)")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")

    # Per-class report using simplified labels
    target_names = [SIMPLIFIED_DECODE.get(i, f"class_{i}")
                    for i in sorted(np.unique(np.concatenate([y_test, y_pred])))]
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names,
                                zero_division=0))

    # Feature importance (top 10)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("  Top 10 Important Features:")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:<25s}  {importances[idx]:.4f}")

    # Confidence check — show predict_proba distribution
    proba = model.predict_proba(X_test)
    max_proba = np.max(proba, axis=1)
    print(f"\n  Confidence Distribution:")
    print(f"    Mean  : {np.mean(max_proba):.3f}")
    print(f"    Min   : {np.min(max_proba):.3f}")
    print(f"    >90%  : {np.sum(max_proba > 0.9) / len(max_proba):.1%}")
    print(f"    >80%  : {np.sum(max_proba > 0.8) / len(max_proba):.1%}")


def main():
    print("=" * 60)
    print("  Random Forest Classifier — Training Pipeline (5-class)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading preprocessed data ...")
    train_df, test_df = load_data()

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label_encoded"].values
    X_test  = test_df[FEATURE_NAMES].values
    y_test  = test_df["label_encoded"].values

    print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}  |  Classes: {len(np.unique(y_train))}")

    # Train
    print("\n[2/3] Training model ...")
    model = train_model(X_train, y_train)

    # Evaluate
    print("\n[3/3] Evaluating model ...")
    evaluate(model, X_test, y_test)

    # Save
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    joblib.dump(model, model_path)
    print(f"\n  ✅ Model saved → {model_path}")


if __name__ == "__main__":
    main()
