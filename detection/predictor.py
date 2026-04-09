"""
detection/predictor.py — OPTIMIZED two-layer hybrid prediction engine.

FINAL ARCHITECTURE:
  Layer 1: Isolation Forest → anomaly score
  Layer 2: Random Forest   → attack classification + predict_proba confidence

DECISION LOGIC (smarter, fewer false positives):
  1. RF classifies → if confidence > 0.80 → "Confirmed Attack"
  2. RF classifies → if confidence 0.50-0.80 → "Suspicious Activity"
  3. IF flags anomaly + RF says benign → "Suspicious Activity"
  4. Both say normal → Benign

EXPLAINABILITY:
  Each prediction includes a human-readable explanation of WHY
  the model made that decision (for viva / forensics).
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR
from features.feature_config import SIMPLIFIED_DECODE, FEATURE_NAMES


# ── Explanation templates per attack type ─────────────────────────────
ATTACK_EXPLANATIONS = {
    "Port Scan": "Multiple ports probed in short time → reconnaissance activity",
    "DoS/DDoS": "High packet rate with flood patterns → denial-of-service attempt",
    "Brute Force": "Repeated auth-port connections → credential guessing attack",
    "Web Attack": "Anomalous HTTP patterns detected → web exploitation attempt",
    "Suspicious Activity": "Anomalous traffic pattern does not match known profiles",
    "Benign": "Normal traffic — no threat indicators detected",
}


class Predictor:
    """
    Optimized two-layer hybrid predictor with explainability.
    """

    def __init__(self):
        rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
        if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")

        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            print(f"  ✅ Loaded Random Forest from {rf_path}")
        else:
            self.rf_model = None
            print("  ⚠ Random Forest model not found")

        if os.path.exists(if_path):
            self.if_model = joblib.load(if_path)
            print(f"  ✅ Loaded Isolation Forest from {if_path}")
        else:
            self.if_model = None
            print("  ⚠ Isolation Forest model not found")

    def _build_explanation(self, label: str, confidence: float,
                           if_anomaly: bool, rf_label: str,
                           top_features: list = None) -> str:
        """
        Build a human-readable explanation for WHY this prediction was made.
        """
        base = ATTACK_EXPLANATIONS.get(label,
               ATTACK_EXPLANATIONS.get("Suspicious Activity"))

        parts = [base]

        if if_anomaly and rf_label and rf_label != "Benign":
            parts.append(f"Both anomaly detector and classifier agree ({confidence:.0%} confidence)")
        elif if_anomaly and (rf_label == "Benign" or rf_label is None):
            parts.append("Anomaly detector flagged unusual behavior, classifier uncertain")
        elif rf_label and rf_label != "Benign":
            parts.append(f"Classifier identified {rf_label} pattern ({confidence:.0%} confidence)")

        if top_features:
            feat_str = ", ".join(top_features[:3])
            parts.append(f"Key indicators: {feat_str}")

        return " | ".join(parts)

    def predict(self, features: np.ndarray) -> dict:
        """
        Optimized two-layer hybrid prediction with explainability.

        Returns dict with:
          label, is_attack, confidence, method, anomaly,
          severity, explanation, class_probabilities
        """
        result = {
            "label": "Unknown",
            "is_attack": False,
            "confidence": 0.0,
            "method": "none",
            "anomaly": False,
            "severity": "none",       # none / low / medium / high / critical
            "explanation": "",
            "class_probabilities": {},
        }

        # ══════════════════════════════════════════════════════════════
        #  LAYER 1: Isolation Forest — Anomaly Detection
        # ══════════════════════════════════════════════════════════════
        if_anomaly = False
        if_score = 0.0
        if self.if_model is not None:
            if_pred = self.if_model.predict(features)[0]
            if_anomaly = bool(if_pred == -1)
            if_score = float(self.if_model.decision_function(features)[0])
            result["anomaly"] = if_anomaly

        # ══════════════════════════════════════════════════════════════
        #  LAYER 2: Random Forest — Attack Classification
        # ══════════════════════════════════════════════════════════════
        rf_label = None
        rf_confidence = 0.0
        rf_class_proba = {}

        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(features)[0]
            rf_proba = self.rf_model.predict_proba(features)[0]
            rf_confidence = float(np.max(rf_proba))
            rf_label = SIMPLIFIED_DECODE.get(int(rf_pred), "Unknown")

            for i, prob in enumerate(rf_proba):
                cls_name = SIMPLIFIED_DECODE.get(i, f"class_{i}")
                rf_class_proba[cls_name] = round(float(prob), 3)

            result["class_probabilities"] = rf_class_proba

        # ══════════════════════════════════════════════════════════════
        #  SMART COMBINED DECISION
        # ══════════════════════════════════════════════════════════════

        if rf_label is not None and rf_label != "Benign":
            # ── RF classifies as attack ──────────────────────────────
            result["label"] = rf_label
            result["confidence"] = rf_confidence
            result["is_attack"] = True

            if rf_confidence >= 0.80:
                # HIGH confidence → Confirmed Attack
                result["severity"] = "critical" if if_anomaly else "high"
                result["method"] = "Hybrid (RF + IF)" if if_anomaly else "Random Forest"
                if if_anomaly:
                    result["confidence"] = min(rf_confidence + 0.03, 1.0)
            else:
                # MEDIUM confidence → Suspicious but not confirmed
                result["label"] = f"Suspicious: {rf_label}"
                result["severity"] = "medium"
                result["method"] = "Random Forest (low confidence)"

        elif if_anomaly:
            # ── RF says Benign but IF detects anomaly ────────────────
            # Use IF anomaly score for confidence instead of static 0.6
            anomaly_conf = min(max(0.5 - if_score, 0.4), 0.75)

            # Check if RF's second-best class gives a hint
            attack_proba = {k: v for k, v in rf_class_proba.items()
                           if k != "Benign"}
            if attack_proba:
                best_attack = max(attack_proba, key=attack_proba.get)
                best_prob = attack_proba[best_attack]

                if best_prob > 0.10:
                    result["label"] = f"Suspicious: {best_attack}"
                    result["confidence"] = round(anomaly_conf + best_prob, 3)
                    result["method"] = f"Hybrid (IF anomaly + RF hints {best_attack})"
                else:
                    result["label"] = "Suspicious Activity"
                    result["confidence"] = round(anomaly_conf, 3)
                    result["method"] = "Isolation Forest (anomaly)"
            else:
                result["label"] = "Suspicious Activity"
                result["confidence"] = round(anomaly_conf, 3)
                result["method"] = "Isolation Forest (anomaly)"

            result["is_attack"] = True
            result["severity"] = "low"

        else:
            # ── Both say normal ──────────────────────────────────────
            result["label"] = "Benign"
            result["is_attack"] = False
            result["confidence"] = rf_confidence if rf_label else 0.95
            result["method"] = "Hybrid (both normal)"
            result["severity"] = "none"

        # ── Build explanation ────────────────────────────────────────
        # Extract the core label (strip "Suspicious: " prefix for lookup)
        core_label = result["label"].replace("Suspicious: ", "")
        result["explanation"] = self._build_explanation(
            label=core_label,
            confidence=result["confidence"],
            if_anomaly=if_anomaly,
            rf_label=rf_label,
        )

        return result
