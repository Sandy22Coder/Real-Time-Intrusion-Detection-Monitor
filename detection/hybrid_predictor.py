"""
detection/hybrid_predictor.py - Hybrid predictor with heuristic backstops.

This module combines the persisted ML models with raw-traffic heuristics so
the system can still produce concrete classifications instead of overusing the
"Suspicious" bucket.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR
from features.feature_config import FEATURE_NAMES, SIMPLIFIED_DECODE


ATTACK_EXPLANATIONS = {
    "Port Scan": "Multiple ports probed in short time -> reconnaissance activity",
    "DoS/DDoS": "High packet rate with flood patterns -> denial-of-service attempt",
    "Brute Force": "Repeated auth-port connections -> credential guessing attack",
    "Web Attack": "Anomalous HTTP patterns detected -> web exploitation attempt",
    "Suspicious Activity": "Anomalous traffic pattern does not match known profiles",
    "Benign": "Normal traffic -> no threat indicators detected",
}


class Predictor:
    """Two-layer predictor with raw-signal heuristics and explanations."""

    def __init__(self):
        rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
        if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

        self.rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
        self.if_model = joblib.load(if_path) if os.path.exists(if_path) else None
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        if self.rf_model is not None and hasattr(self.rf_model, "n_jobs"):
            self.rf_model.n_jobs = 1
        if self.if_model is not None and hasattr(self.if_model, "n_jobs"):
            self.if_model.n_jobs = 1

        print(f"  {'OK' if self.rf_model is not None else '!!'} Random Forest available: {bool(self.rf_model)}")
        print(f"  {'OK' if self.if_model is not None else '!!'} Isolation Forest available: {bool(self.if_model)}")
        print(f"  {'OK' if self.scaler is not None else '!!'} Scaler available: {bool(self.scaler)}")

    def _build_explanation(self, label: str, confidence: float,
                           if_anomaly: bool, rf_label: str | None,
                           heuristic_reason: str = "") -> str:
        parts = [ATTACK_EXPLANATIONS.get(label, ATTACK_EXPLANATIONS["Suspicious Activity"])]

        if if_anomaly and rf_label and rf_label != "Benign":
            parts.append(f"Anomaly detector and classifier both agreed ({confidence:.0%})")
        elif if_anomaly and (rf_label == "Benign" or rf_label is None):
            parts.append("Anomaly detector flagged unusual behavior while the classifier stayed uncertain")
        elif rf_label and rf_label != "Benign":
            parts.append(f"Classifier matched {rf_label} traffic ({confidence:.0%})")

        if heuristic_reason:
            parts.append(heuristic_reason)

        return " | ".join(parts)

    def _feature_signals(self, features: np.ndarray) -> dict:
        if self.scaler is not None:
            try:
                raw = self.scaler.inverse_transform(features)[0]
            except Exception:
                raw = features[0]
        else:
            raw = features[0]
        return {name: float(raw[idx]) for idx, name in enumerate(FEATURE_NAMES)}

    def _heuristic_prediction(self, signals: dict) -> dict | None:
        port = int(signals.get("destination_port", 0))
        duration_s = max(signals.get("flow_duration", 0.0) / 1e6, 0.0)
        fwd = int(round(signals.get("total_fwd_packets", 0.0)))
        bwd = int(round(signals.get("total_bwd_packets", 0.0)))
        pkts_per_s = float(signals.get("flow_pkts_per_s", 0.0))
        bytes_per_s = float(signals.get("flow_bytes_per_s", 0.0))
        syn = int(round(signals.get("syn_flag_count", 0.0)))
        ack = int(round(signals.get("ack_flag_count", 0.0)))
        rst = int(round(signals.get("rst_flag_count", 0.0)))
        psh = int(round(signals.get("fwd_psh_flags", 0.0)))
        avg_size = float(signals.get("avg_packet_size", 0.0))

        if fwd <= 4 and bwd <= 1 and syn >= 1 and ack <= 1 and duration_s <= 0.35 and avg_size <= 120:
            return {
                "label": "Port Scan",
                "confidence": 0.82 if rst else 0.77,
                "severity": "high",
                "method": "Heuristic signature",
                "reason": "Short SYN-heavy probe with almost no return traffic",
            }

        if (
            ((pkts_per_s >= 250 and fwd >= 30) or bytes_per_s >= 180000)
            and bwd <= max(4, int(fwd * 0.25))
            and syn >= 8
        ):
            return {
                "label": "DoS/DDoS",
                "confidence": 0.90,
                "severity": "critical",
                "method": "Heuristic signature",
                "reason": "Forward packet flood dominated the flow at an abnormal rate",
            }

        if port in {21, 22, 25, 110, 143, 3389} and fwd >= 3 and syn >= 1 and duration_s <= 2.0 and (rst >= 1 or bwd <= 2):
            return {
                "label": "Brute Force",
                "confidence": 0.84,
                "severity": "high",
                "method": "Heuristic signature",
                "reason": "Repeated auth-oriented exchange with short resets resembles failed login attempts",
            }

        if port in {80, 443, 8080, 8000} and fwd >= 5 and bwd >= 2 and psh >= 2 and duration_s <= 4.0 and avg_size <= 500:
            return {
                "label": "Web Attack",
                "confidence": 0.81,
                "severity": "high",
                "method": "Heuristic signature",
                "reason": "Interactive web flow with repeated client pushes resembles exploit traffic",
            }

        if bwd >= 2 and ack >= 2 and syn <= 2 and pkts_per_s < 220 and bytes_per_s < 180000:
            return {
                "label": "Benign",
                "confidence": 0.78,
                "severity": "none",
                "method": "Traffic profile heuristic",
                "reason": "Balanced request-response pattern matches ordinary client/server traffic",
            }

        return None

    def predict(self, features: np.ndarray) -> dict:
        result = {
            "label": "Unknown",
            "is_attack": False,
            "confidence": 0.0,
            "method": "none",
            "anomaly": False,
            "severity": "none",
            "explanation": "",
            "class_probabilities": {},
            "signals": {},
        }

        signals = self._feature_signals(features)
        heuristic = self._heuristic_prediction(signals)
        result["signals"] = signals

        if_anomaly = False
        if_score = 0.0
        if self.if_model is not None:
            if_pred = self.if_model.predict(features)[0]
            if_anomaly = bool(if_pred == -1)
            if_score = float(self.if_model.decision_function(features)[0])
            result["anomaly"] = if_anomaly

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

        if heuristic and heuristic["label"] != "Benign" and (rf_label in {None, "Benign"} or rf_confidence < 0.70):
            result["label"] = heuristic["label"]
            result["confidence"] = heuristic["confidence"]
            result["is_attack"] = True
            result["severity"] = heuristic["severity"]
            result["method"] = heuristic["method"]

        elif rf_label is not None and rf_label != "Benign":
            result["label"] = rf_label
            result["confidence"] = rf_confidence
            result["is_attack"] = True

            if rf_confidence >= 0.72 or (heuristic and heuristic["label"] == rf_label):
                result["severity"] = "critical" if if_anomaly else "high"
                result["method"] = "Hybrid (RF + IF)" if if_anomaly else "Random Forest"
                if if_anomaly:
                    result["confidence"] = min(rf_confidence + 0.03, 1.0)
                if heuristic and heuristic["label"] == rf_label:
                    result["confidence"] = max(result["confidence"], heuristic["confidence"])
                    result["method"] += " + Heuristic"
            else:
                if heuristic and heuristic["label"] == rf_label:
                    result["label"] = rf_label
                    result["confidence"] = max(rf_confidence, heuristic["confidence"])
                    result["severity"] = heuristic["severity"]
                    result["method"] = "Random Forest + Heuristic"
                else:
                    result["label"] = f"Suspicious: {rf_label}"
                    result["severity"] = "medium"
                    result["method"] = "Random Forest (low confidence)"

        elif if_anomaly:
            anomaly_conf = min(max(0.5 - if_score, 0.4), 0.75)
            attack_proba = {k: v for k, v in rf_class_proba.items() if k != "Benign"}
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

        elif heuristic and heuristic["label"] == "Benign":
            result["label"] = "Benign"
            result["is_attack"] = False
            result["confidence"] = max(rf_confidence if rf_label else 0.0, heuristic["confidence"])
            result["method"] = "Hybrid (normal) + Heuristic"
            result["severity"] = "none"

        else:
            result["label"] = "Benign"
            result["is_attack"] = False
            result["confidence"] = rf_confidence if rf_label else 0.95
            result["method"] = "Hybrid (both normal)"
            result["severity"] = "none"

        core_label = result["label"].replace("Suspicious: ", "")
        result["explanation"] = self._build_explanation(
            label=core_label,
            confidence=result["confidence"],
            if_anomaly=if_anomaly,
            rf_label=rf_label,
            heuristic_reason=heuristic["reason"] if heuristic else "",
        )

        return result
