"""
detection/attack_correlator.py - Context-aware refinement for repeated attacks.

Single-flow ML predictions are useful, but several common attacks only become
obvious when the same source repeats a pattern over a short time window. This
module upgrades low-confidence detections into presentable, concrete labels.
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict, deque


class AttackCorrelator:
    """Track short-term source behavior and refine tentative predictions."""

    def __init__(self, window_seconds: int = 25):
        self.window_seconds = window_seconds
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def refine(self, src_ip: str, prediction: dict) -> dict:
        now = time.time()
        signals = prediction.get("signals", {})
        if not src_ip or not signals:
            return prediction

        event = {
            "time": now,
            "port": int(signals.get("destination_port", 0)),
            "label": str(prediction.get("label", "")),
            "confidence": float(prediction.get("confidence", 0.0)),
            "pkts_per_s": float(signals.get("flow_pkts_per_s", 0.0)),
            "fwd": int(signals.get("total_fwd_packets", 0)),
            "bwd": int(signals.get("total_bwd_packets", 0)),
            "syn": int(signals.get("syn_flag_count", 0)),
        }

        with self._lock:
            buf = self._events[src_ip]
            buf.append(event)
            while buf and now - buf[0]["time"] > self.window_seconds:
                buf.popleft()
            recent = list(buf)

        distinct_ports = {e["port"] for e in recent if e["port"] > 0}
        auth_hits = sum(1 for e in recent if e["port"] in {21, 22, 25, 110, 143, 3389})
        web_hits = sum(1 for e in recent if e["port"] in {80, 443, 8080, 8000})
        flood_hits = sum(1 for e in recent if e["pkts_per_s"] > 450 or e["fwd"] >= 30)

        label = prediction.get("label", "")
        updated = dict(prediction)

        if len(distinct_ports) >= 6 and any("Scan" in e["label"] or e["syn"] >= 1 for e in recent):
            return self._upgrade(
                updated, "Port Scan", 0.94, "high", "Correlation engine: one source probed many destination ports in a short window"
            )

        if auth_hits >= 5 and any("Brute" in e["label"] or e["fwd"] >= 3 for e in recent):
            return self._upgrade(
                updated, "Brute Force", 0.91, "high", "Correlation engine: repeated authentication-targeted connections from the same source"
            )

        if flood_hits >= 4 and len(recent) >= 5:
            return self._upgrade(
                updated, "DoS/DDoS", 0.93, "critical", "Correlation engine: sustained high-rate burst from the same source"
            )

        if web_hits >= 4 and any("Web" in e["label"] for e in recent):
            return self._upgrade(
                updated, "Web Attack", 0.88, "high", "Correlation engine: repeated malicious-looking web requests from the same source"
            )

        if label.startswith("Suspicious:") and len(recent) >= 3:
            core = label.replace("Suspicious: ", "")
            return self._upgrade(
                updated, core, max(float(updated.get("confidence", 0.0)), 0.78), "medium",
                "Correlation engine: the same suspicious pattern repeated multiple times"
            )

        return updated

    @staticmethod
    def _upgrade(prediction: dict, label: str, confidence: float,
                 severity: str, explanation: str) -> dict:
        prediction["label"] = label
        prediction["is_attack"] = True
        prediction["confidence"] = max(float(prediction.get("confidence", 0.0)), confidence)
        prediction["severity"] = severity
        prediction["method"] = prediction.get("method", "Hybrid")
        if "Correlation" not in prediction["method"]:
            prediction["method"] = f"{prediction['method']} + Correlation"
        base = prediction.get("explanation", "")
        prediction["explanation"] = f"{base} | {explanation}" if base else explanation
        return prediction
