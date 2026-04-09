"""
detection/alert_manager.py — OPTIMIZED alert system with forensics + explainability.

Features:
  - Severity levels (critical / high / medium / low)
  - Human-readable explanations per alert (for viva / forensics)
  - Structured JSON forensics log (logs/forensics.jsonl)
  - Top attacker IP tracking
  - Thread-safe with deque storage
"""
import os
import sys
import json
import datetime
import threading
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ALERT_LOG_FILE, MAX_ALERTS_STORED, LOGS_DIR


class AlertManager:
    """Thread-safe alert manager with console + file + forensics + tracking."""

    def __init__(self):
        self._alerts = deque(maxlen=MAX_ALERTS_STORED)
        self._lock = threading.Lock()
        self._alert_count = 0
        self._attacker_counter = Counter()

        os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
        self._forensics_path = os.path.join(LOGS_DIR, "forensics.jsonl")

    def raise_alert(self, label: str, src_ip: str, dst_ip: str,
                    confidence: float, method: str,
                    action_taken: str = "Logged",
                    anomaly: bool = False,
                    severity: str = "medium",
                    explanation: str = ""):
        """Generate an alert with full context and explanation."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert = {
            "id": self._alert_count + 1,
            "timestamp": timestamp,
            "label": str(label),
            "src_ip": str(src_ip),
            "dst_ip": str(dst_ip),
            "confidence": round(float(confidence), 3),
            "method": str(method),
            "action_taken": str(action_taken),
            "anomaly": bool(anomaly),
            "severity": str(severity),
            "explanation": str(explanation),
        }

        with self._lock:
            self._alerts.append(alert)
            self._alert_count += 1
            self._attacker_counter[src_ip] += 1

        self._print_alert(alert)
        self._log_alert(alert)
        self._log_forensics(alert)

    def _print_alert(self, alert: dict):
        """Print a formatted alert with severity + explanation."""
        severity_icons = {
            "critical": "🔴", "high": "🟠",
            "medium": "🟡", "low": "🔵", "none": "⚪",
        }
        sev_icon = severity_icons.get(alert["severity"], "⚪")
        action_icon = "🚫" if alert["action_taken"] == "IP Blocked" else "📝"

        print(f"\n  ⚠️  ALERT #{alert['id']}  |  {alert['timestamp']}  |  {sev_icon} {alert['severity'].upper()}")
        print(f"      Attack    : {alert['label']}")
        print(f"      Source    : {alert['src_ip']}  →  {alert['dst_ip']}")
        print(f"      Confidence: {alert['confidence']:.1%}")
        print(f"      Model     : {alert['method']}")
        print(f"      Action    : {action_icon} {alert['action_taken']}")
        if alert["explanation"]:
            print(f"      Reason    : {alert['explanation'][:80]}")
        print(f"  {'─' * 55}")

    def _log_alert(self, alert: dict):
        """Append alert to plain-text log file."""
        try:
            with open(ALERT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(
                    f"[{alert['timestamp']}] [{alert['severity'].upper()}] "
                    f"{alert['label']} | "
                    f"{alert['src_ip']} → {alert['dst_ip']} | "
                    f"conf={alert['confidence']} | "
                    f"{alert['method']} | "
                    f"action={alert['action_taken']} | "
                    f"reason={alert['explanation']}\n"
                )
        except OSError:
            pass

    def _log_forensics(self, alert: dict):
        """Append structured JSON to forensics log."""
        try:
            with open(self._forensics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def get_recent_alerts(self, n: int = 50) -> list:
        """Return the most recent *n* alerts (newest first)."""
        with self._lock:
            return list(reversed(list(self._alerts)))[:n]

    @property
    def total_alerts(self) -> int:
        return self._alert_count

    def get_attack_stats(self) -> dict:
        """Return a count of each attack type seen."""
        stats = {}
        with self._lock:
            for alert in self._alerts:
                lbl = alert["label"]
                stats[lbl] = stats.get(lbl, 0) + 1
        return stats

    def get_top_attackers(self, n: int = 10) -> list:
        """Return the top N attacker IPs by frequency."""
        with self._lock:
            return [{"ip": ip, "count": count}
                    for ip, count in self._attacker_counter.most_common(n)]
