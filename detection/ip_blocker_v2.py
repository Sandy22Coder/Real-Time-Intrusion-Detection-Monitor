"""
detection/ip_blocker_v2.py - IP blocking with Windows firewall support.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOGS_DIR


class IPBlocker:
    """Manage blocked IPs with firewall enforcement when available."""

    def __init__(self, confidence_threshold: float = 0.80):
        self._blocked = {}
        self._lock = threading.Lock()
        self.confidence_threshold = confidence_threshold
        self._is_linux = platform.system() == "Linux"
        self._is_windows = platform.system() == "Windows"
        self._block_count = 0
        self._state_path = os.path.join(LOGS_DIR, "blocked_ips.json")
        self.whitelist = {
            "127.0.0.1",
            "0.0.0.0",
            "localhost",
            "192.168.1.1",
            "10.0.0.1",
        }

    def should_block(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold

    def block_ip(self, ip: str, attack_type: str, confidence: float) -> dict:
        if ip in self.whitelist:
            return {"status": "whitelisted", "ip": ip}

        with self._lock:
            if ip in self._blocked:
                self._blocked[ip]["hit_count"] += 1
                self._persist_blocklist()
                return {"status": "already_blocked", "ip": ip, **self._blocked[ip]}

            entry = {
                "ip": ip,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "attack_type": attack_type,
                "confidence": round(confidence, 3),
                "hit_count": 1,
                "enforced": False,
                "mode": "app_containment",
                "details": "",
            }
            self._blocked[ip] = entry
            self._block_count += 1

        enforced, mode, details = self._enforce_block(ip)
        with self._lock:
            self._blocked[ip]["enforced"] = enforced
            self._blocked[ip]["mode"] = mode
            self._blocked[ip]["details"] = details
            self._persist_blocklist()

        action = "BLOCKED (firewall)" if enforced else "BLOCKED (app containment)"
        print(f"  BLOCK {action}: {ip} [{attack_type}, conf={confidence:.0%}]")

        return {
            "status": "blocked",
            "ip": ip,
            "enforced": enforced,
            "attack_type": attack_type,
            "mode": mode,
            "details": details,
        }

    def _enforce_block(self, ip: str) -> tuple[bool, str, str]:
        if self._is_linux:
            try:
                subprocess.run(
                    ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                    check=True, capture_output=True, timeout=5,
                )
                return True, "firewall", "iptables INPUT DROP rule created"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, "app_containment", f"iptables unavailable: {type(exc).__name__}"

        if self._is_windows:
            rule_base = f"IDS_BLOCK_{ip}"
            commands = [
                [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_base}_IN", "dir=in", "action=block",
                    f"remoteip={ip}",
                ],
                [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_base}_OUT", "dir=out", "action=block",
                    f"remoteip={ip}",
                ],
            ]
            try:
                for cmd in commands:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=5)
                return True, "firewall", "Windows Defender Firewall rules added"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, "app_containment", f"netsh firewall rule failed: {type(exc).__name__}"

        return False, "app_containment", "platform firewall integration not configured"

    def unblock_ip(self, ip: str) -> bool:
        with self._lock:
            if ip in self._blocked:
                del self._blocked[ip]
                self._persist_blocklist()
                return True
        return False

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            return ip in self._blocked

    def get_blocked_list(self) -> list:
        with self._lock:
            return sorted(self._blocked.values(), key=lambda x: x["timestamp"], reverse=True)

    @property
    def blocked_count(self) -> int:
        with self._lock:
            return len(self._blocked)

    @property
    def total_blocks(self) -> int:
        return self._block_count

    def _persist_blocklist(self):
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(list(self._blocked.values()), f, ensure_ascii=False, indent=2)
        except OSError:
            pass
