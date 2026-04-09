"""
detection/ip_blocker.py — Auto-response system for blocking attacker IPs.

When a high-confidence attack is detected, the attacker IP is automatically
added to a blocklist.  On Linux this would use iptables; on Windows we
simulate the block and maintain the list for the dashboard.

Features:
  - Thread-safe blocklist management
  - Duplicate prevention (won't block same IP twice)
  - Configurable confidence threshold
  - Whitelist for safe IPs (localhost, gateway, etc.)
"""
import os
import sys
import datetime
import threading
import platform
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IPBlocker:
    """
    Manages a blocklist of attacker IPs with optional OS-level enforcement.
    
    Attributes:
        blocked_ips:    dict mapping IP → {timestamp, reason, attack_type}
        whitelist:      set of IPs that should never be blocked
        confidence_threshold: minimum confidence to auto-block (default 0.70)
    """

    def __init__(self, confidence_threshold: float = 0.85):
        self._blocked = {}           # ip → info dict
        self._lock = threading.Lock()
        self.confidence_threshold = confidence_threshold
        self._is_linux = platform.system() == "Linux"

        # IPs that should NEVER be blocked
        self.whitelist = {
            "127.0.0.1",
            "0.0.0.0",
            "localhost",
            "192.168.1.1",    # typical gateway
            "10.0.0.1",
        }

        self._block_count = 0

    def should_block(self, confidence: float) -> bool:
        """Check if confidence exceeds the auto-block threshold."""
        return confidence >= self.confidence_threshold

    def block_ip(self, ip: str, attack_type: str, confidence: float) -> dict:
        """
        Block an attacker IP address.
        
        Args:
            ip:          IP address to block
            attack_type: type of attack detected
            confidence:  model confidence score

        Returns:
            dict with block result info, or None if IP was already blocked/whitelisted
        """
        if ip in self.whitelist:
            return {"status": "whitelisted", "ip": ip}

        with self._lock:
            if ip in self._blocked:
                # Already blocked — increment hit count
                self._blocked[ip]["hit_count"] += 1
                return {"status": "already_blocked", "ip": ip}

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._blocked[ip] = {
                "ip": ip,
                "timestamp": timestamp,
                "attack_type": attack_type,
                "confidence": round(confidence, 3),
                "hit_count": 1,
                "enforced": False,
            }
            self._block_count += 1

        # Attempt OS-level block
        enforced = self._enforce_block(ip)
        with self._lock:
            self._blocked[ip]["enforced"] = enforced

        action = "BLOCKED (enforced)" if enforced else "BLOCKED (simulated)"
        print(f"  🚫 {action}: {ip}  [{attack_type}, conf={confidence:.0%}]")

        return {
            "status": "blocked",
            "ip": ip,
            "enforced": enforced,
            "attack_type": attack_type,
        }

    def _enforce_block(self, ip: str) -> bool:
        """
        Try to enforce the block at the OS level.
        
        Linux:   iptables -A INPUT -s <ip> -j DROP
        Windows: netsh advfirewall (requires admin)
        
        Returns True if OS-level block was applied, False if simulated.
        """
        if self._is_linux:
            try:
                subprocess.run(
                    ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                    check=True, capture_output=True, timeout=5,
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError, 
                    subprocess.TimeoutExpired):
                return False
        else:
            # Windows: simulate (would need admin + netsh advfirewall)
            # In production, uncomment:
            # subprocess.run([
            #     "netsh", "advfirewall", "firewall", "add", "rule",
            #     f"name=IDS_BLOCK_{ip}", "dir=in", "action=block",
            #     f"remoteip={ip}"
            # ], check=True)
            return False

    def unblock_ip(self, ip: str) -> bool:
        """Remove an IP from the blocklist."""
        with self._lock:
            if ip in self._blocked:
                del self._blocked[ip]
                return True
            return False

    def is_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked."""
        with self._lock:
            return ip in self._blocked

    def get_blocked_list(self) -> list:
        """Return list of all blocked IPs with metadata, sorted by timestamp."""
        with self._lock:
            return sorted(
                self._blocked.values(),
                key=lambda x: x["timestamp"],
                reverse=True,
            )

    @property
    def blocked_count(self) -> int:
        with self._lock:
            return len(self._blocked)

    @property
    def total_blocks(self) -> int:
        return self._block_count
