"""
dashboard/app.py — Enhanced Flask dashboard with attack classification,
system status, IP blocking, and top attacker tracking.

UPGRADED Endpoints:
  GET /              — main dashboard page
  GET /api/stats     — live traffic statistics + system status (JSON)
  GET /api/alerts    — recent alerts with classification (JSON)
  GET /api/traffic   — time-series traffic data for charts (JSON)
  GET /api/top_attackers — top 10 attacker IPs (JSON)
  GET /api/blocked_ips   — currently blocked IPs (JSON)
"""
import os
import sys
import time
import threading
from collections import deque, Counter
from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DASHBOARD_HOST, DASHBOARD_PORT

# ── Flask app setup ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


class DashboardState:
    """
    Shared state between the IDS engine and the Flask dashboard.
    UPGRADED: tracks system status, top attackers, and blocked IPs.
    """

    def __init__(self):
        self.total_packets = 0
        self.total_flows = 0
        self.total_attacks = 0
        self.total_normal = 0
        self.total_blocked = 0
        self.attack_types = {}           # {type: count}
        self.recent_alerts = []          # list of alert dicts
        self.traffic_history = deque(maxlen=60)  # last 60 data points
        self.attacker_counter = Counter()  # IP → hit count
        self.blocked_ips = []            # list of blocked IP dicts
        self._lock = threading.Lock()
        self.start_time = time.time()

        # System status: based on attack rate
        self._recent_attack_times = deque(maxlen=100)

    def update_flow(self, is_attack: bool, label: str = "",
                    src_ip: str = ""):
        with self._lock:
            self.total_flows += 1
            if is_attack:
                self.total_attacks += 1
                self.attack_types[label] = self.attack_types.get(label, 0) + 1
                self._recent_attack_times.append(time.time())
                if src_ip:
                    self.attacker_counter[src_ip] += 1
            else:
                self.total_normal += 1

    def add_alert(self, alert: dict):
        with self._lock:
            self.recent_alerts.append(alert)
            if len(self.recent_alerts) > 200:
                self.recent_alerts = self.recent_alerts[-200:]

    def set_packet_count(self, count: int):
        with self._lock:
            self.total_packets = count

    def set_blocked_ips(self, blocked_list: list):
        """Update the blocked IPs list from IPBlocker."""
        with self._lock:
            self.blocked_ips = blocked_list
            self.total_blocked = len(blocked_list)

    def record_traffic_point(self):
        """Record a traffic snapshot for time-series chart."""
        with self._lock:
            self.traffic_history.append({
                "time": time.strftime("%H:%M:%S"),
                "packets": self.total_packets,
                "attacks": self.total_attacks,
                "normal": self.total_normal,
                "flows": self.total_flows,
            })

    def get_system_status(self) -> dict:
        """
        Determine system status based on recent attack rate.
        
        - Normal:       < 5 attacks in last 30 seconds
        - Warning:      5-20 attacks in last 30 seconds
        - Under Attack: > 20 attacks in last 30 seconds
        """
        with self._lock:
            now = time.time()
            recent = sum(1 for t in self._recent_attack_times
                        if now - t < 30)

            if recent > 20:
                status = "Under Attack"
                level = "critical"
            elif recent > 5:
                status = "Warning"
                level = "warning"
            else:
                status = "Normal"
                level = "normal"

            return {
                "status": status,
                "level": level,
                "attacks_last_30s": recent,
                "total_attacks": self.total_attacks,
                "total_blocked": self.total_blocked,
            }

    def get_stats(self) -> dict:
        with self._lock:
            uptime = int(time.time() - self.start_time)
            return {
                "total_packets": self.total_packets,
                "total_flows": self.total_flows,
                "total_attacks": self.total_attacks,
                "total_normal": self.total_normal,
                "total_blocked": self.total_blocked,
                "attack_types": dict(self.attack_types),
                "uptime_seconds": uptime,
            }

    def get_alerts(self) -> list:
        with self._lock:
            return list(reversed(self.recent_alerts[-50:]))

    def get_traffic(self) -> list:
        with self._lock:
            return list(self.traffic_history)

    def get_top_attackers(self, n: int = 10) -> list:
        with self._lock:
            return [{"ip": ip, "count": count}
                    for ip, count in self.attacker_counter.most_common(n)]


# ── Global state instance (set by main.py) ───────────────────────────────
state = DashboardState()


# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    return jsonify(state.get_stats())


@app.route("/api/alerts")
def api_alerts():
    return jsonify(state.get_alerts())


@app.route("/api/traffic")
def api_traffic():
    return jsonify(state.get_traffic())


@app.route("/api/top_attackers")
def api_top_attackers():
    return jsonify(state.get_top_attackers())


@app.route("/api/blocked_ips")
def api_blocked_ips():
    with state._lock:
        return jsonify(state.blocked_ips)


@app.route("/api/system_status")
def api_system_status():
    return jsonify(state.get_system_status())


def start_dashboard(shared_state: "DashboardState" = None):
    """Start the Flask dashboard in a background thread."""
    global state
    if shared_state:
        state = shared_state

    thread = threading.Thread(
        target=lambda: app.run(
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    thread.start()
    print(f"  🌐 Dashboard started at http://localhost:{DASHBOARD_PORT}")
    return thread
