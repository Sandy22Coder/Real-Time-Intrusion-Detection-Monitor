"""
main.py — Entry point for the UPGRADED AI-Based Network IDS.

Modes:
    python main.py                      # live capture (admin + Npcap required)
    python main.py --demo               # demo with REAL dataset features
    python main.py --dashboard-only     # just the dashboard
"""
import os
import sys
import time
import argparse
import threading
import random
import numpy as np
import pandas as pd
from queue import Queue, Empty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import FLOW_TIMEOUT_SEC, PROCESSED_DATA_DIR
from features.feature_extractor import FeatureExtractor
from detection.hybrid_predictor import Predictor
from detection.attack_correlator import AttackCorrelator
from detection.alert_manager import AlertManager
from detection.ip_blocker_v2 import IPBlocker
from dashboard.app import DashboardState, start_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Network Intrusion Detection & Prevention System"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with simulated traffic")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Start only the dashboard (no capture)")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface to capture on")
    parser.add_argument("--no-block", action="store_true",
                        help="Disable auto IP blocking")
    return parser.parse_args()


def prediction_loop(flow_queue: Queue, extractor: FeatureExtractor,
                    predictor: Predictor, alert_mgr: AlertManager,
                    ip_blocker: IPBlocker, dash_state: DashboardState,
                    running: threading.Event, auto_block: bool):
    """
    Main processing loop with two-layer hybrid prediction and auto-response.
    
    Handles two types of flows:
      - "precomputed": pre-extracted feature vectors (from demo mode)
      - raw flows: packets that need feature extraction (from live capture)
    """
    correlator = AttackCorrelator()

    while running.is_set():
        try:
            flow = flow_queue.get(timeout=1.0)
        except Empty:
            continue

        try:
            src_ip = flow.get("src_ip", "unknown")
            dst_ip = flow.get("dst_ip", "unknown")

            # ── Get features (precomputed or extracted) ──────────────
            if flow.get("type") == "precomputed":
                # Demo mode: features already scaled, ready for prediction
                features = flow["features"]
            else:
                # Live mode: extract from raw packets
                features = extractor.extract(flow)

            # Skip if IP already blocked
            if ip_blocker.is_blocked(src_ip):
                dash_state.update_flow(is_attack=True, label="Blocked IP",
                                       src_ip=src_ip)
                continue

            # ── Two-layer hybrid prediction ──────────────────────────
            result = predictor.predict(features)
            result = correlator.refine(src_ip, result)

            # ── Update dashboard state ───────────────────────────────
            dash_state.update_flow(
                is_attack=result["is_attack"],
                label=result["label"],
                src_ip=src_ip,
            )

            # ── Handle attacks ───────────────────────────────────────
            if result["is_attack"]:
                action_taken = "Logged"

                # Auto-block ONLY confirmed high-confidence attacks
                # Don't block "Suspicious" — only block confirmed types
                is_confirmed = not result["label"].startswith("Suspicious")
                if (auto_block and is_confirmed
                        and ip_blocker.should_block(result["confidence"])):
                    block_result = ip_blocker.block_ip(
                        ip=src_ip,
                        attack_type=result["label"],
                        confidence=result["confidence"],
                    )
                    if block_result["status"] == "blocked":
                        action_taken = (
                            "IP Blocked (Firewall)"
                            if block_result.get("enforced")
                            else "IP Blocked (App Containment)"
                        )
                        dash_state.set_blocked_ips(ip_blocker.get_blocked_list())

                # Raise alert with severity + explanation
                alert_mgr.raise_alert(
                    label=result["label"],
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    confidence=result["confidence"],
                    method=result["method"],
                    action_taken=action_taken,
                    anomaly=result.get("anomaly", False),
                    severity=result.get("severity", "medium"),
                    explanation=result.get("explanation", ""),
                )
                dash_state.add_alert(alert_mgr.get_recent_alerts(1)[0])

        except Exception as e:
            print(f"  ⚠ Prediction error: {e}")


def demo_traffic_loop(flow_queue: Queue, running: threading.Event):
    """
    UPGRADED demo traffic generator using REAL dataset feature vectors.
    
    Instead of synthesizing packets (which don't match RF training patterns),
    this samples actual rows from the preprocessed training data.  The RF model
    sees features identical to what it was trained on, so it classifies attacks
    correctly with proper confidence scores.
    
    Traffic mix:
      - 50% Benign
      - 15% Port Scan
      - 15% DoS/DDoS
      - 12% Brute Force
      -  8% Web Attack
    """
    # Load preprocessed training data (features are already scaled)
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    print(f"  📂 Loading training data for demo: {train_path}")
    train_df = pd.read_csv(train_path)

    # Group by simplified label
    class_groups = {}
    for label in train_df["label_simplified"].unique():
        group = train_df[train_df["label_simplified"] == label]
        class_groups[label] = group[FEATURE_NAMES].values
        print(f"     {label}: {len(group):,} samples available")

    # Weighted class selection for demo traffic
    attack_choices = [
        ("Benign",      0.50),
        ("Port Scan",   0.15),
        ("DoS/DDoS",    0.15),
        ("Brute Force", 0.12),
        ("Web Attack",  0.08),
    ]
    class_labels = [c[0] for c in attack_choices]
    class_weights = [c[1] for c in attack_choices]

    # Filter to classes that exist in the data
    valid = [(l, w) for l, w in zip(class_labels, class_weights) if l in class_groups]
    class_labels = [v[0] for v in valid]
    class_weights = [v[1] for v in valid]
    # Normalize weights
    total_w = sum(class_weights)
    class_weights = [w / total_w for w in class_weights]

    print(f"  ✅ Demo ready with {len(class_labels)} traffic classes")

    # Attacker IP pools (so we see repeat offenders in the dashboard)
    attacker_pools = {
        "Port Scan":    [f"10.200.{random.randint(1,5)}.{random.randint(1,254)}" for _ in range(8)],
        "DoS/DDoS":     [f"172.16.{random.randint(1,3)}.{random.randint(1,254)}" for _ in range(5)],
        "Brute Force":  [f"203.0.113.{random.randint(1,254)}" for _ in range(6)],
        "Web Attack":   [f"198.51.100.{random.randint(1,254)}" for _ in range(4)],
    }

    while running.is_set():
        time.sleep(random.uniform(0.3, 1.2))

        # Choose traffic class
        chosen_label = random.choices(class_labels, weights=class_weights, k=1)[0]

        # Sample a random feature vector from that class
        samples = class_groups[chosen_label]
        idx = random.randint(0, len(samples) - 1)
        features = samples[idx].reshape(1, -1)

        # Generate realistic IP addresses
        if chosen_label == "Benign":
            src_ip = f"192.168.1.{random.randint(2, 254)}"
        else:
            pool = attacker_pools.get(chosen_label, [])
            src_ip = random.choice(pool) if pool else f"192.168.1.{random.randint(2, 254)}"

        dst_ip = f"10.0.0.{random.randint(1, 50)}"

        flow_queue.put({
            "type": "precomputed",
            "features": features,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
        })


def demo_traffic_loop_v2(flow_queue: Queue, running: threading.Event):
    """Presentation-friendly demo traffic with repeated campaigns."""
    from capture.demo_traffic import DemoTrafficGenerator

    generator = DemoTrafficGenerator()
    print("  Demo generator ready with realistic benign traffic and attack bursts")

    while running.is_set():
        time.sleep(generator.next_sleep_interval())
        flow_queue.put(generator.next_flow())


def traffic_stats_loop(dash_state: DashboardState, packet_counter,
                       running: threading.Event):
    """Periodically update traffic history for the dashboard charts."""
    while running.is_set():
        time.sleep(2.0)
        if packet_counter:
            dash_state.set_packet_count(packet_counter())
        dash_state.record_traffic_point()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    print("=" * 60)
    print("  🛡️  AI-Based Network Intrusion Detection & Prevention")
    print("  Detection + Classification + Auto-Response")
    print("=" * 60)

    # ── Shared state ─────────────────────────────────────────────────────
    dash_state = DashboardState()
    running = threading.Event()
    running.set()

    # ── Start dashboard ──────────────────────────────────────────────────
    print("\n[1/5] Starting dashboard ...")
    start_dashboard(dash_state)

    if args.dashboard_only:
        print("\n  Dashboard-only mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down ...")
            return

    # ── Load models ──────────────────────────────────────────────────────
    print("\n[2/5] Loading ML models (two-layer hybrid) ...")
    extractor = FeatureExtractor()
    predictor = Predictor()

    # ── Initialize alert manager + IP blocker ────────────────────────────
    print("\n[3/5] Initializing alert manager & IP blocker ...")
    alert_mgr = AlertManager()
    ip_blocker = IPBlocker(confidence_threshold=0.80)
    auto_block = not args.no_block
    if auto_block:
        print(f"  🚫 Auto-block enabled (threshold: {ip_blocker.confidence_threshold:.0%})")
    else:
        print("  ℹ️  Auto-block disabled")

    flow_queue = Queue(maxsize=1000)

    # ── Start packet capture or demo mode ────────────────────────────────
    packet_counter = None

    if args.demo:
        print("\n[4/5] Starting DEMO traffic generator (real dataset features) ...")
        demo_thread = threading.Thread(
            target=demo_traffic_loop_v2,
            args=(flow_queue, running),
            daemon=True,
        )
        demo_thread.start()
        packet_counter_val = [0]

        def _fake_counter():
            packet_counter_val[0] += random.randint(10, 50)
            return packet_counter_val[0]
        packet_counter = _fake_counter
    else:
        print("\n[4/5] Starting live packet capture ...")
        try:
            from capture.packet_capture import PacketCapture
            capture = PacketCapture(
                flow_queue=flow_queue,
                interface=args.interface,
            )
            capture.start()
            packet_counter = lambda: capture.packet_count
        except Exception as e:
            print(f"\n  ❌ Failed to start capture: {e}")
            print("  💡 Try running as Administrator, or use --demo mode")
            running.clear()
            return

    # ── Start prediction loop ────────────────────────────────────────────
    print("\n[5/5] Starting two-layer prediction engine ...")
    pred_thread = threading.Thread(
        target=prediction_loop,
        args=(flow_queue, extractor, predictor, alert_mgr, ip_blocker,
              dash_state, running, auto_block),
        daemon=True,
    )
    pred_thread.start()

    # ── Traffic stats updater ────────────────────────────────────────────
    stats_thread = threading.Thread(
        target=traffic_stats_loop,
        args=(dash_state, packet_counter, running),
        daemon=True,
    )
    stats_thread.start()

    # ── Ready ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ System is LIVE!")
    print(f"  🌐 Dashboard → http://localhost:5000")
    print("  🛡️ Two-layer detection: IF (anomaly) → RF (classification)")
    if auto_block:
        print("  🚫 Auto-response: blocking attacker IPs")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n  Shutting down gracefully ...")
        running.clear()
        if not args.demo and 'capture' in locals():
            capture.stop()

        # Print summary
        print(f"\n  📊 Session Summary:")
        print(f"     Alerts raised : {alert_mgr.total_alerts}")
        print(f"     IPs blocked   : {ip_blocker.blocked_count}")
        top = alert_mgr.get_top_attackers(3)
        if top:
            print(f"     Top attackers : {top}")
        print("  System stopped. Goodbye! 👋")


if __name__ == "__main__":
    main()
