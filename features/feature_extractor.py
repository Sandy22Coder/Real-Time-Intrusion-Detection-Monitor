"""
features/feature_extractor.py — Convert raw flow packets into the 20-feature
vector expected by the ML models.

This module bridges live capture → model prediction by producing the SAME
feature format used during training (defined in feature_config.py).
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR
from features.feature_config import FEATURE_NAMES
from utils.helpers import safe_division


class FeatureExtractor:
    """
    Extracts 20 features from a flow (list of packet dicts) and applies
    the same StandardScaler used during training.
    """

    def __init__(self):
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  ✅ Loaded scaler from {scaler_path}")
        else:
            self.scaler = None
            print("  ⚠ No scaler found — features will NOT be scaled")

    def extract(self, flow: dict) -> np.ndarray:
        """
        Extract the 20-feature vector from a flow summary.

        Args:
            flow: dict with keys 'packets' (list of packet dicts) and 'key'

        Returns:
            (1, 20) numpy array ready for model.predict()
        """
        packets = flow["packets"]
        key = flow["key"]

        # ── Separate forward and backward packets ────────────────────────
        fwd = [p for p in packets if p["is_forward"]]
        bwd = [p for p in packets if not p["is_forward"]]

        # ── Time calculations ────────────────────────────────────────────
        times = [p["time"] for p in packets]
        flow_duration = max(times) - min(times)
        flow_duration_us = flow_duration * 1e6  # microseconds (matching dataset)

        # Inter-arrival times
        sorted_times = sorted(times)
        iats = [sorted_times[i+1] - sorted_times[i]
                for i in range(len(sorted_times) - 1)]
        mean_iat = np.mean(iats) * 1e6 if iats else 0  # microseconds

        # ── Packet lengths ───────────────────────────────────────────────
        fwd_lengths = [p["length"] for p in fwd]
        bwd_lengths = [p["length"] for p in bwd]
        all_lengths = [p["length"] for p in packets]

        total_len_fwd = sum(fwd_lengths)
        total_len_bwd = sum(bwd_lengths)
        fwd_pkt_len_mean = np.mean(fwd_lengths) if fwd_lengths else 0
        bwd_pkt_len_mean = np.mean(bwd_lengths) if bwd_lengths else 0
        avg_packet_size = np.mean(all_lengths) if all_lengths else 0

        # ── Rates ────────────────────────────────────────────────────────
        total_bytes = sum(all_lengths)
        total_pkts = len(packets)
        flow_bytes_per_s = safe_division(total_bytes, flow_duration)
        flow_pkts_per_s = safe_division(total_pkts, flow_duration)

        # ── Flag counts ──────────────────────────────────────────────────
        syn_count = sum(1 for p in packets if p["flags"].get("SYN", False))
        rst_count = sum(1 for p in packets if p["flags"].get("RST", False))
        ack_count = sum(1 for p in packets if p["flags"].get("ACK", False))
        fwd_psh = sum(1 for p in fwd if p["flags"].get("PSH", False))

        # ── TCP window sizes ─────────────────────────────────────────────
        fwd_wins = [p["win_size"] for p in fwd if p["win_size"] > 0]
        bwd_wins = [p["win_size"] for p in bwd if p["win_size"] > 0]
        init_win_fwd = fwd_wins[0] if fwd_wins else 0
        init_win_bwd = bwd_wins[0] if bwd_wins else 0

        # ── Min segment size (approximate via smallest forward packet) ───
        min_seg_fwd = min(fwd_lengths) if fwd_lengths else 0

        # ── Active time (simplified: total flow duration if active) ──────
        active_mean = flow_duration_us

        # ── Build feature vector (MUST match FEATURE_NAMES order) ────────
        features = np.array([
            min(
                [p["dst_port"] for p in packets if p["dst_port"] > 0] +
                [p["src_port"] for p in packets if p["src_port"] > 0]
            ) if packets else key.dst_port,  # destination_port
            flow_duration_us,      # flow_duration
            len(fwd),              # total_fwd_packets
            len(bwd),              # total_bwd_packets
            total_len_fwd,         # total_len_fwd
            total_len_bwd,         # total_len_bwd
            fwd_pkt_len_mean,      # fwd_pkt_len_mean
            bwd_pkt_len_mean,      # bwd_pkt_len_mean
            flow_bytes_per_s,      # flow_bytes_per_s
            flow_pkts_per_s,       # flow_pkts_per_s
            mean_iat,              # flow_iat_mean
            fwd_psh,               # fwd_psh_flags
            syn_count,             # syn_flag_count
            rst_count,             # rst_flag_count
            ack_count,             # ack_flag_count
            avg_packet_size,       # avg_packet_size
            init_win_fwd,          # init_win_fwd
            init_win_bwd,          # init_win_bwd
            min_seg_fwd,           # min_seg_size_fwd
            active_mean,           # active_mean
        ], dtype=np.float64).reshape(1, -1)

        # Apply the same scaler used during training
        if self.scaler is not None:
            features = self.scaler.transform(features)

        return features
