"""
features/feature_config.py — Common feature set for training & real-time prediction.

This is the SINGLE SOURCE OF TRUTH for feature names.  Both the dataset
preprocessor and the real-time feature extractor import from here so that
the feature vector is always in the same order and format.

UPGRADE: Added simplified 5-category attack labels for cleaner classification.
"""

# ── 20 features used by the ML models ────────────────────────────────────
# Each tuple: (our_name, dataset_column_name)
FEATURE_COLUMNS = [
    ("destination_port",    " Destination Port"),
    ("flow_duration",       " Flow Duration"),
    ("total_fwd_packets",   " Total Fwd Packets"),
    ("total_bwd_packets",   " Total Backward Packets"),
    ("total_len_fwd",       " Total Length of Fwd Packets"),
    ("total_len_bwd",       " Total Length of Bwd Packets"),
    ("fwd_pkt_len_mean",    " Fwd Packet Length Mean"),
    ("bwd_pkt_len_mean",    " Bwd Packet Length Mean"),
    ("flow_bytes_per_s",    " Flow Bytes/s"),
    ("flow_pkts_per_s",     " Flow Packets/s"),
    ("flow_iat_mean",       " Flow IAT Mean"),
    ("fwd_psh_flags",       " Fwd PSH Flags"),
    ("syn_flag_count",      " SYN Flag Count"),
    ("rst_flag_count",      " RST Flag Count"),
    ("ack_flag_count",      " ACK Flag Count"),
    ("avg_packet_size",     " Average Packet Size"),
    ("init_win_fwd",        " Init_Win_bytes_forward"),
    ("init_win_bwd",        " Init_Win_bytes_backward"),
    ("min_seg_size_fwd",    " min_seg_size_forward"),
    ("active_mean",         " Active Mean"),
]

# Convenient lists
FEATURE_NAMES    = [f[0] for f in FEATURE_COLUMNS]   # names we use internally
DATASET_COLUMNS  = [f[1] for f in FEATURE_COLUMNS]   # original CSV column names

# ── Label column in the dataset ───────────────────────────────────────────
LABEL_COLUMN = " Label"

# ── Original dataset labels → display names ──────────────────────────────
ATTACK_CATEGORIES = {
    "BENIGN":                     "Benign",
    "DDoS":                       "DDoS",
    "DoS Hulk":                   "DoS Hulk",
    "DoS GoldenEye":              "DoS GoldenEye",
    "DoS slowloris":              "DoS Slowloris",
    "DoS Slowhttptest":           "DoS SlowHTTPTest",
    "PortScan":                   "Port Scan",
    "FTP-Patator":                "FTP Brute-Force",
    "SSH-Patator":                "SSH Brute-Force",
    "Web Attack \x96 Brute Force": "Web Brute-Force",
    "Web Attack \x96 XSS":        "Web XSS",
    "Web Attack \x96 Sql Injection": "SQL Injection",
    "Heartbleed":                  "Heartbleed",
}

# ── SIMPLIFIED 5-category labels (for upgraded classifier) ───────────────
# Groups 13 fine-grained labels into 5 actionable categories
SIMPLIFIED_LABELS = {
    "BENIGN":                       "Benign",
    "PortScan":                     "Port Scan",
    "DDoS":                         "DoS/DDoS",
    "DoS Hulk":                     "DoS/DDoS",
    "DoS GoldenEye":                "DoS/DDoS",
    "DoS slowloris":                "DoS/DDoS",
    "DoS Slowhttptest":             "DoS/DDoS",
    "Heartbleed":                   "DoS/DDoS",
    "FTP-Patator":                  "Brute Force",
    "SSH-Patator":                  "Brute Force",
    "Web Attack \x96 Brute Force":  "Web Attack",
    "Web Attack \x96 XSS":         "Web Attack",
    "Web Attack \x96 Sql Injection":"Web Attack",
}

# Simplified categories list (for display / iteration)
SIMPLIFIED_CATEGORY_LIST = ["Benign", "Port Scan", "DoS/DDoS", "Brute Force", "Web Attack"]

# ── Encoding for simplified labels ────────────────────────────────────────
SIMPLIFIED_ENCODE = {label: i for i, label in enumerate(SIMPLIFIED_CATEGORY_LIST)}
SIMPLIFIED_DECODE = {i: label for label, i in SIMPLIFIED_ENCODE.items()}

# Binary mapping: 0 = normal, 1 = attack
LABEL_BINARY = {label: (0 if label == "BENIGN" else 1)
                for label in ATTACK_CATEGORIES}

# ── Legacy encodings (kept for backward compatibility) ────────────────────
LABEL_ENCODE = {label: i for i, label in enumerate(ATTACK_CATEGORIES)}
LABEL_DECODE = {i: label for label, i in LABEL_ENCODE.items()}
