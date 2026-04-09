import os

# ── Project root (auto-detected) ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Preprocessing ─────────────────────────────────────────────────────────
SAMPLE_PER_FILE = 100_000        # rows to sample from each CSV (None = all)
TEST_SIZE = 0.20                 # train/test split ratio
RANDOM_STATE = 42

# ── Model hyperparameters ─────────────────────────────────────────────────
RF_N_ESTIMATORS = 100            # Random Forest trees
RF_MAX_DEPTH = 20                # limit depth for speed
IF_CONTAMINATION = 0.015         # Isolation Forest — lower = fewer false positives
IF_N_ESTIMATORS = 100

# ── Real-time capture ────────────────────────────────────────────────────
FLOW_TIMEOUT_SEC = 2.0           # aggregate packets into flows every N sec
CAPTURE_INTERFACE = None         # None = default interface

# ── Dashboard ─────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
MAX_ALERTS_STORED = 500          # keep last N alerts in memory

# ── Alert thresholds ──────────────────────────────────────────────────────
ALERT_LOG_FILE = os.path.join(LOGS_DIR, "alerts.log")

# ── Ensure directories exist ─────────────────────────────────────────────
for d in [PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
