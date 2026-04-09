# 🛡️ AI-Based Network Intrusion Detection & Prevention System

**Detection + Classification + Auto-Response**

A production-quality IDS that captures live network traffic, classifies attacks using a two-layer hybrid ML model, auto-blocks attacker IPs, and displays everything on a real-time dashboard.

## Architecture

```
Traffic → Capture (Scapy) → Feature Extraction (20 features)
  → Layer 1: Isolation Forest (anomaly detection)
  → Layer 2: Random Forest (5-class attack classification)
  → Auto-Response: IP Blocking
  → Dashboard (Flask + Chart.js)
```

## Features

- **Two-layer hybrid detection**: Isolation Forest (unsupervised anomaly) + Random Forest (supervised classification)
- **5 attack categories**: Benign, Port Scan, DoS/DDoS, Brute Force, Web Attack
- **Auto IP blocking**: high-confidence attacks trigger automatic IP blocks
- **Real-time dashboard**: system status, attack classification chart, top attacker IPs, blocked IP panel
- **Forensics logging**: structured JSON logs in `logs/forensics.jsonl`
- **Demo mode**: simulated traffic with 55% benign + 45% mixed attacks

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess dataset (CIC-IDS-2017)
python -m data.preprocess

# 3. Train models
python -m models.train_random_forest
python -m models.train_isolation_forest

# 4. Run in demo mode
python main.py --demo

# 5. Open dashboard
# → http://localhost:5000
```

## Modes

| Mode | Command | Requires |
|------|---------|----------|
| Demo | `python main.py --demo` | Nothing extra |
| Live | `python main.py` | Admin + Npcap |
| Dashboard only | `python main.py --dashboard-only` | Nothing extra |
| No auto-block | `python main.py --demo --no-block` | Nothing extra |

## Dashboard

- **System Status**: Normal (green) / Warning (amber) / Under Attack (red)
- **Stats**: Total packets, flows, attacks, normal traffic, blocked IPs, uptime
- **Charts**: Traffic over time (line), Attack classification (doughnut)
- **Top Attacking IPs**: Ranked by hit frequency
- **Blocked IPs**: List with attack type and hit count
- **Alert Feed**: Real-time alerts with attack badges, confidence bars, action taken

## Real Attack Testing

```bash
# Port scan
nmap -sS -p 1-1024 <target_ip>

# DoS test
hping3 -S --flood -p 80 <target_ip>

# SSH brute force
hydra -l admin -P wordlist.txt ssh://<target_ip>
```

## Project Structure

```
NS PROJECT/
├── main.py                  # Entry point (3 modes)
├── requirements.txt         # Dependencies
├── config/
│   └── settings.py          # Global configuration
├── data/
│   ├── preprocess.py        # Dataset preprocessing (5-class)
│   ├── processed/           # Train/test CSVs
│   └── *.csv                # CIC-IDS-2017 dataset
├── features/
│   ├── feature_config.py    # 20-feature + label definitions
│   └── feature_extractor.py # Flow → feature vector
├── models/
│   ├── train_random_forest.py  # RF trainer (5-class)
│   ├── train_isolation_forest.py # IF trainer
│   ├── random_forest.pkl    # Trained RF model
│   ├── isolation_forest.pkl # Trained IF model
│   └── scaler.pkl           # StandardScaler
├── capture/
│   └── packet_capture.py    # Scapy live capture
├── detection/
│   ├── predictor.py         # Two-layer hybrid predictor
│   ├── alert_manager.py     # Alert + forensics logging
│   └── ip_blocker.py        # Auto IP blocking
├── dashboard/
│   ├── app.py               # Flask API (6 endpoints)
│   ├── templates/index.html # Dashboard UI
│   └── static/style.css     # Dark theme CSS
├── utils/
│   └── helpers.py           # Shared utilities
└── logs/
    ├── alerts.log           # Plain text alerts
    └── forensics.jsonl      # Structured JSON forensics
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard page |
| `GET /api/stats` | Traffic statistics |
| `GET /api/alerts` | Recent alerts (50) |
| `GET /api/traffic` | Time-series chart data |
| `GET /api/top_attackers` | Top 10 attacker IPs |
| `GET /api/blocked_ips` | Currently blocked IPs |
| `GET /api/system_status` | System health status |

## Requirements

- Python 3.10+
- CIC-IDS-2017 dataset (CSV files in `data/`)
- For live capture: Administrator/root + Npcap (Windows) or libpcap (Linux)
