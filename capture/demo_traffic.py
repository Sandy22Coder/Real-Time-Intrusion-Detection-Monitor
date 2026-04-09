"""
capture/demo_traffic.py - Scripted demo traffic generator for presentations.

The generator produces packet-level flow summaries that look realistic enough
for the feature extractor, dashboard, and mitigation flow to behave
convincingly without requiring the original CIC dataset.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from capture.packet_capture import FlowKey


@dataclass
class Campaign:
    label: str
    src_ip: str
    remaining_flows: int
    target_ip: str


class DemoTrafficGenerator:
    """Create realistic benign traffic and attack bursts for demo mode."""

    def __init__(self, seed: int | None = 42):
        self.rng = random.Random(seed)
        self.campaign: Campaign | None = None
        self.internal_clients = [f"192.168.1.{i}" for i in range(10, 70)]
        self.internal_servers = [f"10.0.0.{i}" for i in range(2, 20)]
        self.scan_sources = [f"203.0.113.{i}" for i in range(10, 18)]
        self.ddos_sources = [f"172.16.4.{i}" for i in range(20, 30)]
        self.brute_sources = [f"198.51.100.{i}" for i in range(40, 48)]
        self.web_sources = [f"45.83.12.{i}" for i in range(60, 68)]
        self.common_ports = [80, 443, 53, 22, 25, 110, 143, 3306, 8080]

    def next_sleep_interval(self) -> float:
        """Human-looking pacing: calm background, faster attack bursts."""
        if self.campaign:
            return self.rng.uniform(0.12, 0.35)
        return self.rng.uniform(0.35, 1.1)

    def next_flow(self) -> dict:
        """Return a raw flow summary ready for feature extraction."""
        if self.campaign is None and self.rng.random() < 0.22:
            self._start_campaign()

        if self.campaign:
            flow = self._generate_attack_flow(self.campaign)
            self.campaign.remaining_flows -= 1
            if self.campaign.remaining_flows <= 0:
                self.campaign = None
            return flow

        return self._generate_benign_flow()

    def _start_campaign(self):
        label = self.rng.choices(
            population=["Port Scan", "DoS/DDoS", "Brute Force", "Web Attack"],
            weights=[0.28, 0.24, 0.24, 0.24],
            k=1,
        )[0]
        if label == "Port Scan":
            src_ip = self.rng.choice(self.scan_sources)
            remaining = self.rng.randint(8, 14)
        elif label == "DoS/DDoS":
            src_ip = self.rng.choice(self.ddos_sources)
            remaining = self.rng.randint(6, 10)
        elif label == "Brute Force":
            src_ip = self.rng.choice(self.brute_sources)
            remaining = self.rng.randint(7, 12)
        else:
            src_ip = self.rng.choice(self.web_sources)
            remaining = self.rng.randint(5, 9)

        self.campaign = Campaign(
            label=label,
            src_ip=src_ip,
            remaining_flows=remaining,
            target_ip=self.rng.choice(self.internal_servers),
        )

    def _generate_benign_flow(self) -> dict:
        profile = self.rng.choices(
            population=["web", "dns", "ssh"],
            weights=[0.58, 0.24, 0.18],
            k=1,
        )[0]
        src_ip = self.rng.choice(self.internal_clients)
        dst_ip = self.rng.choice(self.internal_servers)

        if profile == "dns":
            return self._udp_exchange(src_ip, dst_ip, port=53, req_len=78, resp_len=162)
        if profile == "ssh":
            return self._tcp_conversation(
                src_ip=src_ip,
                dst_ip=dst_ip,
                dst_port=22,
                req_sizes=[74, 90, 120, 98],
                resp_sizes=[74, 110, 94, 86],
                spacing=(0.02, 0.12),
                push_every=2,
            )
        return self._tcp_conversation(
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=self.rng.choice([80, 443]),
            req_sizes=[78, 780, 920, 1100, 260],
            resp_sizes=[74, 1450, 1680, 1320, 240],
            spacing=(0.05, 0.35),
            push_every=4,
        )

    def _generate_attack_flow(self, campaign: Campaign) -> dict:
        if campaign.label == "Port Scan":
            return self._port_scan_flow(campaign.src_ip, campaign.target_ip)
        if campaign.label == "DoS/DDoS":
            return self._dos_flow(campaign.src_ip, campaign.target_ip)
        if campaign.label == "Brute Force":
            return self._brute_force_flow(campaign.src_ip, campaign.target_ip)
        return self._web_attack_flow(campaign.src_ip, campaign.target_ip)

    def _port_scan_flow(self, src_ip: str, dst_ip: str) -> dict:
        dst_port = self.rng.choice([21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 3306, 3389, 8080])
        base = time.time()
        packets = [
            self._packet(base, src_ip, dst_ip, 42000 + self.rng.randint(0, 4000), dst_port, 60,
                         {"SYN": True}, 1024, True),
        ]
        if self.rng.random() < 0.7:
            packets.append(
                self._packet(base + self.rng.uniform(0.002, 0.03), dst_ip, src_ip, dst_port,
                             packets[0]["src_port"], 54, {"RST": True, "ACK": True}, 0, False)
            )
        return self._wrap_flow(packets, proto=6)

    def _dos_flow(self, src_ip: str, dst_ip: str) -> dict:
        dst_port = self.rng.choice([80, 443, 8080])
        count = self.rng.randint(35, 70)
        base = time.time()
        packets = []
        src_port = 50000 + self.rng.randint(0, 1000)
        for i in range(count):
            ts = base + i * self.rng.uniform(0.0008, 0.004)
            packets.append(
                self._packet(ts, src_ip, dst_ip, src_port, dst_port, self.rng.randint(58, 90),
                             {"SYN": i % 3 == 0, "ACK": i % 5 == 0}, 1024, True)
            )
            if i % 10 == 0:
                packets.append(
                    self._packet(ts + 0.0004, dst_ip, src_ip, dst_port, src_port, 60,
                                 {"ACK": True}, 8192, False)
                )
        return self._wrap_flow(packets, proto=6)

    def _brute_force_flow(self, src_ip: str, dst_ip: str) -> dict:
        dst_port = self.rng.choice([21, 22, 3389])
        attempts = self.rng.randint(4, 7)
        base = time.time()
        packets = []
        src_port = 43000 + self.rng.randint(0, 3000)
        for i in range(attempts):
            t = base + i * self.rng.uniform(0.03, 0.12)
            packets.extend([
                self._packet(t, src_ip, dst_ip, src_port, dst_port, 60, {"SYN": True}, 2048, True),
                self._packet(t + 0.01, dst_ip, src_ip, dst_port, src_port, 60, {"SYN": True, "ACK": True}, 8192, False),
                self._packet(t + 0.02, src_ip, dst_ip, src_port, dst_port, 74, {"ACK": True, "PSH": True}, 2048, True),
                self._packet(t + 0.03, dst_ip, src_ip, dst_port, src_port, 66, {"RST": True, "ACK": True}, 0, False),
            ])
        return self._wrap_flow(packets, proto=6)

    def _web_attack_flow(self, src_ip: str, dst_ip: str) -> dict:
        dst_port = self.rng.choice([80, 443, 8080])
        base = time.time()
        src_port = 46000 + self.rng.randint(0, 3000)
        packets = [
            self._packet(base, src_ip, dst_ip, src_port, dst_port, 60, {"SYN": True}, 4096, True),
            self._packet(base + 0.01, dst_ip, src_ip, dst_port, src_port, 60, {"SYN": True, "ACK": True}, 8192, False),
            self._packet(base + 0.02, src_ip, dst_ip, src_port, dst_port, 74, {"ACK": True}, 4096, True),
        ]
        for i in range(self.rng.randint(4, 7)):
            t = base + 0.04 + i * self.rng.uniform(0.02, 0.08)
            packets.append(
                self._packet(t, src_ip, dst_ip, src_port, dst_port, self.rng.randint(180, 420),
                             {"ACK": True, "PSH": True}, 4096, True)
            )
            packets.append(
                self._packet(t + 0.01, dst_ip, src_ip, dst_port, src_port, self.rng.randint(120, 220),
                             {"ACK": True, "PSH": i % 2 == 0}, 8192, False)
            )
        return self._wrap_flow(packets, proto=6)

    def _tcp_conversation(self, src_ip: str, dst_ip: str, dst_port: int,
                          req_sizes: list[int], resp_sizes: list[int],
                          spacing: tuple[float, float], push_every: int = 2) -> dict:
        base = time.time()
        src_port = 40000 + self.rng.randint(0, 5000)
        packets = [
            self._packet(base, src_ip, dst_ip, src_port, dst_port, 60, {"SYN": True}, 4096, True),
            self._packet(base + 0.01, dst_ip, src_ip, dst_port, src_port, 60, {"SYN": True, "ACK": True}, 8192, False),
            self._packet(base + 0.02, src_ip, dst_ip, src_port, dst_port, 74, {"ACK": True}, 4096, True),
        ]
        t = base + 0.04
        for i, (req, resp) in enumerate(zip(req_sizes, resp_sizes)):
            delta = self.rng.uniform(*spacing)
            packets.append(
                self._packet(t, src_ip, dst_ip, src_port, dst_port, req,
                             {"ACK": True, "PSH": i % push_every == 0}, 4096, True)
            )
            packets.append(
                self._packet(t + delta, dst_ip, src_ip, dst_port, src_port, resp,
                             {"ACK": True, "PSH": True}, 8192, False)
            )
            t += delta + self.rng.uniform(*spacing)
        return self._wrap_flow(packets, proto=6)

    def _udp_exchange(self, src_ip: str, dst_ip: str, port: int,
                      req_len: int, resp_len: int) -> dict:
        base = time.time()
        src_port = 50000 + self.rng.randint(0, 3000)
        packets = [
            self._packet(base, src_ip, dst_ip, src_port, port, req_len, {}, 0, True),
            self._packet(base + self.rng.uniform(0.01, 0.08), dst_ip, src_ip, port, src_port, resp_len, {}, 0, False),
        ]
        return self._wrap_flow(packets, proto=17)

    def _wrap_flow(self, packets: list[dict], proto: int) -> dict:
        first = packets[0]
        key = FlowKey(first["src_ip"], first["dst_ip"], first["src_port"], first["dst_port"], proto)
        return {
            "key": key,
            "packets": packets,
            "src_ip": first["src_ip"],
            "dst_ip": first["dst_ip"],
        }

    @staticmethod
    def _packet(ts: float, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                length: int, flags: dict, win_size: int, is_forward: bool) -> dict:
        return {
            "time": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "length": length,
            "flags": flags,
            "win_size": win_size,
            "is_forward": is_forward,
        }
