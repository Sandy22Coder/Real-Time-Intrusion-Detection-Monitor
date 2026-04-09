import time
import threading
from collections import defaultdict
from queue import Queue

from scapy.all import sniff, IP, TCP, UDP

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import FLOW_TIMEOUT_SEC, CAPTURE_INTERFACE


class FlowKey:
    """A 5-tuple key that identifies a bidirectional network flow."""

    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto):
        # Sort IPs so A→B and B→A map to the same flow
        if (src_ip, src_port) > (dst_ip, dst_port):
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port,
                      self.dst_port, self.proto))

    def __eq__(self, other):
        return (self.src_ip == other.src_ip and
                self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and
                self.dst_port == other.dst_port and
                self.proto == other.proto)

    def __repr__(self):
        return (f"{self.src_ip}:{self.src_port} ↔ "
                f"{self.dst_ip}:{self.dst_port} [{self.proto}]")


class PacketCapture:
    """
    Captures live packets and aggregates them into flows.
    
    Usage:
        q = Queue()
        cap = PacketCapture(flow_queue=q)
        cap.start()        # starts capture + aggregation threads
        flow = q.get()     # blocks until a flow summary is ready
    """

    def __init__(self, flow_queue: Queue, interface=None, timeout=None):
        self.flow_queue = flow_queue
        self.interface = interface or CAPTURE_INTERFACE
        self.timeout = timeout or FLOW_TIMEOUT_SEC
        self._flows = defaultdict(list)    # FlowKey → [packet_info, ...]
        self._lock = threading.Lock()
        self._running = False
        self._packet_count = 0

    # ── Packet callback ──────────────────────────────────────────────────
    def _process_packet(self, pkt):
        """Called by Scapy for each captured packet."""
        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto = ip.proto  # 6=TCP, 17=UDP

        src_port = 0
        dst_port = 0
        flags = {}

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            flags = {
                "SYN": bool(tcp.flags & 0x02),
                "ACK": bool(tcp.flags & 0x10),
                "RST": bool(tcp.flags & 0x04),
                "PSH": bool(tcp.flags & 0x08),
                "FIN": bool(tcp.flags & 0x01),
                "URG": bool(tcp.flags & 0x20),
            }
            win_size = tcp.window
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport
            win_size = 0
        else:
            win_size = 0

        pkt_info = {
            "time": time.time(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "proto": proto,
            "length": len(pkt),
            "flags": flags,
            "win_size": win_size,
            "is_forward": True,  # will be adjusted in feature extraction
        }

        key = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)

        with self._lock:
            # Mark direction — forward = same as first packet's direction
            existing = self._flows[key]
            if existing:
                first = existing[0]
                pkt_info["is_forward"] = (pkt_info["src_ip"] == first["src_ip"])
            self._flows[key].append(pkt_info)
            self._packet_count += 1

    # ── Flow aggregation ─────────────────────────────────────────────────
    def _aggregator_loop(self):
        """Periodically flush flows older than timeout to the queue."""
        while self._running:
            time.sleep(self.timeout)
            self._flush_flows()

    def _flush_flows(self):
        """Move completed flows from buffer to the output queue."""
        with self._lock:
            if not self._flows:
                return
            flows_snapshot = dict(self._flows)
            self._flows.clear()

        for key, packets in flows_snapshot.items():
            if len(packets) < 2:
                continue  # skip trivially small flows
            flow_summary = {
                "key": key,
                "packets": packets,
                "src_ip": packets[0]["src_ip"],
                "dst_ip": packets[0]["dst_ip"],
            }
            self.flow_queue.put(flow_summary)

    # ── Start / stop ─────────────────────────────────────────────────────
    def start(self):
        """Start capture and aggregation in background threads."""
        self._running = True

        # Aggregation thread
        agg_thread = threading.Thread(target=self._aggregator_loop, daemon=True)
        agg_thread.start()

        # Sniff thread (blocking call in its own thread)
        sniff_thread = threading.Thread(
            target=lambda: sniff(
                iface=self.interface,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda _: not self._running,
            ),
            daemon=True,
        )
        sniff_thread.start()
        print(f"  📡 Packet capture started  (interface={self.interface or 'default'}, "
              f"flow_timeout={self.timeout}s)")

    def stop(self):
        """Stop capturing."""
        self._running = False
        self._flush_flows()
        print("  📡 Packet capture stopped")

    @property
    def packet_count(self):
        return self._packet_count
