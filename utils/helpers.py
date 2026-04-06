"""
utils/helpers.py — Shared utility functions used across modules.
"""
import datetime
import re


def timestamp_now() -> str:
    """Return current timestamp as a readable string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_valid_ipv4(ip: str) -> bool:
    """Check if a string is a valid IPv4 address."""
    pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(pattern, ip):
        return False
    return all(0 <= int(octet) <= 255 for octet in ip.split("."))


def format_number(n: int) -> str:
    """Format large numbers with comma separators.  e.g. 1234567 → '1,234,567'"""
    return f"{n:,}"


def safe_division(a: float, b: float, default: float = 0.0) -> float:
    """Divide a/b, returning *default* when b is zero."""
    return a / b if b != 0 else default
