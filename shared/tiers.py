"""
shared/tiers.py
Single source of truth for the Tier enum.
All other modules must import Tier from here.
"""
from enum import IntEnum


class Tier(IntEnum):
    LITE = 0   # Qwen3.5-0.8B  port 8081
    MID  = 1   # Qwen3.5-2B    port 8082
    FULL = 2   # Qwen3.5-4B    port 8083
