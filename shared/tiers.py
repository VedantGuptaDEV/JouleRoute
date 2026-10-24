"""Model tiers. This is the only definition of Tier in the project."""
from enum import IntEnum


class Tier(IntEnum):
    LITE = 0
    MID = 1
    FULL = 2

    @property
    def key(self) -> str:
        """Lowercase name used in config.yaml and CSV columns ("lite", "mid", "full")."""
        return self.name.lower()


TIER_KEYS = [t.key for t in Tier]
