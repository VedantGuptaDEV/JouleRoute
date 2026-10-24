"""Answer cache (Readme §5.C): LRU with TTL, keyed on the normalized query."""
import hashlib
import re
import string
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from shared.tiers import Tier

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip()


def query_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


@dataclass
class CachedAnswer:
    text: str
    tier: Tier
    tokens_out: int
    stored_at: float


class AnswerCache:
    def __init__(self, size: int, ttl_s: float):
        self.size = size
        self.ttl_s = ttl_s
        self._items: OrderedDict[str, CachedAnswer] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, text: str) -> CachedAnswer | None:
        key = query_hash(text)
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if time.time() - item.stored_at > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return item

    def put(self, text: str, answer: str, tier: Tier, tokens_out: int):
        key = query_hash(text)
        with self._lock:
            self._items[key] = CachedAnswer(answer, tier, tokens_out, time.time())
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)
