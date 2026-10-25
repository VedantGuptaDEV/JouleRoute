"""Prompt templates per query category.

The router was trained on answers to queries wrapped in exactly these templates, so the app
must use the same ones.
"""
from __future__ import annotations

import re

# Fixed subset of CLINC150 intents that an on-device assistant would handle.
INTENT_LABELS = [
    "alarm", "timer", "reminder", "weather", "time", "date", "timezone",
    "play_music", "next_song", "change_volume", "make_call", "text",
    "calendar", "todo_list", "shopping_list", "translate", "definition",
    "spelling", "calculator", "measurement_conversion", "directions", "traffic",
    "share_location", "recipe", "restaurant_reservation", "book_flight",
    "tell_joke", "fun_fact", "flip_coin", "roll_dice",
]

SYSTEM_PROMPTS = {
    "intent": (
        "Classify the user's request into exactly one of these intents: "
        + ", ".join(INTENT_LABELS)
        + ". Reply with only the intent name, nothing else."
    ),
    "classification": "Reply with only the label asked for, in lowercase, nothing else.",
    "factual_qa": "Answer with only the short answer (a few words). No explanation.",
    "rewrite": "Reply with only the corrected text, nothing else.",
    "summary": "Reply with a single-sentence summary only.",
    "math": (
        "Solve the problem. Show brief working, then end with a final line "
        "of the form 'Answer: <number>'."
    ),
    "code": "Reply with only the Python code in a single ```python code block. No explanation.",
}

CATEGORIES = list(SYSTEM_PROMPTS)


def build_messages(category: str, text: str) -> list[dict[str, str]]:
    """Chat messages for one query, using the category's template."""
    if category not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown category '{category}', expected one of {CATEGORIES}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[category]},
        {"role": "user", "content": text},
    ]


# ---------------------------------------------------------------------------
# Natural-language tier override detection
# ---------------------------------------------------------------------------
# Matches phrases like:
#   "use full model", "use the full model", "run on full",
#   "force lite", "switch to mid", "with lite model",
#   "answer using full", "respond with mid", "lite:", "full -"
# The tier keyword must appear as a whole word.
# Returns (tier_key, cleaned_text) where tier_key is "lite" | "mid" | "full" | None.

_TIER_PATTERN = re.compile(
    r"""
    (?:
        # verb-led phrases: "use / run / force / switch to / answer with / respond with ..."
        (?:use|run\s+on|force|switch\s+to|answer\s+(?:with|using)|respond\s+with|using|with)\s+
        (?:the\s+)?
        (?P<tier1>full|mid|lite)
        (?:\s+model)?
    |
        # "on full", "on lite", "on mid"
        \bon\s+(?P<tier2>full|mid|lite)\b
    |
        # bare prefix: "full:", "lite -", "mid —"
        \b(?P<tier3>full|mid|lite)\s*[:—\-]
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_tier_override(text: str) -> tuple[str | None, str]:
    """Detect and strip a natural-language tier directive from *text*.

    Returns ``(tier_key, cleaned_text)``.
    ``tier_key`` is ``"lite"``, ``"mid"``, ``"full"``, or ``None`` (no directive found).
    ``cleaned_text`` is the original text with the directive removed and extra
    whitespace collapsed — ready to send to the gateway as the actual query.

    Examples
    --------
    >>> parse_tier_override("use full model — what is photosynthesis?")
    ('full', 'what is photosynthesis?')
    >>> parse_tier_override("what is 2+2?")
    (None, 'what is 2+2?')
    """
    m = _TIER_PATTERN.search(text)
    if m is None:
        return None, text

    tier = (m.group("tier1") or m.group("tier2") or m.group("tier3")).lower()
    # Remove the matched span from the text and tidy up.
    cleaned = text[: m.start()] + text[m.end() :]
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;—-")
    return tier, cleaned
