"""Prompt templates per query category.

Shared by router/data/generate_answers.py and the app: correctness labels are only
valid if the app wraps queries with exactly these templates.
"""

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
