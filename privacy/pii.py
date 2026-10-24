"""PII detector (Readme §5.D).

Layer 1: regexes with checksum validation (email, Indian mobile, Aadhaar + Verhoeff,
PAN, card numbers + Luhn, IFSC, IPv4).
Layer 2 (optional, pii.ner_enabled): spaCy small English NER, only if layer 1 found nothing.
Layer 3: keywords for sensitive intent (passwords, OTPs, medical words).

Tuned toward flagging when in doubt: a false negative is worse than a false positive.
"""
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Verhoeff tables (dihedral group D5), used by the Aadhaar check digit.
_VD = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VP = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    """Check digit to append to `digits` so the result passes verhoeff_valid."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[(i + 1) % 8][int(ch)]]
    return str(_VINV[c])


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# Order matters: longer / checksummed patterns claim their span first, so a card number
# is not also reported as a phone number.
_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    ("card", re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"), lambda m: luhn_valid(_digits(m))),
    ("aadhaar", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
     lambda m: verhoeff_valid(_digits(m))),
    ("phone", re.compile(r"(?<![\d+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), None),
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE), None),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE), None),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
]

SENSITIVE_KEYWORDS = [
    "password", "passcode", "passwd", "otp", "one time password", "pin", "cvv", "cvc",
    "account number", "bank account", "ssn", "aadhaar", "aadhar", "pan card", "passport",
    "diagnosis", "diagnosed", "prescription", "medication", "hiv", "cancer", "diabetes",
    "pregnant", "pregnancy", "depression", "anxiety", "therapy", "therapist", "std",
]
_KEYWORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in SENSITIVE_KEYWORDS) + r")\b", re.IGNORECASE)
_NER_LABELS = {"PERSON", "GPE", "LOC"}


@dataclass
class PIIResult:
    flag: bool
    types: list[str] = field(default_factory=list)
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (start, end, type)


class PIIDetector:
    def __init__(self, ner_enabled: bool = False):
        self.nlp = None
        if ner_enabled:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError) as e:
                log.warning("pii.ner_enabled is set but spaCy en_core_web_sm is unavailable (%s); NER layer off", e)

    def detect(self, text: str) -> PIIResult:
        spans: list[tuple[int, int, str]] = []

        def free(start: int, end: int) -> bool:
            return all(end <= s or start >= e for s, e, _ in spans)

        for kind, pattern, validator in _PATTERNS:
            for m in pattern.finditer(text):
                if free(m.start(), m.end()) and (validator is None or validator(m.group())):
                    spans.append((m.start(), m.end(), kind))

        if not spans and self.nlp is not None:
            for ent in self.nlp(text).ents:
                if ent.label_ in _NER_LABELS:
                    spans.append((ent.start_char, ent.end_char, ent.label_.lower()))

        for m in _KEYWORD_RE.finditer(text):
            if free(m.start(), m.end()):
                spans.append((m.start(), m.end(), "sensitive_keyword"))

        spans.sort()
        types = sorted({kind for _, _, kind in spans})
        return PIIResult(flag=bool(spans), types=types, spans=spans)
