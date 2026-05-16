import re

WORD_OK_RE = re.compile(r"^[a-z]+$")
MIN_WORD_LEN = 3

# spaCy POS tags we exclude from the dictionary.
BAD_POS = {"PROPN", "SYM", "PUNCT", "X", "SPACE"}

_LEMMA_STRIP_RE = re.compile(r"^[^a-z]+|[^a-z]+$")


def normalize_for_match(text: str) -> str:
    """Token text -> lowercase, [a-z] only. e.g. 'Mrs.' -> 'mrs', 'can't' -> 'cant'."""
    return re.sub(r"[^A-Za-z]", "", text).lower()


def accept_word_form(word: str) -> bool:
    """Whether the normalized word can be a dictionary headword."""
    if len(word) < MIN_WORD_LEN:
        return False
    return bool(WORD_OK_RE.match(word))


def sanitize_lemma(lemma: str) -> str | None:
    """Strip leading/trailing non-letter chars from a spaCy lemma
    (e.g. 'robot-' -> 'robot'). Returns None if non-[a-z] chars remain
    inside (e.g. 'didn\\'t', 'ph.d.', 'mistake,-,somewheres') — those are
    not valid dictionary headwords.
    """
    cleaned = _LEMMA_STRIP_RE.sub("", lemma.lower())
    if cleaned and WORD_OK_RE.match(cleaned):
        return cleaned
    return None
