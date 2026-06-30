import re


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim boundaries."""
    return re.sub(r"\s+", " ", str(text).strip())
