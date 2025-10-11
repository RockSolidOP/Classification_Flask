import os
import re
from typing import Dict


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """Sanitize strings for directory/file names."""
    if value is None:
        return "_"
    value = value.strip()
    if not value:
        return "_"
    return _SAFE_CHARS_RE.sub("_", value)


def normalize_family(name: str, mapping: Dict[str, str]) -> str:
    if not name:
        return name
    return mapping.get(name, name)


def normalize_text(text: str, ops) -> str:
    if text is None:
        return ""
    out = text
    if "lowercase" in ops:
        out = out.lower()
    if "collapse_whitespace" in ops:
        out = re.sub(r"\s+", " ", out).strip()
    return out


STOPWORDS = {
    # minimal English stopwords; extend as needed
    "the", "and", "or", "of", "a", "to", "in", "on", "for", "by", "with",
    "is", "are", "as", "at", "be", "this", "that", "from", "an", "it",
}


def tokenize(text: str):
    # basic tokenization; keep alphanumerics and underscores
    return re.findall(r"[a-z0-9_]+", text)


def os_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

