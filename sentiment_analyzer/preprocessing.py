"""Text preprocessing helpers for e-commerce reviews."""

from __future__ import annotations

import html
import re
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize raw review text before tokenization."""

    text = html.unescape(text or "")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Convert review text into normalized word tokens."""

    normalized = normalize_text(text)
    return [token for token in TOKEN_RE.findall(normalized) if token not in STOPWORDS]


def join_tokens(tokens: Iterable[str]) -> str:
    """Create readable text from tokens for debugging or reports."""

    return " ".join(tokens)
