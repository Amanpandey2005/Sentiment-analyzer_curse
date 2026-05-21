"""Reusable text normalization for training and inference."""

from __future__ import annotations

import html
import re

WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def normalize_text(text: str) -> str:
    """Normalize customer review text before vectorization."""

    cleaned = html.unescape(text or "").strip().lower()
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned
