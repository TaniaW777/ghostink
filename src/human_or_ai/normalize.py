"""Text normalization applied to both classes, to remove formatting artifacts that leak the label."""

from __future__ import annotations

import re
import unicodedata

_SPLIT_NT_RE = re.compile(r"(\w) n't\b")
_SPLIT_CLITIC_RE = re.compile(r"(\w) '(s|re|ll|ve|m|d)\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_URL_PLACEHOLDER_RE = re.compile(r"\burl_?\d+\b", flags=re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize a text so that human and AI texts share the same surface conventions.

    Steps, in this order: NFKC, literal backslash-n to line break, straight quotes, rejoined
    contractions, URLs and URL placeholders removed, spacing around punctuation fixed, line
    breaks dropped (non-empty lines joined with a space), whitespace collapsed.
    """
    # 1. Unicode compatibility normalization.
    t = unicodedata.normalize("NFKC", text)
    # 2. The two characters "\" + "n" become a real line break.
    t = t.replace("\\n", "\n")
    # 3. Typographic quotes become straight quotes.
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    # 4. Rejoin tokenized contractions: "do n't" -> "don't", "it 's" -> "it's".
    t = _SPLIT_NT_RE.sub(r"\1n't", t)
    t = _SPLIT_CLITIC_RE.sub(r"\1'\2", t)
    # 5. Remove URLs and their placeholder tokens (URL_0, url12, ...).
    t = _URL_RE.sub(" ", t)
    t = _URL_PLACEHOLDER_RE.sub(" ", t)
    # 6. No space before closing punctuation, no space after an opening bracket.
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN_RE.sub(r"\1", t)
    # 7. Drop layout: strip each line, drop empty lines, join with a space.
    t = " ".join(line.strip() for line in t.splitlines() if line.strip())
    # 8. Collapse whitespace.
    return _WHITESPACE_RE.sub(" ", t).strip()
