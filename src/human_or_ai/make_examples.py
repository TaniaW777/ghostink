"""Write app/examples.json: two test v2 texts and two public-domain texts from Project Gutenberg.

Usage:
    python -m human_or_ai.make_examples
"""

from __future__ import annotations

import json
import urllib.request

from human_or_ai.utils import load_config, resolve

MIN_WORDS = 80
# Official mirror of the Project Gutenberg Literary Archive Foundation (www.gutenberg.org answered 503).
DECLARATION_URL = "https://gutenberg.pglaf.org/cache/epub/1/pg1.txt"
CAESAR_URL = "https://gutenberg.pglaf.org/cache/epub/10657/pg10657.txt"


def download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "human-or-ai-examples"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8-sig").replace("\r\n", "\n")


def paragraph_starting_at(text: str, start: int) -> str:
    """The paragraph (lines up to the next blank line) that begins at index ``start``, on one line."""
    end = text.find("\n\n", start)
    block = text[start:] if end == -1 else text[start:end]
    return " ".join(line.strip() for line in block.split("\n"))


def declaration_preamble(text: str) -> str:
    """Preamble of the Declaration of Independence: the paragraph "We hold these truths ..."."""
    return paragraph_starting_at(text, text.index("We hold these truths"))


def caesar_book_one(text: str) -> str:
    """First paragraph of Book I of "The War in Gaul" (McDevitte and Bohn translation)."""
    book = text.index("BOOK I\n", text.index("THE WAR IN GAUL"))
    start = book + len("BOOK I\n")
    while text[start] == "\n":
        start += 1
    return paragraph_starting_at(text, start)


def first_long_text(df, label: int) -> str:
    for text in df.loc[df["label"] == label, "text"]:
        if len(str(text).split()) > MIN_WORDS:
            return str(text)
    raise ValueError(f"no text of more than {MIN_WORDS} words with label {label}")


def main() -> None:
    from human_or_ai.data import load_splits

    test = load_splits(load_config(resolve("configs/v2.yaml"))).test
    examples = [
        {
            "label": "Human-written text (test v2)",
            "source": "test v2, first human text over 80 words",
            "text": first_long_text(test, 0),
        },
        {
            "label": "AI-generated text (test v2)",
            "source": "test v2, first AI text over 80 words",
            "text": first_long_text(test, 1),
        },
        {
            "label": "Declaration of Independence (1776)",
            "source": f"Project Gutenberg #1, {DECLARATION_URL}",
            "text": declaration_preamble(download_text(DECLARATION_URL)),
        },
        {
            "label": "Caesar, The Gallic War, Book I",
            "source": f"Project Gutenberg #10657, {CAESAR_URL}",
            "text": caesar_book_one(download_text(CAESAR_URL)),
        },
    ]
    path = resolve("app/examples.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(examples)} examples to {path}")


if __name__ == "__main__":
    main()
