from human_or_ai.normalize import normalize_text

EXAMPLES = [
    ("I do n't know", "I don't know"),
    ("I ca n't go", "I can't go"),
    ("it 's fine , really .", "it's fine, really."),
    ("see URL_0 for details", "see for details"),
    ("see https://x.org/a?b=1 now", "see now"),
    ("\n\n  Title\n\nBody text.  ", "Title Body text."),
    ("line one\\nline two", "line one line two"),
    ("“Hello” ‘world’", "\"Hello\" 'world'"),
]


def test_examples():
    assert normalize_text("I do n't know") == "I don't know"
    assert normalize_text("I ca n't go") == "I can't go"
    assert normalize_text("it 's fine , really .") == "it's fine, really."
    assert normalize_text("see URL_0 for details") == "see for details"
    assert normalize_text("see https://x.org/a?b=1 now") == "see now"
    assert normalize_text("\n\n  Title\n\nBody text.  ") == "Title Body text."
    assert normalize_text("line one\\nline two") == "line one line two"
    assert normalize_text("“Hello” ‘world’") == "\"Hello\" 'world'"


def test_idempotent():
    for raw, _ in EXAMPLES:
        once = normalize_text(raw)
        assert normalize_text(once) == once
