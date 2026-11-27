import pandas as pd

from human_or_ai.shortcuts import pattern_counts, shortcut_table, standardize


def test_split_contractions_are_counted_separately():
    counts = pattern_counts(["I do n't know , it 's fine", "I don't know, it's fine"]).iloc
    assert counts[0]["split n't (do n't)"] == 1
    assert counts[0]["split 's (it 's)"] == 1
    assert counts[0]["space before , . ; : ! ? )"] == 1
    assert counts[0]["joined contraction (don't, it's)"] == 0
    assert counts[1]["split n't (do n't)"] == 0
    assert counts[1]["joined contraction (don't, it's)"] == 2


def test_url_placeholder_and_literal_newline():
    counts = pattern_counts([r"see URL_0 and url_12\nnext", "real\nbreak"]).iloc
    assert counts[0]["URL_ placeholder"] == 2
    assert counts[0]["literal \\n (backslash + n)"] == 1
    assert counts[0]["real line break"] == 0
    assert counts[1]["real line break"] == 1


def test_standardize_matches_v1_and_creates_nt_token():
    # This is why "nt" exists in the v1 vocabulary: "do n't" -> "do nt".
    assert standardize("I do n't <br/>know!").split() == ["i", "do", "nt", "know"]
    assert standardize("I don't").split() == ["i", "dont"]


def test_shortcut_table_detects_a_perfect_artifact():
    df = pd.DataFrame({"text": ["do n't go", "it 's ok", "do not go", "it is ok"], "label": [0, 0, 1, 1]})
    table = shortcut_table(df).set_index("pattern")
    assert table.loc["split n't (do n't)", "human % texts"] == 50.0
    assert table.loc["split n't (do n't)", "ai % texts"] == 0.0
    assert table.loc["split n't (do n't)", "points_to"] == "human"
