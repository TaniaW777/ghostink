import pandas as pd

from human_or_ai.data import build_splits_v2, drop_near_duplicates

DATA_CFG = {"abstracts_val_fraction": 0.2, "near_duplicate_key_chars": 200}


def _recent(texts, labels):
    return pd.DataFrame({"text": texts, "label": labels, "source": "recent", "title": None})


def _abstracts(n_titles: int, prefix: str) -> pd.DataFrame:
    # Two abstracts per title (one human, one AI), all with different texts.
    rows = []
    for i in range(n_titles):
        title = f"{prefix} title {i}"
        rows.append({"text": f"Human abstract number {i} about {title}.", "label": 0, "title": title})
        rows.append({"text": f"Generated abstract number {i} about {title}.", "label": 1, "title": title})
    return pd.DataFrame(rows).assign(source="abstracts")[["text", "label", "source", "title"]]


def _build():
    recent = {
        "train": _recent([f"recent train text {i}" for i in range(10)], [0, 1] * 5),
        "validation": _recent([f"recent val text {i}" for i in range(4)], [0, 1] * 2),
        "test": _recent([f"recent test text {i}" for i in range(4)], [0, 1] * 2),
    }
    return build_splits_v2(recent, _abstracts(20, "train"), _abstracts(5, "test"), DATA_CFG, seed=42)


def test_no_title_shared_between_train_and_val():
    splits, _ = _build()
    train_titles = set(splits.train.loc[splits.train["source"] == "abstracts", "title"])
    val_titles = set(splits.val.loc[splits.val["source"] == "abstracts", "title"])
    assert train_titles and val_titles
    assert not train_titles & val_titles


def test_case_and_punctuation_variants_are_near_duplicates():
    df = pd.DataFrame(
        {
            "text": ["Hello, World! This is a test.", "hello world this is a TEST"],
            "label": [0, 1],
            "source": "recent",
            "title": None,
        }
    )
    kept = drop_near_duplicates(df, 200)
    assert len(kept) == 1
    assert kept["text"].iloc[0] == "Hello, World! This is a test."
