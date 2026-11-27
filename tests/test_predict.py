"""Detector on a tiny v1 model trained in a temporary folder (no real artifacts, no network)."""

import json

import numpy as np
import pytest

from human_or_ai.calibrate import VERDICTS

SENTENCES = [
    ("the cat sat on the mat today", 0),
    ("my dog runs in the park", 0),
    ("we had fun at the beach", 0),
    ("i like my old red bike", 0),
    ("the sun was hot today", 0),
    ("furthermore it is important to note", 1),
    ("in conclusion this delves into key aspects", 1),
    ("overall it is crucial to consider", 1),
    ("moreover this highlights key aspects", 1),
    ("it is important to delve into this", 1),
]


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    import keras
    import tensorflow as tf

    from human_or_ai.models import build_export_model, build_v1, build_vectorizer, freeze_vectorizer
    from human_or_ai.predict import Detector

    keras.utils.set_random_seed(0)
    texts = np.array([s for s, _ in SENTENCES])
    labels = np.array([y for _, y in SENTENCES], dtype="int32")
    vectorizer = build_vectorizer(max_tokens=20, sequence_length=12)
    vectorizer.adapt(texts)
    model = build_v1(max_tokens=20, sequence_length=12, embedding_dim=4)
    model.fit(vectorizer(tf.constant(texts)), labels, epochs=1, verbose=0)
    export = build_export_model(freeze_vectorizer(vectorizer), model)

    folder = tmp_path_factory.mktemp("mini_v1")
    export.save(folder / "model.keras")
    calibration = {"t_human": 0.4, "t_ai": 0.6, "min_words": 8, "target_fpr": 0.02, "target_fnr": 0.02}
    (folder / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    return Detector.load(folder)


def test_predict_outputs(detector):
    long_text = " ".join(s for s, _ in SENTENCES)
    results = detector.predict([long_text, "the cat sat on the mat today and we had fun"])
    assert len(results) == 2
    for r in results:
        assert r["verdict"] in VERDICTS
        assert 0.0 <= r["p_ai"] <= 1.0
        assert r["n_words"] >= 8
        assert len(r["signals"]) <= 10
        scores = [abs(s) for _, s in r["signals"]]
        assert scores == sorted(scores, reverse=True)


def test_short_text_not_enough(detector):
    (r,) = detector.predict(["the cat sat on mat"])
    assert r["n_words"] == 5
    assert r["verdict"] == "Not enough text"
    assert 0.0 <= r["p_ai"] <= 1.0
