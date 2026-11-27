"""Prediction API: raw texts in, P(AI), verdict and word signals out.

Usage:
    python -m human_or_ai.predict "some text"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human_or_ai.calibrate import decide
from human_or_ai.normalize import normalize_text
from human_or_ai.utils import resolve

N_SIGNALS = 10


class Detector:
    """Saved raw-text model plus its decision thresholds."""

    def __init__(self, export_model, calibration: dict):
        from human_or_ai.shortcuts import word_scores

        self.model = export_model
        self.calibration = calibration
        # Score of each vocabulary word: embedding vector · output weights (positive = AI).
        scores = word_scores(export_model)
        self._scores = dict(zip(scores["word"], scores["score"].astype(float), strict=True))

    @classmethod
    def load(cls, artifacts_dir: str | Path = "artifacts/v2") -> Detector:
        import keras

        import human_or_ai.models  # noqa: F401  registers custom_standardization for loading

        directory = resolve(artifacts_dir)
        model = keras.models.load_model(directory / "model.keras", compile=False)
        calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        return cls(model, calibration)

    def signals(self, normalized_text: str) -> list[tuple[str, float]]:
        """At most N_SIGNALS words of the text, by decreasing absolute score."""
        from human_or_ai.shortcuts import standardize

        words = dict.fromkeys(standardize(normalized_text).split())
        scored = [(w, self._scores[w]) for w in words if w in self._scores]
        return sorted(scored, key=lambda ws: abs(ws[1]), reverse=True)[:N_SIGNALS]

    def predict(self, texts: list[str]) -> list[dict]:
        from human_or_ai.diagnose import predict_texts

        normalized = [normalize_text(t) for t in texts]
        p_ai = predict_texts(self.model, normalized)
        c = self.calibration
        out = []
        for text, p in zip(normalized, p_ai, strict=True):
            n_words = len(text.split())
            p = float(p)
            out.append(
                {
                    "p_ai": p,
                    "verdict": decide(p, n_words, c["t_human"], c["t_ai"], c["min_words"]),
                    "n_words": n_words,
                    "signals": self.signals(text),
                }
            )
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict whether a text was written by a human or an AI.")
    parser.add_argument("text", help="Text to analyse.")
    parser.add_argument("--artifacts-dir", default="artifacts/v2")
    args = parser.parse_args(argv)
    detector = Detector.load(args.artifacts_dir)
    print(json.dumps(detector.predict([args.text])[0], indent=2))


if __name__ == "__main__":
    main()
