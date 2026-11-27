import numpy as np
import pytest

from human_or_ai.evaluate import compute_metrics, expected_calibration_error, length_bins, reliability_bins


def test_perfect_predictions():
    m = compute_metrics([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
    assert m["accuracy"] == 1.0
    assert m["fpr_human"] == 0.0
    assert m["tpr_ai"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn_human_ok": 2, "fp_human_flagged_ai": 0, "fn_ai_missed": 0, "tp_ai_ok": 2}


def test_fpr_counts_humans_flagged_as_ai():
    # 1 of 4 humans flagged as AI, 1 of 2 AI texts missed.
    m = compute_metrics([0, 0, 0, 0, 1, 1], [0.9, 0.2, 0.1, 0.3, 0.8, 0.4])
    assert m["fpr_human"] == pytest.approx(0.25)
    assert m["tpr_ai"] == pytest.approx(0.5)


def test_ece_zero_when_calibrated():
    # In the 0.2-0.3 bin, P(AI)=0.25 and exactly 1 of 4 texts is AI.
    y = [1, 0, 0, 0]
    p = [0.25] * 4
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_large_when_overconfident():
    # Always 99% sure it is AI, but only half are AI: gap of 0.49.
    y = [1, 0] * 50
    p = [0.99] * 100
    assert expected_calibration_error(y, p) == pytest.approx(0.49)


def test_reliability_bins_put_one_in_last_bin():
    bins = reliability_bins([1], [1.0], n_bins=10)
    assert bins["count"].iloc[-1] == 1
    assert bins["count"].sum() == 1


def test_length_bins_edges():
    texts = ["w " * 49, "w " * 50, "w " * 149, "w " * 150, "w " * 300]
    assert list(length_bins(texts)) == ["<50", "50-150", "50-150", "150-300", ">300"]


def test_single_class_has_no_auc():
    m = compute_metrics(np.zeros(3), [0.1, 0.2, 0.9])
    assert m["roc_auc"] is None
    assert m["tpr_ai"] is None
