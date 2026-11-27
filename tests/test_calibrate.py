import numpy as np
import pytest

from human_or_ai.calibrate import choose_thresholds, decide


def test_decide_too_short():
    assert decide(0.99, n_words=49, t_human=0.2, t_ai=0.8, min_words=50) == "Not enough text"
    assert decide(0.01, n_words=0, t_human=0.2, t_ai=0.8, min_words=50) == "Not enough text"


def test_decide_exactly_min_words_is_decided():
    assert decide(0.99, n_words=50, t_human=0.2, t_ai=0.8, min_words=50) == "AI"


def test_decide_p_equal_t_ai():
    assert decide(0.8, n_words=100, t_human=0.2, t_ai=0.8, min_words=50) == "AI"


def test_decide_p_equal_t_human():
    assert decide(0.2, n_words=100, t_human=0.2, t_ai=0.8, min_words=50) == "Human"


def test_decide_between_thresholds():
    assert decide(0.5, n_words=100, t_human=0.2, t_ai=0.8, min_words=50) == "Uncertain"
    assert decide(0.79, n_words=100, t_human=0.2, t_ai=0.8, min_words=50) == "Uncertain"
    assert decide(0.21, n_words=100, t_human=0.2, t_ai=0.8, min_words=50) == "Uncertain"


def test_choose_thresholds_hand_made():
    # 5 human texts and 5 AI texts.
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.0, 0.1, 0.2, 0.6, 0.9, 0.3, 0.4, 0.7, 0.8, 1.0])
    # Human scores sorted: 0.0 0.1 0.2 0.6 0.9. Linear quantile at 0.75: position 0.75 * 4 = 3,
    # i.e. the 4th value, 0.6; max(0.5, 0.6) = 0.6.
    # AI scores sorted: 0.3 0.4 0.7 0.8 1.0. Quantile at 0.25: position 1, i.e. 0.4; min(0.5, 0.4) = 0.4.
    t = choose_thresholds(y, p, target_fpr=0.25, target_fnr=0.25)
    assert t["t_ai"] == pytest.approx(0.6)
    assert t["t_human"] == pytest.approx(0.4)


def test_choose_thresholds_clamped_to_half():
    # Well separated classes: raw quantiles would be 0.2 (human) and 0.8 (AI), clamped to 0.5.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    t = choose_thresholds(y, p, target_fpr=0.0, target_fnr=0.0)
    assert t == {"t_human": 0.5, "t_ai": 0.5}
