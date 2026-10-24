import numpy as np
import pandas as pd
import pytest

from router.data.grade import extract_number, grade_alias, grade_exact
from router.data.make_labels import assign_splits, cheapest_correct_tier
from router.train.common import expected_cost
from router.train.select_threshold import (
    clopper_pearson_upper, crc_tau, fixed_sequence_tau_min, gate, mode_presets, routing_loss,
)
from shared.tiers import Tier


@pytest.mark.parametrize("lite, mid, full, expected", [
    (1, 1, 1, Tier.LITE), (1, 0, 0, Tier.LITE), (0, 1, 1, Tier.MID),
    (0, 0, 1, Tier.FULL), (0, 0, 0, Tier.FULL),
])
def test_cheapest_correct_tier(lite, mid, full, expected):
    assert cheapest_correct_tier(lite, mid, full) == expected


def test_splits_cover_everything_with_expected_sizes():
    df = pd.DataFrame({"label": [0] * 60 + [1] * 20 + [2] * 20})
    split = assign_splits(df, {"train": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}, seed=42)
    assert split.value_counts().to_dict() == {"train": 60, "calib": 15, "test": 15, "val": 10}


def test_gate_falls_back_to_full_below_tau():
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    assert gate(probs, 0.6).tolist() == [Tier.LITE, Tier.FULL, Tier.FULL]
    assert gate(probs, 0.5).tolist() == [Tier.LITE, Tier.LITE, Tier.FULL]


def test_routing_loss_counts_only_harm_vs_full():
    correct = np.array([[0, 0, 1], [0, 0, 0], [1, 1, 1]])
    routed = np.array([Tier.LITE, Tier.LITE, Tier.LITE])
    assert routing_loss(routed, correct).tolist() == [1, 0, 0]


def test_expected_cost_matches_hand_computation():
    C = np.array([[0, 1, 2], [4, 0, 1], [6, 4, 0]], dtype=float)
    probs = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert expected_cost(probs, np.array([2, 0]), C) == pytest.approx((0.5 * 6 + 0.5 * 4 + 2) / 2)


def test_clopper_pearson_upper_bound():
    assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100))
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0
    assert clopper_pearson_upper(3, 100, 0.05) > 0.03


def test_risk_control_on_synthetic_monotone_loss():
    # Loss rate falls as tau rises, on 1000 queries.
    taus = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rates = [0.10, 0.05, 0.025, 0.01, 0.0]
    n = 1000
    losses = [np.r_[np.ones(int(n * r)), np.zeros(n - int(n * r))] for r in rates]
    # 1% loss: upper bound ~0.017 <= 0.03 passes; 2.5% loss: upper bound ~0.034 fails.
    assert fixed_sequence_tau_min(taus, losses, alpha=0.03, delta=0.05) == pytest.approx(0.8)
    # CRC bounds only the expected loss, so it accepts 2.5% ((25 + 1) / 1001 <= 0.03).
    assert crc_tau(taus, losses, alpha=0.03) == pytest.approx(0.7)


def test_risk_control_reports_none_when_nothing_is_safe():
    taus = np.array([0.5, 0.9])
    losses = [np.ones(100), np.ones(100)]
    assert fixed_sequence_tau_min(taus, losses, alpha=0.03, delta=0.05) is None
    assert crc_tau(taus, losses, alpha=0.03) is None


def test_presets_never_below_tau_min():
    presets = mode_presets(0.62, 0.99)
    assert min(presets.values()) >= 0.62
    assert presets["saver"] <= presets["balanced"] <= presets["performance"]


def test_graders():
    assert grade_exact("Shopping_List.", "shopping_list")
    assert grade_exact("not spam", "ham")
    assert not grade_exact("negative", "positive")
    assert grade_alias("It is a pulsar.", '["pulsar"]')
    assert not grade_alias("a lunar lobe", '["pulsar"]')
    assert extract_number("so 63 + 10 = 73\nAnswer: 1,073") == 1073
    assert extract_number("the total is 42.") == 42
