"""Smoke and interface tests for the neural Deep CFR implementation."""

from __future__ import annotations

import math

from src.cfr.cfr import CFR
from src.cfr.deep_cfr import DeepCFR, DeepCFRConfig, InfoSetEncoder
from src.cfr.run_deep_cfr import run_comparison
from src.cfr.figgie_lite import (
    ACTION_COUNT,
    SPLITS,
    apply_buy,
    card_consistency_posterior,
    encode_joint,
    initial_hand_from_history,
)


def test_encoder_is_fixed_width_and_distinguishes_public_history():
    encoder = InfoSetEncoder(ticks=2)
    hand = SPLITS[0][0][0]
    empty = encoder.encode(0, hand, ())
    history = encoder.encode(0, hand, (7,))
    assert empty.shape == history.shape == (encoder.size,)
    assert not empty.equal(history)


def test_card_consistency_posterior_recognizes_a_three_card_suit():
    posterior = card_consistency_posterior((3, 0, 0, 0))
    assert posterior == [1.0, 0.0, 0.0, 0.0]


def test_public_transfers_recover_the_initial_private_hand():
    initial = [[2, 0, 0, 1], [0, 1, 1, 1]]
    after_0, cash, success_0 = apply_buy(initial, [100.0, 100.0], 0, 2)
    after_1, _, success_1 = apply_buy(after_0, cash, 1, 1)
    history = (encode_joint(2, 1, success_0, success_1),)
    assert initial_hand_from_history(0, tuple(after_1[0]), history) == tuple(initial[0])
    assert initial_hand_from_history(1, tuple(after_1[1]), history) == tuple(initial[1])


def test_deep_cfr_produces_a_valid_neural_average_strategy():
    learner = DeepCFR(
        DeepCFRConfig(
            ticks=1, hidden_size=16, memory_capacity=500,
            advantage_steps=2, strategy_steps=4, train_every=4, batch_size=16,
        ),
        seed=7,
    )
    learner.iterate(12)
    strategy = learner.average_strategy()
    hand = SPLITS[0][0][0]
    probs = strategy.probs(0, hand, ())
    assert len(probs) == ACTION_COUNT
    assert all(probability >= 0 for probability in probs)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-6)
    assert all(size > 0 for size in learner.memory_sizes())
    metrics = CFR(ticks=1, seed=0).exploitability(strategy)
    assert math.isfinite(metrics["eps"])
    assert metrics["eps"] >= 0


def test_comparison_runner_reports_exact_metrics():
    result = run_comparison(
        [5],
        iterations=4,
        config=DeepCFRConfig(
            ticks=1, hidden_size=8, memory_capacity=100,
            advantage_steps=1, strategy_steps=1, train_every=2, batch_size=8,
        ),
    )
    assert result["config"]["seeds"] == [5]
    assert len(result["rows"]) == 1
    assert math.isfinite(result["deep_cfr"]["mean"])
    assert math.isfinite(result["tabular_cfr"]["mean"])
