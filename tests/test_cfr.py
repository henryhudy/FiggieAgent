"""Tests for the reduced-game CFR: game mechanics, convergence, exploitability
invariants, and the full-engine agent wrapper."""

from __future__ import annotations

import json

import pytest

from src.agents.cfr_agent import CFRAgent, _clamp_hand
from src.cfr.cfr import CFR, DictStrategy, play_match
from src.cfr.figgie_lite import (
    ACTION_COUNT,
    REACHABLE_HANDS,
    SPLITS,
    apply_buy,
    deck_counts,
    encode_joint,
    payoff,
)


def test_buy_transfers_card_and_cash():
    hands = [[2, 0, 0, 1], [0, 1, 1, 1]]
    cash = [100.0, 100.0]
    h, c, success = apply_buy(hands, cash, 0, 2)  # P0 buys suit 1
    assert success
    assert h[0] == [2, 1, 0, 1]
    assert h[1] == [0, 0, 1, 1]
    assert c == [94.0, 106.0]


def test_failed_buy_is_noop():
    hands = [[0, 3, 0, 0], [3, 0, 0, 0]]
    cash = [100.0, 100.0]
    h, c, success = apply_buy(hands, cash, 0, 2)  # P0 has no suit-1 card
    assert not success
    assert h == [[0, 3, 0, 0], [3, 0, 0, 0]]
    assert c == [100.0, 100.0]


def test_hand_index_covers_post_buy_hands():
    seen = set()
    for goal in range(4):
        for h0, h1 in SPLITS[goal]:
            for a0 in range(ACTION_COUNT):
                for a1 in range(ACTION_COUNT):
                    h, c, s0 = apply_buy([list(h0), list(h1)], [100.0, 100.0], 0, a0)
                    h, c, s1 = apply_buy(h, c, 1, a1)
                    for hand in (tuple(h[0]), tuple(h[1])):
                        assert hand in REACHABLE_HANDS
                        seen.add(hand)
    assert len(seen) > 0


def test_encode_joint_roundtrip():
    for a0 in range(ACTION_COUNT):
        for a1 in range(ACTION_COUNT):
            for s0 in (False, True):
                for s1 in (False, True):
                    j = encode_joint(a0, a1, s0, s1)
                    assert ((j >> 2) // ACTION_COUNT, (j >> 2) % ACTION_COUNT) == (a0, a1)
                    assert bool((j >> 1) & 1) == s0
                    assert bool(j & 1) == s1


def test_ticks1_converges():
    cfr = CFR(ticks=1, seed=0)
    cfr.iterate(20000)
    ex = cfr.exploitability(DictStrategy(cfr.average_strategy()))
    assert ex["eps"] < 0.05
    assert ex["eps0"] >= 0 and ex["eps1"] >= 0


def test_exploitability_nonnegative_ticks2():
    cfr = CFR(ticks=2, seed=0)
    cfr.iterate(20000)
    ex = cfr.exploitability(DictStrategy(cfr.average_strategy()))
    assert ex["eps0"] >= 0
    assert ex["eps1"] >= 0
    assert ex["eps"] == pytest.approx(ex["eps0"] + ex["eps1"])


def test_best_response_beats_reference_values():
    """A best response must be at least as good as the reference itself."""
    cfr = CFR(ticks=2, seed=0)
    cfr.iterate(30000)
    avg = DictStrategy(cfr.average_strategy())
    value = cfr.strategy_value(avg)
    br1 = cfr.best_response(1, avg)
    # P1's avg achieves -value (P1 units); the BR must match or exceed it.
    assert br1 >= -value - 1e-9


def test_head_to_head_best_responder_advantage():
    cfr = CFR(ticks=1, seed=0)
    cfr.iterate(30000)
    avg = DictStrategy(cfr.average_strategy())
    a = play_match(avg, avg, 1, 2000, seed=3)
    assert -2.0 < a < 2.0


def test_deal_splits_match_deck_counts():
    for goal in range(4):
        dc = deck_counts(goal)
        for h0, h1 in SPLITS[goal]:
            assert tuple(h0[s] + h1[s] for s in range(4)) == dc


def test_policy_json_structure(tmp_path):
    cfr = CFR(ticks=2, seed=0)
    cfr.iterate(3000)
    from src.cfr.run import build_policy
    path = tmp_path / "policy.json"
    data = build_policy(cfr, 2)
    with open(path, "w") as f:
        json.dump(data, f)
    assert data["ticks"] == 2
    assert len(data["policy"]) == 4
    assert len(data["policy"]["0"]) == 2  # one row per tick
    assert all(len(row) == len(REACHABLE_HANDS) for row in data["policy"]["0"])


def test_cfr_agent_uses_policy(tmp_path):
    cfr = CFR(ticks=2, seed=0)
    cfr.iterate(3000)
    from src.cfr.run import build_policy
    path = tmp_path / "policy.json"
    with open(path, "w") as f:
        json.dump(build_policy(cfr, 2), f)
    agent = CFRAgent(str(path), seed=0)
    assert agent.ticks == 2
    assert len(agent.policy) == 4
    assert len(agent._hand_index) == len(REACHABLE_HANDS)


def test_clamp_hand_projects_to_reduced_space():
    for hand in ([9, 1, 0, 0], [0, 7, 3, 0], [10, 0, 0, 0], [0, 0, 0, 10]):
        clamped = _clamp_hand(hand)
        assert sum(clamped) <= 6
        assert all(0 <= c <= 3 for c in clamped)
        assert clamped in REACHABLE_HANDS
