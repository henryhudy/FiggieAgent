"""Belief tracker: prior match, exact posterior concentration on a
fully-informative flow, pruning of impossible assignments, and
world-consistency during live random play."""

from __future__ import annotations

import math
import random

import numpy as np

from src.agents.random import RandomAgent
from src.belief.tracker import BeliefTracker
from src.env.deck import ALL_ASSIGNMENTS, Assignment, N_SUITS, hand_likelihood
from src.env.engine import Engine
from src.env.state import RoundConfig, RoundState, Trade

N_BIG = 8192


def prior_over_assignments(own_hand, n_particles=N_BIG):
    w = np.array([hand_likelihood(own_hand, a.counts) for a in ALL_ASSIGNMENTS], dtype=float)
    return w / w.sum()


def deal_state(seed=0):
    eng = Engine(RoundConfig(), random.Random(seed))
    return eng.new_round()


def make_state(counts):
    names = "CLUBS", "SPADES", "HEARTS", "DIAMONDS"
    assignment = Assignment.from_counts(dict(zip(names, counts)))
    hands = [[2, 3, 3, 2], [4, 2, 1, 3], [3, 3, 2, 2], [3, 2, 4, 1]]
    state = RoundState(config=RoundConfig(n_players=4), assignment=assignment, hands=hands, cash=[350.0] * 4, quotes={})
    return state, ALL_ASSIGNMENTS.index(assignment)


def test_prior_matches_hand_likelihood():
    state = deal_state(1)
    tr = BeliefTracker(seed=2, n_particles=N_BIG)
    tr.init(state, 0)
    post = tr.assignment_posterior()
    prior = prior_over_assignments(state.hands[0])
    assert np.abs(post - prior).max() < 0.05
    assert tr.consistent


def test_synthetic_flow_pins_assignment_exactly():
    state, true_ai = make_state((12, 10, 10, 8))
    tr = BeliefTracker(seed=3, n_particles=N_BIG)
    tr.init(state, 0)

    needed = {1: [4, 2, 1, 0], 2: [3, 3, 2, 0], 3: [3, 2, 4, 0]}
    i = 0
    for suit in range(N_SUITS):
        for p in (1, 2, 3):
            for _ in range(needed[p][suit]):
                tr.update(Trade(tick=i, buyer=0, seller=p, suit=suit, price=10.0, aggressor=p))
                i += 1

    post = tr.assignment_posterior()
    assert tr.consistent
    assert int(np.argmax(post)) == true_ai
    assert post[true_ai] == 1.0
    for ai, a in enumerate(ALL_ASSIGNMENTS):
        if ai != true_ai:
            assert post[ai] == 0.0
    gm = tr.goal_marginals()
    assert math.isclose(gm.sum(), 1.0)
    assert int(np.argmax(gm)) == ALL_ASSIGNMENTS[true_ai].goal_suit


def test_sale_beyond_possible_holding_prunes_assignments():
    state, _ = make_state((12, 10, 10, 8))
    tr = BeliefTracker(seed=4, n_particles=N_BIG)
    tr.init(state, 0)

    for i in range(9):
        tr.update(Trade(tick=i, buyer=0, seller=1, suit=0, price=10.0, aggressor=1))

    post = tr.assignment_posterior()
    assert tr.consistent
    for ai, a in enumerate(ALL_ASSIGNMENTS):
        if a.counts[0] < 12:
            assert post[ai] == 0.0
    surviving = [post[ai] for ai, a in enumerate(ALL_ASSIGNMENTS) if a.counts[0] == 12]
    assert math.isclose(sum(surviving), 1.0)
    gm = tr.goal_marginals()
    assert int(np.argmax(gm)) == 1
    assert gm[1] > 0.9


def test_goal_marginals_sum_to_one_always():
    state = deal_state(7)
    tr = BeliefTracker(seed=8, n_particles=2048)
    tr.init(state, 2)
    assert math.isclose(tr.goal_marginals().sum(), 1.0)


def test_random_games_worlds_stay_consistent():
    engine = Engine(RoundConfig(n_ticks=40), random.Random(11))
    rng = random.Random(12)
    agents = [RandomAgent(rng.randrange(1 << 30)) for _ in range(4)]
    for r in range(4):
        state = engine.new_round()
        for p, a in enumerate(agents):
            a.new_round(state, p)
        seat = r % 4
        tr = BeliefTracker(seed=13 + r, n_particles=4096)
        tr.init(state, seat)
        for _ in range(engine.config.n_ticks):
            actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
            engine.step(state, actions)
            for a in agents:
                a.observe(state)
            tr.observe(state)
            assert tr.consistent
            for _ in range(2):
                w_assignment, w_hands = tr.sample_world()
                for p in range(4):
                    for s in range(N_SUITS):
                        assert w_hands[p][s] >= 0
                for s in range(N_SUITS):
                    assert sum(h[s] for h in w_hands) == w_assignment.counts[s]
                assert w_hands[seat] == state.hands[seat]
