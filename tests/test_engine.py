"""Engine invariants, trade execution, and payout conservation."""

from __future__ import annotations

import random

from src.agents.random import RandomAgent
from src.env.deck import ALL_ASSIGNMENTS, Assignment, TOTAL_CARDS, SUIT_INDEX
from src.env.engine import Engine
from src.env.state import (
    Action,
    ASK,
    BID,
    KIND_ACCEPT_ASK,
    KIND_ACCEPT_BID,
    KIND_BUNDLE_BID,
    KIND_PASS,
    KIND_POST_ASK,
    KIND_POST_BID,
    KIND_WITHDRAW,
    RoundConfig,
    RoundState,
)

CLUB, SPADE, HEART, DIAMOND = (SUIT_INDEX[s] for s in ("CLUBS", "SPADES", "HEARTS", "DIAMONDS"))

ASSIGNMENT = Assignment.from_counts({"CLUBS": 12, "SPADES": 8, "HEARTS": 10, "DIAMONDS": 10})

HANDS_4P = [
    [5, 2, 2, 1],
    [3, 3, 3, 1],
    [2, 2, 3, 3],
    [2, 1, 2, 5],
]


def make_state(config: RoundConfig | None = None, hands: list[list[int]] | None = None) -> RoundState:
    config = config or RoundConfig()
    hands = [list(h) for h in (hands or HANDS_4P)]
    n = len(hands)
    cash = [config.start_cash - config.pot / n] * n
    return RoundState(config, ASSIGNMENT, hands, cash, {})


def test_assignment_goal_derivation():
    assert ASSIGNMENT.goal_suit == SPADE
    assert ASSIGNMENT.counts[SPADE] in (8, 10)
    assert ASSIGNMENT.counts[CLUB] == 12


def test_all_assignments_consistent():
    assert len(ALL_ASSIGNMENTS) == 12
    for a in ALL_ASSIGNMENTS:
        assert sorted(a.counts) == [8, 10, 10, 12]
        assert a.goal_suit != a.twelve_suit
        assert a.counts[a.twelve_suit] == 12
        assert a.counts[a.goal_suit] in (8, 10)


def test_deal_invariants():
    engine = Engine(RoundConfig(), random.Random(0))
    for _ in range(50):
        state = engine.new_round()
        assert sum(sum(hand) for hand in state.hands) == TOTAL_CARDS
        assert all(sum(hand) == 10 for hand in state.hands)
        for s in range(4):
            assert sum(hand[s] for hand in state.hands) == state.assignment.counts[s]


def test_accept_ask_transfer():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.best_ask(CLUB) == (5.0, 0)
    engine.step(state, [Action(KIND_PASS), Action(KIND_ACCEPT_ASK, CLUB), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 4 and state.hands[1][CLUB] == 4
    assert state.cash[0] == 305.0 and state.cash[1] == 295.0
    assert state.best_ask(CLUB) is None
    assert len(state.trades) == 1


def test_accept_bid_transfer():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_BID, CLUB, 7.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_PASS), Action(KIND_ACCEPT_BID, CLUB), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 6 and state.hands[1][CLUB] == 2
    assert state.cash[0] == 293.0 and state.cash[1] == 307.0
    assert state.best_bid(CLUB) is None


def test_no_self_trade():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_ACCEPT_ASK, CLUB), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 5
    assert state.cash[0] == 300.0
    assert state.best_ask(CLUB) == (5.0, 0)
    assert state.trades == []


def test_bid_cross_executes_at_ask():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_PASS), Action(KIND_POST_BID, CLUB, 7.0), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 4 and state.hands[1][CLUB] == 4
    assert state.cash[0] == 305.0 and state.cash[1] == 295.0
    assert state.best_ask(CLUB) is None
    assert state.best_bid(CLUB) == (7.0, 1)


def test_ask_cross_executes_at_bid():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_BID, CLUB, 8.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_PASS), Action(KIND_POST_ASK, CLUB, 6.0), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 6 and state.hands[1][CLUB] == 2
    assert state.cash[0] == 292.0 and state.cash[1] == 308.0
    assert state.best_bid(CLUB) is None
    assert state.best_ask(CLUB) == (6.0, 1)


def test_withdraw():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_BID, CLUB, 7.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.best_bid(CLUB) == (7.0, 0)
    engine.step(state, [Action(KIND_WITHDRAW, CLUB), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.best_bid(CLUB) is None


def test_quote_replacement():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 7.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.quotes[0][CLUB] == (ASK, 7.0)


def test_insufficient_cash_blocks_post():
    state = make_state()
    state.cash[1] = 0.0
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_PASS), Action(KIND_POST_BID, CLUB, 7.0), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.best_bid(CLUB) is None


def test_sell_without_cards_blocked():
    state = make_state()
    state.hands[1] = [0, 0, 0, 0]
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_PASS), Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.best_ask(CLUB) is None


def test_settle_conservation():
    state = make_state()
    deltas = Engine(rng=random.Random(0)).settle(state)
    assert sum(deltas) == 0.0


def test_settle_matches_official_example():
    config = RoundConfig(n_players=5)
    hands = [
        [3, 2, 2, 1],
        [3, 2, 1, 2],
        [2, 1, 3, 2],
        [2, 2, 2, 2],
        [2, 1, 2, 3],
    ]
    state = make_state(config, hands)
    deltas = Engine(rng=random.Random(0)).settle(state)
    assert deltas == [20.0, 20.0, -30.0, 20.0, -30.0]


def test_full_round_random_conservation():
    engine = Engine(RoundConfig(), random.Random(1))
    agents = [RandomAgent(i) for i in range(4)]
    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        engine.step(state, actions)
    assert sum(sum(h) for h in state.hands) == TOTAL_CARDS
    assert sum(state.cash) == 4 * 300.0
    deltas = engine.settle(state)
    assert abs(sum(deltas)) < 1e-6


def test_no_negative_hands_or_cash_during_random_play():
    engine = Engine(RoundConfig(n_ticks=60), random.Random(21))
    agents = [RandomAgent(i) for i in range(4)]
    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        engine.step(state, actions)
        for p in range(4):
            assert min(state.hands[p]) >= 0, f"negative hand {state.hands[p]}"
            assert state.cash[p] >= 0, f"negative cash {state.cash[p]}"


def test_stale_ask_after_same_tick_sell_cleared():
    hands = [[1, 2, 4, 3], [5, 3, 2, 1], [3, 3, 2, 2], [3, 2, 2, 4]]
    state = make_state(hands=hands)
    engine = Engine(rng=random.Random(0))
    engine.step(state, [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_POST_BID, CLUB, 2.0), Action(KIND_PASS), Action(KIND_PASS)])
    engine.step(state, [Action(KIND_ACCEPT_BID, CLUB), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 0
    assert state.best_ask(CLUB) is None
    engine.step(state, [Action(KIND_PASS), Action(KIND_PASS), Action(KIND_ACCEPT_ASK, CLUB), Action(KIND_PASS)])
    assert state.hands[0][CLUB] == 0


def test_bundle_bid_posts_and_crosses():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(
        state,
        [Action(KIND_PASS), Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_POST_ASK, SPADE, 6.0), Action(KIND_PASS)],
    )
    engine.step(
        state,
        [Action(KIND_BUNDLE_BID, suit=CLUB, suit2=SPADE, price=15.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)],
    )
    # p0 buys one CLUB + one SPADE from p1 and p2 at the resting asks (5 + 6).
    assert state.hands[0][CLUB] == 6 and state.hands[0][SPADE] == 3
    assert state.hands[1][CLUB] == 2 and state.hands[2][SPADE] == 1
    assert state.cash[0] == 289.0 and state.cash[1] == 305.0 and state.cash[2] == 306.0
    assert state.best_ask(CLUB) is None and state.best_ask(SPADE) is None
    assert not state.bundles[0]
    assert len(state.trades) == 2 and all(t.bundle for t in state.trades)


def test_bundle_bid_too_cheap_does_not_fill():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(
        state,
        [Action(KIND_PASS), Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_POST_ASK, SPADE, 6.0), Action(KIND_PASS)],
    )
    engine.step(
        state,
        [Action(KIND_BUNDLE_BID, suit=CLUB, suit2=SPADE, price=10.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)],
    )
    assert state.hands[0][CLUB] == 5 and state.hands[0][SPADE] == 2
    assert state.trades == []
    assert state.bundles[0][(CLUB, SPADE)] == 10.0


def test_bundle_bid_cannot_self_trade():
    state = make_state()
    engine = Engine(rng=random.Random(0))
    engine.step(
        state,
        [Action(KIND_POST_ASK, CLUB, 5.0), Action(KIND_PASS), Action(KIND_POST_ASK, SPADE, 6.0), Action(KIND_PASS)],
    )
    engine.step(
        state,
        [Action(KIND_BUNDLE_BID, suit=CLUB, suit2=SPADE, price=15.0), Action(KIND_PASS), Action(KIND_PASS), Action(KIND_PASS)],
    )
    # p0 posted both asks; its own bundle bid must not buy from itself.
    assert state.hands[0][CLUB] == 5 and state.hands[0][SPADE] == 2
    assert state.bundles[0][(CLUB, SPADE)] == 15.0
    assert state.trades == []


def test_bundle_round_conservation_and_legality():
    engine = Engine(RoundConfig(n_ticks=40), random.Random(3))
    rng = random.Random(4)
    state = engine.new_round()
    for _ in range(engine.config.n_ticks):
        actions = []
        for p in range(4):
            legal = engine.legal_actions(state, p)
            actions.append(rng.choice(legal))
        engine.step(state, actions)
        for p in range(4):
            assert min(state.hands[p]) >= 0, f"negative hand {state.hands[p]}"
            assert state.cash[p] >= 0, f"negative cash {state.cash[p]}"
    assert sum(sum(h) for h in state.hands) == TOTAL_CARDS
    assert sum(state.cash) == 4 * 300.0
    assert abs(sum(engine.settle(state))) < 1e-6
