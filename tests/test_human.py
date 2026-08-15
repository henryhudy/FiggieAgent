"""Human-agent menu sanity: every offered action is engine-legal."""

from __future__ import annotations

import random

from src.agents.human import HumanAgent
from src.env.engine import Engine
from src.env.state import RoundConfig
from src.learning.obs import BUNDLE_PAIRS


def _state():
    engine = Engine(RoundConfig(n_ticks=30), random.Random(0))
    return engine, engine.new_round()


def test_menu_subset_of_legal():
    engine, state = _state()
    for seat in range(4):
        human = HumanAgent(seed=7, show_belief=False)
        human.new_round(state, seat)
        menu = human._menu(state)
        legal = engine.legal_actions(state, seat)
        assert menu and menu[0][0] == "PASS"
        for label, action in menu:
            assert action in legal, f"{label} not legal for P{seat}: {action}"


def test_menu_covers_bundle_actions():
    engine, state = _state()
    human = HumanAgent(seed=7, show_belief=False)
    human.new_round(state, 0)
    menu = human._menu(state)
    kinds = {a.kind for _, a in menu}
    assert "BUNDLE_BID" in kinds
    bundle_pairs = {tuple(sorted((a.suit, a.suit2))) for _, a in menu if a.kind == "BUNDLE_BID"}
    assert bundle_pairs.issubset(set(BUNDLE_PAIRS))


def test_choose_action_returns_from_menu():
    engine, state = _state()
    human = HumanAgent(seed=7, show_belief=False)
    human.new_round(state, 0)
    legal = engine.legal_actions(state, 0)
    for label, action in human._menu(state):
        assert action in legal
