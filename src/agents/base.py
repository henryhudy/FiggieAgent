"""Agent interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..env.engine import Engine
    from ..env.state import Action, RoundState


class Agent:
    name = "base"

    def new_round(self, state: "RoundState", player: int) -> None:
        pass

    def observe(self, state: "RoundState") -> None:
        pass

    def choose_action(self, engine: "Engine", state: "RoundState", player: int) -> "Action":
        raise NotImplementedError
