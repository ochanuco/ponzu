"""Conversation state machine (DESIGN section 4.1, ADR-008).

An explicit transition table instead of an implicit event chain, so
cancellation/timeout handling is consistent and no code path can leave the
assistant stuck in `LISTENING` or `SPEAKING` (DESIGN section 8).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from itertools import pairwise

from ponzu.core.logging import log_event


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class InvalidTransition(Exception):
    """Raised when a transition isn't in the allowed-transition table.

    Not a `PonzuError`: an illegal transition is a bug in the caller, not a
    recoverable adapter failure the orchestrator routes through `ERROR`.
    """


# ADR-008 primary path, plus "any state -> ERROR" and "ERROR -> IDLE".
_PRIMARY_PATH: tuple[State, ...] = (
    State.IDLE,
    State.LISTENING,
    State.TRANSCRIBING,
    State.THINKING,
    State.SPEAKING,
    State.IDLE,
)


def _build_allowed() -> dict[State, frozenset[State]]:
    allowed: dict[State, set[State]] = {state: set() for state in State}
    for src, dst in pairwise(_PRIMARY_PATH):
        allowed[src].add(dst)
    for state in State:
        if state is not State.ERROR:
            allowed[state].add(State.ERROR)
    allowed[State.ERROR] = {State.IDLE}
    return {state: frozenset(dests) for state, dests in allowed.items()}


_ALLOWED: dict[State, frozenset[State]] = _build_allowed()


class StateMachine:
    """Holds the current `State` and enforces `_ALLOWED` on every change."""

    def __init__(
        self,
        *,
        initial: State = State.IDLE,
        logger: logging.Logger | None = None,
    ) -> None:
        self._state = initial
        self._logger = logger or logging.getLogger("ponzu.state")
        self._callbacks: list[Callable[[State, State], None]] = []

    @property
    def state(self) -> State:
        return self._state

    def on_change(self, callback: Callable[[State, State], None]) -> None:
        """Subscribe to transitions; called with `(previous, new)` after each one.

        Exists so a future UI (menu bar status, etc.) can observe state
        without the state machine knowing about it (DESIGN section 4.8
        "future capability" / ADR-008 consequences).
        """
        self._callbacks.append(callback)

    def transition(self, new_state: State) -> None:
        """Move to `new_state`, or raise `InvalidTransition` if not allowed."""
        if new_state not in _ALLOWED[self._state]:
            raise InvalidTransition(
                f"{self._state.value} -> {new_state.value} is not an allowed transition"
            )
        previous = self._state
        self._state = new_state
        log_event(
            self._logger,
            "state_changed",
            from_state=previous.value,
            to_state=new_state.value,
        )
        for callback in self._callbacks:
            callback(previous, new_state)
