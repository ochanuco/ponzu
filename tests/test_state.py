"""Tests for the conversation state machine (DESIGN section 4.1, ADR-008)."""

from __future__ import annotations

import pytest

from ponzu.core.state import InvalidTransition, State, StateMachine


def test_primary_path_is_legal() -> None:
    machine = StateMachine()
    path = [
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]
    for next_state in path:
        machine.transition(next_state)
        assert machine.state == next_state


@pytest.mark.parametrize(
    "source",
    [State.IDLE, State.LISTENING, State.TRANSCRIBING, State.THINKING, State.SPEAKING],
)
def test_any_state_can_transition_to_error(source: State) -> None:
    machine = StateMachine(initial=source)
    machine.transition(State.ERROR)
    assert machine.state == State.ERROR


def test_error_can_only_return_to_idle() -> None:
    machine = StateMachine(initial=State.ERROR)
    machine.transition(State.IDLE)
    assert machine.state == State.IDLE


@pytest.mark.parametrize(
    "target",
    [State.LISTENING, State.TRANSCRIBING, State.THINKING, State.SPEAKING, State.ERROR],
)
def test_error_cannot_transition_to_anything_but_idle(target: State) -> None:
    machine = StateMachine(initial=State.ERROR)
    with pytest.raises(InvalidTransition):
        machine.transition(target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (State.IDLE, State.TRANSCRIBING),
        (State.IDLE, State.SPEAKING),
        (State.LISTENING, State.THINKING),
        (State.LISTENING, State.SPEAKING),
        (State.LISTENING, State.IDLE),
        (State.TRANSCRIBING, State.SPEAKING),
        (State.TRANSCRIBING, State.IDLE),
    ],
)
def test_illegal_transitions_are_rejected(source: State, target: State) -> None:
    machine = StateMachine(initial=source)
    with pytest.raises(InvalidTransition):
        machine.transition(target)
    # A rejected transition must not mutate state.
    assert machine.state == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (State.IDLE, State.THINKING),
        (State.THINKING, State.IDLE),
    ],
)
def test_text_path_transitions_are_allowed(source: State, target: State) -> None:
    # ADR-008 "Text Transition": `ponzu chat` takes typed input, so LISTENING and
    # TRANSCRIBING are skipped rather than faked. THINKING -> IDLE covers a text
    # turn with no spoken output.
    machine = StateMachine(initial=source)
    machine.transition(target)
    assert machine.state == target


def test_on_change_callback_receives_previous_and_new_state() -> None:
    machine = StateMachine()
    seen: list[tuple[State, State]] = []
    machine.on_change(lambda previous, new: seen.append((previous, new)))

    machine.transition(State.LISTENING)
    machine.transition(State.ERROR)

    assert seen == [
        (State.IDLE, State.LISTENING),
        (State.LISTENING, State.ERROR),
    ]


def test_follow_up_transition_is_allowed() -> None:
    # ADR-015: after speaking, a configured follow-up window returns to
    # LISTENING so a second utterance needs no wake word.
    machine = StateMachine(initial=State.SPEAKING)
    machine.transition(State.LISTENING)
    assert machine.state is State.LISTENING
