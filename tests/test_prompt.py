from __future__ import annotations

from ponzu.core.prompt import DEFAULT_PERSONA, ConversationContext


def test_build_starts_with_persona_and_ends_with_utterance() -> None:
    ctx = ConversationContext()
    messages = ctx.build("いま何時")

    assert messages[0].role == "system"
    assert messages[0].content == DEFAULT_PERSONA
    assert messages[-1] == messages[-1].__class__(role="user", content="いま何時")


def test_history_is_replayed_between_persona_and_utterance() -> None:
    ctx = ConversationContext()
    ctx.record("こんにちは", "こんにちは。")

    messages = ctx.build("元気")

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "こんにちは"
    assert messages[2].content == "こんにちは。"
    assert messages[3].content == "元気"


def test_window_evicts_oldest_turns_but_never_the_persona() -> None:
    ctx = ConversationContext(max_turns=2)
    for i in range(5):
        ctx.record(f"q{i}", f"a{i}")

    messages = ctx.build("now")

    assert ctx.turn_count == 2
    assert messages[0].role == "system"  # persona survives eviction
    # Only the two most recent exchanges remain.
    assert [m.content for m in messages[1:-1]] == ["q3", "a3", "q4", "a4"]


def test_clear_drops_history_only() -> None:
    ctx = ConversationContext()
    ctx.record("q", "a")
    ctx.clear()

    messages = ctx.build("next")

    assert ctx.turn_count == 0
    assert len(messages) == 2
    assert messages[0].content == DEFAULT_PERSONA


def test_persona_forbids_claiming_unimplemented_actions() -> None:
    # DESIGN section 4.6 lists "no claim of actions not actually performed" as a
    # persona requirement, and the MVP genuinely has no skills (ROADMAP Phase 3).
    # If this constraint is ever dropped from the persona text, the assistant
    # starts inventing capabilities it does not have.
    assert "実行したかのように言わない" in DEFAULT_PERSONA
