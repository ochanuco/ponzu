from __future__ import annotations

import pytest

from ponzu.adapters import (
    AdapterTimeout,
    AdapterUnavailable,
    AudioBuffer,
    ModelResponse,
    Transcript,
)
from ponzu.core.orchestrator import Orchestrator
from ponzu.core.state import State
from ponzu.wakeword import ManualWakeWord


class FakeLLM:
    def __init__(self, text: str = "はい。", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[list] = []

    def generate(self, messages, *, timeout_s=None):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return ModelResponse(text=self.text, model="fake", duration_ms=1)


class FakeSTT:
    def __init__(self, text: str = "こんにちは", error: Exception | None = None):
        self.text = text
        self.error = error

    def transcribe(self, audio):
        if self.error is not None:
            raise self.error
        return Transcript(text=self.text, language="ja", duration_ms=100)


class FakeTTS:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.spoken: list[str] = []

    def synthesize(self, text):
        if self.error is not None:
            raise self.error
        self.spoken.append(text)
        return AudioBuffer(pcm=b"\x00\x00" * 16, sample_rate=24000)


class FakeMic:
    def __init__(self, pcm: bytes = b"\x00\x00" * 1600) -> None:
        self.pcm = pcm
        self.captures = 0

    def capture_utterance(self, *, max_duration_ms, silence_timeout_ms):
        self.captures += 1
        return AudioBuffer(pcm=self.pcm, sample_rate=16000)


class FakeSpeaker:
    def __init__(self) -> None:
        self.played: list[AudioBuffer] = []
        self.cancels = 0
        self.is_playing = False

    def play(self, audio):
        self.played.append(audio)

    def cancel(self):
        self.cancels += 1


def trace(orchestrator: Orchestrator) -> list[State]:
    seen: list[State] = []
    orchestrator.on_state_change(lambda _old, new: seen.append(new))
    return seen


# ---------------------------------------------------------------- text turns


def test_text_turn_returns_response_and_ends_idle() -> None:
    orch = Orchestrator(llm=FakeLLM("はい。"))
    seen = trace(orch)

    result = orch.text_turn("こんにちは")

    assert result.ok
    assert result.response == "はい。"
    assert orch.state is State.IDLE
    # ADR-008 text transition: LISTENING and TRANSCRIBING are skipped, not faked.
    assert seen == [State.THINKING, State.IDLE]


def test_text_turn_counts_characters_not_content() -> None:
    orch = Orchestrator(llm=FakeLLM("ok"))
    result = orch.text_turn("12345")

    assert result.metrics.input_chars == 5
    assert result.metrics.output_chars == 2
    event = result.metrics.as_event()
    assert event["event"] == "turn_completed"
    assert "12345" not in str(event)


def test_text_turn_does_not_speak_by_default() -> None:
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(llm=FakeLLM(), tts=tts, audio_out=speaker)

    orch.text_turn("こんにちは")

    assert tts.spoken == []
    assert speaker.played == []


def test_text_turn_speaks_when_requested() -> None:
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(llm=FakeLLM("どうも。"), tts=tts, audio_out=speaker)
    seen = trace(orch)

    orch.text_turn("こんにちは", speak=True)

    assert tts.spoken == ["どうも。"]
    assert len(speaker.played) == 1
    assert seen == [State.THINKING, State.SPEAKING, State.IDLE]


def test_history_accumulates_across_turns() -> None:
    llm = FakeLLM()
    orch = Orchestrator(llm=llm)

    orch.text_turn("一回目")
    orch.text_turn("二回目")

    # Second call carries the first exchange: system + user + assistant + user.
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    assert llm.calls[1][1].content == "一回目"


# ----------------------------------------------------------------- recovery


@pytest.mark.parametrize(
    "error", [AdapterUnavailable("ollama down"), AdapterTimeout("too slow")]
)
def test_recoverable_llm_failure_returns_to_idle(error: Exception) -> None:
    orch = Orchestrator(llm=FakeLLM(error=error))
    seen = trace(orch)

    result = orch.text_turn("こんにちは")

    # DESIGN section 8: the assistant must not get stuck, and must not raise.
    assert not result.ok
    assert type(error).__name__ in result.error
    assert orch.state is State.IDLE
    assert seen == [State.THINKING, State.ERROR, State.IDLE]


def test_recovery_releases_audio_output() -> None:
    speaker = FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM(error=AdapterUnavailable("down")),
        tts=FakeTTS(),
        audio_out=speaker,
    )

    orch.text_turn("こんにちは")

    # DESIGN section 8 step 3: release audio resources before returning to IDLE.
    assert speaker.cancels >= 1


def test_tts_failure_recovers_from_speaking() -> None:
    orch = Orchestrator(
        llm=FakeLLM(),
        tts=FakeTTS(error=AdapterUnavailable("voicevox down")),
        audio_out=FakeSpeaker(),
    )
    seen = trace(orch)

    result = orch.text_turn("こんにちは", speak=True)

    assert not result.ok
    assert orch.state is State.IDLE
    assert seen == [State.THINKING, State.SPEAKING, State.ERROR, State.IDLE]


# --------------------------------------------------------------- voice turns


def test_voice_turn_runs_the_full_pipeline() -> None:
    mic, stt, tts, speaker = FakeMic(), FakeSTT("いま何時"), FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM("わかりません。"),
        stt=stt,
        tts=tts,
        audio_in=mic,
        audio_out=speaker,
    )
    seen = trace(orch)

    result = orch.voice_turn()

    assert result.ok
    assert result.utterance == "いま何時"
    assert tts.spoken == ["わかりません。"]
    assert seen == [
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]


def test_empty_transcript_skips_the_model() -> None:
    llm = FakeLLM()
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("   "),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )
    seen = trace(orch)

    result = orch.voice_turn()

    # DESIGN section 8 lists an empty transcript as recoverable: there is
    # nothing to answer, so the model is never called. Section 4.1 requires
    # recoverable failures to pass through ERROR rather than shortcutting to
    # IDLE, so the trace records that the turn did not complete normally.
    assert result.ok
    assert llm.calls == []
    assert seen == [State.LISTENING, State.TRANSCRIBING, State.ERROR, State.IDLE]


def test_stt_failure_recovers_to_idle() -> None:
    orch = Orchestrator(
        llm=FakeLLM(),
        stt=FakeSTT(error=AdapterTimeout("stt stalled")),
        audio_in=FakeMic(),
    )

    result = orch.voice_turn()

    assert not result.ok
    assert orch.state is State.IDLE


def test_voice_turn_without_audio_adapters_is_a_wiring_bug() -> None:
    orch = Orchestrator(llm=FakeLLM())
    # Not a recoverable TurnResult — missing adapters mean the caller wired it
    # wrong, which should be loud rather than logged and swallowed.
    with pytest.raises(RuntimeError):
        orch.voice_turn()


# ------------------------------------------------------------------- looping


def test_wake_word_triggers_a_turn() -> None:
    detector = ManualWakeWord()
    mic, tts, speaker = FakeMic(), FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM("はい。"),
        stt=FakeSTT("ぽんず"),
        tts=tts,
        audio_in=mic,
        audio_out=speaker,
        wake_word=detector,
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    assert mic.captures == 1
    assert tts.spoken == ["はい。"]
    assert orch.state is State.IDLE


def test_loop_survives_an_unexpected_error() -> None:
    class Exploding:
        def transcribe(self, audio):
            raise ValueError("not a PonzuError")

    detector = ManualWakeWord()
    orch = Orchestrator(
        llm=FakeLLM(), stt=Exploding(), audio_in=FakeMic(), wake_word=detector
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(None)

    # An unexpected exception must not escape onto the detector thread and kill
    # the loop; the orchestrator resets and waits for the next wake word.
    assert orch.state is State.IDLE


def test_stop_releases_resources() -> None:
    detector = ManualWakeWord()
    speaker = FakeSpeaker()
    orch = Orchestrator(llm=FakeLLM(), audio_out=speaker, wake_word=detector)
    detector.start()

    orch.stop()

    assert speaker.cancels == 1
