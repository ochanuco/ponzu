from __future__ import annotations

import itertools
import logging

import pytest

from ponzu.adapters import (
    AdapterTimeout,
    AdapterUnavailable,
    AudioBuffer,
    ModelResponse,
    Transcript,
)
from ponzu.core.orchestrator import _MAX_CONSECUTIVE_FOLLOW_UPS, Orchestrator
from ponzu.core.prompt import ConversationContext
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


class FakeStreamingLLM:
    """A `LanguageModel` that exposes `generate_stream` but not `generate`.

    ADR-014: only the voice loop streams, so this fake exists to prove the
    orchestrator picks the streaming path when it's offered, and never falls
    back to a whole-answer call it doesn't have.
    """

    def __init__(
        self,
        fragments: list[str],
        *,
        fail_before: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.fragments = fragments
        # If set, raise `error` instead of yielding fragments[fail_before:].
        self.fail_before = fail_before
        self.error = error
        self.calls: list[list] = []

    def generate_stream(self, messages, *, timeout_s=None):
        self.calls.append(list(messages))
        for i, fragment in enumerate(self.fragments):
            if self.fail_before is not None and i == self.fail_before:
                raise self.error
            yield fragment


class FakeStreamingLLMWithThinking:
    """A `LanguageModel` whose `generate_stream` also accepts `on_thinking`.

    ADR-015. Distinct from `FakeStreamingLLM` above, which deliberately does
    NOT accept the parameter -- that fake is what proves the orchestrator
    never forces every adapter to support it.
    """

    def __init__(self, thinking: list[str], fragments: list[str]) -> None:
        self.thinking = thinking
        self.fragments = fragments

    def generate_stream(self, messages, *, timeout_s=None, on_thinking=None):
        if on_thinking is not None:
            for fragment in self.thinking:
                on_thinking(fragment)
        yield from self.fragments


class FakeSTT:
    def __init__(
        self, text: str | list[str] = "こんにちは", error: Exception | None = None
    ):
        # A list lets a follow-up test give each successive `transcribe` call
        # a different transcript (e.g. real speech, then silence); a plain
        # string keeps every existing single-turn caller unchanged.
        self._texts = [text] if isinstance(text, str) else list(text)
        self.error = error
        self.calls = 0

    def transcribe(self, audio):
        if self.error is not None:
            raise self.error
        # Repeats the last entry once exhausted, so a caller need not size the
        # list to the exact number of calls a follow-up chain will make.
        index = min(self.calls, len(self._texts) - 1)
        text = self._texts[index]
        self.calls += 1
        return Transcript(text=text, language="ja", duration_ms=100)


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
        # One entry appended per call, so a test can assert the orchestrator
        # passed `follow_up_ms` on a follow-up capture and `None` on the
        # original wake-triggered one (ADR-015).
        self.speech_start_timeout_calls: list[int | None] = []

    def capture_utterance(
        self, *, max_duration_ms, silence_timeout_ms, speech_start_timeout_ms=None
    ):
        self.captures += 1
        self.speech_start_timeout_calls.append(speech_start_timeout_ms)
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


# ------------------------------------------------------ streaming voice turns
# ADR-014: `ponzu start` speaks the first sentence while the model is still
# generating. Fragments below deliberately don't align with sentence
# boundaries, to prove `_SentenceSplitter` reassembles across them.


def test_streaming_turn_speaks_each_sentence_in_order_as_it_completes() -> None:
    fragments = ["こんに", "ちは。二番", "目の文です", "。三番目でした", "。"]
    llm = FakeStreamingLLM(fragments)
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("何か言って"),
        tts=tts,
        audio_in=FakeMic(),
        audio_out=speaker,
    )
    seen = trace(orch)

    result = orch.voice_turn()

    assert result.ok
    assert tts.spoken == ["こんにちは。", "二番目の文です。", "三番目でした。"]
    assert len(speaker.played) == 3
    # Exactly one THINKING -> SPEAKING transition for the whole turn, not one
    # per sentence -- the transition table forbids SPEAKING -> SPEAKING.
    assert seen.count(State.SPEAKING) == 1
    assert seen == [
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]


def test_streaming_turn_records_the_full_reply_not_per_sentence() -> None:
    fragments = ["こんに", "ちは。二番", "目の文です", "。三番目でした", "。"]
    context = ConversationContext()
    orch = Orchestrator(
        llm=FakeStreamingLLM(fragments),
        context=context,
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )

    orch.voice_turn()

    # `build` returns [system, *history, user]; the assistant message just
    # before the trailing user probe is what got recorded for the turn.
    messages = context.build("次の質問")
    assert messages[-2].role == "assistant"
    assert messages[-2].content == "こんにちは。二番目の文です。三番目でした。"


def test_streaming_turn_with_no_terminator_still_speaks_the_remainder() -> None:
    fragments = ["ただの続き", "だけで終わります"]
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeStreamingLLM(fragments),
        stt=FakeSTT("何か言って"),
        tts=tts,
        audio_in=FakeMic(),
        audio_out=speaker,
    )

    result = orch.voice_turn()

    assert result.ok
    assert tts.spoken == ["ただの続きだけで終わります"]
    assert len(speaker.played) == 1


def test_streaming_turn_first_audio_ms_is_present_and_before_llm_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real time.sleep is never used here; a strictly-increasing fake clock
    # guarantees first_audio_ms (recorded mid-stream, before the third
    # sentence is even generated) is smaller than llm_ms (recorded once the
    # whole stream is exhausted), without depending on wall-clock timing.
    counter = itertools.count()
    monkeypatch.setattr(
        "ponzu.core.orchestrator.time.monotonic", lambda: float(next(counter))
    )

    fragments = ["一文目。", "二文目。", "三文目。"]
    orch = Orchestrator(
        llm=FakeStreamingLLM(fragments),
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )

    result = orch.voice_turn()

    assert result.ok
    first_audio_ms = result.metrics.extra["first_audio_ms"]
    assert first_audio_ms >= 0
    assert first_audio_ms < result.metrics.llm_ms


def test_streaming_turn_falls_back_to_generate_when_llm_has_no_generate_stream() -> (
    None
):
    assert not hasattr(FakeLLM(), "generate_stream")
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM("わかりません。"),
        stt=FakeSTT("いま何時"),
        tts=tts,
        audio_in=FakeMic(),
        audio_out=speaker,
    )

    result = orch.voice_turn()

    assert result.ok
    assert tts.spoken == ["わかりません。"]
    assert orch.state is State.IDLE


def test_streaming_turn_mid_stream_failure_speaks_queued_audio_then_errors() -> None:
    fragments = ["一文目。", "二文目。", "三文目。"]
    llm = FakeStreamingLLM(
        fragments, fail_before=2, error=AdapterUnavailable("ollama died mid-stream")
    )
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("何か言って"),
        tts=tts,
        audio_in=FakeMic(),
        audio_out=speaker,
    )
    seen = trace(orch)

    result = orch.voice_turn()

    # The two sentences already committed to must finish; the third, which
    # never arrived, obviously does not.
    assert tts.spoken == ["一文目。", "二文目。"]
    assert len(speaker.played) == 2
    assert not result.ok
    assert "AdapterUnavailable" in result.error
    assert orch.state is State.IDLE
    # ADR-014: cutting the assistant off mid-sentence to report the failure
    # would be worse than letting what's already queued finish.
    assert speaker.cancels == 0
    assert seen == [
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.ERROR,
        State.IDLE,
    ]


def test_streaming_turn_failure_before_first_sentence_uses_normal_recovery() -> None:
    # Nothing was ever spoken, so this must behave exactly like the existing
    # non-streaming recovery path -- including calling cancel().
    llm = FakeStreamingLLM(
        ["何か話す前に壊れます"],
        fail_before=0,
        error=AdapterTimeout("ollama stalled"),
    )
    speaker = FakeSpeaker()
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=speaker,
    )
    seen = trace(orch)

    result = orch.voice_turn()

    assert not result.ok
    assert "AdapterTimeout" in result.error
    assert orch.state is State.IDLE
    assert speaker.cancels >= 1
    assert seen == [
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.ERROR,
        State.IDLE,
    ]


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


def test_loop_logs_no_traceback_or_message_for_an_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DESIGN section 7: only the exception type may reach the log, same rule
    # `_recover` already follows for a `PonzuError`. `_log.exception(...)`
    # would attach a traceback the `text` formatter prints verbatim, so it
    # must not be called here.
    secret_message = "this exception message must never reach the log"

    class Exploding:
        def transcribe(self, audio):
            raise ValueError(secret_message)

    detector = ManualWakeWord()
    orch = Orchestrator(
        llm=FakeLLM(), stt=Exploding(), audio_in=FakeMic(), wake_word=detector
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    with caplog.at_level(logging.INFO):
        detector.trigger(None)

    assert secret_message not in caplog.text
    assert "Traceback" not in caplog.text
    record = next(
        r for r in caplog.records if getattr(r, "ponzu_event", None) == "turn_failed"
    )
    assert record.ponzu_fields["reason"] == "ValueError"
    assert record.exc_info is None


def test_stop_releases_resources() -> None:
    detector = ManualWakeWord()
    speaker = FakeSpeaker()
    orch = Orchestrator(llm=FakeLLM(), audio_out=speaker, wake_word=detector)
    detector.start()

    orch.stop()

    assert speaker.cancels == 1


def test_loop_exits_when_detector_stops_on_its_own() -> None:
    # ADR-010 / DESIGN section 8: a detector that has reached end-of-stream can
    # never fire again. Waiting on it forever is the stuck state the spec
    # forbids, so run_forever must return rather than spin.
    class DeadDetector:
        def __init__(self) -> None:
            self.stopped = 0

        def start(self) -> None: ...

        def stop(self) -> None:
            self.stopped += 1

        def on_detected(self, callback) -> None: ...

        @property
        def is_running(self) -> bool:
            return False

    detector = DeadDetector()
    orch = Orchestrator(llm=FakeLLM(), wake_word=detector)

    orch.run_forever()  # must return, not hang

    assert detector.stopped >= 1
    assert orch.state is State.IDLE


# --------------------------------------------------------- visible reasoning
# ADR-015: `generate_stream`'s `on_thinking` callback is what lets the CLI
# print reasoning while the model is still producing it.


def test_on_thinking_subscriber_receives_fragments_in_order() -> None:
    llm = FakeStreamingLLMWithThinking(
        thinking=["まず", "考えて", "みます"], fragments=["わかりました。"]
    )
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )
    seen: list[str] = []
    orch.on_thinking(seen.append)

    result = orch.voice_turn()

    assert result.ok
    assert seen == ["まず", "考えて", "みます"]


def test_on_thinking_with_no_subscriber_does_not_break_the_turn() -> None:
    llm = FakeStreamingLLMWithThinking(thinking=["hmm"], fragments=["はい。"])
    orch = Orchestrator(
        llm=llm,
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )
    # No `on_thinking` subscription at all -- default behaviour must be
    # unaffected by the feature existing.

    result = orch.voice_turn()

    assert result.ok
    assert result.response == "はい。"


def test_fake_llm_without_on_thinking_support_still_works() -> None:
    # `FakeStreamingLLM` deliberately has no `on_thinking` parameter at all;
    # the orchestrator must not force every adapter to accept the kwarg just
    # because a subscriber is registered.
    fragments = ["こんにちは。"]
    orch = Orchestrator(
        llm=FakeStreamingLLM(fragments),
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=FakeMic(),
        audio_out=FakeSpeaker(),
    )
    orch.on_thinking(lambda _fragment: None)

    result = orch.voice_turn()

    assert result.ok
    assert result.response == "こんにちは。"


# --------------------------------------------------------- follow-up window
# ADR-015: after speaking, a configured follow-up window keeps listening with
# no wake word needed. Driven through `_on_wake` (as `test_wake_word_triggers_
# a_turn` already does above) since that's where the chaining lives.


def test_follow_up_after_success_runs_a_second_turn_with_no_wake_word() -> None:
    detector = ManualWakeWord()
    mic = FakeMic()
    # Two real utterances, then silence -- the third capture ends the chain
    # instead of running indefinitely, isolating "exactly one follow-up ran".
    stt = FakeSTT(["最初の発話", "二回目の発話", "   "])
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM("はい。"),
        stt=stt,
        tts=tts,
        audio_in=mic,
        audio_out=speaker,
        wake_word=detector,
        follow_up_ms=4000,
    )
    seen = trace(orch)
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    # Wake-triggered turn + one follow-up turn that actually ran, then a third
    # capture that found silence and stopped the chain.
    assert mic.captures == 3
    assert tts.spoken == ["はい。", "はい。"]
    assert orch.state is State.IDLE
    # The follow-up continues directly from SPEAKING -- not through an IDLE
    # that never really happened (ADR-008's trace would otherwise lie).
    pairs = list(itertools.pairwise(seen))
    assert (State.SPEAKING, State.LISTENING) in pairs
    # First capture uses the configured default (None here); the follow-up
    # capture is given `follow_up_ms` instead of `speech_start_timeout_ms`.
    assert mic.speech_start_timeout_calls[0] is None
    assert mic.speech_start_timeout_calls[1] == 4000


def test_follow_up_window_with_silence_returns_to_idle() -> None:
    detector = ManualWakeWord()
    mic = FakeMic()
    # First capture: a real utterance. Second (the follow-up): nothing said.
    stt = FakeSTT(["最初の発話", "   "])
    tts, speaker = FakeTTS(), FakeSpeaker()
    orch = Orchestrator(
        llm=FakeLLM("はい。"),
        stt=stt,
        tts=tts,
        audio_in=mic,
        audio_out=speaker,
        wake_word=detector,
        follow_up_ms=4000,
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    assert mic.captures == 2  # the original turn, plus one follow-up attempt
    assert tts.spoken == ["はい。"]  # the follow-up never spoke -- nothing to say
    assert orch.state is State.IDLE


def test_follow_up_ms_zero_never_opens_a_window() -> None:
    detector = ManualWakeWord()
    mic = FakeMic()
    orch = Orchestrator(
        llm=FakeLLM("はい。"),
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=mic,
        audio_out=FakeSpeaker(),
        wake_word=detector,
        follow_up_ms=0,  # the default -- explicit here for clarity
    )
    seen = trace(orch)
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    assert mic.captures == 1
    assert orch.state is State.IDLE
    pairs = list(itertools.pairwise(seen))
    assert (State.SPEAKING, State.LISTENING) not in pairs


def test_failed_turn_does_not_open_a_follow_up_window() -> None:
    detector = ManualWakeWord()
    mic = FakeMic()
    orch = Orchestrator(
        llm=FakeLLM(error=AdapterUnavailable("ollama down")),
        stt=FakeSTT("何か言って"),
        tts=FakeTTS(),
        audio_in=mic,
        audio_out=FakeSpeaker(),
        wake_word=detector,
        follow_up_ms=4000,
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    # A failed turn must not be mistaken for an invitation to keep listening.
    assert mic.captures == 1
    assert orch.state is State.IDLE


def test_follow_up_chain_is_capped_at_max_consecutive_follow_ups() -> None:
    detector = ManualWakeWord()
    mic = FakeMic()
    # A single string repeats forever (FakeSTT's "repeat the last entry"
    # behaviour) -- room noise that never stops, in other words -- so nothing
    # but the cap can end this chain.
    orch = Orchestrator(
        llm=FakeLLM("はい。"),
        stt=FakeSTT("ずっと話し続けます"),
        tts=FakeTTS(),
        audio_in=mic,
        audio_out=FakeSpeaker(),
        wake_word=detector,
        follow_up_ms=4000,
    )
    detector.on_detected(orch._on_wake)
    detector.start()

    detector.trigger(0.9)

    # The wake-triggered turn, plus exactly the capped number of follow-ups --
    # not one more, even though every capture "heard" more speech.
    assert mic.captures == 1 + _MAX_CONSECUTIVE_FOLLOW_UPS
    assert orch.state is State.IDLE
