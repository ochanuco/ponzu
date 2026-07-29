"""Conversation orchestration (DESIGN section 4.1).

The orchestrator owns the ADR-008 state machine and calls adapters. It holds
every policy the adapters are forbidden to hold: prompt assembly, context,
timeouts, and error recovery.

It depends only on the protocols in ``ponzu.adapters``. No backend is imported
here, which is what allows ``ponzu chat`` to run a real turn with no audio
adapters constructed at all.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ponzu.adapters import (
    AudioInput,
    AudioOutput,
    LanguageModel,
    PonzuError,
    SpeechRecognizer,
    SpeechSynthesizer,
    TurnMetrics,
    WakeWordDetector,
)
from ponzu.core.logging import chars, log_event
from ponzu.core.prompt import ConversationContext
from ponzu.core.state import State, StateMachine

__all__ = ["Orchestrator", "TurnResult"]

_log = logging.getLogger(__name__)

# ADR-014: terminators that always cut a sentence immediately, regardless of
# what follows. Japanese prose has no "3.5"-style ambiguity, so 。！？ (and a
# bare newline, which separates rather than terminates) never need lookahead.
_JA_TERMINATORS = "。！？"
# ASCII terminators only cut when followed by whitespace or the stream ends.
# `.` immediately followed by a non-space character (as in "3.5") is left
# alone -- see `_SentenceSplitter._find_cut` below.
_ASCII_TERMINATORS = ".!?"


def _elapsed_ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


@dataclass(frozen=True, slots=True)
class _Cut:
    """A sentence boundary found in `_SentenceSplitter`'s buffer.

    `end` is where the emitted sentence stops (terminator included for a
    punctuation cut, excluded for a bare newline); `skip` is where the next
    sentence starts scanning from, which skips the newline itself so it is
    never attached to either side.
    """

    end: int
    skip: int


class _SentenceSplitter:
    """Cuts complete sentences out of streamed text fragments (ADR-014).

    Fed one model fragment at a time; returns whichever sentences became
    complete as a result. An ASCII terminator sitting at the very end of the
    buffer is ambiguous -- "3." could be the end of a sentence or the start of
    "3.5" -- so it is left pending until either more text arrives (revealing
    what follows) or `flush()` is called at the end of the stream.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, fragment: str) -> list[str]:
        self._buffer += fragment
        sentences: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence, remainder = self._buffer[: cut.end], self._buffer[cut.skip :]
            # A run of consecutive newlines (or a newline right at the start of
            # the buffer) would otherwise emit an empty "sentence".
            if sentence:
                sentences.append(sentence)
            self._buffer = remainder.lstrip(" \t")
        return sentences

    def flush(self) -> str:
        """Return and clear whatever is left when the stream has ended.

        Covers both an answer with no terminator at all, and a trailing ASCII
        terminator that never resolved because no more text ever arrived.
        """
        remainder, self._buffer = self._buffer, ""
        return remainder

    def _find_cut(self) -> _Cut | None:
        buf = self._buffer
        for i, ch in enumerate(buf):
            if ch in _JA_TERMINATORS:
                return _Cut(end=i + 1, skip=i + 1)
            if ch == "\n":
                return _Cut(end=i, skip=i + 1)
            if ch in _ASCII_TERMINATORS:
                if i + 1 >= len(buf):
                    # Could still turn into "3.5" once more text arrives --
                    # wait rather than guessing.
                    return None
                if buf[i + 1].isspace():
                    return _Cut(end=i + 1, skip=i + 1)
                # Followed directly by a non-space character (a digit in
                # "3.5"): not a boundary, keep scanning past it.
        return None


class _StreamingTurnFailed(Exception):
    """Internal control-flow signal from `_stream_turn` to `voice_turn`.

    Not a `PonzuError`: it never leaves the orchestrator. It exists only to
    carry whether any sentence had already been spoken, which decides how the
    turn recovers (ADR-014: audio already queued must finish before erroring).
    """

    def __init__(self, original: PonzuError, *, spoke_any: bool) -> None:
        super().__init__(str(original))
        self.original = original
        self.spoke_any = spoke_any


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Outcome of one conversational turn.

    ``error`` is set instead of raising, because DESIGN section 8 requires the
    loop to keep running after a recoverable failure. Callers inspect this;
    they do not catch.
    """

    utterance: str
    response: str
    metrics: TurnMetrics
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Orchestrator:
    """Drives one turn, or the continuous wake-word loop.

    Adapters are injected rather than constructed here so that ``chat`` can
    supply only a language model, ``doctor`` can supply none, and tests can
    supply fakes. The audio-related parameters are genuinely optional: a text
    turn never touches them.
    """

    def __init__(
        self,
        *,
        llm: LanguageModel,
        context: ConversationContext | None = None,
        stt: SpeechRecognizer | None = None,
        tts: SpeechSynthesizer | None = None,
        audio_in: AudioInput | None = None,
        audio_out: AudioOutput | None = None,
        wake_word: WakeWordDetector | None = None,
        llm_timeout_s: float | None = None,
        max_utterance_ms: int = 15_000,
        silence_timeout_ms: int = 1_200,
    ) -> None:
        self._llm = llm
        self._context = context if context is not None else ConversationContext()
        self._stt = stt
        self._tts = tts
        self._audio_in = audio_in
        self._audio_out = audio_out
        self._wake_word = wake_word
        self._llm_timeout_s = llm_timeout_s
        self._max_utterance_ms = max_utterance_ms
        self._silence_timeout_ms = silence_timeout_ms
        self._machine = StateMachine()
        self._stop_requested = False
        self._turn_callback: Callable[[TurnResult], None] | None = None

    @property
    def state(self) -> State:
        return self._machine.state

    def on_state_change(self, callback: Callable[[State, State], None]) -> None:
        """Subscribe to transitions, for a future status indicator (ADR-008)."""
        self._machine.on_change(callback)

    def on_turn(self, callback: Callable[[TurnResult], None]) -> None:
        """Subscribe to completed turns during ``run_forever``.

        DESIGN section 8 requires a failure to produce a message for the user
        when possible, but the orchestrator is a library and must not write to
        a terminal. ``run_forever`` otherwise swallows every ``TurnResult``,
        leaving the caller with nothing to report — this is the seam that lets
        the CLI say what went wrong.
        """
        self._turn_callback = callback

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def text_turn(self, utterance: str, *, speak: bool = False) -> TurnResult:
        """Run a turn from typed input — the path used by ``ponzu chat``.

        Skips LISTENING and TRANSCRIBING entirely rather than faking them, so
        the logged state trace reflects what actually happened.
        """
        metrics = TurnMetrics(input_chars=chars(utterance))
        try:
            response = self._think(utterance, metrics)
            if speak:
                self._speak(response, metrics)
        except PonzuError as exc:
            return self._recover(utterance, metrics, exc)

        self._machine.transition(State.IDLE)
        self._finish(metrics)
        return TurnResult(utterance=utterance, response=response, metrics=metrics)

    def voice_turn(self) -> TurnResult:
        """Run a full capture → transcribe → think → speak turn.

        Requires the audio and STT adapters; calling it without them is a wiring
        bug, not a recoverable runtime failure, so it raises rather than
        returning a failed ``TurnResult``.
        """
        if self._audio_in is None or self._stt is None:
            raise RuntimeError("voice_turn requires audio input and an STT adapter")

        metrics = TurnMetrics()
        utterance = ""
        try:
            self._machine.transition(State.LISTENING)
            audio = self._audio_in.capture_utterance(
                max_duration_ms=self._max_utterance_ms,
                silence_timeout_ms=self._silence_timeout_ms,
            )
            metrics.extra["audio_ms"] = audio.duration_ms

            self._machine.transition(State.TRANSCRIBING)
            started = time.monotonic()
            transcript = self._stt.transcribe(audio)
            metrics.stt_ms = _elapsed_ms(started)

            # DESIGN section 8 lists an empty transcript as a recoverable
            # failure, and section 4.1 requires every recoverable failure to
            # pass through ERROR on its way back to IDLE. The model is not
            # called — there is nothing to answer — but the state trace still
            # records that the turn did not complete normally.
            if transcript.is_empty:
                log_event(_log, "turn_skipped", reason="empty_transcript")
                self._reset_to_idle()
                return TurnResult(utterance="", response="", metrics=metrics)

            utterance = transcript.text
            metrics.input_chars = chars(utterance)

            stream_fn = self._stream_capable()
            if stream_fn is not None:
                try:
                    response = self._stream_turn(utterance, metrics, stream_fn)
                except _StreamingTurnFailed as failed:
                    if failed.spoke_any:
                        return self._recover_after_partial_speech(
                            utterance, metrics, failed.original
                        )
                    return self._recover(utterance, metrics, failed.original)
            else:
                response = self._think(utterance, metrics)
                self._speak(response, metrics)
        except PonzuError as exc:
            return self._recover(utterance, metrics, exc)

        self._machine.transition(State.IDLE)
        self._finish(metrics)
        return TurnResult(utterance=utterance, response=response, metrics=metrics)

    # ------------------------------------------------------------------
    # Wake-word loop
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Idle until the wake word fires, run a turn, return to idle.

        The detector invokes its callback on its own thread (ADR-010), so the
        turn runs there too. Turns are therefore serialised by the detector, not
        by a lock here.
        """
        if self._wake_word is None:
            raise RuntimeError("run_forever requires a wake-word detector")

        self._stop_requested = False
        self._wake_word.on_detected(self._on_wake)
        self._wake_word.start()
        log_event(_log, "loop_started")
        try:
            # ADR-010: also exit when the detector stops on its own. A detector
            # that has reached end-of-stream will never fire again, and waiting
            # on it is exactly the stuck state DESIGN section 8 forbids.
            while not self._stop_requested and self._wake_word.is_running:
                time.sleep(0.1)
            if not self._stop_requested:
                log_event(_log, "loop_ended", reason="detector_stopped")
        except KeyboardInterrupt:
            log_event(_log, "loop_interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the loop and release audio resources (DESIGN section 8)."""
        self._stop_requested = True
        if self._wake_word is not None:
            self._wake_word.stop()
        if self._audio_out is not None:
            self._audio_out.cancel()
        log_event(_log, "loop_stopped")

    def _on_wake(self, confidence: float | None) -> None:
        log_event(_log, "wake_detected", confidence=confidence)
        # A turn must never kill the loop; voice_turn already converts
        # recoverable failures into a TurnResult, and anything else is logged
        # here so the detector thread survives to hear the next wake word.
        try:
            result = self.voice_turn()
        except Exception as exc:  # noqa: BLE001 - last line of defence for the loop
            # Exception type only, same as `_recover` below: this is not a
            # PonzuError, but the text/traceback still must not reach the log
            # under the `text` format (DESIGN section 7). The message itself
            # still travels in the returned TurnResult for the CLI to print.
            log_event(_log, "turn_failed", reason=type(exc).__name__)
            self._reset_to_idle()
            result = TurnResult(
                utterance="",
                response="",
                metrics=TurnMetrics(),
                error=f"{type(exc).__name__}: {exc}",
            )

        if self._turn_callback is not None:
            # A subscriber that raises must not take the loop down with it.
            try:
                self._turn_callback(result)
            except Exception:  # noqa: BLE001
                _log.exception("turn subscriber raised")

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _think(self, utterance: str, metrics: TurnMetrics) -> str:
        self._machine.transition(State.THINKING)
        started = time.monotonic()
        result = self._llm.generate(
            self._context.build(utterance), timeout_s=self._llm_timeout_s
        )
        metrics.llm_ms = _elapsed_ms(started)
        metrics.output_chars = chars(result.text)
        self._context.record(utterance, result.text)
        return result.text

    def _speak(self, text: str, metrics: TurnMetrics) -> None:
        if self._tts is None or self._audio_out is None:
            return
        self._machine.transition(State.SPEAKING)
        started = time.monotonic()
        audio = self._tts.synthesize(text)
        metrics.tts_ms = _elapsed_ms(started)
        self._audio_out.play(audio)

    def _stream_capable(self) -> Callable[..., Iterable[str]] | None:
        """The streaming path is only available with both TTS and playback.

        Mirrors `_speak`'s existing no-op-without-adapters behaviour: a turn
        wired without TTS/audio output falls back to `_think`, which also
        skips speaking entirely.
        """
        if self._tts is None or self._audio_out is None:
            return None
        stream_fn = getattr(self._llm, "generate_stream", None)
        return stream_fn if callable(stream_fn) else None

    def _stream_turn(
        self,
        utterance: str,
        metrics: TurnMetrics,
        stream_fn: Callable[..., Iterable[str]],
    ) -> str:
        """ADR-014: synthesize and speak sentences while generation continues.

        `THINKING -> SPEAKING` fires exactly once, on the first sentence --
        the transition table forbids `SPEAKING -> SPEAKING`, so every later
        sentence is synthesised and played without a further state change.
        On failure, `voice_turn` decides how to recover based on whether any
        sentence was already spoken (`_StreamingTurnFailed.spoke_any`).
        """
        assert self._tts is not None
        assert self._audio_out is not None

        self._machine.transition(State.THINKING)
        turn_started = time.monotonic()
        splitter = _SentenceSplitter()
        parts: list[str] = []
        spoken = False
        tts_ms_total = 0
        first_audio_ms: int | None = None

        def _speak_sentence(sentence: str) -> None:
            nonlocal spoken, tts_ms_total, first_audio_ms
            if not spoken:
                self._machine.transition(State.SPEAKING)
                spoken = True
            tts_started = time.monotonic()
            audio = self._tts.synthesize(sentence)  # type: ignore[union-attr]
            tts_ms_total += _elapsed_ms(tts_started)
            if first_audio_ms is None:
                first_audio_ms = _elapsed_ms(turn_started)
            self._audio_out.play(audio)  # type: ignore[union-attr]

        try:
            for fragment in stream_fn(
                self._context.build(utterance), timeout_s=self._llm_timeout_s
            ):
                parts.append(fragment)
                for sentence in splitter.feed(fragment):
                    _speak_sentence(sentence)
            # The LLM stream itself is done at this point; whatever remains is
            # flushed below without generation still running behind it.
            metrics.llm_ms = _elapsed_ms(turn_started)

            remainder = splitter.flush()
            if remainder:
                _speak_sentence(remainder)
        except PonzuError as exc:
            metrics.tts_ms = tts_ms_total
            if first_audio_ms is not None:
                metrics.extra["first_audio_ms"] = first_audio_ms
            raise _StreamingTurnFailed(exc, spoke_any=spoken) from exc

        metrics.tts_ms = tts_ms_total
        if first_audio_ms is not None:
            metrics.extra["first_audio_ms"] = first_audio_ms
        full_reply = "".join(parts)
        metrics.output_chars = chars(full_reply)
        self._context.record(utterance, full_reply)
        return full_reply

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover(
        self, utterance: str, metrics: TurnMetrics, exc: PonzuError
    ) -> TurnResult:
        """DESIGN section 8: log sanitized, release audio, return to IDLE.

        The exception *type* is logged, not its message. Adapter messages name
        endpoints and models, which is fine, but this path also catches errors
        whose text originates from a backend response, and DESIGN section 7
        forbids that reaching a log.
        """
        self._machine.transition(State.ERROR)
        log_event(
            _log,
            "turn_failed",
            reason=type(exc).__name__,
            state=self._machine.state.value,
        )
        self._reset_to_idle()
        return TurnResult(
            utterance=utterance,
            response="",
            metrics=metrics,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _recover_after_partial_speech(
        self, utterance: str, metrics: TurnMetrics, exc: PonzuError
    ) -> TurnResult:
        """ADR-014: a sentence was already queued/playing when the stream failed.

        Deliberately does not call `_reset_to_idle()` -- that would cancel
        playback, and cutting the assistant off mid-sentence to report a
        backend failure is worse than letting the sentence it already
        committed to finish. `SPEAKING -> ERROR -> IDLE` is legal on its own
        (ADR-008: any state may go to ERROR, and ERROR may go to IDLE).
        """
        self._machine.transition(State.ERROR)
        log_event(
            _log,
            "turn_failed",
            reason=type(exc).__name__,
            state=self._machine.state.value,
        )
        self._machine.transition(State.IDLE)
        return TurnResult(
            utterance=utterance,
            response="",
            metrics=metrics,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _reset_to_idle(self) -> None:
        """Guarantee the assistant is never stuck in LISTENING or SPEAKING."""
        if self._audio_out is not None:
            self._audio_out.cancel()
        if self._machine.state is not State.ERROR:
            self._machine.transition(State.ERROR)
        self._machine.transition(State.IDLE)

    def _finish(self, metrics: TurnMetrics) -> None:
        log_event(_log, **metrics.as_event())
