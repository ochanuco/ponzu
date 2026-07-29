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
from collections.abc import Callable
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


def _elapsed_ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


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

    @property
    def state(self) -> State:
        return self._machine.state

    def on_state_change(self, callback: Callable[[State, State], None]) -> None:
        """Subscribe to transitions, for a future status indicator (ADR-008)."""
        self._machine.on_change(callback)

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
            self.voice_turn()
        except Exception:  # noqa: BLE001 - last line of defence for the loop
            log_event(_log, "turn_failed", reason="unexpected")
            _log.exception("unhandled error during turn")
            self._reset_to_idle()

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

    def _reset_to_idle(self) -> None:
        """Guarantee the assistant is never stuck in LISTENING or SPEAKING."""
        if self._audio_out is not None:
            self._audio_out.cancel()
        if self._machine.state is not State.ERROR:
            self._machine.transition(State.ERROR)
        self._machine.transition(State.IDLE)

    def _finish(self, metrics: TurnMetrics) -> None:
        log_event(_log, **metrics.as_event())
