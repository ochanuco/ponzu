"""Adapter contracts.

Every replaceable component named in ADR-007 is defined here as a
``typing.Protocol``. The orchestrator depends only on these names; concrete
engines live under ``ponzu.wakeword``, ``ponzu.stt``, ``ponzu.llm``,
``ponzu.tts``, and ``ponzu.audio`` and are selected by configuration.

Nothing in this module may import a backend, an audio library, or ``httpx``.
Keeping it dependency-free is what lets the test suite and ``ponzu chat`` run on
a machine with no audio stack (ADR-009).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "AdapterTimeout",
    "AdapterUnavailable",
    "AudioBuffer",
    "AudioInput",
    "AudioOutput",
    "LanguageModel",
    "Message",
    "ModelResponse",
    "PonzuError",
    "ProbeResult",
    "Probeable",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "Transcript",
    "TurnMetrics",
    "WakeWordDetector",
]


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Raw PCM audio crossing an adapter boundary.

    DESIGN section 4.3 requires the format at adapter boundaries to be explicit
    rather than assumed, so the rate and width travel with the samples.
    """

    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2  # bytes per sample; 2 == 16-bit signed

    @property
    def duration_ms(self) -> int:
        frame = self.channels * self.sample_width
        if frame == 0 or self.sample_rate == 0:
            return 0
        return int(len(self.pcm) / frame / self.sample_rate * 1000)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Result of speech recognition (DESIGN section 4.4)."""

    text: str
    language: str | None = None
    confidence: float | None = None  # unavailable on some backends
    duration_ms: int | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation handed to the language model."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model: str
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of a single ``ponzu doctor`` check (ADR-011).

    ``detail`` is written to the user's terminal, so it must never contain
    transcripts, prompts, model output, or credentials (DESIGN section 7).
    """

    component: str
    status: Literal["ok", "warn", "fail", "skipped"]
    detail: str = ""
    remedy: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.status == "fail"


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class Probeable(Protocol):
    """Implemented by any adapter that ``ponzu doctor`` can check.

    ``probe`` must not raise: a dependency being absent or unreachable is a
    normal result to report, not an error to propagate.
    """

    def probe(self) -> ProbeResult: ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """ADR-010. The engine behind this interface is expected to change.

    ``start`` returns immediately; detection is delivered through the callback
    registered with ``on_detected``. Implementations must not persist captured
    audio (DESIGN section 4.2).
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def on_detected(self, callback: Callable[[float | None], None]) -> None:
        """Register a callback receiving detection confidence, if available."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the detector can still deliver detections.

        A detector that has reached end-of-stream is indistinguishable from an
        idle one without this: the loop just never fires again. ADR-010 records
        why it is part of the interface.
        """
        ...


@runtime_checkable
class AudioInput(Protocol):
    """Microphone capture and end-of-utterance handling (DESIGN section 4.3)."""

    def capture_utterance(
        self, *, max_duration_ms: int, silence_timeout_ms: int
    ) -> AudioBuffer:
        """Record until speech ends or ``max_duration_ms`` elapses.

        Returns an empty buffer if nothing was captured; that is a recoverable
        outcome, not an error.
        """
        ...


@runtime_checkable
class AudioOutput(Protocol):
    """Playback with cancellation support (DESIGN section 4.8)."""

    def play(self, audio: AudioBuffer) -> None:
        """Play to completion, or return early if ``cancel`` is called."""
        ...

    def cancel(self) -> None:
        """Stop playback. Safe to call when nothing is playing."""
        ...

    @property
    def is_playing(self) -> bool: ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    def transcribe(self, audio: AudioBuffer) -> Transcript: ...


@runtime_checkable
class LanguageModel(Protocol):
    """Transport only.

    Persona, prompt construction, memory policy, and tool execution stay in the
    orchestration layer (DESIGN section 4.5).
    """

    def generate(
        self, messages: Iterable[Message], *, timeout_s: float | None = None
    ) -> ModelResponse: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> AudioBuffer: ...


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class PonzuError(Exception):
    """Base class for errors the orchestrator knows how to recover from."""


class AdapterUnavailable(PonzuError):
    """A backend is not installed, not running, or not reachable."""


class AdapterTimeout(PonzuError):
    """A backend accepted the request but did not answer in time."""


@dataclass(slots=True)
class TurnMetrics:
    """Per-turn timings emitted as one structured log line (DESIGN section 7).

    Character counts only — never the text itself.
    """

    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    input_chars: int = 0
    output_chars: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_event(self) -> dict[str, int | str]:
        event: dict[str, int | str] = {"event": "turn_completed"}
        event.update(
            stt_ms=self.stt_ms,
            llm_ms=self.llm_ms,
            tts_ms=self.tts_ms,
            input_chars=self.input_chars,
            output_chars=self.output_chars,
        )
        event.update(self.extra)
        return event
