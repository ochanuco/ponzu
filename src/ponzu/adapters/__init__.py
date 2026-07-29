"""Adapter package (ADR-007).

Re-exports the contract types and protocols from `ponzu.adapters.base` so
callers can write `from ponzu.adapters import LanguageModel` instead of
reaching into the submodule. This module must stay as dependency-free as
`base` itself -- concrete engines live in `ponzu.wakeword`, `ponzu.stt`,
`ponzu.llm`, `ponzu.tts`, and `ponzu.audio`, not here.
"""

from __future__ import annotations

from ponzu.adapters.base import (
    AdapterTimeout,
    AdapterUnavailable,
    AudioBuffer,
    AudioInput,
    AudioOutput,
    LanguageModel,
    Message,
    ModelResponse,
    PonzuError,
    Probeable,
    ProbeResult,
    SpeechRecognizer,
    SpeechSynthesizer,
    Transcript,
    TurnMetrics,
    WakeWordDetector,
)

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
