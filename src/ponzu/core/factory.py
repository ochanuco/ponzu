"""Adapter construction from a `Config` (ADR-011 callers: doctor/chat/start).

Plain module-level dispatch dicts, one per adapter kind that has a `provider`
string in configuration -- mirroring how `ponzu.wakeword.PROVIDERS` already
resolves `WakeWordConfig.provider` to a class (ADR-010: adding a real engine
should be a dict entry, not an orchestrator or factory change).

Construction only wires dependencies together and must never touch hardware
or the network. Every concrete adapter already imports its backend lazily
inside its methods rather than at construction time (see e.g.
`ponzu.audio.capture`, `ponzu.stt.whisper`), which is what lets `ponzu chat`
and this module's own tests run with the `audio`/`stt` extras absent
(ADR-009). Nothing here may add eager validation (e.g. calling `.probe()`)
that would break that property.
"""

from __future__ import annotations

from ponzu.adapters import (
    AudioInput,
    AudioOutput,
    LanguageModel,
    SpeechRecognizer,
    SpeechSynthesizer,
    WakeWordDetector,
)
from ponzu.audio.capture import MicrophoneInput
from ponzu.audio.playback import SpeakerOutput
from ponzu.core.config import Config, ConfigError
from ponzu.core.orchestrator import Orchestrator
from ponzu.core.prompt import ConversationContext
from ponzu.llm.ollama import OllamaLanguageModel
from ponzu.stt.whisper import WhisperRecognizer
from ponzu.tts.voicevox import VoicevoxSpeechSynthesizer
from ponzu.wakeword import PROVIDERS as _WAKE_WORD_PROVIDERS

__all__ = [
    "build_audio_in",
    "build_audio_out",
    "build_llm",
    "build_orchestrator",
    "build_stt",
    "build_tts",
    "build_wake_word",
]

_LLM_PROVIDERS: dict[str, type] = {"ollama": OllamaLanguageModel}

_TTS_PROVIDERS: dict[str, type] = {"voicevox": VoicevoxSpeechSynthesizer}

# The config default names the on-disk model *format* ("whisper_cpp"), not
# the Python package backing it -- `WhisperRecognizer` is actually a
# faster-whisper adapter (DESIGN section 4.4 lists `whisper.cpp` and
# `faster-whisper` as candidate backends; faster-whisper is what got
# implemented). Renaming the config default to match is a spec-level change
# (DESIGN section 5.1) and out of scope here; this dict just maps the
# existing string to the class that currently implements it.
_STT_PROVIDERS: dict[str, type] = {"whisper_cpp": WhisperRecognizer}


def _dispatch(providers: dict[str, type], provider: str, kind: str) -> type:
    """Resolve `provider` in `providers`, or raise a `ConfigError` naming both
    the bad value and the valid options -- so a config typo is diagnosable
    from the error message alone, without reading this module's source.
    """
    try:
        return providers[provider]
    except KeyError:
        valid = ", ".join(sorted(providers))
        raise ConfigError(
            f"unknown {kind} provider {provider!r}; valid options: {valid}"
        ) from None


def build_llm(config: Config) -> LanguageModel:
    cls = _dispatch(_LLM_PROVIDERS, config.llm.provider, "llm")
    return cls(config.llm)


def build_tts(config: Config) -> SpeechSynthesizer:
    cls = _dispatch(_TTS_PROVIDERS, config.tts.provider, "tts")
    return cls(config.tts)


def build_stt(config: Config) -> SpeechRecognizer:
    cls = _dispatch(_STT_PROVIDERS, config.stt.provider, "stt")
    return cls(config.stt)


def build_audio_in(config: Config) -> AudioInput:
    # Unlike llm/tts/stt/wake_word, `AudioConfig` has no `provider` field --
    # there is exactly one capture backend (DESIGN section 4.3), so there is
    # nothing to dispatch on.
    return MicrophoneInput(config.audio)


def build_audio_out(config: Config) -> AudioOutput:
    # Same reasoning as `build_audio_in`: one playback backend (DESIGN
    # section 4.8), no `provider` key to dispatch on. `config` is accepted
    # for a uniform `build_*(config)` signature even though `SpeakerOutput`
    # takes no config today.
    del config
    return SpeakerOutput()


def build_wake_word(config: Config) -> WakeWordDetector:
    provider = config.wake_word.provider
    cls = _dispatch(_WAKE_WORD_PROVIDERS, provider, "wake_word")
    # `KeyboardWakeWord` takes the configured phrase; `ManualWakeWord` (and
    # any future no-argument substitute) takes nothing. This is the one place
    # that needs to know about that difference -- ADR-010 still only requires
    # a dict entry, not an orchestrator change, to add a real engine.
    if provider == "keyboard":
        return cls(config.wake_word.phrase)
    return cls()


def build_orchestrator(
    config: Config, *, voice: bool, speak: bool = False
) -> Orchestrator:
    """Build the orchestrator for `ponzu chat` (`voice=False`) or `ponzu
    start` (`voice=True`).

    `voice=False` wires only the language model and context, so a text turn
    can run -- and `voice_turn` deliberately raises if called anyway (see
    `Orchestrator.voice_turn`) -- with no audio or STT adapter constructed at
    all, keeping `ponzu chat` runnable with no audio hardware present
    (DESIGN section 12).

    `speak` adds the output half only: TTS plus playback, still no microphone
    and no STT. This is what `ponzu chat --speak` needs -- typed input, spoken
    reply -- and it is why `speak` is separate from `voice` rather than folded
    into it.
    """
    llm = build_llm(config)
    context = ConversationContext()

    if not voice:
        return Orchestrator(
            llm=llm,
            context=context,
            tts=build_tts(config) if speak else None,
            audio_out=build_audio_out(config) if speak else None,
            llm_timeout_s=config.llm.timeout_s,
        )

    return Orchestrator(
        llm=llm,
        context=context,
        stt=build_stt(config),
        tts=build_tts(config),
        audio_in=build_audio_in(config),
        audio_out=build_audio_out(config),
        wake_word=build_wake_word(config),
        llm_timeout_s=config.llm.timeout_s,
        max_utterance_ms=config.audio.max_utterance_ms,
        silence_timeout_ms=config.audio.silence_timeout_ms,
    )
