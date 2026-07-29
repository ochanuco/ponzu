"""Tests for ponzu.core.factory (ADR-011 callers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ponzu.audio.capture import MicrophoneInput
from ponzu.audio.playback import SpeakerOutput
from ponzu.core import factory
from ponzu.core.config import ConfigError, load_config
from ponzu.llm.ollama import OllamaLanguageModel
from ponzu.stt.whisper import WhisperRecognizer
from ponzu.tts.voicevox import VoicevoxSpeechSynthesizer
from ponzu.wakeword.keyboard import KeyboardWakeWord


@pytest.fixture
def default_config(tmp_path: Path):
    # No file present at this path -- load_config falls back to the in-code
    # defaults, so these tests exercise exactly the default provider wiring.
    return load_config(tmp_path / "does-not-exist.yaml")


# --------------------------------------------------------------- build_* dispatch


def test_build_llm_returns_ollama_for_default_config(default_config) -> None:
    llm = factory.build_llm(default_config)
    assert isinstance(llm, OllamaLanguageModel)


def test_build_tts_returns_voicevox_for_default_config(default_config) -> None:
    tts = factory.build_tts(default_config)
    assert isinstance(tts, VoicevoxSpeechSynthesizer)


def test_build_stt_returns_whisper_recognizer_for_default_config(
    default_config,
) -> None:
    # DESIGN 5.1 / config default: stt.provider == "whisper_cpp" names the
    # model *format*, not the Python package -- it must still resolve to
    # WhisperRecognizer (faster-whisper).
    stt = factory.build_stt(default_config)
    assert isinstance(stt, WhisperRecognizer)


def test_build_audio_in_returns_microphone_input(default_config) -> None:
    assert isinstance(factory.build_audio_in(default_config), MicrophoneInput)


def test_build_audio_out_returns_speaker_output(default_config) -> None:
    assert isinstance(factory.build_audio_out(default_config), SpeakerOutput)


def test_build_wake_word_returns_keyboard_for_default_config(default_config) -> None:
    detector = factory.build_wake_word(default_config)
    assert isinstance(detector, KeyboardWakeWord)


# ----------------------------------------------------------- unknown providers


def test_build_llm_unknown_provider_names_bad_value_and_options(default_config) -> None:
    import dataclasses

    bad_config = dataclasses.replace(default_config.llm, provider="not-a-real-llm")
    cfg = dataclasses.replace(default_config, llm=bad_config)

    with pytest.raises(ConfigError, match="not-a-real-llm") as excinfo:
        factory.build_llm(cfg)
    assert "ollama" in str(excinfo.value)


def test_build_tts_unknown_provider_names_bad_value_and_options(default_config) -> None:
    import dataclasses

    cfg = dataclasses.replace(
        default_config, tts=dataclasses.replace(default_config.tts, provider="nope")
    )

    with pytest.raises(ConfigError, match="nope") as excinfo:
        factory.build_tts(cfg)
    assert "voicevox" in str(excinfo.value)


def test_build_stt_unknown_provider_names_bad_value_and_options(default_config) -> None:
    import dataclasses

    cfg = dataclasses.replace(
        default_config, stt=dataclasses.replace(default_config.stt, provider="nope")
    )

    with pytest.raises(ConfigError, match="nope") as excinfo:
        factory.build_stt(cfg)
    assert "whisper_cpp" in str(excinfo.value)


def test_build_wake_word_unknown_provider_names_bad_value_and_options(
    default_config,
) -> None:
    import dataclasses

    cfg = dataclasses.replace(
        default_config,
        wake_word=dataclasses.replace(default_config.wake_word, provider="nope"),
    )

    with pytest.raises(ConfigError, match="nope") as excinfo:
        factory.build_wake_word(cfg)
    message = str(excinfo.value)
    assert "keyboard" in message
    assert "manual" in message


# -------------------------------------------------------------- build_orchestrator


def test_build_orchestrator_voice_false_has_no_audio_adapters(default_config) -> None:
    orchestrator = factory.build_orchestrator(default_config, voice=False)

    # Proves no audio/STT adapters were wired for `ponzu chat`: voice_turn is
    # a wiring bug without them, and raises rather than failing silently
    # (Orchestrator.voice_turn's own contract).
    with pytest.raises(RuntimeError):
        orchestrator.voice_turn()


def test_chat_without_speak_wires_no_synthesizer(default_config) -> None:
    orchestrator = factory.build_orchestrator(default_config, voice=False, speak=False)

    # Nothing to synthesize with, so a speak=True turn is silent rather than
    # failing — Orchestrator._speak returns early when tts is None.
    assert orchestrator._tts is None
    assert orchestrator._audio_out is None


def test_chat_with_speak_wires_output_but_still_no_microphone(default_config) -> None:
    orchestrator = factory.build_orchestrator(default_config, voice=False, speak=True)

    # `ponzu chat --speak` is typed input with a spoken reply. Wiring the output
    # half is the whole point of the flag — without this the flag parses and
    # then does nothing. The input half must stay unwired so the command still
    # runs on a machine with no microphone (ADR-011).
    assert orchestrator._tts is not None
    assert orchestrator._audio_out is not None
    with pytest.raises(RuntimeError):
        orchestrator.voice_turn()


def test_build_orchestrator_voice_true_wires_every_adapter(default_config) -> None:
    orchestrator = factory.build_orchestrator(default_config, voice=True)

    # audio_in and stt are both non-None now, so voice_turn gets past the
    # "wiring bug" RuntimeError check entirely. It still fails -- there is no
    # real microphone in this environment -- but that failure is a recoverable
    # PonzuError the orchestrator itself catches and turns into a failed
    # TurnResult (not a raise), which is exactly what proves every adapter
    # parameter was actually supplied rather than left as None.
    result = orchestrator.voice_turn()
    assert not result.ok


# ---------------------------------------------------- no hardware/network touch


def test_building_every_adapter_succeeds_without_optional_extras(
    default_config,
) -> None:
    """Regression test: construction must never touch hardware or network.

    This must pass even on a machine with the `audio` and `stt` extras
    absent (ADR-009) -- exactly the environment this test suite normally
    runs in. Every concrete adapter imports its backend lazily inside
    methods, not at construction time, so building them here must succeed
    regardless of what is actually installed.
    """
    factory.build_llm(default_config)
    factory.build_tts(default_config)
    factory.build_stt(default_config)
    factory.build_audio_in(default_config)
    factory.build_audio_out(default_config)
    factory.build_wake_word(default_config)
    factory.build_orchestrator(default_config, voice=True)
