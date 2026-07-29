"""Tests for ponzu.wakeword.whisper_gate (ADR-013, DESIGN section 4.2).

`sounddevice` (ADR-009's `audio` extra) and `faster_whisper` (the `stt`
extra) are both optional and absent in this test environment -- exactly what
CI runs. Coverage here works the same two ways as `tests/test_audio.py` and
`tests/test_stt.py`: against the real (missing) dependencies for the
`AdapterUnavailable` / `ProbeResult` failure paths every environment must
support, and against fakes injected into `sys.modules` (sounddevice) and
`ponzu.wakeword.whisper_gate.WhisperRecognizer` (the recogniser) that drive
the actual gate/match logic the way real backends would.

No real hardware, no real model, nothing that can hang: every thread join or
event wait below has an explicit timeout, and the fakes never block.
"""

from __future__ import annotations

import importlib
import struct
import sys
import threading
import types

import pytest

from ponzu.adapters import (
    AdapterUnavailable,
    AudioBuffer,
    Probeable,
    ProbeResult,
    Transcript,
    WakeWordDetector,
)
from ponzu.core.config import AudioConfig, WakeWordConfig

_WAIT_S = 1.0

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _wake_config(**overrides) -> WakeWordConfig:
    base = {
        "phrase": "ぽんず",
        "sensitivity": 0.6,
        "provider": "whisper",
        "model": "tiny",
        "max_window_ms": 3000,
        "silence_timeout_ms": 600,
        "variants": ["ぽんず", "ポンズ", "ポン酢", "ぽん酢"],
        "max_distance": 1,
    }
    base.update(overrides)
    return WakeWordConfig(**base)


def _audio_config(**overrides) -> AudioConfig:
    base = {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "max_utterance_ms": 10000,
        "silence_timeout_ms": 1200,
        "speech_start_timeout_ms": 2500,
    }
    base.update(overrides)
    return AudioConfig(**base)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _loud_pcm(frames: int, amplitude: int = 3000) -> bytes:
    return struct.pack(f"<{frames}h", *([amplitude] * frames))


def _silent_pcm(frames: int) -> bytes:
    return bytes(frames * 2)


def _loud_then_silent_reader(loud_count: int):
    """`read_chunks(frames)` callable: `loud_count` loud chunks, then silence
    forever -- enough for the gate to open one window, close it, and idle.
    """
    counter = {"n": 0}

    def read_chunks(frames: int) -> bytes:
        counter["n"] += 1
        if counter["n"] <= loud_count:
            return _loud_pcm(frames)
        return _silent_pcm(frames)

    return read_chunks


def _fake_input_sounddevice(read_chunks):
    """Fake `sounddevice` exposing the blocking-mode `RawInputStream` API
    `ponzu.wakeword.whisper_gate` uses -- same shape as
    `tests/test_audio.py`'s helper, plus an `events` log of stream
    open/close so a test can assert the stream is closed before a callback
    runs (ADR-013's microphone-ownership requirement). The gate opens a new
    `RawInputStream` once per window, so this fake must tolerate being
    instantiated more than once across a single test.
    """
    calls: list[int] = []
    events: list[str] = []

    class FakeStream:
        def __init__(self, *, samplerate, channels, dtype, device):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.device = device

        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("close")
            return False

        def read(self, frames):
            calls.append(frames)
            return read_chunks(frames), False

    module = types.ModuleType("sounddevice")
    module.RawInputStream = FakeStream
    module.calls = calls
    module.events = events
    return module


def _fake_recognizer_factory(transcripts: list[Transcript], *, on_transcribe=None):
    """Fake `WhisperRecognizer` replacement: `.transcribe()` returns
    `transcripts` in order (repeating the last one once exhausted, since a
    test only cares about the first window or two), and records every
    `AudioBuffer` it was called with. `on_transcribe`, if given, fires after
    each call -- used to synchronise a test with "a window was transcribed"
    without polling or sleeping.
    """
    received: list[AudioBuffer] = []

    class FakeRecognizer:
        def __init__(self, config, *, logger=None) -> None:
            self.config = config

        def transcribe(self, audio: AudioBuffer) -> Transcript:
            received.append(audio)
            idx = len(received) - 1
            transcript = transcripts[idx] if idx < len(transcripts) else transcripts[-1]
            if on_transcribe is not None:
                on_transcribe()
            return transcript

        def probe(self) -> ProbeResult:
            return ProbeResult(component="stt.whisper", status="ok")

    return FakeRecognizer, received


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_imports_without_optional_dependencies() -> None:
    # ADR-009: sounddevice/faster_whisper are absent here; import must
    # still succeed -- the whole suite depends on this.
    importlib.import_module("ponzu.wakeword.whisper_gate")


# ---------------------------------------------------------------------------
# Failure paths (extras missing)
# ---------------------------------------------------------------------------


def test_probe_fails_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    result = detector.probe()

    assert result.status == "fail"
    assert result.remedy


def test_probe_never_raises_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    # Merely surviving the call is the assertion (Probeable contract).
    WhisperWakeWord(_wake_config(), _audio_config()).probe()


def test_probe_fails_without_faster_whisper(hide_module, monkeypatch) -> None:
    # sounddevice must resolve so the probe reaches the faster-whisper half.
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    hide_module("faster_whisper")
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    result = detector.probe()

    assert result.status == "fail"
    assert result.remedy
    assert result.component == "wake_word.whisper_gate"


def test_run_raises_adapter_unavailable_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(_wake_config(), _audio_config())

    # `_run` is the capture-thread entry point `start()` spawns; calling it
    # directly (synchronously) proves the failure is `AdapterUnavailable`,
    # not a bare `ImportError`, on the exact path `start()` would hit.
    with pytest.raises(AdapterUnavailable):
        detector._run()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_protocols() -> None:
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    assert isinstance(detector, WakeWordDetector)
    assert isinstance(detector, Probeable)


# ---------------------------------------------------------------------------
# start()/stop() lifecycle -- no recogniser needed, nothing ever matches
# ---------------------------------------------------------------------------


def test_stop_without_start_is_a_noop() -> None:
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    detector.stop()  # must not raise or hang

    assert detector.is_running is False


def test_stop_terminates_thread_within_timeout(monkeypatch) -> None:
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    fake_sd = _fake_input_sounddevice(lambda frames: _silent_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    detector.start()
    assert detector.is_running is True

    detector.stop()

    assert detector.is_running is False


def test_double_stop_is_a_noop(monkeypatch) -> None:
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    fake_sd = _fake_input_sounddevice(lambda frames: _silent_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    detector = WhisperWakeWord(_wake_config(), _audio_config())
    detector.start()

    detector.stop()
    detector.stop()  # must not raise or hang

    assert detector.is_running is False


# ---------------------------------------------------------------------------
# Full cycle: fake sounddevice + fake recogniser
# ---------------------------------------------------------------------------


def test_loud_then_silence_transcribes_once_with_accumulated_window(
    monkeypatch,
) -> None:
    from ponzu.wakeword import whisper_gate

    # 3 loud chunks (90ms) then silence; silence_timeout_ms=600 == 20 chunks
    # of trailing silence before the window closes -- 23 chunks * 30ms.
    fake_sd = _fake_input_sounddevice(_loud_then_silent_reader(3))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    transcribed = threading.Event()
    FakeRecognizer, received = _fake_recognizer_factory(
        [Transcript(text="ぽんず", confidence=0.9)], on_transcribe=transcribed.set
    )
    monkeypatch.setattr(whisper_gate, "WhisperRecognizer", FakeRecognizer)

    detector = whisper_gate.WhisperWakeWord(
        _wake_config(silence_timeout_ms=600), _audio_config()
    )
    detector.on_detected(lambda confidence: None)
    detector.start()

    assert transcribed.wait(timeout=_WAIT_S), "transcribe() was never called"
    detector.stop()

    assert len(received) == 1
    assert received[0].duration_ms == 690  # (3 + 20) * 30ms


def test_matching_transcript_fires_callback_exactly_once_with_stream_closed(
    monkeypatch,
) -> None:
    from ponzu.wakeword import whisper_gate

    fake_sd = _fake_input_sounddevice(_loud_then_silent_reader(3))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    FakeRecognizer, received = _fake_recognizer_factory(
        [Transcript(text="ぽんず", confidence=0.9)]
    )
    monkeypatch.setattr(whisper_gate, "WhisperRecognizer", FakeRecognizer)

    fired: list[float | None] = []
    stream_state_at_callback: list[str] = []
    matched = threading.Event()

    def on_detected(confidence: float | None) -> None:
        fired.append(confidence)
        # ADR-013's ownership bug: the callback must never run while the
        # gate still holds the microphone stream open.
        stream_state_at_callback.append(fake_sd.events[-1])
        matched.set()

    detector = whisper_gate.WhisperWakeWord(
        _wake_config(silence_timeout_ms=600), _audio_config()
    )
    detector.on_detected(on_detected)
    detector.start()

    assert matched.wait(timeout=_WAIT_S), "callback was never fired"
    detector.stop()

    assert fired == [0.9]
    assert stream_state_at_callback == ["close"]
    assert len(received) == 1


def test_non_matching_transcript_does_not_fire_callback(monkeypatch) -> None:
    from ponzu.wakeword import whisper_gate

    fake_sd = _fake_input_sounddevice(_loud_then_silent_reader(3))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    transcribed = threading.Event()
    FakeRecognizer, received = _fake_recognizer_factory(
        [Transcript(text="こんにちは", confidence=0.9)], on_transcribe=transcribed.set
    )
    monkeypatch.setattr(whisper_gate, "WhisperRecognizer", FakeRecognizer)

    fired: list[float | None] = []
    detector = whisper_gate.WhisperWakeWord(
        _wake_config(silence_timeout_ms=600), _audio_config()
    )
    detector.on_detected(lambda confidence: fired.append(confidence))
    detector.start()

    assert transcribed.wait(timeout=_WAIT_S), "transcribe() was never called"
    detector.stop()

    assert fired == []
    assert len(received) >= 1


def test_max_window_ms_bounds_window_with_continuous_speech(monkeypatch) -> None:
    from ponzu.wakeword import whisper_gate

    # Continuous speech, huge silence_timeout_ms: only max_window_ms can end
    # the window.
    fake_sd = _fake_input_sounddevice(lambda frames: _loud_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    transcribed = threading.Event()
    FakeRecognizer, received = _fake_recognizer_factory(
        [Transcript(text="ぽんず", confidence=0.9)], on_transcribe=transcribed.set
    )
    monkeypatch.setattr(whisper_gate, "WhisperRecognizer", FakeRecognizer)

    detector = whisper_gate.WhisperWakeWord(
        _wake_config(max_window_ms=300, silence_timeout_ms=10_000), _audio_config()
    )
    detector.on_detected(lambda confidence: None)
    detector.start()

    assert transcribed.wait(timeout=_WAIT_S), "transcribe() was never called"
    detector.stop()

    assert received[0].duration_ms == 300  # 10 chunks * 30ms, the ceiling


# ---------------------------------------------------------------------------
# _matches(): confidence handling and variant matching
# ---------------------------------------------------------------------------


def _detector(**wake_overrides):
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    return WhisperWakeWord(_wake_config(**wake_overrides), _audio_config())


def test_confidence_below_sensitivity_does_not_match() -> None:
    detector = _detector(sensitivity=0.6)
    transcript = Transcript(text="ぽんず", confidence=0.3)

    assert detector._matches(transcript) is False


def test_confidence_none_matches_regardless_of_sensitivity() -> None:
    # The backend did not provide a confidence at all -- accepted rather
    # than silently dropped (base.py `Transcript.confidence`).
    detector = _detector(sensitivity=0.99)
    transcript = Transcript(text="ぽんず", confidence=None)

    assert detector._matches(transcript) is True


def test_confidence_at_or_above_sensitivity_matches() -> None:
    detector = _detector(sensitivity=0.6)
    transcript = Transcript(text="ぽんず", confidence=0.6)

    assert detector._matches(transcript) is True


def test_non_matching_text_does_not_match() -> None:
    detector = _detector()
    transcript = Transcript(text="こんにちは", confidence=0.9)

    assert detector._matches(transcript) is False


def test_empty_transcript_does_not_match() -> None:
    detector = _detector()
    transcript = Transcript(text="", confidence=0.9)

    assert detector._matches(transcript) is False


@pytest.mark.parametrize(
    "text", ["ぽんず", "ポンズ", "ポン酢", "ぽん酢", "ポン酢です。", "ぽんずー"]
)
def test_each_default_variant_matches(text: str) -> None:
    # Default variants mix hiragana, katakana, and kanji -- normalisation
    # (katakana -> hiragana fold, punctuation strip) is what makes every one
    # of them, plus incidental trailing punctuation, match.
    detector = _detector()
    transcript = Transcript(text=text, confidence=0.9)

    assert detector._matches(transcript) is True


def test_phrase_itself_matches_even_if_absent_from_variants() -> None:
    from ponzu.wakeword.whisper_gate import WhisperWakeWord

    detector = WhisperWakeWord(
        _wake_config(phrase="ぽんず", variants=["ポン酢"]), _audio_config()
    )

    assert detector._matches(Transcript(text="ぽんず", confidence=0.9)) is True


def test_wake_gate_event_names_audio_and_transcribe_times_apart(monkeypatch, caplog):
    """DESIGN section 7 fixes the log contract; these two fields once collided.

    `wake_gate` and `stt_result` both carried a `duration_ms` on adjacent
    lines, meaning transcription wall clock in one and audio length in the
    other. Anyone reading the log compared them as if they were the same
    quantity.
    """
    import logging

    from ponzu.wakeword import whisper_gate

    fake_sd = _fake_input_sounddevice(_loud_then_silent_reader(3))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    transcribed = threading.Event()
    FakeRecognizer, _ = _fake_recognizer_factory(
        [Transcript(text="ぽんず", confidence=0.9)], on_transcribe=transcribed.set
    )
    monkeypatch.setattr(whisper_gate, "WhisperRecognizer", FakeRecognizer)

    detector = whisper_gate.WhisperWakeWord(_wake_config(), _audio_config())
    detector.on_detected(lambda confidence: None)

    with caplog.at_level(logging.INFO):
        detector.start()
        assert transcribed.wait(timeout=_WAIT_S), "transcribe() was never called"
        detector.stop()

    events = [
        r.__dict__["ponzu_fields"]
        for r in caplog.records
        if r.__dict__.get("ponzu_event") == "wake_gate"
    ]
    assert events, "no wake_gate event was logged"
    fields = events[0]
    assert "duration_ms" not in fields
    assert set(fields) == {"transcribe_ms", "audio_ms", "chars", "matched"}
    # And still no transcript text anywhere in the record (DESIGN section 7).
    assert "ぽんず" not in caplog.text
