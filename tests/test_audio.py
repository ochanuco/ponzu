"""Tests for ponzu.audio (DESIGN section 4.3/4.8, ADR-009).

`sounddevice` is an optional extra and is absent in this test environment
(ADR-009 consequences), so most coverage here works two ways:

- against the real (missing) dependency, exercising the `AdapterUnavailable`
  / `ProbeResult` failure paths every environment must support, and
- against a fake `sounddevice` module injected into `sys.modules`, which
  drives the actual capture/playback logic (silence detection, timeouts,
  cancellation) the way a real backend would.
"""

from __future__ import annotations

import importlib
import struct
import sys
import threading
import time
import types

import pytest

from ponzu.adapters import AdapterUnavailable, AudioBuffer
from ponzu.core.config import AudioConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_input_sounddevice(read_chunks):
    """Fake `sounddevice` exposing the blocking-mode `RawInputStream` API
    `ponzu.audio.capture` uses -- the raw variant, whose `read()` returns a PCM
    byte buffer rather than a numpy array. `read_chunks(frames)` is called once
    per `stream.read()`; every call is recorded on `module.calls`.
    """
    calls: list[int] = []
    module = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, *, samplerate, channels, dtype, device):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, frames):
            calls.append(frames)
            return read_chunks(frames), False

    module.RawInputStream = InputStream
    module.calls = calls
    return module


def _loud_pcm(frames: int, amplitude: int = 3000) -> bytes:
    return struct.pack(f"<{frames}h", *([amplitude] * frames))


def _silent_pcm(frames: int) -> bytes:
    return bytes(frames * 2)


def _fake_output_sounddevice(
    write_log, *, started: threading.Event | None = None, delay_s: float = 0.0
):
    """Fake `sounddevice` exposing the blocking-mode `RawOutputStream` API
    `ponzu.audio.playback` uses. Every `stream.write()` call appends the
    chunk to `write_log`; `started` (if given) is set after the first write
    so a test can synchronize a cancelling thread with playback actually
    having begun.
    """
    module = types.ModuleType("sounddevice")

    class OutputStream:
        def __init__(self, *, samplerate, channels, dtype):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, chunk):
            # The adapter uses RawOutputStream, whose write() takes a PCM byte
            # buffer. Rejecting anything else keeps the fake honest: an earlier
            # version accepted whatever it was given, so playback shipped
            # broken against the real `OutputStream` while the suite was green.
            if not isinstance(chunk, bytes | bytearray | memoryview):
                raise TypeError(
                    f"RawOutputStream.write expects a byte buffer, "
                    f"got {type(chunk).__name__}"
                )
            write_log.append(chunk)
            if started is not None:
                started.set()
            if delay_s:
                time.sleep(delay_s)

    module.RawOutputStream = OutputStream
    return module


def _fake_device_query_sounddevice(*, has_input: bool, has_output: bool):
    module = types.ModuleType("sounddevice")
    module.__dict__["_devices"] = [
        {
            "name": "fake device",
            "max_input_channels": 1 if has_input else 0,
            "max_output_channels": 1 if has_output else 0,
        }
    ]

    def query_devices():
        return module._devices

    module.query_devices = query_devices
    return module


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_modules_import_without_optional_dependencies() -> None:
    # ADR-009: sounddevice/numpy are absent here; import must still succeed.
    importlib.import_module("ponzu.audio.capture")
    importlib.import_module("ponzu.audio.playback")


# ---------------------------------------------------------------------------
# AudioBuffer arithmetic
# ---------------------------------------------------------------------------


def test_audio_buffer_duration_ms_arithmetic() -> None:
    one_second = AudioBuffer(
        pcm=b"\x00\x00" * 16000, sample_rate=16000, channels=1, sample_width=2
    )
    assert one_second.duration_ms == 1000

    empty = AudioBuffer(pcm=b"", sample_rate=16000, channels=1, sample_width=2)
    assert empty.duration_ms == 0


# ---------------------------------------------------------------------------
# MicrophoneInput: failure paths (no sounddevice installed)
# ---------------------------------------------------------------------------


def _config(**overrides) -> AudioConfig:
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


def test_capture_utterance_raises_adapter_unavailable_without_sounddevice(
    hide_module,
) -> None:
    hide_module("sounddevice")
    from ponzu.audio.capture import MicrophoneInput

    mic = MicrophoneInput(_config())
    with pytest.raises(AdapterUnavailable):
        mic.capture_utterance(max_duration_ms=1000, silence_timeout_ms=200)


def test_capture_probe_fails_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.audio.capture import MicrophoneInput

    result = MicrophoneInput(_config()).probe()

    assert result.status == "fail"
    assert result.remedy


# ---------------------------------------------------------------------------
# MicrophoneInput: driven through a fake sounddevice
# ---------------------------------------------------------------------------


def test_capture_utterance_stops_on_max_duration(monkeypatch) -> None:
    from ponzu.audio.capture import MicrophoneInput

    fake_sd = _fake_input_sounddevice(lambda frames: _loud_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    mic = MicrophoneInput(_config(sample_rate=16000))
    buffer = mic.capture_utterance(max_duration_ms=300, silence_timeout_ms=10_000)

    # 300ms / 30ms chunks == 10 reads; silence_timeout never triggers because
    # every chunk is "loud".
    assert len(fake_sd.calls) == 10
    assert buffer.pcm != b""
    assert buffer.duration_ms == 300


def test_capture_utterance_stops_on_silence(monkeypatch) -> None:
    from ponzu.audio.capture import MicrophoneInput

    call_count = {"n": 0}

    def read_chunks(frames: int) -> bytes:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return _loud_pcm(frames)
        return _silent_pcm(frames)

    fake_sd = _fake_input_sounddevice(read_chunks)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    mic = MicrophoneInput(_config(sample_rate=16000))
    buffer = mic.capture_utterance(max_duration_ms=5000, silence_timeout_ms=90)

    # 2 loud chunks + 3 silent chunks (90ms / 30ms) == 5 reads -- far short of
    # the ~167 reads a 5000ms max_duration would allow, proving silence (not
    # the max-duration timeout) ended capture.
    assert len(fake_sd.calls) == 5
    assert buffer.pcm != b""


def test_capture_utterance_returns_empty_buffer_when_nothing_captured(
    monkeypatch,
) -> None:
    from ponzu.audio.capture import MicrophoneInput

    fake_sd = _fake_input_sounddevice(lambda frames: _silent_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    mic = MicrophoneInput(_config(sample_rate=16000))
    buffer = mic.capture_utterance(max_duration_ms=90, silence_timeout_ms=1000)

    # Pure silence for the whole max_duration_ms window: base.py treats this
    # as a normal recoverable outcome, not an error.
    assert buffer.pcm == b""
    assert buffer.duration_ms == 0
    assert len(fake_sd.calls) == 3  # ceil(90 / 30)


def test_capture_probe_reports_ok_and_fail_on_device_presence(monkeypatch) -> None:
    from ponzu.audio.capture import MicrophoneInput

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _fake_device_query_sounddevice(has_input=True, has_output=False),
    )
    ok_result = MicrophoneInput(_config()).probe()
    assert ok_result.status == "ok"

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _fake_device_query_sounddevice(has_input=False, has_output=True),
    )
    fail_result = MicrophoneInput(_config()).probe()
    assert fail_result.status == "fail"
    assert fail_result.remedy


# ---------------------------------------------------------------------------
# SpeakerOutput: failure paths (no sounddevice installed)
# ---------------------------------------------------------------------------


def test_play_raises_adapter_unavailable_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.audio.playback import SpeakerOutput

    speaker = SpeakerOutput()
    audio = AudioBuffer(pcm=b"\x00\x00" * 100, sample_rate=16000)
    with pytest.raises(AdapterUnavailable):
        speaker.play(audio)


def test_playback_probe_fails_without_sounddevice(hide_module) -> None:
    hide_module("sounddevice")
    from ponzu.audio.playback import SpeakerOutput

    result = SpeakerOutput().probe()
    assert result.status == "fail"
    assert result.remedy


# ---------------------------------------------------------------------------
# SpeakerOutput: driven through a fake sounddevice
# ---------------------------------------------------------------------------


def test_cancel_before_play_is_a_noop() -> None:
    from ponzu.audio.playback import SpeakerOutput

    speaker = SpeakerOutput()
    assert speaker.is_playing is False

    speaker.cancel()  # must not raise

    assert speaker.is_playing is False


def test_cancel_during_play_stops_it_early(monkeypatch) -> None:
    from ponzu.audio.playback import SpeakerOutput

    write_log: list[bytes] = []
    started = threading.Event()
    fake_sd = _fake_output_sounddevice(write_log, started=started, delay_s=0.02)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    # 1 second of audio at 16kHz mono/int16 -> ~34 30ms chunks if uninterrupted.
    audio = AudioBuffer(
        pcm=b"\x10\x00" * 16000, sample_rate=16000, channels=1, sample_width=2
    )
    speaker = SpeakerOutput()

    thread = threading.Thread(target=speaker.play, args=(audio,))
    thread.start()

    assert started.wait(timeout=2.0), "playback never started"
    assert speaker.is_playing is True
    speaker.cancel()
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    assert speaker.is_playing is False
    assert 0 < len(write_log) < 34  # stopped well before the full buffer


def test_playback_probe_reports_ok_and_fail_on_device_presence(monkeypatch) -> None:
    from ponzu.audio.playback import SpeakerOutput

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _fake_device_query_sounddevice(has_input=False, has_output=True),
    )
    ok_result = SpeakerOutput().probe()
    assert ok_result.status == "ok"

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _fake_device_query_sounddevice(has_input=True, has_output=False),
    )
    fail_result = SpeakerOutput().probe()
    assert fail_result.status == "fail"
    assert fail_result.remedy


def test_capture_gives_up_when_speech_never_starts(monkeypatch) -> None:
    """DESIGN section 4.3: bail out instead of running the full max_duration.

    Regression: `silence_timeout_ms` is only consulted once speech has been
    detected, so a user who said the wake word and then waited to see whether
    anything happened sat through ten seconds of dead air and got an empty
    transcript. It read as a broken assistant.
    """
    from ponzu.audio.capture import MicrophoneInput

    fake_sd = _fake_input_sounddevice(lambda frames: _silent_pcm(frames))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    mic = MicrophoneInput(_config(speech_start_timeout_ms=600))
    buffer = mic.capture_utterance(max_duration_ms=10_000, silence_timeout_ms=1200)

    # 600ms / 30ms chunks == 20 reads, not the 333 that max_duration would take.
    assert len(fake_sd.calls) == 20
    assert buffer.pcm == b""  # speech never started, so nothing is returned
