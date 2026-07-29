"""Tests for ponzu.stt.whisper (DESIGN section 4.4, ADR-009).

`faster_whisper`/`numpy` are optional extras and absent in this test
environment (ADR-009 consequences), so most coverage here works two ways:
against the real (missing) dependency for the `AdapterUnavailable` /
`ProbeResult` failure paths every environment must support, and against a
fake `faster_whisper` (plus a fake `numpy`) injected into `sys.modules`,
which drives an actual `transcribe()` cycle the way a real backend would.
"""

from __future__ import annotations

import importlib
import struct
import sys
import types

import pytest

from ponzu.adapters import AdapterUnavailable, AudioBuffer
from ponzu.core.config import SttConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_faster_whisper(
    segments: list[tuple[str, float]], detected_language: str | None
):
    """Fake `faster_whisper` exposing just the `WhisperModel` surface
    `ponzu.stt.whisper` uses: construction from a model ref, and
    `.transcribe(samples, language=...)` -> (segments, info).
    """
    module = types.ModuleType("faster_whisper")

    class _Segment:
        def __init__(self, text: str, avg_logprob: float) -> None:
            self.text = text
            self.avg_logprob = avg_logprob

    class _Info:
        def __init__(self, language: str | None) -> None:
            self.language = language

    class WhisperModel:
        def __init__(self, model_ref: str) -> None:
            self.model_ref = model_ref

        def transcribe(self, samples, language=None):
            return (
                iter(_Segment(text, avg_logprob) for text, avg_logprob in segments),
                _Info(detected_language),
            )

    module.WhisperModel = WhisperModel
    return module


def _fake_numpy():
    """Fake `numpy` covering only what `transcribe()` calls: `frombuffer`
    into something with `.astype()` and `/` support, plus the two dtype
    markers used as arguments.
    """
    module = types.ModuleType("numpy")
    module.int16 = "int16"
    module.float32 = "float32"

    class _FakeArray:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def astype(self, dtype):
            return _FakeArray(list(self.values))

        def __truediv__(self, scalar):
            return _FakeArray([v / scalar for v in self.values])

        def __len__(self) -> int:
            return len(self.values)

    def frombuffer(buf: bytes, dtype=None):
        count = len(buf) // 2
        values = list(struct.unpack(f"<{count}h", buf)) if count else []
        return _FakeArray(values)

    module.frombuffer = frombuffer
    return module


def _config(**overrides) -> SttConfig:
    base = {"provider": "whisper_cpp", "model": "tiny", "language": "ja"}
    base.update(overrides)
    return SttConfig(**base)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_imports_without_optional_dependencies() -> None:
    # ADR-009: faster_whisper/numpy are absent here; import must still succeed.
    importlib.import_module("ponzu.stt.whisper")


# ---------------------------------------------------------------------------
# Failure paths (no faster_whisper/numpy installed)
# ---------------------------------------------------------------------------


def test_transcribe_raises_adapter_unavailable_without_faster_whisper(
    hide_module,
) -> None:
    hide_module("faster_whisper")
    from ponzu.stt.whisper import WhisperRecognizer

    recognizer = WhisperRecognizer(_config())
    audio = AudioBuffer(pcm=b"\x00\x00\x00\x00", sample_rate=16000)

    with pytest.raises(AdapterUnavailable):
        recognizer.transcribe(audio)


def test_probe_fails_without_faster_whisper(hide_module) -> None:
    hide_module("faster_whisper")
    from ponzu.stt.whisper import WhisperRecognizer

    result = WhisperRecognizer(_config()).probe()

    assert result.status == "fail"
    assert result.remedy


# ---------------------------------------------------------------------------
# Empty/silent input never raises
# ---------------------------------------------------------------------------


def test_transcribe_empty_audio_returns_empty_transcript_without_import() -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    recognizer = WhisperRecognizer(_config())
    transcript = recognizer.transcribe(AudioBuffer(pcm=b"", sample_rate=16000))

    assert transcript.text == ""
    assert transcript.is_empty
    assert transcript.duration_ms == 0


# ---------------------------------------------------------------------------
# Driven through a fake faster_whisper/numpy
# ---------------------------------------------------------------------------


def test_transcribe_drives_full_cycle_with_fake_backend(monkeypatch) -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    fake_fw = _fake_faster_whisper(
        segments=[(" hello", -0.1), (" world", -0.05)],
        detected_language="en",
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setitem(sys.modules, "numpy", _fake_numpy())

    recognizer = WhisperRecognizer(_config(model="tiny", language="auto"))
    audio = AudioBuffer(pcm=struct.pack("<4h", 100, -100, 200, -200), sample_rate=16000)

    transcript = recognizer.transcribe(audio)

    assert transcript.text == "hello world"
    assert transcript.language == "en"
    assert transcript.confidence is not None
    assert 0.0 <= transcript.confidence <= 1.0
    assert transcript.duration_ms == audio.duration_ms


def test_transcribe_caches_model_across_calls(monkeypatch) -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    load_count = {"n": 0}
    fake_fw = _fake_faster_whisper(segments=[(" hi", -0.2)], detected_language="en")
    real_model_cls = fake_fw.WhisperModel

    class CountingModel(real_model_cls):
        def __init__(self, model_ref: str) -> None:
            load_count["n"] += 1
            super().__init__(model_ref)

    fake_fw.WhisperModel = CountingModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setitem(sys.modules, "numpy", _fake_numpy())

    recognizer = WhisperRecognizer(_config())
    audio = AudioBuffer(pcm=struct.pack("<2h", 100, -100), sample_rate=16000)

    recognizer.transcribe(audio)
    recognizer.transcribe(audio)

    # Model load is slow (module docstring): it must happen once, on first
    # transcribe(), not be paid again on a second call.
    assert load_count["n"] == 1


def test_transcribe_with_no_segments_returns_empty_text_and_no_confidence(
    monkeypatch,
) -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    fake_fw = _fake_faster_whisper(segments=[], detected_language=None)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setitem(sys.modules, "numpy", _fake_numpy())

    recognizer = WhisperRecognizer(_config())
    audio = AudioBuffer(pcm=struct.pack("<2h", 0, 0), sample_rate=16000)

    transcript = recognizer.transcribe(audio)

    assert transcript.text == ""
    # No segments -> nothing honest to base a confidence number on.
    assert transcript.confidence is None


# ---------------------------------------------------------------------------
# probe(): model resolution without downloading anything
# ---------------------------------------------------------------------------


def test_probe_warns_when_model_is_a_bare_name_not_found_locally(monkeypatch) -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        _fake_faster_whisper(segments=[], detected_language=None),
    )

    result = WhisperRecognizer(_config(model="tiny")).probe()

    assert result.status == "warn"
    assert result.remedy


def test_probe_ok_when_model_is_an_existing_local_path(monkeypatch, tmp_path) -> None:
    from ponzu.stt.whisper import WhisperRecognizer

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        _fake_faster_whisper(segments=[], detected_language=None),
    )

    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"fake-model-bytes")

    result = WhisperRecognizer(_config(model=str(model_path))).probe()

    assert result.status == "ok"
