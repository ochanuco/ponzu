"""Tests for the VOICEVOX TTS adapter (DESIGN section 4.7, ADR-004).

Driven entirely through `httpx.MockTransport`; the happy-path test builds a
real WAV in-memory with the stdlib `wave` module so the header-parsing logic
is exercised against actual WAV bytes, not a hand-rolled stand-in.
"""

from __future__ import annotations

import io
import json
import logging
import wave

import httpx
import pytest

from ponzu.adapters import AdapterTimeout, AdapterUnavailable
from ponzu.core.config import TtsConfig
from ponzu.tts.voicevox import VoicevoxSpeechSynthesizer


def _config(**overrides: object) -> TtsConfig:
    base: dict[str, object] = {
        "provider": "voicevox",
        "endpoint": "http://127.0.0.1:50021",
        "speaker_id": 1,
        "speed": 1.0,
        "pitch": 0.0,
        "intonation": 1.0,
        "volume": 1.0,
        "timeout_s": 60.0,
    }
    base.update(overrides)
    return TtsConfig(**base)  # type: ignore[arg-type]


def _synth(handler, **config_overrides: object) -> VoicevoxSpeechSynthesizer:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VoicevoxSpeechSynthesizer(_config(**config_overrides), client=client)


def _make_wav(
    pcm: bytes, *, sample_rate: int, channels: int, sample_width: int
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buf.getvalue()


def test_synthesize_happy_path_round_trips_wav_header() -> None:
    pcm = (b"\x01\x02\x03\x04") * 100
    wav_bytes = _make_wav(pcm, sample_rate=24000, channels=1, sample_width=2)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/audio_query":
            assert request.url.params["text"] == "こんにちは"
            assert request.url.params["speaker"] == "1"
            return httpx.Response(
                200,
                json={
                    "accent_phrases": [],
                    "speedScale": 1.0,
                    "pitchScale": 0.0,
                    "intonationScale": 1.0,
                    "volumeScale": 1.0,
                    "outputSamplingRate": 24000,
                },
            )
        if request.url.path == "/synthesis":
            assert request.url.params["speaker"] == "1"
            body = json.loads(request.content)
            # Mutated with config's voice settings before being sent on.
            assert body["speedScale"] == 1.3
            assert body["pitchScale"] == 0.1
            assert body["intonationScale"] == 1.2
            assert body["volumeScale"] == 0.8
            return httpx.Response(
                200, content=wav_bytes, headers={"content-type": "audio/wav"}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    synth = _synth(handler, speed=1.3, pitch=0.1, intonation=1.2, volume=0.8)
    audio = synth.synthesize("こんにちは")

    assert len(calls) == 2
    assert audio.sample_rate == 24000
    assert audio.channels == 1
    assert audio.sample_width == 2
    assert audio.pcm == pcm


def test_synthesize_connection_refused_raises_adapter_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(AdapterUnavailable) as exc_info:
        _synth(handler).synthesize("hello")

    assert "50021" in str(exc_info.value)


def test_synthesize_connect_timeout_raises_adapter_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    with pytest.raises(AdapterUnavailable):
        _synth(handler).synthesize("hello")


def test_synthesize_read_timeout_raises_adapter_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(AdapterTimeout):
        _synth(handler).synthesize("hello")


def test_synthesize_500_raises_adapter_unavailable_without_leaking_body() -> None:
    secret_text_echo = "the input text must never leak into the exception"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=secret_text_echo)

    with pytest.raises(AdapterUnavailable) as exc_info:
        _synth(handler).synthesize("hello")

    assert "500" in str(exc_info.value)
    assert secret_text_echo not in str(exc_info.value)


def test_synthesize_logs_output_chars_not_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pcm = b"\x00\x00" * 10
    wav_bytes = _make_wav(pcm, sample_rate=24000, channels=1, sample_width=2)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio_query":
            return httpx.Response(200, json={})
        return httpx.Response(200, content=wav_bytes)

    with caplog.at_level(logging.INFO, logger="ponzu.tts.voicevox"):
        _synth(handler).synthesize("a secret utterance")

    record = next(
        r for r in caplog.records if getattr(r, "ponzu_event", None) == "tts_request"
    )
    assert record.ponzu_fields["output_chars"] == len("a secret utterance")
    assert "duration_ms" in record.ponzu_fields
    assert "a secret utterance" not in caplog.text


def test_synthesize_malformed_audio_query_json_raises_adapter_unavailable() -> None:
    secret_text_echo = "the input text must never leak into the exception"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/audio_query"
        return httpx.Response(200, content=secret_text_echo.encode())

    with pytest.raises(AdapterUnavailable) as exc_info:
        _synth(handler).synthesize(secret_text_echo)

    assert secret_text_echo not in str(exc_info.value)


def test_synthesize_non_wav_body_raises_adapter_unavailable() -> None:
    secret_audio_stand_in = b"this is not a wav file at all"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio_query":
            return httpx.Response(200, json={})
        return httpx.Response(200, content=secret_audio_stand_in)

    with pytest.raises(AdapterUnavailable) as exc_info:
        _synth(handler).synthesize("hello")

    assert secret_audio_stand_in.decode() not in str(exc_info.value)


def test_probe_ok_when_speaker_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/speakers"
        return httpx.Response(
            200, json=[{"name": "Zundamon", "styles": [{"id": 1, "name": "Normal"}]}]
        )

    result = _synth(handler).probe()

    assert result.status == "ok"
    assert result.component == "tts"


def test_probe_warn_when_speaker_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[{"name": "Zundamon", "styles": [{"id": 99, "name": "Normal"}]}]
        )

    result = _synth(handler).probe()

    assert result.status == "warn"


def test_probe_fail_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = _synth(handler).probe()

    assert result.status == "fail"


def test_probe_never_raises_on_malformed_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    result = _synth(handler).probe()  # must not raise

    assert result.status == "warn"


def test_probe_never_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    result = _synth(handler).probe()  # must not raise

    assert result.status == "fail"
