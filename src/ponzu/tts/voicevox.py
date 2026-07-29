"""VOICEVOX speech synthesis adapter (DESIGN section 4.7, ADR-004).

VOICEVOX's HTTP API is two calls -- build a synthesis query, then render it --
so this adapter's job stops at "text in, PCM out" while voice identity
(speaker id, speed, pitch, intonation, volume) stays entirely in
configuration (DESIGN section 4.7 "All voice settings should be user
configuration, not hard-coded").
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any

import httpx

from ponzu.adapters import AdapterTimeout, AdapterUnavailable, AudioBuffer, ProbeResult
from ponzu.core.config import TtsConfig
from ponzu.core.logging import log_event

_logger = logging.getLogger("ponzu.tts.voicevox")


class VoicevoxSpeechSynthesizer:
    """`SpeechSynthesizer` + `Probeable` backed by a VOICEVOX Engine instance."""

    def __init__(self, config: TtsConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        # Injected client is what lets tests drive this against
        # `httpx.MockTransport` instead of a running engine.
        self._client = client or httpx.Client(timeout=config.timeout_s)

    def synthesize(self, text: str) -> AudioBuffer:
        start = time.monotonic()

        query = self._request(
            "post",
            "/audio_query",
            params={"text": text, "speaker": self._config.speaker_id},
        ).json()
        # Voice settings are configuration, not adapter behavior (DESIGN
        # section 4.7): mutate the query VOICEVOX generated rather than
        # letting the caller shape it.
        query["speedScale"] = self._config.speed
        query["pitchScale"] = self._config.pitch
        query["intonationScale"] = self._config.intonation
        query["volumeScale"] = self._config.volume

        wav_bytes = self._request(
            "post",
            "/synthesis",
            params={"speaker": self._config.speaker_id},
            json=query,
        ).content

        duration_ms = int((time.monotonic() - start) * 1000)
        audio = _read_wav(wav_bytes)

        log_event(
            _logger, "tts_request", output_chars=len(text), duration_ms=duration_ms
        )

        return audio

    def probe(self) -> ProbeResult:
        """`ponzu doctor` check (ADR-011). Never raises."""
        try:
            response = self._client.get(f"{self._config.endpoint}/speakers")
        except httpx.HTTPError:
            return ProbeResult(
                component="tts",
                status="fail",
                detail=f"VOICEVOX unreachable at {self._config.endpoint}",
                remedy="Start the VOICEVOX Engine application.",
            )

        if not response.is_success:
            return ProbeResult(
                component="tts",
                status="fail",
                detail=f"VOICEVOX returned HTTP {response.status_code}",
                remedy="Start the VOICEVOX Engine application.",
            )

        try:
            speakers = response.json()
            style_ids = {
                style.get("id")
                for speaker in speakers
                for style in speaker.get("styles", [])
            }
        except (ValueError, AttributeError, TypeError):
            # Reachable but not the shape we expect -- can't confirm the
            # speaker, but that's not the same as being unreachable.
            return ProbeResult(
                component="tts",
                status="warn",
                detail=f"VOICEVOX at {self._config.endpoint} returned an unexpected payload",
                remedy=f"check tts.speaker_id ({self._config.speaker_id}) against GET /speakers",
            )

        if self._config.speaker_id in style_ids:
            return ProbeResult(
                component="tts",
                status="ok",
                detail=f"speaker {self._config.speaker_id} available",
            )
        return ProbeResult(
            component="tts",
            status="warn",
            detail=f"speaker {self._config.speaker_id} not found among VOICEVOX styles",
            remedy=f"check tts.speaker_id ({self._config.speaker_id}) against GET /speakers",
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Shared error mapping for the two VOICEVOX calls `synthesize` makes.

        Not a cross-adapter base class -- just a private helper so the same
        endpoint-down-vs-slow distinction (spec: same mapping as the LLM
        adapter) isn't written out twice in this one class.
        """
        try:
            response = self._client.request(
                method, f"{self._config.endpoint}{path}", **kwargs
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Checked before the broader TimeoutException below: ConnectTimeout
            # is itself a TimeoutException subclass.
            raise AdapterUnavailable(
                f"VOICEVOX is not reachable at {self._config.endpoint}. "
                "Start the VOICEVOX Engine application."
            ) from exc
        except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise AdapterTimeout(
                f"VOICEVOX at {self._config.endpoint} did not respond in time."
            ) from exc

        if not response.is_success:
            # Status code only -- never the body (DESIGN section 7; same
            # rule as the LLM adapter, since audio_query echoes the text).
            raise AdapterUnavailable(
                f"VOICEVOX returned HTTP {response.status_code} for {path}"
            )
        return response


def _read_wav(data: bytes) -> AudioBuffer:
    """Parse a WAV byte string without assuming a fixed rate.

    VOICEVOX's output sample rate depends on the engine version, so it must
    be read from the header rather than hard-coded.
    """
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
    return AudioBuffer(
        pcm=pcm, sample_rate=sample_rate, channels=channels, sample_width=sample_width
    )
