"""Microphone capture and end-of-utterance detection (DESIGN section 4.3).

`sounddevice` is an optional, hardware-bound extra (ADR-009 consequences):
importing this module must succeed on a machine with no audio stack, so the
dependency is imported lazily inside the methods that need it rather than at
module scope, and a missing import is converted to `AdapterUnavailable`
(never a bare `ImportError`) so callers only have one adapter-failure type to
handle (base.py).
"""

from __future__ import annotations

import array
import logging
from types import ModuleType
from typing import Any

from ponzu.adapters import AdapterUnavailable, AudioBuffer, ProbeResult
from ponzu.core.config import AudioConfig
from ponzu.core.logging import log_event

_REMEDY = "install with: uv sync --extra audio"

# Energy-based end-of-speech detection is what DESIGN section 4.3 asks for;
# a full VAD is explicitly deferred to ROADMAP Phase 2, so this stays a
# simple RMS floor. 16-bit PCM full scale is 32767; this threshold sits well
# above typical room-noise RMS but well below a spoken syllable.
_SILENCE_RMS_THRESHOLD = 400.0

# Read the stream in fixed windows so the RMS check and both timeouts
# (max duration / trailing silence) are evaluated on a regular cadence.
_CHUNK_MS = 30


def _rms(pcm: bytes) -> float:
    """Root-mean-square energy of 16-bit signed little-endian PCM samples."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5


class MicrophoneInput:
    """`AudioInput` + `Probeable` backed by `sounddevice` (DESIGN section 4.3)."""

    def __init__(
        self, config: AudioConfig, *, logger: logging.Logger | None = None
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger("ponzu.audio.capture")

    def _import_sounddevice(self) -> ModuleType:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AdapterUnavailable(
                f"sounddevice is not installed ({_REMEDY})"
            ) from exc
        return sd

    def capture_utterance(
        self, *, max_duration_ms: int, silence_timeout_ms: int
    ) -> AudioBuffer:
        """Record until speech ends or `max_duration_ms` elapses (base.py).

        Stops on whichever comes first: `max_duration_ms` total elapsed, or
        `silence_timeout_ms` of continuous silence *after* speech has
        started. If speech never started, nothing was meaningfully captured
        (base.py: a recoverable outcome, not an error), so an empty buffer
        is returned rather than raw room-noise silence.
        """
        sd = self._import_sounddevice()

        chunk_frames = max(1, int(self._config.sample_rate * _CHUNK_MS / 1000))
        chunks: list[bytes] = []
        speech_started = False
        silence_ms = 0
        elapsed_ms = 0

        # RawInputStream, not InputStream: the raw variant yields a PCM byte
        # buffer, which is what AudioBuffer holds and what `_rms` reads. The
        # non-raw variant returns a numpy array, so it would make capture
        # depend on numpy for no gain and diverge from playback, which uses
        # RawOutputStream for the same reason.
        with sd.RawInputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            device=self._config.input_device,
        ) as stream:
            while elapsed_ms < max_duration_ms:
                data, _overflowed = stream.read(chunk_frames)
                pcm_chunk = bytes(data)
                chunks.append(pcm_chunk)
                elapsed_ms += _CHUNK_MS

                if _rms(pcm_chunk) >= _SILENCE_RMS_THRESHOLD:
                    speech_started = True
                    silence_ms = 0
                elif speech_started:
                    silence_ms += _CHUNK_MS
                    if silence_ms >= silence_timeout_ms:
                        break

        if not speech_started:
            return AudioBuffer(
                pcm=b"",
                sample_rate=self._config.sample_rate,
                channels=1,
                sample_width=2,
            )

        buffer = AudioBuffer(
            pcm=b"".join(chunks),
            sample_rate=self._config.sample_rate,
            channels=1,
            sample_width=2,
        )
        # Duration only -- never the samples themselves (DESIGN section 7 /
        # privacy.persist_audio defaults false).
        log_event(self._logger, "utterance_captured", duration_ms=buffer.duration_ms)
        return buffer

    def probe(self) -> ProbeResult:
        """Report sounddevice availability and input device presence.

        Never raises (base.py `Probeable` contract) -- any failure, including
        ones specific to the local audio stack, is reported as a `fail`
        `ProbeResult` instead.
        """
        try:
            sd = self._import_sounddevice()
        except AdapterUnavailable as exc:
            return ProbeResult(
                component="audio.capture",
                status="fail",
                detail=str(exc),
                remedy=_REMEDY,
            )

        try:
            devices: Any = sd.query_devices()
            has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception as exc:  # noqa: BLE001 - probe() must never raise (base.py)
            return ProbeResult(
                component="audio.capture",
                status="fail",
                detail=f"could not query audio devices: {exc}",
                remedy="check system audio/microphone permissions",
            )

        if not has_input:
            return ProbeResult(
                component="audio.capture",
                status="fail",
                detail="no input device found",
                remedy="connect or enable a microphone",
            )
        return ProbeResult(
            component="audio.capture",
            status="ok",
            detail=f"{len(devices)} audio device(s) available",
        )
