"""Speaker playback with cancellation (DESIGN section 4.8).

`sounddevice` is an optional, hardware-bound extra (ADR-009 consequences);
see `ponzu.audio.capture` for the same lazy-import rationale. Playback runs
on the calling thread in small chunks, checking a `threading.Event` between
chunks, so `cancel()` invoked from another thread actually interrupts an
in-flight `play()` (DESIGN section 4.8 "support cancellation"). A lock
serializes `play()` calls so assistant speech never overlaps (DESIGN section
4.8 "prevent overlapping assistant speech"). Barge-in is explicitly future
work (DESIGN section 4.8) and is not implemented here.
"""

from __future__ import annotations

import logging
import threading
from types import ModuleType
from typing import Any

from ponzu.adapters import AdapterUnavailable, AudioBuffer, ProbeResult

_REMEDY = "install with: uv sync --extra audio"

# Write in small windows so a cancel() lands within roughly one chunk's
# worth of audio instead of waiting for the whole buffer to finish.
_CHUNK_MS = 30


class SpeakerOutput:
    """`AudioOutput` + `Probeable` backed by `sounddevice` (DESIGN section 4.8)."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("ponzu.audio.playback")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._playing = False

    def _import_sounddevice(self) -> ModuleType:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AdapterUnavailable(
                f"sounddevice is not installed ({_REMEDY})"
            ) from exc
        return sd

    def play(self, audio: AudioBuffer) -> None:
        """Play to completion, or return early once `cancel()` is called."""
        sd = self._import_sounddevice()

        # Serializes overlapping play() calls (DESIGN section 4.8); a second
        # caller blocks here until the first playback finishes or is
        # cancelled, rather than mixing two streams together.
        with self._lock:
            self._stop_event.clear()
            self._playing = True
            try:
                self._play_blocking(sd, audio)
            finally:
                self._playing = False

    def _play_blocking(self, sd: ModuleType, audio: AudioBuffer) -> None:
        frame_size = audio.channels * audio.sample_width
        if frame_size == 0:
            return
        frames_total = len(audio.pcm) // frame_size
        chunk_frames = max(1, int(audio.sample_rate * _CHUNK_MS / 1000))

        # RawOutputStream, not OutputStream: the raw variant takes a buffer of
        # PCM bytes, which is exactly what AudioBuffer already holds.
        # `OutputStream.write` requires a numpy array and raises
        # "dtype mismatch: 'bytesN' vs 'int16'" on bytes, so using it would
        # both crash and drag numpy into the playback path for no benefit.
        with sd.RawOutputStream(
            samplerate=audio.sample_rate,
            channels=audio.channels,
            dtype="int16",
        ) as stream:
            offset = 0
            while offset < frames_total:
                if self._stop_event.is_set():
                    break
                end = min(offset + chunk_frames, frames_total)
                stream.write(audio.pcm[offset * frame_size : end * frame_size])
                offset = end

    def cancel(self) -> None:
        """Stop playback. A no-op when nothing is playing (base.py)."""
        self._stop_event.set()

    @property
    def is_playing(self) -> bool:
        return self._playing

    def probe(self) -> ProbeResult:
        """Report sounddevice availability and output device presence.

        Never raises (base.py `Probeable` contract).
        """
        try:
            sd = self._import_sounddevice()
        except AdapterUnavailable as exc:
            return ProbeResult(
                component="audio.playback",
                status="fail",
                detail=str(exc),
                remedy=_REMEDY,
            )

        try:
            devices: Any = sd.query_devices()
            has_output = any(d.get("max_output_channels", 0) > 0 for d in devices)
        except Exception as exc:  # noqa: BLE001 - probe() must never raise (base.py)
            return ProbeResult(
                component="audio.playback",
                status="fail",
                detail=f"could not query audio devices: {exc}",
                remedy="check system audio/output permissions",
            )

        if not has_output:
            return ProbeResult(
                component="audio.playback",
                status="fail",
                detail="no output device found",
                remedy="connect or enable a speaker/output device",
            )
        return ProbeResult(
            component="audio.playback",
            status="ok",
            detail=f"{len(devices)} audio device(s) available",
        )
