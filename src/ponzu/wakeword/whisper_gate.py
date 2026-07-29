"""Acoustic "ぽんず" wake word via an energy gate + faster-whisper (ADR-013).

DESIGN section 4.2's "Implementation" subsection: transcription only runs
once an RMS floor has been crossed, so idling in silence costs almost nothing
beyond reading the stream. Rather than calling faster-whisper directly, the
gate reuses `WhisperRecognizer` (DESIGN section 4.4) for transcription, which
keeps `download_root` handling, the int16 -> float32 conversion, and
confidence handling in one place instead of duplicating them here.

`sounddevice` (ADR-009's `audio` extra) and `faster_whisper` (the `stt`
extra) are both optional and hardware/model-bound; importing this module
must succeed with neither installed. `sounddevice` is therefore imported
lazily here, the same way `ponzu.audio.capture` does it; `faster_whisper`'s
lazy import is `WhisperRecognizer`'s responsibility.
"""

from __future__ import annotations

import array
import dataclasses
import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from types import ModuleType

from ponzu.adapters import AdapterUnavailable, AudioBuffer, ProbeResult, Transcript
from ponzu.core.config import AudioConfig, SttConfig, WakeWordConfig
from ponzu.core.logging import chars, log_event
from ponzu.stt.whisper import WhisperRecognizer

_AUDIO_REMEDY = "install with: uv sync --extra audio"

# Same cadence as `ponzu.audio.capture`: fine-grained enough for the RMS
# check and both timeouts (silence / max window) to be evaluated regularly.
_CHUNK_MS = 30

# Same RMS floor rationale as `ponzu.audio.capture._SILENCE_RMS_THRESHOLD`:
# comfortably above typical room-noise RMS, comfortably below a spoken
# syllable, for 16-bit signed PCM (full scale 32767).
_SILENCE_RMS_THRESHOLD = 400.0

# How long `stop()` waits for the capture thread to notice the stop event
# and exit before giving up -- kept small and constant so `stop()` can never
# hang a caller, matching `KeyboardWakeWord`.
_JOIN_TIMEOUT_S = 1.0

# Japanese and ASCII punctuation stripped before comparing a transcript to a
# configured variant -- a real transcription of a two-syllable phrase is
# often surrounded by trailing punctuation the model invents.
_PUNCTUATION = "、。・！？,.!?"

# Full-width katakana occupies U+30A1-U+30F6; the corresponding hiragana
# block starts 0x60 lower, at U+3041. Folding katakana to hiragana this way
# is what lets "ポンズ" and "ぽんず" compare equal (ADR-013's accepted variants
# mix both scripts). U+30FC (the long-vowel mark "ー") sits just above this
# range and is deliberately left untouched -- it has no hiragana equivalent.
_KATAKANA_START = 0x30A1
_KATAKANA_END = 0x30F6
_HIRAGANA_OFFSET = 0x60


def _rms(pcm: bytes) -> float:
    """Root-mean-square energy of 16-bit signed little-endian PCM samples.

    Duplicated from `ponzu.audio.capture._rms` (four lines) rather than
    imported, so this module does not reach into another module's private
    name for something this small.
    """
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5


def _normalize(text: str) -> str:
    """Fold katakana to hiragana and strip whitespace/punctuation.

    Two transcripts of the same utterance can differ in script (katakana vs.
    hiragana) and in incidental punctuation the model attaches; normalising
    both away is what lets a single configured variant match either form.
    """
    # NFKC first so full-width/half-width forms collapse before folding.
    normalized = unicodedata.normalize("NFKC", text)
    folded = "".join(
        chr(ord(ch) - _HIRAGANA_OFFSET)
        if _KATAKANA_START <= ord(ch) <= _KATAKANA_END
        else ch
        for ch in normalized
    )
    without_punctuation = "".join(ch for ch in folded if ch not in _PUNCTUATION)
    return "".join(without_punctuation.split())


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Both strings here are a few characters long."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _contains_within_distance(haystack: str, needle: str, max_distance: int) -> bool:
    """Whether any substring of `haystack` is within `max_distance` of `needle`.

    A substring scan rather than a whole-string comparison, because the gate
    routinely transcribes the phrase inside a longer utterance ("ねえぽんず"
    comes back as メイポンズ). Windows one shorter and one longer than the
    needle are checked too, so an inserted or dropped mora still matches.
    """
    if needle in haystack:
        return True
    for width in range(max(1, len(needle) - 1), len(needle) + 2):
        for start in range(len(haystack) - width + 1):
            if _edit_distance(haystack[start : start + width], needle) <= max_distance:
                return True
    return False


class WhisperWakeWord:
    """Acoustic "ぽんず" `WakeWordDetector` (ADR-013) + `Probeable`.

    Holds the microphone (via its own capture loop, not `MicrophoneInput` --
    the gate needs to inspect each chunk's RMS *before* deciding whether to
    keep it, which `AudioInput.capture_utterance` does not expose) while
    idling, and releases it before invoking the callback: `voice_turn` opens
    its own capture stream, and two open input streams on one device is
    exactly the ownership bug ADR-013 calls out.

    IMPORTANT for callers, same as `KeyboardWakeWord`: the registered
    callback executes ON THIS DETECTOR'S OWN BACKGROUND THREAD, never on the
    caller's thread.
    """

    def __init__(
        self,
        config: WakeWordConfig,
        audio: AudioConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._audio = audio
        self._logger = logger or logging.getLogger("ponzu.wakeword.whisper_gate")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[float | None], None] | None = None

        # A separate, cheap model from `stt.model` (ADR-013), and always
        # Japanese -- the wake phrase is fixed regardless of `stt.language`.
        gate_stt_config = SttConfig(
            provider="faster_whisper", model=config.model, language="ja"
        )
        self._recognizer = WhisperRecognizer(gate_stt_config, logger=self._logger)

        variants = set(config.variants) | {config.phrase}
        self._normalized_variants = frozenset(
            _normalize(variant) for variant in variants if variant
        )

    # -- WakeWordDetector -----------------------------------------------

    def start(self) -> None:
        """Spawn the daemon capture thread and return immediately.

        Safe to call again while already running: a no-op, since a second
        thread opening the same input stream would race the first.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait briefly for it to exit.

        Safe to call when never started and safe to call more than once,
        matching `KeyboardWakeWord.stop`.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)

    @property
    def is_running(self) -> bool:
        """ADR-010 liveness: false once the capture thread has exited."""
        return self._thread is not None and self._thread.is_alive()

    def on_detected(self, callback: Callable[[float | None], None]) -> None:
        """Register the detection callback.

        Policy: registering again REPLACES the previously registered
        callback, matching `KeyboardWakeWord`/`ManualWakeWord`.
        """
        self._callback = callback

    # -- capture loop -----------------------------------------------------

    def _import_sounddevice(self) -> ModuleType:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AdapterUnavailable(
                f"sounddevice is not installed ({_AUDIO_REMEDY})"
            ) from exc
        return sd

    def _run(self) -> None:
        try:
            self._capture_loop()
        except Exception as exc:  # noqa: BLE001 - last line of defence for the thread
            # The thread is about to die either way. `is_running` going False
            # already satisfies ADR-010's liveness contract, so the loop in
            # `run_forever` will notice and exit -- but without this the only
            # trace is Python's default traceback on stderr, and the reason
            # (a disconnected device, say) is lost from the structured log.
            # Type only, never the message: DESIGN section 7.
            #
            # Logged and swallowed rather than re-raised. Re-raising in a daemon
            # thread only prints a traceback the user cannot act on; the thread
            # dies either way, `is_running` goes False, and `run_forever` exits
            # and reports through the CLI. The structured event is what carries
            # the reason.
            log_event(self._logger, "wake_gate_failed", reason=type(exc).__name__)

    def _capture_loop(self) -> None:
        sd = self._import_sounddevice()
        chunk_frames = max(1, int(self._audio.sample_rate * _CHUNK_MS / 1000))

        while not self._stop_event.is_set():
            window = self._capture_window(sd, chunk_frames)
            if self._stop_event.is_set():
                # Stopped mid-capture: whatever was accumulated is discarded
                # rather than transcribed on the way out.
                return
            if window is not None:
                self._process_window(window)

    def _capture_window(self, sd: ModuleType, chunk_frames: int) -> bytes | None:
        """Open the mic, accumulate one candidate window, close, and return it.

        Returns `None` if no speech was detected before `stop()` was
        signalled or the read loop otherwise ended with nothing accumulated.
        Below-threshold chunks are discarded, not accumulated, until speech
        has started -- this is what keeps idling in silence nearly free
        (ADR-013).
        """
        chunks: list[bytes] = []
        speech_started = False
        silence_ms = 0
        window_ms = 0

        # RawInputStream, not InputStream: the raw variant yields a PCM byte
        # buffer, matching `ponzu.audio.capture.MicrophoneInput` and what
        # `AudioBuffer`/`_rms` expect -- no numpy dependency for capture
        # itself.
        with sd.RawInputStream(
            samplerate=self._audio.sample_rate,
            channels=1,
            dtype="int16",
            device=self._audio.input_device,
        ) as stream:
            # Wall clock alongside the audio clock, for the same reason as
            # `ponzu.audio.capture`: summing `_CHUNK_MS` assumes every read
            # returns on time, and a device delivering slower than real time
            # makes the window run far past `max_window_ms`.
            # Started when speech does, NOT when the loop does. This loop
            # idles in silence for as long as nobody is talking, so a
            # reference taken up front measures the wait, not the window --
            # after a minute of quiet the budget below was already blown and
            # the first speech chunk closed the window instantly, producing
            # 30 ms of audio and a gate that never fired again.
            window_started_at: float | None = None
            while not self._stop_event.is_set():
                data, _overflowed = stream.read(chunk_frames)
                pcm_chunk = bytes(data)

                if _rms(pcm_chunk) >= _SILENCE_RMS_THRESHOLD:
                    if not speech_started:
                        window_started_at = time.monotonic()
                    speech_started = True
                    silence_ms = 0
                elif not speech_started:
                    # Silence before speech has ever started: this chunk is
                    # not worth keeping (ADR-013's near-free idling).
                    continue
                else:
                    silence_ms += _CHUNK_MS

                chunks.append(pcm_chunk)
                window_ms += _CHUNK_MS

                if speech_started and silence_ms >= self._config.silence_timeout_ms:
                    break
                if window_ms >= self._config.max_window_ms:
                    break
                if (
                    window_started_at is not None
                    and (time.monotonic() - window_started_at) * 1000
                    >= self._config.max_window_ms
                ):
                    break
        # Stream is closed at this point (the `with` block has exited) --
        # transcription below never runs while the mic is still open.

        if not speech_started or not chunks:
            return None
        return b"".join(chunks)

    def _process_window(self, pcm: bytes) -> None:
        audio = AudioBuffer(
            pcm=pcm, sample_rate=self._audio.sample_rate, channels=1, sample_width=2
        )
        start = time.monotonic()
        transcript = self._recognizer.transcribe(audio)
        transcribe_ms = int((time.monotonic() - start) * 1000)

        matched = self._matches(transcript)

        # DESIGN section 7: durations and a character count only, never the
        # transcript itself. `WhisperRecognizer.transcribe` already logs its
        # own `stt_result` event; this one adds the gate's match outcome,
        # which the STT layer has no reason to know about.
        #
        # `transcribe_ms` and `audio_ms` are named apart on purpose. Both this
        # event and `stt_result` used to carry a `duration_ms`, meaning wall
        # clock here and audio length there, on adjacent log lines.
        log_event(
            self._logger,
            "wake_gate",
            transcribe_ms=transcribe_ms,
            audio_ms=audio.duration_ms,
            chars=chars(transcript.text),
            matched=matched,
        )

        if matched and self._callback is not None:
            # The stream is already closed (the `with` block in
            # `_capture_window` exited before this method was ever called),
            # so the callback -- which runs a whole turn and opens its own
            # capture stream -- never contends with this detector for the
            # microphone (ADR-013). `_run`'s loop reopens a stream on its
            # next iteration once this call returns.
            self._callback(transcript.confidence)

    def _matches(self, transcript: Transcript) -> bool:
        if transcript.is_empty:
            return False
        if (
            transcript.confidence is not None
            and transcript.confidence < self._config.sensitivity
        ):
            return False
        # `confidence is None` means the backend did not provide one at all
        # (e.g. no segments): there is nothing to compare against
        # `sensitivity`, so the match is accepted rather than silently
        # dropped (base.py `Transcript.confidence` / DESIGN section 4.4).
        normalized = _normalize(transcript.text)
        # Exact-variant match first: it is cheap and covers the spellings a
        # user explicitly configured.
        if any(variant in normalized for variant in self._normalized_variants):
            return True
        # Then the distance match. ADR-013 measured that no whisper model
        # transcribes "ぽんず" correctly -- base returns コンズ -- so exact
        # matching alone never fires in practice.
        return _contains_within_distance(
            normalized, _normalize(self._config.phrase), self._config.max_distance
        )

    # -- Probeable ----------------------------------------------------------

    def probe(self) -> ProbeResult:
        """Report sounddevice/faster-whisper availability and model validity.

        Never raises (base.py `Probeable` contract). Delegates the
        faster-whisper/model-resolution half entirely to
        `WhisperRecognizer.probe` -- same backend, same `_MODEL_SIZES`
        reasoning -- relabelled under this component's name.
        """
        try:
            self._import_sounddevice()
        except AdapterUnavailable as exc:
            return ProbeResult(
                component="wake_word.whisper_gate",
                status="fail",
                detail=str(exc),
                remedy=_AUDIO_REMEDY,
            )

        stt_probe = self._recognizer.probe()
        return dataclasses.replace(stt_probe, component="wake_word.whisper_gate")
