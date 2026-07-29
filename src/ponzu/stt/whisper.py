"""faster-whisper speech recognition (DESIGN section 4.4).

`faster_whisper` (and the `numpy` array type it consumes) is an optional
extra (ADR-009 consequences); see `ponzu.audio.capture` for the same
lazy-import rationale. The model is loaded lazily on first `transcribe()`
call and cached on the instance -- model load is slow, and `ponzu doctor`
(which only calls `probe()`) must not pay that cost.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import ModuleType
from typing import Any

from ponzu.adapters import AdapterUnavailable, AudioBuffer, ProbeResult, Transcript
from ponzu.core import paths
from ponzu.core.config import SttConfig
from ponzu.core.logging import chars, log_event

_REMEDY = "install with: uv sync --extra stt"

# Size names faster-whisper downloads and caches itself. Anything not in this
# set and not an existing path is treated by faster-whisper as a Hugging Face
# repository id, so `probe` can tell a plausible config from a broken one
# without touching the network (ADR-009).
_MODEL_SIZES: frozenset[str] = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-small.en",
        "distil-medium.en",
        "distil-large-v2",
        "distil-large-v3",
    }
)

# int16 full-scale magnitude, used to normalize PCM samples into the
# float32 [-1.0, 1.0] range faster-whisper expects.
_INT16_FULL_SCALE = 32768.0


def _confidence_from_logprobs(logprobs: list[float]) -> float | None:
    """Turn faster-whisper's per-segment `avg_logprob` into a 0..1 score.

    `avg_logprob` is a mean per-token log probability, not a true joint
    probability, so `exp(avg_logprob)` is only an approximation -- but it is
    an honest one (it is monotonic in the model's own confidence and is
    bounded into [0, 1]), unlike inventing a number from nothing. With no
    segments at all there is nothing to base a number on, so `None` is
    returned instead (Transcript.confidence: "unavailable on some backends").
    """
    if not logprobs:
        return None
    mean_logprob = sum(logprobs) / len(logprobs)
    return min(1.0, max(0.0, math.exp(mean_logprob)))


class WhisperRecognizer:
    """`SpeechRecognizer` + `Probeable` backed by faster-whisper (DESIGN 4.4)."""

    def __init__(
        self, config: SttConfig, *, logger: logging.Logger | None = None
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger("ponzu.stt.whisper")
        self._model: Any | None = None

    def _import_faster_whisper(self) -> ModuleType:
        try:
            import faster_whisper
        except ImportError as exc:
            raise AdapterUnavailable(
                f"faster-whisper is not installed ({_REMEDY})"
            ) from exc
        return faster_whisper

    def _load_model(self) -> Any:
        # Cached on the instance so only the first transcribe() pays the
        # (slow) model-load cost; see module docstring.
        if self._model is None:
            faster_whisper = self._import_faster_whisper()
            # ADR-006: user data -- including a downloaded model -- lives
            # under the Ponzu data directory, not faster-whisper's default
            # Hugging Face cache. `models_dir()` must exist before it can be
            # used as a download target.
            paths.ensure_data_dirs()
            self._model = faster_whisper.WhisperModel(
                self._config.model, download_root=str(paths.models_dir())
            )
        return self._model

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        """Transcribe `audio`; empty/silent input yields empty text, not an error."""
        if not audio.pcm:
            return Transcript(text="", language=None, confidence=None, duration_ms=0)

        try:
            import numpy as np
        except ImportError as exc:
            raise AdapterUnavailable(f"numpy is not installed ({_REMEDY})") from exc

        model = self._load_model()

        samples = (
            np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32)
            / _INT16_FULL_SCALE
        )

        configured_language = self._config.language
        language = None if configured_language in ("", "auto") else configured_language
        segments, info = model.transcribe(samples, language=language)

        texts: list[str] = []
        logprobs: list[float] = []
        for segment in segments:
            texts.append(segment.text)
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                logprobs.append(avg_logprob)

        text = "".join(texts).strip()
        detected_language = (
            getattr(info, "language", None) or configured_language or None
        )

        # Character count only -- never the transcript itself (DESIGN
        # section 7 "Transcript character count" / "Disabled by default").
        log_event(
            self._logger,
            "stt_result",
            duration_ms=audio.duration_ms,
            chars=chars(text),
        )

        return Transcript(
            text=text,
            language=detected_language,
            confidence=_confidence_from_logprobs(logprobs),
            duration_ms=audio.duration_ms,
        )

    def probe(self) -> ProbeResult:
        """Report faster-whisper availability and whether `config.model` resolves.

        Never raises (base.py `Probeable` contract) and never downloads
        anything -- a bare model name is reported as "will be downloaded",
        not fetched.
        """
        try:
            self._import_faster_whisper()
        except AdapterUnavailable as exc:
            return ProbeResult(
                component="stt.whisper",
                status="fail",
                detail=str(exc),
                remedy=_REMEDY,
            )

        model_ref = self._config.model
        try:
            is_local = Path(model_ref).exists()
        except OSError:
            is_local = False

        if is_local:
            return ProbeResult(
                component="stt.whisper",
                status="ok",
                detail=f"local model at {model_ref}",
            )
        if model_ref in _MODEL_SIZES:
            return ProbeResult(
                component="stt.whisper",
                status="warn",
                detail=f"model size {model_ref!r} will be downloaded on first use",
                remedy="pre-download it to avoid a slow first turn",
            )
        # Not a size name and not a path that exists. faster-whisper resolves
        # anything else as a Hugging Face repository id, which fails at the
        # first transcription rather than here -- so this is a fail, not a
        # warn. A `models/ggml-*.bin` path from whisper.cpp lands here
        # (ADR-009).
        return ProbeResult(
            component="stt.whisper",
            status="fail",
            detail=(
                f"model {model_ref!r} is neither a known size name nor an "
                "existing path; faster-whisper would resolve it as a Hugging "
                "Face repository id"
            ),
            remedy=f"set stt.model to one of: {', '.join(sorted(_MODEL_SIZES))}",
        )
