"""Configuration model and loader (DESIGN section 5, ADR-006).

Defaults live in code so the app runs with nothing installed beyond the
package itself; the repository only ships an *example* file
(`config/default.example.yaml`), never a live one. The user's real config, if
any, lives at `paths.config_path()` and is deep-merged on top of the
defaults.
"""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ponzu.adapters.base import PonzuError
from ponzu.core import paths


class ConfigError(PonzuError):
    """Raised for a missing, unparsable, or invalid user configuration.

    Includes unknown-key rejection: a typo in the user's YAML (e.g.
    `sensitivty`) must be surfaced, not silently ignored.
    """


@dataclass(frozen=True, slots=True)
class WakeWordConfig:
    phrase: str
    sensitivity: float
    provider: str


@dataclass(frozen=True, slots=True)
class SttConfig:
    provider: str
    model: str
    language: str


@dataclass(frozen=True, slots=True)
class LlmConfig:
    provider: str
    endpoint: str
    model: str
    timeout_s: float


@dataclass(frozen=True, slots=True)
class TtsConfig:
    provider: str
    endpoint: str
    speaker_id: int
    speed: float
    pitch: float
    intonation: float
    volume: float
    timeout_s: float


@dataclass(frozen=True, slots=True)
class AudioConfig:
    input_device: int | None
    output_device: int | None
    sample_rate: int
    max_utterance_ms: int
    silence_timeout_ms: int


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    persist_audio: bool
    persist_transcripts: bool
    persist_conversations: bool


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str
    format: str


@dataclass(frozen=True, slots=True)
class Config:
    wake_word: WakeWordConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    privacy: PrivacyConfig
    logging: LoggingConfig


# In-code mirror of config/default.example.yaml (DESIGN section 5.1 plus the
# keys the implementation needs). Kept in code, not read from the example
# file, so the defaults still apply on a machine where the repo checkout
# (and therefore the example file) isn't present -- e.g. an installed wheel.
_DEFAULTS: dict[str, Any] = {
    "wake_word": {"phrase": "ぽんず", "sensitivity": 0.6, "provider": "keyboard"},
    "stt": {
        # ADR-009: the backend is faster-whisper. `model` is either a size name
        # it downloads and caches, or a directory holding a CTranslate2 model --
        # anything else is resolved as a Hugging Face repository id.
        "provider": "faster_whisper",
        "model": "small",
        "language": "ja",
    },
    "llm": {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "model": "qwen3:30b",
        # Sized for the cold load, not the steady state. On an M1 Max / 64 GB
        # the default qwen3:30b takes ~27 s to load 18 GB before answering at
        # all, then settles at ~4-5 s per warm turn (DESIGN section 11). The
        # original 30 s could not cover the cold start.
        "timeout_s": 120.0,
    },
    "tts": {
        "provider": "voicevox",
        "endpoint": "http://127.0.0.1:50021",
        "speaker_id": 0,
        "speed": 1.0,
        "pitch": 0.0,
        "intonation": 1.0,
        "volume": 1.0,
        # DESIGN section 8 lists "VOICEVOX unavailable" as a recoverable
        # failure, which requires a bounded wait rather than the HTTP client's
        # default. Synthesis of a long reply is slower than a chat completion
        # round trip, hence the larger value than llm.timeout_s.
        "timeout_s": 60.0,
    },
    "audio": {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "max_utterance_ms": 10000,
        "silence_timeout_ms": 1200,
    },
    "privacy": {
        "persist_audio": False,
        "persist_transcripts": False,
        "persist_conversations": False,
    },
    "logging": {"level": "info", "format": "json"},
}

# (section, key) pairs holding filesystem paths or network endpoints, where
# a user is likely to write `~` or `$HOME`-style references.
_EXPANDABLE: tuple[tuple[str, str], ...] = (
    ("stt", "model"),
    ("llm", "endpoint"),
    ("tts", "endpoint"),
)


def _deep_merge(
    defaults: dict[str, Any],
    overlay: dict[str, Any],
    unknown: list[str],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Merge `overlay` onto `defaults`, recording unknown keys as `prefix.key`.

    Only keys already present in `defaults` at the same nesting level are
    accepted; everything else is collected in `unknown` so the caller can
    report every typo at once instead of failing on the first.
    """
    merged = dict(defaults)
    for key, value in overlay.items():
        dotted = f"{prefix}{key}"
        if key not in defaults:
            unknown.append(dotted)
            continue
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(
                    f"expected a mapping for {dotted!r}, got {type(value).__name__}"
                )
            merged[key] = _deep_merge(
                defaults[key], value, unknown, prefix=f"{dotted}."
            )
        else:
            merged[key] = value
    return merged


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(path: Path | None = None) -> Config:
    """Load configuration, overlaying the user's YAML on in-code defaults.

    `path` overrides the default lookup at `paths.config_path()` (mainly for
    tests); with no file at either location, the defaults alone are used, so
    the app runs with zero configuration present.
    """
    effective_path = path if path is not None else paths.config_path()
    merged = copy.deepcopy(_DEFAULTS)

    if effective_path.exists():
        try:
            raw = yaml.safe_load(effective_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {effective_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{effective_path} must contain a YAML mapping at the top level"
            )
        unknown: list[str] = []
        merged = _deep_merge(merged, raw, unknown)
        if unknown:
            raise ConfigError(
                "unknown configuration key(s): " + ", ".join(sorted(unknown))
            )

    for section, key in _EXPANDABLE:
        merged[section][key] = _expand(merged[section][key])

    return Config(
        wake_word=WakeWordConfig(**merged["wake_word"]),
        stt=SttConfig(**merged["stt"]),
        llm=LlmConfig(**merged["llm"]),
        tts=TtsConfig(**merged["tts"]),
        audio=AudioConfig(**merged["audio"]),
        privacy=PrivacyConfig(**merged["privacy"]),
        logging=LoggingConfig(**merged["logging"]),
    )


def write_default_config(dest: Path) -> None:
    """Copy the bundled example config to `dest` for first-run setup."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.bundled_example_config(), dest)
