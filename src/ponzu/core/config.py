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
    # ADR-013: the whisper gate's own model, kept separate from `stt.model`
    # so the gate stays cheap while transcription accuracy is unaffected.
    model: str
    # Longest audio window the gate will transcribe in one attempt.
    max_window_ms: int
    # Trailing silence that ends a window before `max_window_ms` is reached.
    silence_timeout_ms: int
    # Accepted transcriptions besides `phrase` itself (ADR-013: a phrase this
    # short is genuinely ambiguous to an ASR model, so the accepted set is
    # configuration, not a hidden constant).
    variants: list[str]
    # Maximum edit distance from the normalised phrase. ADR-013 measured that
    # no whisper model transcribes "ぽんず" correctly, so exact matching never
    # fires; distance 1 works and distance 2 is unusable.
    max_distance: int


@dataclass(frozen=True, slots=True)
class SttConfig:
    provider: str
    model: str
    language: str
    # Vocabulary bias passed to the recogniser (DESIGN section 4.4). Empty
    # disables it.
    initial_prompt: str


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
    # Bounds the wait for speech to begin; `silence_timeout_ms` only applies
    # once speech has already started (DESIGN section 4.3).
    speech_start_timeout_ms: int
    # ADR-015: after speaking, how long to keep listening with no wake word
    # before giving up and returning to IDLE. `0` disables the feature
    # entirely -- a false-trigger risk (room noise crossing the capture's RMS
    # floor with no wake word standing between it and a turn), so it must be
    # possible to turn off.
    follow_up_ms: int


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
class UiConfig:
    """What the terminal shows. Not logging.

    `show_thinking` started out under `logging`, which was misleading:
    reasoning is never written to a log whatever this says (DESIGN section 7
    excludes model output), so a `logging` key implying otherwise would send a
    reader looking in the wrong place. This controls what is printed to a
    terminal the user is already watching, which is a different act.
    """

    # ADR-015: a reasoning model spends ~99% of a turn thinking before the
    # first character of the answer exists (ADR-014). Printing it fills that
    # silence. Default True because an assistant that looks frozen is the
    # worse failure.
    show_thinking: bool


@dataclass(frozen=True, slots=True)
class Config:
    wake_word: WakeWordConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    privacy: PrivacyConfig
    logging: LoggingConfig
    ui: UiConfig


# In-code mirror of config/default.example.yaml (DESIGN section 5.1 plus the
# keys the implementation needs). Kept in code, not read from the example
# file, so the defaults still apply on a machine where the repo checkout
# (and therefore the example file) isn't present -- e.g. an installed wheel.
_DEFAULTS: dict[str, Any] = {
    "wake_word": {
        "phrase": "ぽんず",
        # Compared against exp(mean(avg_logprob)), which measured ~0.42 for
        # every utterance tried, correct or not (ADR-013). This is a floor
        # against garbage, not a discriminating threshold, so it sits below the
        # observed band rather than at the 0.6 the original DESIGN example
        # suggested -- 0.6 rejected everything, including correct matches.
        "sensitivity": 0.3,
        # ADR-013: acoustic detection via a whisper gate is now the default.
        # "keyboard" (ADR-010) and "manual" remain available substitutes.
        "provider": "whisper",
        # Kept separate from stt.model so the gate stays cheap (ADR-013).
        # `base`, not `tiny`: tiny hears the phrase as コンゼ, which is edit
        # distance 2 away and would never fire. base gives コンズ (distance 1).
        "model": "base",
        "max_window_ms": 3000,
        "silence_timeout_ms": 600,
        # Extra spellings accepted verbatim, on top of the distance match.
        "variants": ["ぽんず", "ポンズ", "ポン酢", "ぽん酢"],
        # Maximum edit distance from the normalised phrase (ADR-013). Measured:
        # 1 -> 5/7 detected, 1/24 false positives; 2 -> 7/7 but 11/24 false
        # positives (こんにちは, こんばんは, そんな, ...). Do not raise this.
        "max_distance": 1,
    },
    "stt": {
        # ADR-009: the backend is faster-whisper. `model` is either a size name
        # it downloads and caches, or a directory holding a CTranslate2 model --
        # anything else is resolved as a Hugging Face repository id.
        "provider": "faster_whisper",
        "model": "small",
        "language": "ja",
        # Seeds the recogniser's vocabulary (DESIGN section 4.4). Proper nouns
        # are where it fails hardest, and the assistant then reasons
        # confidently about the wrong word. Users should add their own terms.
        "initial_prompt": "ぽんず",
    },
    "llm": {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        # The non-thinking variant of the same 30B-A3B model (ADR-016).
        # Ollama's template implements no thinking switch, so the reasoning
        # variant cannot be told to skip it -- a separate tag is the switch.
        "model": "qwen3:30b-instruct",
        # Sized for the cold load, not the steady state: ~27 s to load 18 GB
        # before answering at all. Warm turns are now sub-second (ADR-016), but
        # the cold start is unchanged and is what this has to cover.
        "timeout_s": 120.0,
    },
    "tts": {
        "provider": "voicevox",
        "endpoint": "http://127.0.0.1:50021",
        # 冥鳴ひまり / ノーマル. Resolves the "VOICEVOX speaker" entry in
        # DESIGN section 11. Style ids are engine-assigned, so `ponzu doctor`
        # checks this against GET /speakers rather than trusting it.
        "speaker_id": 14,
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
        # How long to wait for speech to *begin* after the wake word. Without
        # this the capture runs the full max_utterance_ms when the user says
        # nothing -- ten seconds of dead air that reads as a broken assistant.
        #
        # 5s, not the 2.5s tried first: the user has to notice the assistant
        # woke before they start talking, and 2.5s expired while they were
        # still reacting to the cue. This bounds dead air without racing the
        # person it is waiting for.
        "speech_start_timeout_ms": 5000,
        # ADR-015: how long to keep listening after speaking, with no wake
        # word needed, before giving up and returning to IDLE. 4000, not
        # speech_start_timeout_ms's 5000 -- a follow-up already has the
        # user's attention, so it does not need as generous a reaction
        # window. `0` disables the feature entirely.
        "follow_up_ms": 4000,
    },
    "privacy": {
        "persist_audio": False,
        "persist_transcripts": False,
        "persist_conversations": False,
    },
    "logging": {"level": "info", "format": "json"},
    "ui": {
        # ADR-015: on by default -- see UiConfig.show_thinking.
        "show_thinking": True,
    },
}

# (section, key) pairs holding filesystem paths or network endpoints, where
# a user is likely to write `~` or `$HOME`-style references.
_EXPANDABLE: tuple[tuple[str, str], ...] = (
    ("stt", "model"),
    ("llm", "endpoint"),
    ("tts", "endpoint"),
)


def _scalar_type_error(dotted: str, default: Any, value: Any) -> str | None:
    """Type-check one leaf overlay value against its default's type.

    Returns an error message, or `None` if `value` is acceptable. Dicts are
    handled by the caller before this is reached; this only ever sees leaves.

    Mirrors YAML's own coercion where it matters (an integer literal like
    `timeout_s: 120` must still satisfy a `float` default -- this is exactly
    what `config/default.example.yaml` ships), while keeping `bool` distinct
    from `int`/`float` even though Python's `bool` is an `int` subclass:
    `persist_audio: true` must satisfy a bool field, but `sensitivity: true`
    must not silently satisfy a numeric one.
    """
    if default is None:
        # Only `audio.input_device` / `audio.output_device` default to `None`
        # in `_DEFAULTS`; the dataclass fields behind them declare
        # `int | None`, so an int is the one non-null type to accept here
        # rather than skipping validation for every `None` default.
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            return None
        return f"expected an int or null for {dotted!r}, got {type(value).__name__}"

    if isinstance(default, bool):
        if isinstance(value, bool):
            return None
        return f"expected a bool for {dotted!r}, got {type(value).__name__}"

    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected a number for {dotted!r}, got {type(value).__name__}"
        return None

    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected an int for {dotted!r}, got {type(value).__name__}"
        return None

    if isinstance(default, str):
        if not isinstance(value, str):
            return f"expected a string for {dotted!r}, got {type(value).__name__}"
        return None

    return None


def _deep_merge(
    defaults: dict[str, Any],
    overlay: dict[str, Any],
    unknown: list[str],
    type_errors: list[str],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Merge `overlay` onto `defaults`, recording unknown keys as `prefix.key`.

    Only keys already present in `defaults` at the same nesting level are
    accepted; everything else is collected in `unknown` so the caller can
    report every typo at once instead of failing on the first. Scalars are
    additionally type-checked against the default's type, with every mismatch
    collected into `type_errors` the same way, instead of assigning them
    unchecked and letting a wrong type fail later somewhere confusing.
    """
    merged = dict(defaults)
    for key, value in overlay.items():
        dotted = f"{prefix}{key}"
        if key not in defaults:
            unknown.append(dotted)
            continue
        default_value = defaults[key]
        if isinstance(default_value, dict):
            if not isinstance(value, dict):
                raise ConfigError(
                    f"expected a mapping for {dotted!r}, got {type(value).__name__}"
                )
            merged[key] = _deep_merge(
                default_value, value, unknown, type_errors, prefix=f"{dotted}."
            )
        elif isinstance(default_value, list):
            # New with `wake_word.variants` (ADR-013): the only list-valued
            # default so far, so this checks specifically for a list of
            # strings rather than trying to be a general list validator.
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                type_errors.append(
                    f"expected a list of strings for {dotted!r}, "
                    f"got {type(value).__name__}"
                )
            elif any(not item.strip() for item in value):
                # `WhisperWakeWord._matches` tests each variant as a substring,
                # and every string contains "". A blank entry would therefore
                # wake the assistant on any non-empty transcript at all, which
                # is worse than the typo that produced it.
                type_errors.append(f"{dotted!r} must not contain a blank entry")
            else:
                merged[key] = value
        else:
            error = _scalar_type_error(dotted, default_value, value)
            if error is not None:
                type_errors.append(error)
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
        type_errors: list[str] = []
        merged = _deep_merge(merged, raw, unknown, type_errors)
        if unknown:
            raise ConfigError(
                "unknown configuration key(s): " + ", ".join(sorted(unknown))
            )
        if type_errors:
            raise ConfigError(
                "invalid configuration value(s): " + "; ".join(type_errors)
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
        ui=UiConfig(**merged["ui"]),
    )


def write_default_config(dest: Path) -> None:
    """Copy the bundled example config to `dest` for first-run setup."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.bundled_example_config(), dest)
