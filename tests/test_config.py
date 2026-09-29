"""Tests for ponzu.core.config (DESIGN section 5, ADR-006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ponzu.core.config import ConfigError, load_config, write_default_config
from ponzu.core.paths import bundled_example_config


def test_defaults_load_with_no_file_present(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist" / "config.yaml"
    config = load_config(missing)

    assert config.wake_word.phrase == "ぽんず"
    # ADR-013: acoustic detection via the whisper gate is the default now;
    # "keyboard" (ADR-010) remains available as an explicit opt-in.
    assert config.wake_word.provider == "whisper"
    # ADR-013: `base`, not `tiny` -- tiny hears the phrase as コンゼ, which is
    # edit distance 2 away and would never fire.
    assert config.wake_word.model == "base"
    assert config.wake_word.max_distance == 1
    # Measured ~0.42 for every utterance, so 0.6 rejected correct matches too.
    assert config.wake_word.sensitivity == 0.3
    assert config.wake_word.max_window_ms == 3000
    assert config.wake_word.silence_timeout_ms == 600
    assert config.wake_word.variants == ["ぽんず", "ポンズ", "ポン酢", "ぽん酢"]
    assert config.llm.provider == "ollama"
    # ADR-016: the non-thinking variant. The reasoning one spent 13 s (median)
    # thinking before the first character of an answer.
    assert config.llm.model == "qwen3:30b-instruct"
    assert config.privacy.persist_audio is False
    assert config.audio.sample_rate == 16000
    assert config.logging.format == "json"
    # ADR-015: the follow-up window is on by default (4s) but disableable
    # with 0; reasoning is printed by default (an assistant that looks frozen
    # for ten seconds is the worse failure).
    assert config.audio.follow_up_ms == 4000
    assert config.ui.show_thinking is True
    # ADR-018: off by default -- opening a listening socket is a change in
    # posture and should be asked for, via this key or `--web`.
    assert config.web.enabled is False
    assert config.web.port == 8765
    assert config.web.history == 50


def test_deep_merge_overlays_only_specified_keys(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        """
wake_word:
  sensitivity: 0.9
llm:
  model: "custom-model"
"""
    )

    config = load_config(user_config)

    # Overridden.
    assert config.wake_word.sensitivity == 0.9
    assert config.llm.model == "custom-model"
    # Untouched siblings keep their defaults.
    assert config.wake_word.phrase == "ぽんず"
    assert config.wake_word.provider == "whisper"
    assert config.llm.provider == "ollama"
    assert config.llm.endpoint == "http://127.0.0.1:11434"


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("not_a_real_section:\n  foo: 1\n")

    with pytest.raises(ConfigError, match="not_a_real_section"):
        load_config(user_config)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    # Typo: "sensitivty" instead of "sensitivity" must not silently no-op.
    user_config.write_text("wake_word:\n  sensitivty: 0.9\n")

    with pytest.raises(ConfigError, match="wake_word.sensitivty"):
        load_config(user_config)


def test_wrong_scalar_type_is_rejected_with_dotted_key_named(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text('llm:\n  timeout_s: "abc"\n')

    with pytest.raises(ConfigError, match="llm.timeout_s"):
        load_config(user_config)


def test_int_is_accepted_where_a_float_default_is_expected(tmp_path: Path) -> None:
    # config/default.example.yaml ships `llm.timeout_s: 120` (an int literal
    # in YAML) against a `float` dataclass field; that must keep working.
    user_config = tmp_path / "config.yaml"
    user_config.write_text("llm:\n  timeout_s: 45\n")

    config = load_config(user_config)

    assert config.llm.timeout_s == 45.0


def test_bool_is_rejected_where_a_number_is_expected(tmp_path: Path) -> None:
    # bool is an int subclass in Python; `sensitivity: true` must not be
    # silently accepted as a numeric value.
    user_config = tmp_path / "config.yaml"
    user_config.write_text("wake_word:\n  sensitivity: true\n")

    with pytest.raises(ConfigError, match="wake_word.sensitivity"):
        load_config(user_config)


def test_number_is_rejected_where_a_bool_is_expected(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("privacy:\n  persist_audio: 1\n")

    with pytest.raises(ConfigError, match="privacy.persist_audio"):
        load_config(user_config)


def test_mapping_is_rejected_where_a_scalar_is_expected(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("wake_word:\n  sensitivity:\n    nested: 1\n")

    with pytest.raises(ConfigError, match="wake_word.sensitivity"):
        load_config(user_config)


def test_list_overlay_is_accepted_for_wake_word_variants(tmp_path: Path) -> None:
    # ADR-013: `wake_word.variants` is the first list-valued default; a
    # well-formed list of strings must overlay cleanly.
    user_config = tmp_path / "config.yaml"
    user_config.write_text('wake_word:\n  variants: ["ぽんず", "ぽんずさん"]\n')

    config = load_config(user_config)

    assert config.wake_word.variants == ["ぽんず", "ぽんずさん"]


def test_non_list_is_rejected_where_a_list_is_expected(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text('wake_word:\n  variants: "ぽんず"\n')

    with pytest.raises(ConfigError, match="wake_word.variants"):
        load_config(user_config)


def test_list_of_non_strings_is_rejected_for_wake_word_variants(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("wake_word:\n  variants: [1, 2]\n")

    with pytest.raises(ConfigError, match="wake_word.variants"):
        load_config(user_config)


def test_int_is_accepted_for_a_none_defaulted_audio_device(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("audio:\n  input_device: 3\n")

    config = load_config(user_config)

    assert config.audio.input_device == 3


def test_expands_user_and_env_vars_in_path_and_endpoint_values(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PONZU_TEST_HOST", "127.0.0.1")
    user_config = tmp_path / "config.yaml"
    user_config.write_text(
        """
llm:
  endpoint: "http://$PONZU_TEST_HOST:11434"
"""
    )

    config = load_config(user_config)

    assert config.llm.endpoint == "http://127.0.0.1:11434"


def test_bundled_example_config_loads_without_unknown_keys() -> None:
    # The committed example must itself be valid against the schema, or a
    # first-run `write_default_config` copy would immediately fail to load.
    config = load_config(bundled_example_config())
    assert config.wake_word.phrase == "ぽんず"


def test_write_default_config_copies_example(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "config.yaml"
    write_default_config(dest)

    assert dest.is_file()
    config = load_config(dest)
    assert config.tts.provider == "voicevox"


def test_blank_wake_word_variant_is_rejected(tmp_path: Path) -> None:
    """A blank variant would match every transcript.

    `WhisperWakeWord._matches` tests each variant as a substring, and every
    string contains "" -- so one stray empty entry turns the wake word into
    "any speech at all".
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text('wake_word:\n  variants: ["ぽんず", ""]\n')

    with pytest.raises(ConfigError, match="blank"):
        load_config(config_file)


def test_whitespace_only_wake_word_variant_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text('wake_word:\n  variants: ["ぽんず", "   "]\n')

    with pytest.raises(ConfigError, match="blank"):
        load_config(config_file)


# ------------------------------------------------------------- ADR-015 keys


def test_follow_up_ms_can_be_overridden_and_disabled(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("audio:\n  follow_up_ms: 0\n")

    config = load_config(user_config)

    # 0 is the documented way to disable the feature entirely -- must not be
    # rejected as an invalid or "falsy" value.
    assert config.audio.follow_up_ms == 0


def test_follow_up_ms_rejects_a_non_int_value(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text('audio:\n  follow_up_ms: "soon"\n')

    with pytest.raises(ConfigError, match="audio.follow_up_ms"):
        load_config(user_config)


def test_show_thinking_can_be_disabled(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("ui:\n  show_thinking: false\n")

    config = load_config(user_config)

    assert config.ui.show_thinking is False


def test_show_thinking_rejects_a_non_bool_value(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("ui:\n  show_thinking: 1\n")

    with pytest.raises(ConfigError, match="ui.show_thinking"):
        load_config(user_config)


def test_web_settings_can_be_overridden(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("web:\n  enabled: true\n  port: 9000\n  history: 10\n")

    config = load_config(user_config)

    assert config.web.enabled is True
    assert config.web.port == 9000
    assert config.web.history == 10


def test_web_host_is_not_a_configurable_key(tmp_path: Path) -> None:
    """ADR-018: loopback is not configurable -- there is no `web.host` key
    to override, and a typo'd attempt at one must be rejected like any other
    unknown key."""
    user_config = tmp_path / "config.yaml"
    user_config.write_text('web:\n  host: "0.0.0.0"\n')

    with pytest.raises(ConfigError, match="web.host"):
        load_config(user_config)


def test_web_enabled_rejects_a_non_bool_value(tmp_path: Path) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text("web:\n  enabled: 1\n")

    with pytest.raises(ConfigError, match="web.enabled"):
        load_config(user_config)
