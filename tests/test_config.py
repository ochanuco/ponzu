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
    assert config.wake_word.provider == "keyboard"
    assert config.llm.provider == "ollama"
    assert config.llm.model == "qwen3:30b"
    assert config.privacy.persist_audio is False
    assert config.audio.sample_rate == 16000
    assert config.logging.format == "json"


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
    assert config.wake_word.provider == "keyboard"
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
