"""Tests for ponzu.core.paths (ADR-006 data directory rules)."""

from __future__ import annotations

import stat
from pathlib import Path

from ponzu.core import paths


def test_data_dir_defaults_to_application_support(monkeypatch) -> None:
    monkeypatch.delenv("PONZU_DATA_DIR", raising=False)
    assert paths.data_dir() == Path(
        "~/Library/Application Support/Ponzu"
    ).expanduser()


def test_data_dir_honours_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "ponzu-data"
    monkeypatch.setenv("PONZU_DATA_DIR", str(override))
    assert paths.data_dir() == override


def test_derived_paths_live_under_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PONZU_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PONZU_CONFIG", raising=False)
    assert paths.config_path() == tmp_path / "config.yaml"
    assert paths.logs_dir() == tmp_path / "logs"
    assert paths.cache_dir() == tmp_path / "cache"
    assert paths.models_dir() == tmp_path / "models"
    assert paths.audio_dir() == tmp_path / "audio"


def test_config_path_honours_ponzu_config_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PONZU_DATA_DIR", str(tmp_path))
    explicit = tmp_path / "elsewhere" / "config.yaml"
    monkeypatch.setenv("PONZU_CONFIG", str(explicit))
    assert paths.config_path() == explicit


def test_ensure_data_dirs_creates_private_directories(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PONZU_DATA_DIR", str(tmp_path / "ponzu-data"))
    paths.ensure_data_dirs()

    for directory in (
        paths.data_dir(),
        paths.logs_dir(),
        paths.cache_dir(),
        paths.models_dir(),
        paths.audio_dir(),
    ):
        assert directory.is_dir()
        mode = stat.S_IMODE(directory.stat().st_mode)
        assert mode == 0o700


def test_bundled_example_config_exists() -> None:
    example = paths.bundled_example_config()
    assert example.name == "default.example.yaml"
    assert example.is_file()
