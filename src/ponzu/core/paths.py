"""Filesystem locations for user data (ADR-006, DESIGN section 5.2).

Source, examples, and docs live in the git repository. Everything
machine/user-specific -- config, logs, cache, models, audio -- lives outside
it, under a single per-user data directory. Nothing in this module reads or
writes user content; it only computes paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# Mode applied to directories under the data dir (owner rwx only). These can
# hold config, logs, and eventually transcripts/audio -- all "private
# content" under ADR-006 -- so group/other access is deliberately withheld.
_PRIVATE_DIR_MODE = 0o700

# Default macOS location (ADR-006 / DESIGN section 5.2). $PONZU_DATA_DIR
# overrides this, primarily for tests (tmp_path) and non-default setups.
_DEFAULT_DATA_DIR = Path("~/Library/Application Support/Ponzu")


def data_dir() -> Path:
    """Root of the user's Ponzu data directory."""
    override = os.environ.get("PONZU_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DATA_DIR.expanduser()


def config_path() -> Path:
    """Path to the user's config.yaml.

    Honors $PONZU_CONFIG (documented in .env.example); otherwise defaults to
    config.yaml directly under the data directory.
    """
    override = os.environ.get("PONZU_CONFIG")
    if override:
        return Path(override).expanduser()
    return data_dir() / "config.yaml"


def logs_dir() -> Path:
    return data_dir() / "logs"


def cache_dir() -> Path:
    return data_dir() / "cache"


def models_dir() -> Path:
    return data_dir() / "models"


def audio_dir() -> Path:
    return data_dir() / "audio"


def ensure_data_dirs() -> None:
    """Create the data directory tree, restricted to the owner.

    Safe to call repeatedly (e.g. on every CLI invocation) -- existing
    directories are left alone aside from having their mode re-asserted.
    """
    for path in (data_dir(), logs_dir(), cache_dir(), models_dir(), audio_dir()):
        path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
        os.chmod(path, _PRIVATE_DIR_MODE)


def bundled_example_config() -> Path:
    """Path to the repository's committed `config/default.example.yaml`.

    Used to seed a first-run config (see `core.config.write_default_config`)
    and by tests that need a known-good example file.
    """
    # src/ponzu/core/paths.py -> parents[3] is the repository root.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "config" / "default.example.yaml"
