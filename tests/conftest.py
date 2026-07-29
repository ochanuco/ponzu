"""Shared test fixtures.

ADR-009 makes `sounddevice`, `numpy`, and `faster_whisper` optional extras, so
the adapters must degrade to `AdapterUnavailable` when they are missing. That
path has to be covered whether or not the extras happen to be installed on the
machine running the suite: CI deliberately runs without them, while a developer
working on the voice loop has them. Tests that assumed absence passed in CI and
failed the moment someone ran `uv sync --extra audio`.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def hide_module(monkeypatch):
    """Make `import <name>` raise ImportError for the duration of a test.

    A `None` entry in `sys.modules` is the documented way to force an import
    failure, so this simulates a missing extra deterministically instead of
    depending on what is installed.
    """

    def _hide(*names: str) -> None:
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
            monkeypatch.setitem(sys.modules, name, None)

    return _hide
