"""Tests for ponzu.cli (ADR-011: doctor / chat / start)."""

from __future__ import annotations

from pathlib import Path

from ponzu import cli
from ponzu.adapters import TurnMetrics
from ponzu.core import paths
from ponzu.core.orchestrator import TurnResult


class FakeOrchestrator:
    """Stand-in for `Orchestrator`, injected via a monkeypatched factory.

    Real adapters are never constructed in these tests -- `cli.py` is driven
    entirely through `factory.build_orchestrator`, so patching that one seam
    is enough to keep every CLI test free of real stdin, hardware, or network
    access.
    """

    def __init__(self, response: str = "はい。", error: str | None = None) -> None:
        self.calls: list[tuple[str, bool]] = []
        self._response = response
        self._error = error
        self.stopped = False
        self.run_forever_calls = 0
        self.run_forever_error: Exception | None = None

    def text_turn(self, utterance: str, *, speak: bool = False) -> TurnResult:
        self.calls.append((utterance, speak))
        if self._error is not None:
            return TurnResult(
                utterance=utterance,
                response="",
                metrics=TurnMetrics(),
                error=self._error,
            )
        return TurnResult(
            utterance=utterance, response=self._response, metrics=TurnMetrics()
        )

    def run_forever(self) -> None:
        self.run_forever_calls += 1
        if self.run_forever_error is not None:
            raise self.run_forever_error

    def stop(self) -> None:
        self.stopped = True


def _isolate_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Point the default config location at an empty tmp dir.

    Without this, tests that don't pass `--config` would resolve
    `paths.config_path()` against whatever is actually on the machine running
    the suite (ADR-006's real data directory), which is exactly the kind of
    environment leakage `PONZU_DATA_DIR` exists to prevent (see test_paths.py).
    """
    monkeypatch.setenv("PONZU_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PONZU_CONFIG", raising=False)


# --------------------------------------------------------------------- global


def test_no_subcommand_prints_help_and_returns_2(capsys) -> None:
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "usage" in captured.out.lower()


# ---------------------------------------------------------------------- doctor


def test_doctor_with_broken_config_reports_fail_row_without_raising(
    tmp_path: Path, capsys
) -> None:
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text("wake_word:\n  sensitivty: 0.9\n")  # typo'd key

    exit_code = cli.main(["--config", str(bad_config), "doctor"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "config" in captured.out
    assert "sensitivty" in captured.out


def test_doctor_with_default_config_reports_fail_rows_and_never_raises(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)

    exit_code = cli.main(["doctor"])
    captured = capsys.readouterr()

    # No user config file, no live Ollama/VOICEVOX, and the audio/stt extras
    # absent (ADR-009) -- at least one row must fail, and a remedy must be
    # printed for it, all without raising.
    assert exit_code == 1
    assert "REMEDY" in captured.out
    for component in (
        "config",
        "llm",
        "tts",
        "stt",
        "audio-in",
        "audio-out",
        "wake-word",
    ):
        assert component in captured.out


def test_doctor_write_config_writes_then_does_not_overwrite(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    config_path = paths.config_path()
    assert not config_path.exists()

    cli.main(["doctor", "--write-config"])
    first_output = capsys.readouterr().out
    assert config_path.exists()
    assert "wrote default config" in first_output
    written_at = config_path.stat().st_mtime

    cli.main(["doctor", "--write-config"])
    second_output = capsys.readouterr().out
    assert "already exists" in second_output
    assert config_path.stat().st_mtime == written_at


# ------------------------------------------------------------------------ chat


def test_chat_with_utterance_runs_exactly_one_turn(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    fake = FakeOrchestrator(response="こんにちは!")
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    exit_code = cli.main(["chat", "テスト"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert fake.calls == [("テスト", False)]
    assert captured.out.strip() == "こんにちは!"


def test_chat_one_shot_failure_is_reported_and_nonzero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    fake = FakeOrchestrator(error="AdapterUnavailable: ollama down")
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    exit_code = cli.main(["chat", "テスト"])
    captured = capsys.readouterr()

    # The one-shot form has no session to protect, and a script that got no
    # answer must not be told the command succeeded.
    assert exit_code == 1
    assert captured.out == ""
    assert "ollama down" in captured.err


def test_chat_repl_survives_a_failed_turn(monkeypatch, tmp_path: Path, capsys) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    fake = FakeOrchestrator(error="AdapterUnavailable: ollama down")
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )
    lines = iter(["\u4e00\u56de\u76ee", "\u4e8c\u56de\u76ee", "/exit"])
    monkeypatch.setattr("builtins.input", lambda: next(lines))

    exit_code = cli.main(["chat"])
    captured = capsys.readouterr()

    # DESIGN section 8: a recoverable failure is reported, not fatal -- the
    # REPL keeps taking input and still exits cleanly.
    assert exit_code == 0
    assert len(fake.calls) == 2
    assert captured.err.count("ollama down") == 2


def test_chat_repl_exits_on_slash_exit(monkeypatch, tmp_path: Path) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    fake = FakeOrchestrator(response="ok")
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )
    inputs = iter(["こんにちは", "/exit"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    exit_code = cli.main(["chat"])

    assert exit_code == 0
    assert fake.calls == [("こんにちは", False)]


def test_chat_repl_exits_on_eof(monkeypatch, tmp_path: Path) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    fake = FakeOrchestrator()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    def _raise_eof() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", lambda: _raise_eof())

    exit_code = cli.main(["chat"])

    assert exit_code == 0
    assert fake.calls == []


# ----------------------------------------------------------------------- start


def _pretend_tty(monkeypatch) -> None:
    """`ponzu start` refuses a non-interactive stdin (ADR-010), and pytest's
    stdin is not a TTY. Tests exercising the normal path must say so."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)


def test_start_banner_names_keyboard_provider_and_its_limitation(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    fake = FakeOrchestrator()
    fake.run_forever_error = KeyboardInterrupt()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    exit_code = cli.main(["start"])
    captured = capsys.readouterr()

    assert exit_code == 0
    # ADR-010: users must not be left guessing why saying "ぽんず" does
    # nothing -- the keyboard substitute's limitation must be stated plainly.
    assert "keyboard" in captured.out.lower()
    assert "enter" in captured.out.lower()


def test_start_returns_0_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    fake = FakeOrchestrator()
    fake.run_forever_error = KeyboardInterrupt()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    exit_code = cli.main(["start"])

    assert exit_code == 0
    assert fake.run_forever_calls == 1
    assert fake.stopped is True


def test_start_refuses_non_interactive_stdin(monkeypatch, tmp_path: Path, capsys):
    _isolate_data_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
    built: list[object] = []
    monkeypatch.setattr(
        cli.factory,
        "build_orchestrator",
        lambda cfg, *, voice, speak=False: built.append(1),
    )

    exit_code = cli.main(["start"])
    captured = capsys.readouterr()

    # Regression: the keyboard substitute reads stdin, so with no TTY it can
    # never fire. Starting anyway left the loop spinning forever with no way to
    # respond and no way to exit -- the stuck state DESIGN section 8 forbids.
    assert exit_code == 1
    assert "interactive terminal" in captured.err
    assert built == []  # bailed out before constructing anything
