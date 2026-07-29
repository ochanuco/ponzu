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
        self.turn_callback = None
        self.state_callback = None
        self.thinking_callback = None

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

    def on_turn(self, callback) -> None:
        self.turn_callback = callback

    def on_state_change(self, callback) -> None:
        self.state_callback = callback

    def on_thinking(self, callback) -> None:
        self.thinking_callback = callback

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

    # Deliberately environment-independent. Whether Ollama is running or the
    # extras are installed varies per machine, and an earlier version asserted
    # `exit_code == 1`, which broke the moment the dependencies were actually
    # set up. What must hold everywhere: doctor probes every component, prints
    # a row for each, and never raises.
    assert exit_code in (0, 1)
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


def test_doctor_exit_code_is_driven_by_fail_rows(monkeypatch, tmp_path: Path) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)

    # The exit-code rule itself is tested with fixed rows so it holds no matter
    # what is installed: warn does not fail the command, fail does.
    monkeypatch.setattr(
        cli,
        "_probe_adapters",
        lambda cfg: [cli.ProbeResult(component="llm", status="warn", detail="x")],
    )
    assert cli.main(["doctor"]) == 0

    monkeypatch.setattr(
        cli,
        "_probe_adapters",
        lambda cfg: [cli.ProbeResult(component="llm", status="fail", detail="x")],
    )
    assert cli.main(["doctor"]) == 1


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


def test_doctor_write_config_honours_explicit_config_path(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    # `--write-config` must target the same file `--config` tells `doctor` to
    # read, not the default data-dir location -- otherwise one invocation
    # writes one file and reports on another.
    _isolate_data_dir(monkeypatch, tmp_path)
    custom_path = tmp_path / "custom" / "config.yaml"
    default_path = paths.config_path()

    exit_code = cli.main(["--config", str(custom_path), "doctor", "--write-config"])
    output = capsys.readouterr().out

    assert custom_path.exists()
    assert not default_path.exists()
    assert f"wrote default config to {custom_path}" in output
    assert exit_code in (0, 1)


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


def _use_keyboard_provider(monkeypatch, tmp_path: Path) -> None:
    """Write a config selecting the keyboard substitute.

    ADR-013 made the whisper gate the default, so tests about the keyboard
    provider have to ask for it explicitly rather than relying on the default.
    """
    (tmp_path / "kb.yaml").write_text('wake_word:\n  provider: "keyboard"\n')


def _pretend_deps_ok(monkeypatch) -> None:
    """`ponzu start` preflights every adapter and refuses to enter a loop that
    can only fail. The optional extras are absent here, so tests exercising the
    normal path must stand in for a healthy environment."""
    monkeypatch.setattr(cli, "_probe_adapters", lambda cfg: [])


def test_start_banner_tells_the_user_what_triggers_a_turn(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    _pretend_deps_ok(monkeypatch)
    fake = FakeOrchestrator()
    fake.run_forever_error = KeyboardInterrupt()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    exit_code = cli.main(["start"])
    captured = capsys.readouterr()

    assert exit_code == 0
    # ADR-013 is now the default, so the banner must name the phrase. It also
    # states the gate's limitation, so a missed utterance reads as a known
    # trade-off rather than a broken install.
    assert "whisper" in captured.out.lower()
    assert "ぽんず" in captured.out


def test_start_banner_states_the_keyboard_substitute_limitation(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    _pretend_deps_ok(monkeypatch)
    _use_keyboard_provider(monkeypatch, tmp_path)
    fake = FakeOrchestrator()
    fake.run_forever_error = KeyboardInterrupt()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    cli.main(["--config", str(tmp_path / "kb.yaml"), "start"])
    captured = capsys.readouterr()

    # ADR-010: this provider does not listen, and users must not be left
    # guessing why saying "ぽんず" does nothing.
    assert "keyboard" in captured.out.lower()
    assert "enter" in captured.out.lower()


def test_start_returns_0_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    _pretend_deps_ok(monkeypatch)
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
    _use_keyboard_provider(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
    built: list[object] = []
    monkeypatch.setattr(
        cli.factory,
        "build_orchestrator",
        lambda cfg, *, voice, speak=False: built.append(1),
    )

    exit_code = cli.main(["--config", str(tmp_path / "kb.yaml"), "start"])
    captured = capsys.readouterr()

    # Regression: the keyboard substitute reads stdin, so with no TTY it can
    # never fire. Starting anyway left the loop spinning forever with no way to
    # respond and no way to exit -- the stuck state DESIGN section 8 forbids.
    assert exit_code == 1
    assert "interactive terminal" in captured.err
    assert built == []  # bailed out before constructing anything


def test_start_refuses_when_a_required_component_is_unavailable(
    monkeypatch, tmp_path: Path, capsys
):
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    monkeypatch.setattr(
        cli,
        "_probe_adapters",
        lambda cfg: [
            cli.ProbeResult(component="llm", status="ok"),
            cli.ProbeResult(
                component="audio-in",
                status="fail",
                detail="sounddevice is not installed",
                remedy="uv sync --extra audio",
            ),
        ],
    )
    built: list[object] = []
    monkeypatch.setattr(
        cli.factory,
        "build_orchestrator",
        lambda cfg, *, voice, speak=False: built.append(1),
    )

    exit_code = cli.main(["start"])
    captured = capsys.readouterr()

    # Regression: the loop used to start anyway, so every Enter produced an
    # identical AdapterUnavailable turn forever, with nothing on screen saying
    # what to install.
    assert exit_code == 1
    assert "audio-in" in captured.out
    assert "uv sync --extra audio" in captured.out
    assert built == []


def test_start_reports_a_failed_turn_to_the_user(monkeypatch, tmp_path: Path, capsys):
    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    _pretend_deps_ok(monkeypatch)
    fake = FakeOrchestrator()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    cli.main(["start"])
    # DESIGN section 8 step 2: the person at the terminal must get a message,
    # not just a JSON `turn_failed` event naming an exception type.
    assert fake.turn_callback is not None
    fake.turn_callback(
        TurnResult(
            utterance="",
            response="",
            metrics=TurnMetrics(),
            error="AdapterUnavailable: sounddevice is not installed",
        )
    )
    captured = capsys.readouterr()

    assert "sounddevice is not installed" in captured.err
    assert "ponzu doctor" in captured.err


def test_start_announces_that_it_is_listening(monkeypatch, tmp_path: Path, capsys):
    """A woken assistant must say so.

    Regression: waking printed nothing but JSON, so a user who said the wake
    word and then paused saw no sign it had heard them -- indistinguishable
    from no response at all.
    """
    from ponzu.core.state import State

    _isolate_data_dir(monkeypatch, tmp_path)
    _pretend_tty(monkeypatch)
    _pretend_deps_ok(monkeypatch)
    fake = FakeOrchestrator()
    fake.run_forever_error = KeyboardInterrupt()
    monkeypatch.setattr(
        cli.factory, "build_orchestrator", lambda cfg, *, voice, speak=False: fake
    )

    cli.main(["start"])
    assert fake.state_callback is not None
    fake.state_callback(State.IDLE, State.LISTENING)
    captured = capsys.readouterr()

    assert "listening" in captured.out.lower()
