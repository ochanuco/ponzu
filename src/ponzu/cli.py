"""Command-line entry point (ADR-011: `doctor`, `chat`, `start`).

`argparse` only -- `typer`/`click` are not project dependencies (ADR-009 keeps
the base install minimal). `main` returns an exit code rather than calling
`sys.exit` so tests can drive it directly with an argv list and inspect the
result.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from ponzu.adapters import AdapterUnavailable, Probeable, ProbeResult
from ponzu.core import factory, paths
from ponzu.core.config import Config, ConfigError, load_config, write_default_config
from ponzu.core.logging import setup_logging
from ponzu.core.orchestrator import Orchestrator, TurnResult
from ponzu.core.state import State

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ponzu", description="Ponzu: a local-first voice assistant (ADR-011)."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="override the config file path"
    )
    parser.add_argument(
        "--log-level", default=None, help="override logging.level (e.g. debug, info)"
    )
    parser.add_argument(
        "--log-format",
        choices=("json", "text"),
        default=None,
        help="override logging.format",
    )

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser(
        "doctor",
        help="probe configuration and every dependency without starting the loop",
    )
    doctor.add_argument(
        "--write-config",
        action="store_true",
        help="write a default config file if none exists yet",
    )

    chat = subparsers.add_parser(
        "chat", help="text-in/text-out conversation against the LLM adapter"
    )
    chat.add_argument(
        "utterance",
        nargs="?",
        default=None,
        help="run exactly one turn with this text instead of starting the REPL",
    )
    chat.add_argument(
        "--speak", action="store_true", help="also synthesize and play the reply"
    )

    subparsers.add_parser("start", help="run the full wake-word voice loop")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Global flags apply before the subcommand runs. Log level/format default
    # to whatever the resolved config says (DESIGN section 7); a config that
    # fails to load falls back to the built-in defaults here rather than
    # being reported twice -- each subcommand below does its own load and
    # reports a config problem in the way appropriate to it (doctor: a row;
    # chat/start: a stderr message).
    try:
        logging_defaults = load_config(args.config).logging
    except ConfigError:
        logging_defaults = None
    level = args.log_level or (logging_defaults.level if logging_defaults else "info")
    fmt = args.log_format or (logging_defaults.format if logging_defaults else "json")
    setup_logging(level, fmt)

    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "chat":
        return cmd_chat(args)
    if args.command == "start":
        return cmd_start(args)

    # No subcommand: nothing to run, so this is exactly the "print help"
    # case, not an error worth a traceback.
    parser.print_help()
    return 2


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _safe_probe(component: str, build: Callable[[], Probeable]) -> ProbeResult:
    """Build one adapter and probe it, converting any failure into a row.

    `Probeable.probe()` itself already promises never to raise, but
    *constructing* the adapter can (an unknown provider raises `ConfigError`)
    -- ADR-011 requires `doctor` to report that as a fail row, not crash.
    The adapter's own `.component` label (e.g. `stt.whisper`) is discarded in
    favor of `component` so the printed table always uses the fixed column
    order from ADR-011, whether or not construction succeeded.
    """
    try:
        adapter = build()
    except Exception as exc:  # noqa: BLE001 - doctor must never propagate
        return ProbeResult(component=component, status="fail", detail=str(exc))
    try:
        result = adapter.probe()
    except Exception as exc:  # noqa: BLE001 - same guarantee for probe() itself
        return ProbeResult(component=component, status="fail", detail=str(exc))
    return ProbeResult(
        component=component,
        status=result.status,
        detail=result.detail,
        remedy=result.remedy,
    )


def _print_table(rows: list[ProbeResult]) -> None:
    header = ("COMPONENT", "STATUS", "DETAIL", "REMEDY")
    body = [
        (
            row.component,
            row.status,
            row.detail,
            row.remedy if row.status != "ok" else "",
        )
        for row in rows
    ]
    table = [header, *body]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
        )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Probe every component independently; never start the loop (ADR-011)."""
    effective_path = args.config if args.config is not None else paths.config_path()

    if args.write_config:
        # Target the same path `load_config` below is about to read (honours
        # `--config`), so this invocation initialises the file it then
        # reports on instead of writing to the default location silently.
        if effective_path.exists():
            print(f"config already exists at {effective_path}; not overwritten")
        else:
            write_default_config(effective_path)
            print(f"wrote default config to {effective_path}")

    try:
        cfg: Config = load_config(args.config)
    except ConfigError as exc:
        # A broken config is exactly what `doctor` is for surfacing -- report
        # it as a fail row and stop; there is no config to build adapters
        # from, so the adapter rows would be meaningless.
        _print_table([ProbeResult(component="config", status="fail", detail=str(exc))])
        return 1

    exists = effective_path.exists()
    detail = (
        f"loaded from {effective_path}"
        if exists
        else f"no user config at {effective_path}; using built-in defaults"
    )
    rows: list[ProbeResult] = [
        ProbeResult(component="config", status="ok", detail=detail)
    ]
    rows.extend(_probe_adapters(cfg))

    _print_table(rows)
    return 1 if any(row.is_blocking for row in rows) else 0


def _probe_adapters(cfg: Config) -> list[ProbeResult]:
    """Probe every adapter in the fixed ADR-011 column order.

    Shared with `start`'s preflight so the two commands can never disagree
    about what is broken.
    """
    return [
        _safe_probe("llm", lambda: factory.build_llm(cfg)),
        _safe_probe("tts", lambda: factory.build_tts(cfg)),
        _safe_probe("stt", lambda: factory.build_stt(cfg)),
        _safe_probe("audio-in", lambda: factory.build_audio_in(cfg)),
        _safe_probe("audio-out", lambda: factory.build_audio_out(cfg)),
        _safe_probe("wake-word", lambda: factory.build_wake_word(cfg)),
    ]


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def _run_turn(orchestrator: Orchestrator, utterance: str, speak: bool) -> bool:
    """Run one turn, print the reply, and report whether it succeeded.

    The REPL ignores the return value: DESIGN section 8 requires a recoverable
    failure to be reported without ending the session. The one-shot form uses
    it as the exit code, because a script that gets no answer must not be told
    the command succeeded.
    """
    result = orchestrator.text_turn(utterance, speak=speak)
    if result.ok:
        print(result.response)
        return True
    print(result.error, file=sys.stderr)
    return False


def _chat_repl(orchestrator: Orchestrator, speak: bool) -> int:
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return 0
        if line == "/exit":
            return 0
        _run_turn(orchestrator, line, speak)


def cmd_chat(args: argparse.Namespace) -> int:
    """stdin/stdout REPL over `Orchestrator.text_turn` (ADR-011)."""
    try:
        cfg: Config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        # `--speak` wires TTS and playback but still no microphone or STT, so
        # chat stays runnable on a machine with no input device (ADR-011).
        orchestrator = factory.build_orchestrator(cfg, voice=False, speak=args.speak)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.utterance is not None:
        return 0 if _run_turn(orchestrator, args.utterance, args.speak) else 1

    return _chat_repl(orchestrator, args.speak)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


_STATE_CUES: dict[State, str] = {
    State.LISTENING: "🎤 listening...",
    State.THINKING: "…thinking",
}


def _announce_state(_previous: State, new: State) -> None:
    """Print a one-line cue for the states the user is waiting through."""
    cue = _STATE_CUES.get(new)
    if cue is not None:
        print(cue, flush=True)


def _report_turn(result: TurnResult) -> None:
    """Print one turn's outcome for the person sitting at the terminal."""
    if result.ok:
        if result.response:
            print(result.response)
        return
    print(f"turn failed: {result.error}", file=sys.stderr)
    print("Run `ponzu doctor` to check dependencies.", file=sys.stderr)


def cmd_start(args: argparse.Namespace) -> int:
    """Run the full voice loop (ADR-011)."""
    try:
        cfg: Config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    provider = cfg.wake_word.provider
    print(f"ponzu start: wake-word provider = {provider!r}")
    if provider == "whisper":
        # ADR-013. Say what actually triggers a turn, and be honest that this
        # gate is less reliable than a purpose-built detector so a missed
        # utterance reads as a known limitation rather than a broken install.
        print(
            f'say "{cfg.wake_word.phrase}" to start a turn. Detection runs the '
            f"{cfg.wake_word.model!r} model behind an energy gate (ADR-013), so "
            "a quiet or distant utterance may be missed."
        )
    elif provider == "keyboard":
        # ADR-010: the keyboard substitute does not do acoustic detection.
        # Users must not be left guessing why saying "ぽんず" does nothing.
        print(
            "keyboard wake word: press Enter to trigger a turn; this provider "
            'does not listen for "ぽんず" (ADR-010).'
        )
        if not sys.stdin.isatty():
            # The substitute reads stdin, so with no terminal it can never fire.
            # Starting anyway would look like it is running while being unable
            # to ever respond -- the stuck state DESIGN section 8 forbids.
            print(
                "error: the keyboard wake word needs an interactive terminal, "
                "but stdin is not a TTY. Run `ponzu start` from a terminal, or "
                "use `ponzu chat` for a non-interactive session.",
                file=sys.stderr,
            )
            return 1

    # Preflight. Every turn in the loop uses the same adapters, so a component
    # that is already known to be unavailable produces a loop that can only
    # fail, identically, forever. Refusing up front is the same reasoning as
    # the non-TTY guard above.
    blocking = [row for row in _probe_adapters(cfg) if row.is_blocking]
    if blocking:
        print(
            "error: cannot start, required components are unavailable:", file=sys.stderr
        )
        _print_table(blocking)
        print("Run `ponzu doctor` for the full report.", file=sys.stderr)
        return 1

    try:
        orchestrator = factory.build_orchestrator(cfg, voice=True)
    except (ConfigError, AdapterUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        print("Run `ponzu doctor` to check dependencies.", file=sys.stderr)
        return 1

    # DESIGN section 8 step 2: a failure should produce a message when
    # possible. Without this the terminal shows only a JSON `turn_failed`
    # event naming the exception type, which does not tell the user what to fix.
    orchestrator.on_turn(_report_turn)
    # Without a cue the terminal shows only JSON while the assistant waits for
    # an utterance, so a user who woke it and then paused had no way to tell it
    # was listening -- it read as no response at all (DESIGN section 8 asks for
    # a message on failure; this is the same problem before the failure).
    orchestrator.on_state_change(_announce_state)

    try:
        orchestrator.run_forever()
    except KeyboardInterrupt:
        orchestrator.stop()
        return 0
    except AdapterUnavailable as exc:
        print(str(exc), file=sys.stderr)
        print("Run `ponzu doctor` to check dependencies.", file=sys.stderr)
        orchestrator.stop()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
