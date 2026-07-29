"""Tests for the wake-word substitutes (ADR-010)."""

from __future__ import annotations

import io
import sys
import threading

from ponzu.adapters import Probeable, WakeWordDetector
from ponzu.wakeword.keyboard import KeyboardWakeWord
from ponzu.wakeword.manual import ManualWakeWord

_WAIT_S = 1.0


# ---------------------------------------------------------------------------
# ManualWakeWord
# ---------------------------------------------------------------------------


def test_manual_trigger_fires_callback_with_float_confidence() -> None:
    received: list[float | None] = []
    detector = ManualWakeWord()
    detector.on_detected(received.append)

    detector.start()
    detector.trigger(0.87)

    assert received == [0.87]


def test_manual_trigger_fires_callback_with_none_confidence() -> None:
    received: list[float | None] = []
    detector = ManualWakeWord()
    detector.on_detected(received.append)

    detector.start()
    detector.trigger(None)

    assert received == [None]


def test_manual_trigger_does_not_fire_before_start() -> None:
    received: list[float | None] = []
    detector = ManualWakeWord()
    detector.on_detected(received.append)

    detector.trigger(0.5)

    assert received == []


def test_manual_trigger_does_not_fire_after_stop() -> None:
    received: list[float | None] = []
    detector = ManualWakeWord()
    detector.on_detected(received.append)

    detector.start()
    detector.stop()
    detector.trigger(0.5)

    assert received == []


def test_manual_satisfies_protocols() -> None:
    detector = ManualWakeWord()
    assert isinstance(detector, WakeWordDetector)
    assert isinstance(detector, Probeable)


def test_manual_probe_is_ok_and_never_raises() -> None:
    result = ManualWakeWord().probe()
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# KeyboardWakeWord
# ---------------------------------------------------------------------------


def test_keyboard_blank_line_triggers_callback_with_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    event = threading.Event()
    received: list[float | None] = []

    def on_detected(confidence: float | None) -> None:
        received.append(confidence)
        event.set()

    detector = KeyboardWakeWord(phrase="ぽんず")
    detector.on_detected(on_detected)
    detector.start()

    event.wait(timeout=_WAIT_S)

    assert event.is_set()
    assert received == [None]

    detector.stop()


def test_keyboard_configured_phrase_triggers_callback(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("ぽんず\n"))
    event = threading.Event()
    received: list[float | None] = []

    def on_detected(confidence: float | None) -> None:
        received.append(confidence)
        event.set()

    detector = KeyboardWakeWord(phrase="ぽんず")
    detector.on_detected(on_detected)
    detector.start()

    event.wait(timeout=_WAIT_S)

    assert event.is_set()
    assert received == [None]

    detector.stop()


def test_keyboard_stop_terminates_thread_within_timeout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    detector = KeyboardWakeWord(phrase="ぽんず")
    detector.on_detected(lambda confidence: None)
    detector.start()

    detector.stop()

    assert detector._thread is not None
    assert not detector._thread.is_alive()


def test_keyboard_double_stop_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    detector = KeyboardWakeWord(phrase="ぽんず")
    detector.start()

    detector.stop()
    detector.stop()  # must not raise or hang


def test_keyboard_stop_without_start_is_a_no_op() -> None:
    detector = KeyboardWakeWord(phrase="ぽんず")
    detector.stop()  # must not raise or hang


def test_keyboard_satisfies_protocols() -> None:
    detector = KeyboardWakeWord()
    assert isinstance(detector, WakeWordDetector)
    assert isinstance(detector, Probeable)


def test_keyboard_probe_is_ok_when_stdin_is_a_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    result = KeyboardWakeWord().probe()
    assert result.status == "ok"


def test_keyboard_probe_is_warn_when_stdin_is_not_a_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    result = KeyboardWakeWord().probe()
    assert result.status == "warn"
