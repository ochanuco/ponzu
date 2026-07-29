"""Keyboard wake-word substitute (ADR-010) -- default provider for `ponzu start`.

DESIGN section 4.2 asks the wake-word engine to detect "ぽんず" acoustically,
operate continuously at low CPU cost, expose confidence where available,
support configurable sensitivity, and never persist microphone audio. This
substitute intentionally does NOT satisfy the acoustic detection requirement
-- ADR-010 records that gap deliberately. It also never touches a microphone:
detection here is driven entirely by stdin text, so there is no audio capture
of any kind to persist.

ADR-010 picks this as the MVP default specifically so `ponzu start` is
runnable without a real acoustic engine installed.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

from ponzu.adapters import ProbeResult

# ``WakeWordConfig.sensitivity`` exists for a future acoustic provider. This
# substitute has no acoustic model to tune, so the value is accepted (via
# config plumbing elsewhere) but never read here.

# How long `stop()` waits for the reader thread to notice the stop event and
# exit before giving up. Kept small and constant so `stop()` can never hang a
# caller (e.g. a test, or `ponzu`'s own shutdown path).
_JOIN_TIMEOUT_S = 1.0


class KeyboardWakeWord:
    """Stdin-driven ``WakeWordDetector`` substitute.

    A bare Enter keypress (an empty line) OR the configured phrase typed
    literally fires the detected callback with confidence ``None`` -- a
    keyboard match is not a confidence score, unlike a real acoustic engine.

    Satisfies ``ponzu.adapters.WakeWordDetector`` and ``ponzu.adapters.Probeable``
    structurally (both are ``@runtime_checkable`` Protocols).

    IMPORTANT for callers: the registered callback executes ON THIS
    DETECTOR'S OWN BACKGROUND THREAD (the stdin-reading daemon thread spawned
    by ``start()``), never on the caller's thread. Whatever orchestrates
    against this interface must not assume it is running on the main thread
    inside the callback.
    """

    def __init__(self, phrase: str = "") -> None:
        self._phrase = phrase
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[float | None], None] | None = None

    def start(self) -> None:
        """Spawn the daemon reader thread and return immediately.

        Safe to call again while already running: a no-op, since a second
        thread reading the same stdin would race the first.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # `sys.stdin.readline()` blocks on real, interactive stdin, so a
        # `threading.Event.wait(timeout)` alone cannot interrupt it once
        # there is no more input coming. We handle the two situations that
        # matter in practice instead:
        #
        # 1. Test-driven input (e.g. `io.StringIO`): once exhausted,
        #    `readline()` returns "" repeatedly *without blocking*, which we
        #    treat as "end of stream, stop reading" -- so the loop always
        #    terminates on its own well within `stop()`'s join timeout.
        # 2. Real interactive stdin with no more input: `readline()` can
        #    legitimately block past a `stop()` call. That is a known,
        #    accepted limitation of a keyboard substitute (there is no
        #    portable, dependency-free way to interrupt a blocking stdin
        #    read from another thread). The thread is created as a daemon
        #    specifically so this can never hang process exit.
        while not self._stop_event.is_set():
            line = sys.stdin.readline()
            if line == "":
                # EOF / exhausted stream -- nothing more will ever arrive.
                return
            stripped = line.strip()
            if (stripped == "" or stripped == self._phrase) and (
                self._callback is not None
            ):
                self._callback(None)

    def stop(self) -> None:
        """Signal the reader thread to stop and wait briefly for it to exit.

        Safe to call when never started (no thread exists) and safe to call
        more than once (joining an already-finished thread is a no-op); both
        cases are guarded so `stop()` never raises or hangs.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)

    def on_detected(self, callback: Callable[[float | None], None]) -> None:
        """Register the detection callback.

        Policy: registering again REPLACES the previously registered callback
        rather than raising or stacking multiple callbacks. This matches
        ``ManualWakeWord.on_detected`` -- both substitutes use the same
        single-slot-replace policy so callers can rely on one behavior
        regardless of provider.
        """
        self._callback = callback

    def probe(self) -> ProbeResult:
        """"ok" on an interactive terminal, "warn" (never "fail") otherwise.

        A non-interactive stdin (piped input, a non-interactive shell, a
        service manager with no controlling terminal) means the keyboard
        substitute cannot receive keypresses, but that is a configuration
        fact to surface via `ponzu doctor`, not a hard failure that should
        block startup.
        """
        try:
            interactive = sys.stdin.isatty()
        except OSError:
            # `isatty()` practically never raises, but `probe()` must never
            # propagate an exception regardless (see `Probeable`).
            interactive = False
        if interactive:
            return ProbeResult(component="wake_word.keyboard", status="ok")
        return ProbeResult(
            component="wake_word.keyboard",
            status="warn",
            detail="stdin is not interactive (piped input or non-interactive shell)",
            remedy="run `ponzu start` from an interactive terminal to use keyboard wake word",
        )
