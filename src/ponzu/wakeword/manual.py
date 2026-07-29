"""Manual wake-word substitute (ADR-010).

DESIGN section 4.2 asks the wake-word engine to detect "ぽんず" acoustically,
operate continuously at low CPU cost, expose confidence where available,
support configurable sensitivity, and never persist microphone audio. This
substitute intentionally does NOT satisfy the acoustic detection requirement
-- that gap is recorded by ADR-010, not an oversight here. It also never
touches a microphone at all: there is no audio capture, no thread, and no I/O
of any kind. ``trigger`` is a plain synchronous call meant for tests and for
``ponzu chat``, where a human (or a test) decides exactly when "wake word
detected" should fire.
"""

from __future__ import annotations

from collections.abc import Callable

from ponzu.adapters import ProbeResult

# ``WakeWordConfig.sensitivity`` exists for a future acoustic provider. This
# substitute has no acoustic model to tune, so the value is accepted (via
# config plumbing elsewhere) but never read here.


class ManualWakeWord:
    """Programmatic ``WakeWordDetector`` for tests and ``ponzu chat``.

    Satisfies ``ponzu.adapters.WakeWordDetector`` and ``ponzu.adapters.Probeable``
    structurally (both are ``@runtime_checkable`` Protocols) -- no explicit
    inheritance is required, only matching method signatures.

    There is no background thread here, unlike ``KeyboardWakeWord``: `trigger`
    runs synchronously on the caller's own thread. That is exactly why a plain
    ``bool`` is safe for the started/stopped flag in this class specifically --
    there is no concurrent writer to race against, so no ``threading.Event`` or
    lock is needed. Do not copy this shortcut into a class that has a thread.
    """

    def __init__(self) -> None:
        self._started = False
        self._callback: Callable[[float | None], None] | None = None

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    @property
    def is_running(self) -> bool:
        """ADR-010 liveness. There is no stream to end, so this tracks
        `start`/`stop` exactly — a programmatic detector never dies on its own.
        """
        return self._started

    def on_detected(self, callback: Callable[[float | None], None]) -> None:
        """Register the detection callback.

        Policy: registering again REPLACES the previously registered callback
        rather than raising or stacking multiple callbacks. This matches
        ``KeyboardWakeWord.on_detected`` -- both substitutes use the same
        single-slot-replace policy so callers can rely on one behavior
        regardless of provider.
        """
        self._callback = callback

    def trigger(self, confidence: float | None = None) -> None:
        """Synchronously invoke the registered callback, as if woken.

        No-op (not an error) in each of these cases:
        - no callback has been registered yet (nothing to invoke -- the brief
          leaves this case unspecified, so we choose "silently do nothing"
          over raising, matching ``Probeable.probe``'s "don't raise for a
          normal absence" spirit elsewhere in this codebase);
        - called before ``start()``;
        - called after ``stop()``.
        """
        if not self._started or self._callback is None:
            return
        self._callback(confidence)

    def probe(self) -> ProbeResult:
        """Always healthy: there is nothing external to fail.

        No microphone, no subprocess, no file -- so "ok" is the only
        meaningful status for this substitute (DESIGN section 4.2's "avoid
        persisting microphone audio" is trivially satisfied: there is no
        microphone access at all).
        """
        return ProbeResult(component="wake_word.manual", status="ok")
