"""Wake-word providers (ADR-010, ADR-013).

Re-exports the concrete ``WakeWordDetector`` implementations and a plain
``PROVIDERS`` mapping so ``ponzu.core.config.WakeWordConfig.provider`` (a
string) can be resolved to a class with ``PROVIDERS[provider]`` instead of an
if/elif chain. ``whisper`` (ADR-013's acoustic gate) is the default; a future
dedicated acoustic engine (e.g. Porcupine, ADR-013's "Consequences") remains a
drop-in addition to this mapping -- no orchestrator change required to adopt
it (ADR-010).
"""

from __future__ import annotations

from ponzu.wakeword.keyboard import KeyboardWakeWord
from ponzu.wakeword.manual import ManualWakeWord
from ponzu.wakeword.whisper_gate import WhisperWakeWord

PROVIDERS: dict[str, type] = {
    "keyboard": KeyboardWakeWord,
    "manual": ManualWakeWord,
    "whisper": WhisperWakeWord,
}

__all__ = ["PROVIDERS", "KeyboardWakeWord", "ManualWakeWord", "WhisperWakeWord"]
