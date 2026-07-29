"""Wake-word substitute providers (ADR-010).

Re-exports the two concrete ``WakeWordDetector`` substitutes and a plain
``PROVIDERS`` mapping so ``ponzu.core.config.WakeWordConfig.provider`` (a
string) can be resolved to a class with ``PROVIDERS[provider]`` instead of an
if/elif chain. A real acoustic engine is meant to be a drop-in addition to
this mapping -- no orchestrator change required to adopt it (ADR-010).
"""

from __future__ import annotations

from ponzu.wakeword.keyboard import KeyboardWakeWord
from ponzu.wakeword.manual import ManualWakeWord

PROVIDERS: dict[str, type] = {"keyboard": KeyboardWakeWord, "manual": ManualWakeWord}

__all__ = ["PROVIDERS", "KeyboardWakeWord", "ManualWakeWord"]
