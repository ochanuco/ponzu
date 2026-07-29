"""Audio capture/playback adapters (DESIGN section 4.3/4.8, ADR-009).

Only two backends exist so far (`ponzu.audio.capture`, `ponzu.audio.playback`);
this module stays a thin package marker rather than a registry so adding a
second backend doesn't force an abstraction decision before it's needed.
Importing this package must never require `sounddevice`/`numpy` (ADR-009):
neither submodule imports them at module scope.
"""

from __future__ import annotations
