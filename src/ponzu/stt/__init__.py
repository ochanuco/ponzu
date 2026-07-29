"""STT adapters (DESIGN section 4.4, ADR-009).

Only one backend exists so far (`ponzu.stt.whisper`); this module stays a
thin package marker rather than a registry so adding a second backend
doesn't force an abstraction decision before it's needed. Importing this
package must never require `faster_whisper`/`numpy` (ADR-009): the submodule
does not import them at module scope.
"""

from __future__ import annotations
