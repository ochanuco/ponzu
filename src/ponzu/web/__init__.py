"""Read-only web conversation view (ADR-018).

Re-exports `ConversationView` so callers write `from ponzu.web import
ConversationView` instead of reaching into the submodule.
"""

from __future__ import annotations

from ponzu.web.server import ConversationView

__all__ = ["ConversationView"]
