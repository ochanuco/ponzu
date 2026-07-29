"""Core services: configuration, data paths, logging, and the state machine.

These modules have no adapter or orchestrator knowledge (DESIGN section 6) --
they are the substrate the orchestrator and CLI are built on.
"""

from __future__ import annotations
