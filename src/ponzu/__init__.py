"""Ponzu: a local-first voice assistant.

See docs/DESIGN.md for the overall architecture and docs/ADR.md for the
decisions behind it. This package only exposes a version marker; adapters,
core services, and the CLI are imported directly from their submodules so
that importing `ponzu` never pulls in optional/hardware-bound dependencies
(ADR-009).
"""

from __future__ import annotations

__version__ = "0.1.0"
