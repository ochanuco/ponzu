"""Structured logging with mandatory redaction (DESIGN section 7, SECURITY.md).

Callers never attach free-form user content to a log record. `log_event`
takes an event name plus keyword fields (counts, durations, model names --
the allowed list in DESIGN section 7); `DENIED_FIELDS` is enforced by a
`logging.Filter` so that even a future caller who passes a forbidden field
name by mistake has it stripped before it reaches a handler, rather than
relying on every call site remembering not to.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

# Field names that must never carry real content into a log record: raw
# transcripts, prompts, model output, audio, and auth material (DESIGN
# section 7 "Disabled by default" list; SECURITY.md "Never Commit" /
# "Recommended .gitignore" for the same categories at rest). Matched
# case-insensitively so `Authorization` and `authorization` both hit.
DENIED_FIELDS: frozenset[str] = frozenset(
    {
        "text",
        "transcript",
        "transcripts",
        "prompt",
        "prompts",
        "messages",
        "message_content",
        "response",
        "response_text",
        "output",
        "llm_output",
        "audio",
        "pcm",
        "raw_audio",
        "authorization",
        "auth_header",
        "auth_headers",
        "headers",
        "api_key",
        "token",
        "password",
        "secret",
        "credentials",
    }
)

_REDACTED = "[REDACTED]"


def chars(text: str) -> int:
    """Character count, for logging in place of the text itself."""
    return len(text)


def redact(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `fields` with any denied key's value replaced."""
    return {
        key: (_REDACTED if key.lower() in DENIED_FIELDS else value)
        for key, value in fields.items()
    }


class RedactionFilter(logging.Filter):
    """Scrubs the structured fields attached by `log_event` on every record.

    Attached to the handler (not just relied on at the call site) so
    redaction happens regardless of how a record was produced.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        fields = getattr(record, "ponzu_fields", None)
        if fields:
            record.ponzu_fields = redact(fields)
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line (DESIGN section 7 example)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": getattr(record, "ponzu_event", record.getMessage()),
        }
        payload.update(getattr(record, "ponzu_fields", None) or {})
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Human-readable equivalent: base line plus `key=value` fields."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "ponzu_fields", None)
        if not fields:
            return base
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        return f"{base} {rendered}"


def setup_logging(level: str = "info", fmt: str = "json") -> None:
    """Configure the `ponzu` logger tree.

    `fmt` is `"json"` (default, machine-parseable) or `"text"` (for a
    terminal). Idempotent: safe to call again, e.g. after re-reading config.
    """
    if fmt not in ("json", "text"):
        raise ValueError(f"unknown log format: {fmt!r}")

    logger = logging.getLogger("ponzu")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.addFilter(RedactionFilter())
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured event: `{"event": event, **fields}` (redacted)."""
    logger.info(event, extra={"ponzu_event": event, "ponzu_fields": fields})
