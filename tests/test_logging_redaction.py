"""Tests for the redaction filter (DESIGN section 7, SECURITY.md)."""

from __future__ import annotations

import json
import logging

from ponzu.core.logging import (
    DENIED_FIELDS,
    RedactionFilter,
    chars,
    log_event,
    redact,
    setup_logging,
)


def test_chars_counts_characters_not_bytes() -> None:
    assert chars("ぽんず") == 3
    assert chars("") == 0


def test_redact_replaces_denied_fields_only() -> None:
    fields = {
        "transcript": "こんにちは、元気ですか",
        "authorization": "Bearer secret-token",
        "stt_ms": 820,
        "input_chars": 18,
    }

    cleaned = redact(fields)

    assert cleaned["transcript"] == "[REDACTED]"
    assert cleaned["authorization"] == "[REDACTED]"
    # Allowed fields pass through untouched.
    assert cleaned["stt_ms"] == 820
    assert cleaned["input_chars"] == 18


def test_denied_fields_constant_covers_spec_categories() -> None:
    # DESIGN section 7 "Disabled by default": raw audio, full transcripts,
    # full prompts, full LLM responses, secrets, auth headers.
    for expected in ("transcript", "prompt", "response", "audio", "authorization"):
        assert expected in DENIED_FIELDS


def test_redaction_filter_mutates_record_in_place() -> None:
    record = logging.LogRecord(
        name="ponzu.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="turn_completed",
        args=(),
        exc_info=None,
    )
    record.ponzu_fields = {"prompt": "raw user prompt text", "llm_ms": 2410}

    result = RedactionFilter().filter(record)

    assert result is True  # never drops the record itself
    assert record.ponzu_fields["prompt"] == "[REDACTED]"
    assert record.ponzu_fields["llm_ms"] == 2410


def test_log_event_end_to_end_never_emits_denied_content(capsys) -> None:
    setup_logging(level="info", fmt="json")
    logger = logging.getLogger("ponzu.test.end_to_end")

    log_event(
        logger,
        "turn_completed",
        transcript="this must never reach the log",
        stt_ms=820,
        input_chars=18,
    )

    captured = capsys.readouterr()
    line = (captured.err or captured.out).strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["event"] == "turn_completed"
    assert payload["transcript"] == "[REDACTED]"
    assert payload["stt_ms"] == 820
    assert payload["input_chars"] == 18
    assert "this must never reach the log" not in line
