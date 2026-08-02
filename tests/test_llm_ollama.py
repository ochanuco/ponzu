"""Tests for the Ollama LLM adapter (DESIGN section 4.5, ADR-005).

Driven entirely through `httpx.MockTransport` -- no real Ollama server, no
extra test dependency beyond httpx itself.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from ponzu.adapters import AdapterTimeout, AdapterUnavailable, Message
from ponzu.core.config import LlmConfig
from ponzu.llm.ollama import OllamaLanguageModel


def _config(**overrides: object) -> LlmConfig:
    base: dict[str, object] = {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "model": "qwen3:30b",
        "timeout_s": 5.0,
    }
    base.update(overrides)
    return LlmConfig(**base)  # type: ignore[arg-type]


def _model(handler, **config_overrides: object) -> OllamaLanguageModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaLanguageModel(_config(**config_overrides), client=client)


def test_generate_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:30b"
        assert body["stream"] is False
        assert body["messages"] == [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hi"},
        ]
        return httpx.Response(
            200,
            json={
                "model": "qwen3:30b",
                "message": {"role": "assistant", "content": "hello"},
            },
        )

    model = _model(handler)
    result = model.generate(
        [
            Message(role="system", content="be concise"),
            Message(role="user", content="hi"),
        ]
    )

    assert result.text == "hello"
    assert result.model == "qwen3:30b"
    assert result.duration_ms is not None
    assert result.duration_ms >= 0


def test_generate_connection_refused_raises_adapter_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    model = _model(handler)

    with pytest.raises(AdapterUnavailable) as exc_info:
        model.generate([Message(role="user", content="hi")])

    assert "11434" in str(exc_info.value)
    assert "ollama serve" in str(exc_info.value)


def test_generate_connect_timeout_raises_adapter_unavailable() -> None:
    # ConnectTimeout is a TimeoutException subclass; it must still map to
    # AdapterUnavailable (server absent), not AdapterTimeout (server slow).
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    model = _model(handler)

    with pytest.raises(AdapterUnavailable):
        model.generate([Message(role="user", content="hi")])


def test_generate_read_timeout_raises_adapter_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    model = _model(handler)

    with pytest.raises(AdapterTimeout):
        model.generate([Message(role="user", content="hi")])


def test_generate_500_raises_adapter_unavailable_without_leaking_body() -> None:
    secret_prompt_echo = "this prompt text must never leak into the exception"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=secret_prompt_echo)

    model = _model(handler)

    with pytest.raises(AdapterUnavailable) as exc_info:
        model.generate([Message(role="user", content="hi")])

    assert "500" in str(exc_info.value)
    assert secret_prompt_echo not in str(exc_info.value)


def test_generate_logs_message_count_not_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": "qwen3:30b", "message": {"content": "reply text"}}
        )

    model = _model(handler)

    with caplog.at_level(logging.INFO, logger="ponzu.llm.ollama"):
        model.generate([Message(role="user", content="a very secret user prompt")])

    record = next(
        r for r in caplog.records if getattr(r, "ponzu_event", None) == "llm_request"
    )
    assert record.ponzu_fields["message_count"] == 1
    assert record.ponzu_fields["model"] == "qwen3:30b"
    assert "duration_ms" in record.ponzu_fields
    assert "a very secret user prompt" not in caplog.text
    assert "reply text" not in caplog.text


def test_generate_malformed_json_raises_adapter_unavailable() -> None:
    secret_prompt_echo = "this prompt text must never leak into the exception"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=secret_prompt_echo.encode())

    model = _model(handler)

    with pytest.raises(AdapterUnavailable) as exc_info:
        model.generate([Message(role="user", content="hi")])

    assert secret_prompt_echo not in str(exc_info.value)


def test_generate_null_message_raises_adapter_unavailable() -> None:
    # `{"message": null}` is well-formed JSON but `.get("message", {})` hands
    # back `None`, not `{}`; `.get("content")` on that raises AttributeError.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "qwen3:30b", "message": None})

    model = _model(handler)

    with pytest.raises(AdapterUnavailable):
        model.generate([Message(role="user", content="hi")])


def test_probe_ok_when_model_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"model": "qwen3:30b"}]})

    result = _model(handler).probe()

    assert result.status == "ok"
    assert result.component == "llm"


def test_probe_warn_when_model_not_pulled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"model": "llama3:8b"}]})

    result = _model(handler).probe()

    assert result.status == "warn"
    assert "ollama pull qwen3:30b" == result.remedy


def test_probe_fail_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = _model(handler).probe()

    assert result.status == "fail"
    assert result.remedy == "ollama serve"


def test_probe_never_raises_on_malformed_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    result = _model(handler).probe()  # must not raise

    assert result.status == "warn"


def test_probe_never_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    result = _model(handler).probe()  # must not raise

    assert result.status == "fail"


# --------------------------------------------------------------- generate_stream
# ADR-014: `generate_stream` is what lets the voice loop speak the first
# sentence while the model keeps generating. Driven the same way as
# `generate` above, but the mock response body is newline-delimited JSON.


def _ndjson(*objects: dict[str, object]) -> bytes:
    return b"\n".join(json.dumps(obj).encode() for obj in objects)


def test_generate_stream_happy_path_yields_content_only() -> None:
    body = _ndjson(
        {"model": "qwen3:30b", "message": {"content": "Hello"}, "done": False},
        {"model": "qwen3:30b", "message": {"content": ", world"}, "done": False},
        {"model": "qwen3:30b", "message": {"content": ""}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body)

    model = _model(handler)
    chunks = list(model.generate_stream([Message(role="user", content="hi")]))

    assert chunks == ["Hello", ", world"]


def test_generate_stream_thinking_only_chunk_yields_nothing() -> None:
    # ADR-009/ADR-012: reasoning arrives in a separate `thinking` field. A
    # chunk carrying only that must not fall back to it -- the assistant would
    # end up speaking its reasoning aloud.
    body = _ndjson(
        {"message": {"thinking": "let me consider this"}, "done": False},
        {"message": {"content": "answer"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)
    chunks = list(model.generate_stream([Message(role="user", content="hi")]))

    assert chunks == ["answer"]


def test_generate_stream_malformed_json_line_raises_adapter_unavailable() -> None:
    secret_prompt_echo = "this prompt text must never leak into the exception"
    body = (
        json.dumps({"message": {"content": "partial"}, "done": False}).encode()
        + b"\n"
        + secret_prompt_echo.encode()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)

    with pytest.raises(AdapterUnavailable) as exc_info:
        list(model.generate_stream([Message(role="user", content="hi")]))

    assert secret_prompt_echo not in str(exc_info.value)


def test_generate_stream_connect_error_raises_adapter_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    model = _model(handler)

    with pytest.raises(AdapterUnavailable):
        list(model.generate_stream([Message(role="user", content="hi")]))


def test_generate_stream_read_timeout_raises_adapter_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    model = _model(handler)

    with pytest.raises(AdapterTimeout):
        list(model.generate_stream([Message(role="user", content="hi")]))


# --------------------------------------------------------- on_thinking (ADR-015)
# The adapter never yields `thinking` from the iterator (asserted above); these
# cover the separate `on_thinking` callback that receives it instead.


def test_generate_stream_thinking_chunk_calls_on_thinking_and_yields_nothing() -> None:
    body = _ndjson(
        {"message": {"thinking": "let me consider this"}, "done": False},
        {"message": {"content": "answer"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)
    seen: list[str] = []
    chunks = list(
        model.generate_stream(
            [Message(role="user", content="hi")], on_thinking=seen.append
        )
    )

    assert chunks == ["answer"]
    assert seen == ["let me consider this"]


def test_generate_stream_chunk_with_both_fields_yields_only_content() -> None:
    body = _ndjson(
        {
            "message": {"thinking": "hmm", "content": "par"},
            "done": False,
        },
        {"message": {"content": "tial"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)
    seen: list[str] = []
    chunks = list(
        model.generate_stream(
            [Message(role="user", content="hi")], on_thinking=seen.append
        )
    )

    assert chunks == ["par", "tial"]
    assert seen == ["hmm"]


def test_generate_stream_with_no_callback_drops_thinking_silently() -> None:
    body = _ndjson(
        {"message": {"thinking": "reasoning nobody asked for"}, "done": False},
        {"message": {"content": "answer"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)

    # No on_thinking passed at all -- must not raise, must not yield thinking.
    chunks = list(model.generate_stream([Message(role="user", content="hi")]))

    assert chunks == ["answer"]


def test_generate_stream_thinking_never_reaches_a_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DESIGN section 7 excludes model output from logs, and reasoning is model
    # output (ADR-015). The CLI may print it to a terminal, but the adapter
    # itself must never let it reach `log_event`.
    secret_reasoning = "this reasoning text must never reach a log record"
    body = _ndjson(
        {"message": {"thinking": secret_reasoning}, "done": False},
        {"message": {"content": "answer"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)

    with caplog.at_level(logging.INFO, logger="ponzu.llm.ollama"):
        list(
            model.generate_stream(
                [Message(role="user", content="hi")], on_thinking=lambda _f: None
            )
        )

    assert secret_reasoning not in caplog.text
    for record in caplog.records:
        fields = getattr(record, "ponzu_fields", None) or {}
        assert secret_reasoning not in str(fields)


def test_generate_stream_logs_counts_not_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _ndjson(
        {"message": {"content": "ABCDE"}, "done": False},
        {"message": {"content": "FGHIJ"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    model = _model(handler)

    with caplog.at_level(logging.INFO, logger="ponzu.llm.ollama"):
        list(model.generate_stream([Message(role="user", content="a secret prompt")]))

    record = next(
        r for r in caplog.records if getattr(r, "ponzu_event", None) == "llm_stream"
    )
    assert record.ponzu_fields["message_count"] == 2
    assert record.ponzu_fields["output_chars"] == 10
    assert record.ponzu_fields["model"] == "qwen3:30b"
    assert "duration_ms" in record.ponzu_fields
    assert "a secret prompt" not in caplog.text
    assert "ABCDE" not in caplog.text
    assert "FGHIJ" not in caplog.text
