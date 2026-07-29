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
            json={"model": "qwen3:30b", "message": {"role": "assistant", "content": "hello"}},
        )

    model = _model(handler)
    result = model.generate(
        [Message(role="system", content="be concise"), Message(role="user", content="hi")]
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


def test_generate_logs_message_count_not_content(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": "qwen3:30b", "message": {"content": "reply text"}}
        )

    model = _model(handler)

    with caplog.at_level(logging.INFO, logger="ponzu.llm.ollama"):
        model.generate([Message(role="user", content="a very secret user prompt")])

    record = next(r for r in caplog.records if getattr(r, "ponzu_event", None) == "llm_request")
    assert record.ponzu_fields["message_count"] == 1
    assert record.ponzu_fields["model"] == "qwen3:30b"
    assert "duration_ms" in record.ponzu_fields
    assert "a very secret user prompt" not in caplog.text
    assert "reply text" not in caplog.text


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
