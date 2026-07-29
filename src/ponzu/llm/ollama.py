"""Ollama-compatible local LLM adapter (DESIGN section 4.5, ADR-005).

Transport only: this module knows how to reach an Ollama-compatible HTTP API
and turn its reply into a `ModelResponse`. Persona, prompt construction, and
conversation memory are explicitly out of scope here (DESIGN section 4.5
"Non-Responsibilities") and stay in the orchestration layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import httpx

from ponzu.adapters import (
    AdapterTimeout,
    AdapterUnavailable,
    Message,
    ModelResponse,
    ProbeResult,
)
from ponzu.core.config import LlmConfig
from ponzu.core.logging import log_event

_logger = logging.getLogger("ponzu.llm.ollama")


class OllamaLanguageModel:
    """`LanguageModel` + `Probeable` backed by an Ollama-compatible server."""

    def __init__(self, config: LlmConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        # Accepting an injected client is what lets tests drive this adapter
        # with `httpx.MockTransport` instead of a real Ollama server; the
        # default still respects the configured timeout (DESIGN section 4.5).
        self._client = client or httpx.Client(timeout=config.timeout_s)

    def generate(
        self, messages: Iterable[Message], *, timeout_s: float | None = None
    ) -> ModelResponse:
        payload_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        payload = {
            "model": self._config.model,
            "messages": payload_messages,
            "stream": False,  # streaming is explicitly deferred, DESIGN section 4.5
        }

        start = time.monotonic()
        try:
            response = self._client.post(
                f"{self._config.endpoint}/api/chat",
                json=payload,
                timeout=timeout_s if timeout_s is not None else self._config.timeout_s,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Checked before the broader TimeoutException below: ConnectTimeout
            # is itself a TimeoutException subclass, and a refused/absent
            # server is a "go start it" problem, not a "it's slow" problem.
            raise AdapterUnavailable(
                f"Ollama is not reachable at {self._config.endpoint}. "
                "Start it with `ollama serve`."
            ) from exc
        except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise AdapterTimeout(
                f"Ollama at {self._config.endpoint} did not respond in time."
            ) from exc
        duration_ms = int((time.monotonic() - start) * 1000)

        if not response.is_success:
            # Status code only -- the body can echo the prompt back (DESIGN
            # section 7 forbids that in logs, and it must not leak into an
            # exception message either).
            raise AdapterUnavailable(
                f"Ollama returned HTTP {response.status_code} for "
                f"{self._config.endpoint}/api/chat"
            )

        try:
            data = response.json()
            # `.get("message", {})`'s default only applies when the key is
            # absent; `{"message": null}` is well-formed JSON that still
            # yields `None` here, and `.get("content", ...)` on `None` raises
            # AttributeError -- caught below along with a non-JSON body,
            # rather than treated as a valid empty reply.
            text = data.get("message", {}).get("content", "")
            model = data.get("model", self._config.model)
        except (ValueError, AttributeError, TypeError) as exc:
            # Never the response body in the exception (DESIGN section 7) --
            # same property the status-code path above already keeps.
            raise AdapterUnavailable(
                f"Ollama at {self._config.endpoint} returned an unexpected payload"
            ) from exc

        log_event(
            _logger,
            "llm_request",
            model=model,
            duration_ms=duration_ms,
            message_count=len(payload_messages),
        )

        return ModelResponse(text=text, model=model, duration_ms=duration_ms)

    def probe(self) -> ProbeResult:
        """`ponzu doctor` check (ADR-011). Never raises."""
        try:
            response = self._client.get(
                f"{self._config.endpoint}/api/tags", timeout=self._config.timeout_s
            )
        except httpx.HTTPError:
            return ProbeResult(
                component="llm",
                status="fail",
                detail=f"Ollama unreachable at {self._config.endpoint}",
                remedy="ollama serve",
            )

        if not response.is_success:
            return ProbeResult(
                component="llm",
                status="fail",
                detail=f"Ollama returned HTTP {response.status_code}",
                remedy="ollama serve",
            )

        try:
            data = response.json()
            available = {
                entry.get("model") or entry.get("name")
                for entry in data.get("models", [])
            }
        except (ValueError, AttributeError, TypeError):
            # Reachable but the payload wasn't the shape we expect -- treat
            # as "can't confirm the model is there" rather than raising.
            return ProbeResult(
                component="llm",
                status="warn",
                detail=f"Ollama at {self._config.endpoint} returned an unexpected payload",
                remedy=f"ollama pull {self._config.model}",
            )

        if self._config.model in available:
            return ProbeResult(
                component="llm",
                status="ok",
                detail=f"model {self._config.model} available",
            )
        return ProbeResult(
            component="llm",
            status="warn",
            detail=f"model {self._config.model} is not pulled",
            remedy=f"ollama pull {self._config.model}",
        )
