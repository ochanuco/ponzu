"""Ollama-compatible local LLM adapter (DESIGN section 4.5, ADR-005).

Transport only: this module knows how to reach an Ollama-compatible HTTP API
and turn its reply into a `ModelResponse`. Persona, prompt construction, and
conversation memory are explicitly out of scope here (DESIGN section 4.5
"Non-Responsibilities") and stay in the orchestration layer.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator

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

    def generate_stream(
        self, messages: Iterable[Message], *, timeout_s: float | None = None
    ) -> Iterator[str]:
        """Yield `message.content` fragments as Ollama streams them (ADR-014).

        Ollama's streaming `/api/chat` response is newline-delimited JSON, one
        object per line, each carrying an incremental `message`. `think: false`
        is not set (ADR-009 / ADR-012): this model separates its reasoning into
        a `thinking` field, and a chunk that carries only `thinking` must yield
        nothing rather than falling back to it, or the assistant would speak
        its reasoning aloud.
        """
        payload_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        payload = {
            "model": self._config.model,
            "messages": payload_messages,
            "stream": True,
        }
        effective_timeout = (
            timeout_s if timeout_s is not None else self._config.timeout_s
        )

        start = time.monotonic()
        message_count = 0
        output_chars = 0
        model_name = self._config.model

        try:
            with self._client.stream(
                "POST",
                f"{self._config.endpoint}/api/chat",
                json=payload,
                timeout=effective_timeout,
            ) as response:
                if not response.is_success:
                    # Status code only -- same rule as `generate` (DESIGN
                    # section 7).
                    raise AdapterUnavailable(
                        f"Ollama returned HTTP {response.status_code} for "
                        f"{self._config.endpoint}/api/chat"
                    )

                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError as exc:
                        # A malformed NDJSON line must not kill the stream with
                        # a raw ValueError, and the line itself (which may echo
                        # prompt content) must never reach the exception
                        # message (DESIGN section 7).
                        raise AdapterUnavailable(
                            f"Ollama at {self._config.endpoint} sent a "
                            "malformed stream chunk"
                        ) from exc

                    message_count += 1
                    model_name = data.get("model", model_name)
                    message = data.get("message") or {}
                    content = (
                        message.get("content") if isinstance(message, dict) else None
                    )
                    if content:
                        output_chars += len(content)
                        yield content

                    if data.get("done"):
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise AdapterUnavailable(
                f"Ollama is not reachable at {self._config.endpoint}. "
                "Start it with `ollama serve`."
            ) from exc
        except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise AdapterTimeout(
                f"Ollama at {self._config.endpoint} did not respond in time."
            ) from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        log_event(
            _logger,
            "llm_stream",
            model=model_name,
            duration_ms=duration_ms,
            message_count=message_count,
            output_chars=output_chars,
        )

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
