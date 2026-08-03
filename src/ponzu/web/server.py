"""A read-only conversation view, on the standard library (ADR-018).

The console keeps its current job -- operational logs (DESIGN section 7).
This module gives the conversation a second surface: `ponzu start --web` (or
`web.enabled` in config) serves a browser page showing who said what, over
`http.server` plus server-sent events. No new dependency, no extra (ADR-009).

Hard constraints, straight from the ADR -- not up for negotiation here:

- Loopback only, bound to `127.0.0.1`, and there is no config key that can
  change that (DESIGN section 9). This serves transcripts of everything said
  in the room.
- Read-only: two routes, `GET /` and `GET /events`. No form, no POST, no text
  input.
- In-memory ring buffer only. History dies with the process --
  `privacy.persist_transcripts` defaults false (DESIGN section 5.1), and
  persisting it here would be exactly that.
- Never logged. `web_started` and `web_client_connected` carry counts and
  ports only, never utterance or response text (DESIGN section 7). Showing
  content to a user who is present is not logging it (ADR-015); writing it to
  disk would be.

ADR-010's lifecycle lesson applies here too: a thread still holding a
resource at interpreter exit is how the CoreAudio deadlock happened. The
server thread and every per-connection handler thread are daemons, and
`stop()` unblocks any `/events` stream promptly (`_stopping` plus a condition
notify) rather than relying on the daemon flag alone to paper over a hang.
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ponzu.adapters import AdapterUnavailable
from ponzu.core.logging import log_event

__all__ = ["ConversationView"]

_log = logging.getLogger(__name__)

# ADR-018: loopback only, not configurable. This is not a default -- there is
# deliberately no parameter or config key that can override it.
_HOST = "127.0.0.1"

# How often an idle stream sends a keepalive comment, so proxies/browsers do
# not time out a connection that has nothing new to report. Also doubles as
# the interval at which a stream notices `stop()` and a dead client (the next
# write after a client has gone away is what actually raises).
_KEEPALIVE_S = 15.0
_TICK_S = 1.0


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """One SSE frame. `data` must already be safe to embed as HTML text --
    see `_escaped_turn` -- `json.dumps` only handles the transport encoding,
    not the HTML-safety of the values inside it."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()


def _escaped_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Escape a stored turn for the wire. Transcripts are arbitrary user (and
    model) speech, and they end up as HTML text in the browser -- a
    transcript containing `</script>` or `<img onerror=...>` must not reach
    the page unescaped."""
    return {
        "ts": turn["ts"],
        "utterance": html.escape(turn["utterance"]),
        "response": html.escape(turn["response"]),
    }


_PAGE_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>ぽんず</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, system-ui, "Hiragino Sans", sans-serif;
    background: #111214;
    color: #e8e8e8;
  }
  header {
    position: sticky;
    top: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    background: #1b1c1f;
    border-bottom: 1px solid #2c2d31;
  }
  header h1 { font-size: 1rem; margin: 0; font-weight: 600; }
  #state {
    font-size: 0.85rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: #2c2d31;
    text-transform: capitalize;
  }
  #state.listening, #state.thinking, #state.speaking { background: #3a4a2c; }
  #log {
    max-width: 720px;
    margin: 0 auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }
  .turn { display: flex; flex-direction: column; gap: 0.3rem; }
  .bubble {
    max-width: 80%;
    padding: 0.5rem 0.8rem;
    border-radius: 0.9rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
  }
  .user { align-self: flex-end; background: #2a5db0; color: #fff; }
  .ponzu { align-self: flex-start; background: #2c2d31; color: #e8e8e8; }
  .label {
    font-size: 0.7rem;
    opacity: 0.6;
    padding: 0 0.2rem;
  }
  .user-row { display: flex; flex-direction: column; align-items: flex-end; }
  .ponzu-row { display: flex; flex-direction: column; align-items: flex-start; }
  #empty { opacity: 0.5; padding: 1rem; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>ぽんず -- conversation view</h1>
  <span id="state">idle</span>
</header>
<div id="log"><p id="empty">no turns yet</p></div>
<script>
(function () {
  "use strict";
  var log = document.getElementById("log");
  var empty = document.getElementById("empty");
  var stateEl = document.getElementById("state");

  function addTurn(turn) {
    if (empty) { empty.remove(); empty = null; }

    var wrap = document.createElement("div");
    wrap.className = "turn";

    var userRow = document.createElement("div");
    userRow.className = "user-row";
    var userLabel = document.createElement("div");
    userLabel.className = "label";
    userLabel.textContent = "あなた";
    var userBubble = document.createElement("div");
    userBubble.className = "bubble user";
    userBubble.innerHTML = turn.utterance;
    userRow.appendChild(userLabel);
    userRow.appendChild(userBubble);

    var ponzuRow = document.createElement("div");
    ponzuRow.className = "ponzu-row";
    var ponzuLabel = document.createElement("div");
    ponzuLabel.className = "label";
    ponzuLabel.textContent = "ぽんず";
    var ponzuBubble = document.createElement("div");
    ponzuBubble.className = "bubble ponzu";
    ponzuBubble.innerHTML = turn.response;
    ponzuRow.appendChild(ponzuLabel);
    ponzuRow.appendChild(ponzuBubble);

    wrap.appendChild(userRow);
    wrap.appendChild(ponzuRow);
    log.appendChild(wrap);
    window.scrollTo(0, document.body.scrollHeight);
  }

  function setState(state) {
    stateEl.textContent = state;
    stateEl.className = state;
  }

  var source = new EventSource("/events");
  source.addEventListener("turn", function (e) { addTurn(JSON.parse(e.data)); });
  source.addEventListener("state", function (e) {
    setState(JSON.parse(e.data).state);
  });
})();
</script>
</body>
</html>
"""
_PAGE_BYTES = _PAGE_HTML.encode("utf-8")


class ConversationView:
    """Owns the in-memory transcript buffer and the loopback HTTP server.

    Subscribed the same way the CLI subscribes to the orchestrator
    (`on_turn` -> `record_turn`, `on_state_change` -> `record_state`) --
    ADR-018's whole point is that no new seam was needed for this.
    """

    def __init__(
        self, *, port: int, history: int, logger: logging.Logger | None = None
    ) -> None:
        self._port = port
        self._logger = logger if logger is not None else _log
        self._condition = threading.Condition()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=history)
        self._next_seq = 0
        self._state = "idle"
        self._stopping = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._connection_count = 0

    @property
    def url(self) -> str:
        return f"http://{_HOST}:{self._port}/"

    @property
    def port(self) -> int:
        return self._port

    # ------------------------------------------------------------------
    # Recording -- the seams the CLI wires to `on_turn` / `on_state_change`.
    # ------------------------------------------------------------------

    def record_turn(self, utterance: str, response: str) -> None:
        """Append a completed turn to the ring buffer (both sides, timestamped).

        Only successful turns carry text here (the CLI only calls this for a
        `TurnResult` where `.ok` is true); a failed turn still moves the
        state machine, which `record_state` reports on its own.
        """
        with self._condition:
            self._next_seq += 1
            self._buffer.append(
                {
                    "seq": self._next_seq,
                    "ts": time.time(),
                    "utterance": utterance,
                    "response": response,
                }
            )
            self._condition.notify_all()

    def record_state(self, state: str) -> None:
        """Record the current conversation state for the page's live status."""
        with self._condition:
            self._state = state
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the server on a daemon thread, bound to loopback. Returns
        immediately -- `serve_forever` runs on its own thread."""
        if self._server is not None:
            return  # already running; starting twice is a no-op.

        handler = self._build_handler()
        try:
            server = ThreadingHTTPServer((_HOST, self._port), handler)
        except OSError as exc:
            raise AdapterUnavailable(
                f"cannot serve the conversation view: port {self._port} is "
                "already in use. Set a different `web.port` in config, or "
                "stop whatever else is listening on it."
            ) from exc

        # ADR-010's lesson: a non-daemon thread still holding a socket open at
        # interpreter exit is how the CoreAudio deadlock happened. Both the
        # server's own thread (below) and every per-request thread it spawns
        # (ThreadingMixIn.daemon_threads) must be daemons.
        server.daemon_threads = True
        self._stopping.clear()
        # A short poll interval, not `serve_forever`'s 0.5s default: `stop()`
        # below has to wait for this loop to notice `shutdown()`, and a
        # daemon thread still winding down is exactly the ADR-010 lifecycle
        # risk `stop()` exists to avoid lingering.
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05),
            name="ponzu-web",
            daemon=True,
        )
        self._server = server
        self._thread = thread
        thread.start()
        log_event(self._logger, "web_started", port=self._port)

    def stop(self, timeout: float = 5.0) -> None:
        """Shut the server down and release the port. Safe when never
        started, and safe to call more than once."""
        server, thread = self._server, self._thread
        if server is None:
            return
        self._server = None
        self._thread = None

        # Wakes any `/events` stream blocked in `_stream`'s wait so it exits
        # promptly instead of relying solely on the daemon flag to avoid
        # blocking process exit (ADR-010).
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()

        server.shutdown()
        server.server_close()  # actually releases the listening port.
        if thread is not None:
            thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # HTTP handling
    # ------------------------------------------------------------------

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        view = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "PonzuWeb/1"

            def log_message(self, format: str, *args: object) -> None:
                # http.server logs every request to stderr by default, which
                # would interleave with the structured JSON logs. Silenced
                # rather than routed through `log_event`: per-request access
                # logging is not one of the two allowed events here.
                pass

            def do_GET(self) -> None:
                if self.path == "/":
                    view._serve_index(self)
                elif self.path == "/events":
                    view._serve_events(self)
                else:
                    self.send_error(404)

            def _reject_method(self) -> None:
                self.send_error(405)

            do_POST = _reject_method
            do_PUT = _reject_method
            do_DELETE = _reject_method
            do_PATCH = _reject_method
            do_HEAD = _reject_method
            do_OPTIONS = _reject_method

        return _Handler

    def _serve_index(self, request: BaseHTTPRequestHandler) -> None:
        request.send_response(200)
        request.send_header("Content-Type", "text/html; charset=utf-8")
        request.send_header("Content-Length", str(len(_PAGE_BYTES)))
        request.end_headers()
        request.wfile.write(_PAGE_BYTES)

    def _serve_events(self, request: BaseHTTPRequestHandler) -> None:
        request.send_response(200)
        request.send_header("Content-Type", "text/event-stream")
        request.send_header("Cache-Control", "no-cache")
        request.send_header("Connection", "close")
        request.end_headers()
        with self._condition:
            self._connection_count += 1
            total = self._connection_count
        # Counts and ports only -- never what was said (DESIGN section 7).
        log_event(self._logger, "web_client_connected", total=total)
        try:
            for chunk in self._stream():
                request.wfile.write(chunk)
                request.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected; nothing left to do.

    def _stream(self) -> Iterator[bytes]:
        """Existing buffer + current state first, then live updates, with a
        periodic keepalive comment so an idle connection is not dropped."""
        last_seq_sent = 0
        last_state_sent: str | None = None
        last_keepalive = time.monotonic()
        first = True

        while not self._stopping.is_set():
            with self._condition:
                if not first:
                    self._condition.wait(timeout=_TICK_S)
                    if self._stopping.is_set():
                        break
                turns = [t for t in self._buffer if t["seq"] > last_seq_sent]
                state = self._state
            first = False

            for turn in turns:
                last_seq_sent = turn["seq"]
                yield _sse("turn", _escaped_turn(turn))

            if state != last_state_sent:
                last_state_sent = state
                yield _sse("state", {"state": html.escape(state)})

            now = time.monotonic()
            if now - last_keepalive >= _KEEPALIVE_S:
                last_keepalive = now
                yield b": keepalive\n\n"
