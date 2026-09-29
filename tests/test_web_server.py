"""Tests for ponzu.web.server (ADR-018: read-only conversation view).

Driven over real HTTP on an ephemeral port -- `port=0` is not available
through `Config` (ADR-018 forbids a configurable host, and the port itself
still has to be a real, fixed number for the server to bind), so each test
picks a free port itself with `socket` and passes it to `ConversationView`.

Every wait has a bound: SSE frames are read line-by-line off a socket with a
short connection timeout, never with an unbounded `.read()`, so a bug that
stops the server from sending data fails the test instead of hanging it.
"""

from __future__ import annotations

import http.client
import logging
import socket
import urllib.error
import urllib.request

import pytest

from ponzu.adapters import AdapterUnavailable
from ponzu.web.server import ConversationView

_TIMEOUT = 5.0


def _free_port() -> int:
    """A port nothing is listening on right now, for tests to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read_sse_frames(port: int, count: int, timeout: float = _TIMEOUT) -> list[str]:
    """Read exactly `count` complete SSE frames from `/events`, then
    disconnect without waiting for the server to close the stream itself
    (it never does while running).

    Reads line-by-line rather than a fixed-size `.read()`: the connection
    stays open indefinitely, so requesting more bytes than have actually
    been written yet would block until the socket timeout regardless of
    whether anything is wrong. `readline()` only blocks for data that is
    already in flight to complete the current line.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/events")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"

        frames: list[str] = []
        current: list[str] = []
        while len(frames) < count:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8")
            if text in ("\n", "\r\n"):
                frames.append("".join(current))
                current = []
            else:
                current.append(text)
        return frames
    finally:
        conn.close()


@pytest.fixture
def make_view():
    """Builds a started `ConversationView` on a free port, and stops every
    one created by a test during teardown -- including ones a test already
    stopped itself, exercising `stop()`'s double-call safety along the way.
    """
    created: list[ConversationView] = []

    def _make(history: int = 50) -> ConversationView:
        view = ConversationView(port=_free_port(), history=history)
        view.start()
        created.append(view)
        return view

    yield _make

    for view in created:
        view.stop()


# --------------------------------------------------------------------- routes


def test_index_returns_200_and_html(make_view) -> None:
    view = make_view()

    response = urllib.request.urlopen(view.url, timeout=_TIMEOUT)

    assert response.status == 200
    assert response.headers.get("Content-Type", "").startswith("text/html")
    body = response.read().decode("utf-8")
    assert "<!doctype html>" in body.lower()
    assert "ぽんず" in body


def test_events_returns_200_with_event_stream_content_type(make_view) -> None:
    view = make_view()

    frames = _read_sse_frames(view.port, count=1)  # the initial state event

    assert frames  # content-type/status are asserted inside the helper


def test_unknown_path_returns_404(make_view) -> None:
    view = make_view()

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{view.url}nope", timeout=_TIMEOUT)

    assert exc_info.value.code == 404


def test_post_to_index_returns_405(make_view) -> None:
    view = make_view()

    conn = http.client.HTTPConnection("127.0.0.1", view.port, timeout=_TIMEOUT)
    try:
        conn.request("POST", "/")
        response = conn.getresponse()
        assert response.status == 405
        response.read()
    finally:
        conn.close()


def test_post_to_events_returns_405(make_view) -> None:
    view = make_view()

    conn = http.client.HTTPConnection("127.0.0.1", view.port, timeout=_TIMEOUT)
    try:
        conn.request("POST", "/events")
        response = conn.getresponse()
        assert response.status == 405
        response.read()
    finally:
        conn.close()


# ----------------------------------------------------------------------- SSE


def test_recorded_turn_appears_in_the_event_stream(make_view) -> None:
    view = make_view()
    view.record_turn("体重は？", "86.9キロです")

    frames = _read_sse_frames(view.port, count=2)  # initial state, then the turn
    combined = "\n".join(frames)

    assert "体重は" in combined
    assert "86.9" in combined


def test_state_change_appears_in_the_event_stream(make_view) -> None:
    view = make_view()
    view.record_state("listening")

    frames = _read_sse_frames(view.port, count=1)

    assert "listening" in frames[0]


def test_turn_with_script_and_img_onerror_is_escaped(make_view) -> None:
    """Transcripts are arbitrary user speech, and they end up as HTML text in
    the browser -- neither the page nor the SSE payload may ever carry the
    raw markup, on either side of the conversation."""
    utterance = "<script>alert(1)</script>"
    response_text = '<img src=x onerror="alert(2)">'
    view = make_view()
    view.record_turn(utterance, response_text)

    frames = _read_sse_frames(view.port, count=2)
    combined = "\n".join(frames)

    assert "<script>alert(1)</script>" not in combined
    assert "<img src=x onerror" not in combined
    assert "&lt;script&gt;" in combined
    assert "&lt;img src=x onerror" in combined

    # The index page never embeds transcript content at all -- confirm the
    # raw markup is absent there too, not just in the stream.
    page = urllib.request.urlopen(view.url, timeout=_TIMEOUT).read().decode("utf-8")
    assert utterance not in page
    assert response_text not in page


# --------------------------------------------------------------- ring buffer


def test_ring_buffer_is_bounded_and_keeps_the_newest(make_view) -> None:
    view = make_view(history=3)
    for i in range(5):
        view.record_turn(f"utterance-{i}", f"response-{i}")

    frames = _read_sse_frames(view.port, count=1 + 3)  # state + the newest 3
    combined = "\n".join(frames)

    assert "utterance-0" not in combined
    assert "utterance-1" not in combined
    for i in (2, 3, 4):
        assert f"utterance-{i}" in combined


# -------------------------------------------------------------------- logging


def test_conversation_text_never_reaches_a_log_record(make_view, caplog) -> None:
    caplog.set_level(logging.INFO, logger="ponzu.web.server")
    view = make_view()
    view.record_turn("体重は？", "86.9キロです")

    # Triggers `web_client_connected`; one line is enough, then disconnect.
    conn = http.client.HTTPConnection("127.0.0.1", view.port, timeout=_TIMEOUT)
    try:
        conn.request("GET", "/events")
        response = conn.getresponse()
        response.readline()
    finally:
        conn.close()

    assert caplog.records  # something was actually logged (web_started etc.)
    for record in caplog.records:
        fields = getattr(record, "ponzu_fields", None) or {}
        haystack = " ".join([record.getMessage(), *(str(v) for v in fields.values())])
        assert "体重" not in haystack
        assert "86.9" not in haystack
        assert "キロ" not in haystack


# --------------------------------------------------------------- lifecycle


def test_stop_releases_the_port(make_view) -> None:
    view = make_view()
    port = view.port
    # An actual client connection first, the same as a browser having been
    # open against it -- a TIME_WAIT socket from that connection must not be
    # mistaken for the listening port still being held.
    urllib.request.urlopen(view.url, timeout=_TIMEOUT).read()

    view.stop()

    # The real proof this matters: a fresh server binds the same port right
    # away, the same as `ponzu start --web` being restarted straight after
    # Ctrl-C.
    reborn = ConversationView(port=port, history=1)
    try:
        reborn.start()  # raises AdapterUnavailable if the port is still held
    finally:
        reborn.stop()


def test_start_on_an_occupied_port_raises_adapter_unavailable(make_view) -> None:
    view = make_view()
    other = ConversationView(port=view.port, history=1)

    with pytest.raises(AdapterUnavailable, match=str(view.port)):
        other.start()


def test_stop_without_start_is_a_noop() -> None:
    view = ConversationView(port=_free_port(), history=1)

    view.stop()  # never started -- must not raise


def test_stop_twice_is_a_noop(make_view) -> None:
    view = make_view()

    view.stop()
    view.stop()  # already stopped -- must not raise
