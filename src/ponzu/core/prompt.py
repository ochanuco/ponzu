"""Persona and prompt construction (DESIGN section 4.6).

This layer is deliberately outside the LLM adapter. ADR-005 scopes the adapter
to transport only, so persona text, conversation history, and response
constraints are assembled here and handed over as plain messages.

The persona describes ぽんず, not a voice. ADR-004 keeps assistant identity
independent of the TTS engine, so nothing in this module may mention VOICEVOX,
a speaker id, or any other synthesis detail.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ponzu.adapters import Message

__all__ = ["DEFAULT_PERSONA", "ConversationContext"]


# DESIGN section 4.6 specifies the persona direction rather than the wording.
# The traits below are a direct transcription of that list: concise, calm,
# familiar but not overly casual, technically capable, minimal chatter,
# occasional light humor, and no claiming actions it did not perform.
#
# The last constraint is the load-bearing one. The MVP has no skills (ROADMAP
# Phase 3), so the model must not imply it set a timer, sent a message, or
# controlled a device — it currently cannot do any of those.
DEFAULT_PERSONA = """\
あなたは「ぽんず」という名前の音声アシスタントです。

話し方:
- 簡潔に答える。前置きや復唱をしない。
- 落ち着いた口調。過度にくだけた表現や過剰な敬語は使わない。
- 技術的な質問には具体的に答える。
- ときどき軽い冗談は許容されるが、雑談を長引かせない。

制約:
- 音声で読み上げられるため、箇条書き・記号・コードブロックは避け、
  自然な文章で答える。
- 通常は2〜3文以内で答える。
- 実際には実行していない操作を、実行したかのように言わない。
  タイマー、メール送信、機器操作などの機能は現在持っていない。
  できないことは、できないと短く伝える。
- 入力は音声認識を経ているため、誤認識が混ざることがある。
  聞き取れない語や知らない固有名詞が来たときは、
  勝手に「誤字ですね」などと断定して直さず、短く聞き返す。
"""


@dataclass(slots=True)
class ConversationContext:
    """Short-term conversation history for a single session.

    DESIGN section 11 leaves the retention policy open, so this keeps a bounded
    in-memory window and nothing else. Persistence is deliberately absent:
    ``privacy.persist_conversations`` defaults to false (DESIGN section 5.1) and
    long-term memory is ROADMAP Phase 4. There is no disk path here to disable.
    """

    persona: str = DEFAULT_PERSONA
    max_turns: int = 6
    _history: deque[Message] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Two messages per turn (user + assistant), so the deque holds double.
        self._history = deque(maxlen=self.max_turns * 2)

    def build(self, utterance: str) -> list[Message]:
        """Assemble the message list for one turn.

        The persona is rebuilt on every call rather than stored in history, so
        trimming old turns can never evict the system prompt.
        """
        return [
            Message(role="system", content=self.persona),
            *self._history,
            Message(role="user", content=utterance),
        ]

    def record(self, utterance: str, response: str) -> None:
        """Append a completed exchange to the window."""
        self._history.append(Message(role="user", content=utterance))
        self._history.append(Message(role="assistant", content=response))

    def clear(self) -> None:
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2
