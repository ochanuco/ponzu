# Ponzu Initial System Design

## 1. Purpose

Ponzu is a local-first voice assistant intended to operate as a familiar household companion.

The first implementation targets a MacBook Pro M1 Max with 64 GB of memory and uses local speech recognition, local LLM inference, and VOICEVOX speech synthesis.

---

## 2. MVP Requirements

### Functional Requirements

1. Wait in an idle state.
2. Detect the wake word "ぽんず".
3. Capture the user's next utterance.
4. Convert speech to text.
5. Send the text and minimal conversation context to a local LLM.
6. Convert the generated response to speech.
7. Play the response.
8. Return to idle.
9. Recover from component errors without requiring restart.

### Non-Functional Requirements

- Local processing by default
- No paid cloud API required
- Replaceable STT, LLM, and TTS engines
- No personal-data persistence by default
- Clear operational logs without raw user content
- Apple Silicon support
- Low enough latency for conversational use

---

## 3. High-Level Architecture

```text
+------------------+
| Microphone Input |
+--------+---------+
         |
         v
+------------------+
| Wake Word Engine |
+--------+---------+
         |
         v
+------------------+
| Utterance Capture|
| + VAD / Timeout  |
+--------+---------+
         |
         v
+------------------+
| STT Adapter      |
+--------+---------+
         |
         v
+------------------+
| Orchestrator     |
| - State machine  |
| - Prompt builder |
| - Short context  |
+--------+---------+
         |
         v
+------------------+
| Local LLM Adapter|
| Ollama-compatible|
+--------+---------+
         |
         v
+------------------+
| TTS Adapter      |
| VOICEVOX         |
+--------+---------+
         |
         v
+------------------+
| Audio Output     |
+------------------+
```

---

## 4. Component Design

## 4.1 Orchestrator

The orchestrator owns the conversation state machine and invokes adapters.

### Responsibilities

- State transitions
- Timeouts
- Cancellation
- Error recovery
- Context assembly
- Adapter invocation
- Metrics and structured logs

### State Model

```text
IDLE
LISTENING
TRANSCRIBING
THINKING
SPEAKING
ERROR
```

All recoverable failures should transition through `ERROR` and return to `IDLE`.

---

## 4.2 Wake Word Detector

### Interface

```text
start()
stop()
on_detected(callback)
is_running -> bool
```

`is_running` was added by ADR-010: without it the orchestrator cannot tell an
idle detector from one that has died, and the loop waits forever on an engine
that will never fire again.

### Requirements

- Detect "ぽんず"
- Operate continuously with low CPU use
- Expose confidence where available
- Support configurable sensitivity
- Avoid persisting microphone audio

### Implementation

ADR-013: an energy gate in front of the existing `faster-whisper` recogniser,
rather than a dedicated wake-word engine. Transcription runs only once the RMS
threshold has been crossed, so silence is nearly free. `sensitivity` is the
minimum transcript confidence accepted for a match.

The detector holds the microphone while idling and releases it before invoking
the callback, because `voice_turn` opens its own capture stream.

### Future Considerations

- Custom wake-word model
- Prefix phrase such as "ねえ、ぽんず"
- Per-room calibration
- False-positive evaluation corpus

---

## 4.3 Audio Input and Utterance Capture

### Responsibilities

- Open microphone stream
- Buffer audio after wake detection
- Detect end of speech
- Apply maximum utterance timeout
- Give up early when speech never starts
- Normalize audio format for STT

`silence_timeout_ms` only applies once speech has been detected, so an
utterance that never begins would otherwise run the full `max_utterance_ms`.
Ten seconds of nothing reads as a broken assistant, so `speech_start_timeout_ms`
bounds the wait for speech to begin.

```text
capture_utterance(max_duration_ms, silence_timeout_ms,
                  speech_start_timeout_ms=None) -> AudioBuffer
```

`speech_start_timeout_ms` is an optional per-call override of the configured
default, falling back to it when omitted. ADR-015's follow-up window needs a
different budget for "wait for speech to begin" than an ordinary turn does —
that is the same question with a different answer, not a different mechanism,
so it is a parameter rather than a second method.

### Recommended Runtime Format

- Mono PCM
- 16 kHz or adapter-native sample rate
- 16-bit signed samples

The exact format may vary internally, but adapter boundaries should be explicit.

---

## 4.4 Speech-to-Text Adapter

### Candidate Backends

- `whisper.cpp`
- `faster-whisper`
- Apple-native alternatives if appropriate

### Interface

```text
transcribe(audio) -> Transcript
```

### Transcript Model

```text
text
language
confidence
duration_ms
```

Confidence may be unavailable depending on the backend.

### Vocabulary Biasing

`stt.initial_prompt` seeds the recogniser with words it would otherwise not
reach for. Proper nouns are where an open-vocabulary model fails hardest:
"うば茶" came back as "うばちゃん" and "奪茶", and the assistant then confidently
"corrected" the user's spelling.

Measured on the `small` model:

| | Without | With |
| --- | --- | --- |
| うば茶 | うばちゃ | うば茶 |
| うば茶に書いて | うばちゃんに書いて | うば茶に書いて |

It does **not** help the wake gate. A two-mora phrase in isolation gives the
model no context for the bias to act on — `ぽんず` still comes back as `コンズ`
with the prompt set, so ADR-013's edit-distance match stays necessary.

---

## 4.5 Local LLM Adapter

### Initial Backend

Ollama-compatible local HTTP API.

### Interface

```text
generate(messages, options) -> ModelResponse
```

### Responsibilities

- Transport only
- Model selection
- Timeout handling
- Streaming support later

### Non-Responsibilities

- Persona definition
- Tool execution
- Memory policy
- Prompt construction

These remain in the orchestration and application layers.

---

## 4.6 Prompt and Persona Layer

Ponzu's identity must remain independent of its voice provider.

### Initial Persona Direction

- Concise
- Calm
- Familiar but not overly casual
- Technically capable
- Minimal unnecessary chatter
- Occasional light humor
- No claim of actions not actually performed
- Asks rather than guesses when the input looks garbled

Speech recognition errors are unavoidable, so the assistant receives malformed
input as a matter of course. Asserting a confident interpretation of a garbled
proper noun is the failure mode observed in practice — told "うば茶" and given
"こっちゃんのうばっちゃん", it replied that this was a typo and stated the
"correct" spelling, which it had no basis for.

### Prompt Inputs

- System persona
- Current user utterance
- Limited session history
- Optional skill results
- Response constraints

Prompts containing private user data must never be committed to the public repository.

---

## 4.7 TTS Adapter

### Initial Backend

VOICEVOX Engine.

### Interface

```text
synthesize(text, voice_config) -> Audio
```

### Configuration

- Engine endpoint
- Speaker ID
- Speed
- Pitch
- Intonation
- Volume

All voice settings should be user configuration, not hard-coded.

---

## 4.8 Audio Output

### Responsibilities

- Play synthesized audio
- Report completion
- Support cancellation
- Prevent overlapping assistant speech
- Play a sequence of clips back to back without a gap between them

Sentence-level synthesis (ADR-014) hands playback one clip at a time while the
model is still generating, so clips must queue rather than overlap or race.

### Future Capability

Barge-in should stop playback and transition to listening.

---

## 5. Configuration

## 5.1 Public Example Configuration

```yaml
wake_word:
  provider: "whisper"
  phrase: "ぽんず"
  sensitivity: 0.3 # a floor, not a probability -- see ADR-013
  model: "base"

stt:
  provider: "faster_whisper"
  model: "small"

llm:
  provider: "ollama"
  endpoint: "http://127.0.0.1:11434"
  model: "qwen3:30b"

tts:
  provider: "voicevox"
  endpoint: "http://127.0.0.1:50021"
  speaker_id: 14 # 冥鳴ひまり / ノーマル

privacy:
  persist_audio: false
  persist_transcripts: false
  persist_conversations: false
```

The committed repository should contain only an example configuration.

## 5.2 Runtime Configuration Location

```text
~/Library/Application Support/Ponzu/config.yaml
```

Potential future storage:

```text
~/Library/Application Support/Ponzu/
├── config.yaml
├── secrets.env
├── ponzu.db
├── logs/
├── cache/
├── audio/
└── models/
```

---

## 6. Repository Layout

```text
ponzu/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── config/
│   └── default.example.yaml
├── docs/
│   ├── ADR.md
│   ├── DESIGN.md
│   ├── ROADMAP.md
│   └── SECURITY.md
├── src/
│   └── ponzu/
│       ├── cli.py
│       ├── core/          # config, paths, logging, state machine, orchestrator
│       ├── adapters/      # protocol definitions + registry
│       ├── audio/
│       ├── wakeword/
│       ├── stt/
│       ├── llm/
│       ├── tts/
│       └── skills/
├── tests/
└── scripts/
```

The implementation language is **Python managed by `uv`** (see ADR-009). Design
documents live under `docs/`; only `README.md` remains at the repository root.
The `src/ponzu/` package layout keeps the installable module namespace explicit
and matches `uv`'s default src-layout support.

---

## 7. Logging and Observability

### Default Logging

Allowed:

- State transitions
- Component timings
- Model names
- Error categories
- Audio duration
- Transcript character count
- Response character count

Disabled by default:

- Raw audio
- Full transcripts
- Full prompts
- Full LLM responses
- Email or calendar payloads
- Secrets
- Authentication headers

### Example

```json
{
  "event": "turn_completed",
  "stt_ms": 820,
  "llm_ms": 2410,
  "tts_ms": 530,
  "input_chars": 18,
  "output_chars": 42
}
```

---

## 8. Error Handling

### Recoverable Failures

- Wake-word engine restart
- Microphone timeout
- Empty transcript
- STT timeout
- LLM timeout
- VOICEVOX unavailable
- Playback error

### Policy

1. Log a sanitized error.
2. Play a short local fallback sound or message when possible.
3. Release audio resources.
4. Return to `IDLE`.

The assistant must not remain stuck in `LISTENING` or `SPEAKING`.

---

## 9. Security and Privacy Model

### Trust Boundary

The local machine is the primary trusted environment.

External systems are untrusted unless explicitly configured.

### Rules

- Bind local service endpoints to loopback where possible.
- Do not expose Ollama or VOICEVOX directly to the LAN by default.
- Use OS keychain or protected local files for credentials.
- Never log authorization headers.
- Treat transcripts and memory as personal data.
- Require explicit user intent before persisting audio.
- Keep skill permissions separate.

---

## 10. Future Skill Model

A future skill should declare:

```text
name
description
permissions
input_schema
output_schema
confirmation_policy
timeout
```

Example permission classes:

- `local.read`
- `local.write`
- `network.read`
- `network.write`
- `account.read`
- `account.write`
- `device.control`
- `financial.read`
- `financial.write`

Write and high-impact operations should require explicit confirmation.

---

## 11. Open Decisions

### Resolved

| Decision | Outcome | Reference |
| --- | --- | --- |
| Primary implementation language | Python + `uv` | ADR-009 |
| Target platform | macOS on Apple Silicon | ADR-009 |
| Menu bar UI versus CLI-only MVP | CLI-only for the MVP | ADR-011 |
| Wake-word engine | Substitute engine initially, behind a stable interface | ADR-010 |
| STT backend | `faster-whisper`, default model size `small` | ADR-009 |
| ~~Default local LLM~~ | reopened — ADR-012's reasoning was refuted by ADR-014's measurements | ADR-014 |
| VOICEVOX speaker | 冥鳴ひまり / ノーマル (style id 14) | section 5.1 |
| License | MIT | `LICENSE` |

### Still Open

- Default local LLM. A reasoning model spends ~99% of a turn thinking before
  emitting any answer, which streaming cannot hide (ADR-014).
- Packaging and process supervision
- Short-term context retention policy

### Measured Latency

End to end on an M1 Max / 64 GB over three consecutive `ponzu start` turns,
warm model, `qwen3:30b` + `faster-whisper` `small` + VOICEVOX:

| Turn | Utterance | STT | LLM | TTS | Stop speaking → reply starts |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.6 s | 1531 ms | 4833 ms | 1384 ms | 6.4 s |
| 2 | 4.2 s | 874 ms | 4064 ms | 1030 ms | 4.9 s |
| 3 | 6.2 s | 1016 ms | 4114 ms | 1096 ms | 5.1 s |

Cold-loading the 18 GB model costs ~27 s once, before the first turn.

An earlier revision recorded ~35 s per warm turn and concluded the model could
not meet the "low enough latency for conversational use" requirement in
section 2. That measurement used a bare `curl` with no system prompt, where the
model produced ~6,800 characters of reasoning before answering. Through the
real pipeline the persona's response-length constraint (section 4.6) keeps
reasoning short. The earlier conclusion was wrong; see ADR-012 for the decision
this evidence supports.

--- | --- | --- | --- | --- | --- |
  | 1 | 3.6 s | 1531 ms | 4833 ms | 1384 ms | 6.4 s |
  | 2 | 4.2 s | 874 ms | 4064 ms | 1030 ms | 4.9 s |
  | 3 | 6.2 s | 1016 ms | 4114 ms | 1096 ms | 5.1 s |

  Cold-loading the 18 GB model costs ~27 s once, before the first turn.

  An earlier revision of this section recorded ~35 s per warm turn and
  concluded the model could not meet the "low enough latency for conversational
  use" requirement in section 2. That measurement was taken with a bare `curl`
  and no system prompt, where the model produced ~6,800 characters of reasoning
  before answering. Through the real pipeline the persona's response-length
  constraint (section 4.6) keeps reasoning short, and the LLM stage settles
  around 4-5 s. The earlier conclusion was wrong and the model is usable.

  What remains true: with Ollama's `think: false`, `qwen3:30b` stops separating
  its reasoning and leaks it into `message.content`, so the assistant would
  speak "Okay, the user said..." aloud. Reasoning must stay enabled for the
  adapter's `message.content` read to be correct.

  ~5 s from end of speech to start of reply is usable but not comfortable.
  The LLM is the largest single stage, so a smaller model is still the obvious
  lever; streaming (ROADMAP Phase 2) would cut perceived latency without
  changing the model.

- VOICEVOX speaker
- Packaging and process supervision
- Short-term context retention policy
- License

---

## 12. MVP Command Surface

The first milestone delivers exactly three commands (ADR-011).

| Command | Purpose |
| --- | --- |
| `ponzu doctor` | Verify configuration and probe every dependency, reporting per-component status without starting the loop. |
| `ponzu chat` | Text-in/text-out conversation against the LLM adapter. Exercises orchestration without audio hardware. |
| `ponzu start` | The full voice loop described in section 3. |

`ponzu chat` exists so the orchestration, prompt, and LLM layers can be
developed and tested on machines without a microphone, VOICEVOX, or an STT
model present.
