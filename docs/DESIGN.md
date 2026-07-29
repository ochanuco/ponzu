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
```

### Requirements

- Detect "ぽんず"
- Operate continuously with low CPU use
- Expose confidence where available
- Support configurable sensitivity
- Avoid persisting microphone audio

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
- Normalize audio format for STT

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

### Future Capability

Barge-in should stop playback and transition to listening.

---

## 5. Configuration

## 5.1 Public Example Configuration

```yaml
wake_word:
  phrase: "ぽんず"
  sensitivity: 0.6

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
  speaker_id: 0

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

### Still Open

- Default local LLM (example configuration uses `qwen3:30b`)

  Measured on an M1 Max / 64 GB, `qwen3:30b` via Ollama:

  | | |
  | --- | --- |
  | Cold load (18 GB) | ~27 s |
  | Warm turn | ~35 s |
  | Reasoning emitted for an 8-character answer | ~6,800 characters |

  This does not meet the non-functional requirement "low enough latency for
  conversational use" in section 2. The cost is the reasoning trace, not the
  parameter count: the model thinks at length before answering briefly.

  Disabling it is not a fix. With Ollama's `think: false`, `qwen3:30b` stops
  separating its reasoning and leaks it into `message.content` instead, so the
  assistant would speak "Okay, the user said..." aloud. Reasoning must stay
  enabled for the adapter's `message.content` read to be correct.

  Resolving this therefore means choosing a different default model, not
  tuning the current one.

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
