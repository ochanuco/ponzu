# Ponzu Roadmap

## Guiding Principles

- Local-first
- Privacy by default
- Simple MVP before integrations
- Replaceable adapters
- Assistant identity independent of voice provider
- Public code, private personal data

---

## Phase 0: Repository Foundation

### Goals

- Create the public `ponzu` repository
- Establish project boundaries
- Prevent accidental secret or personal-data commits

### Deliverables

- `README.md`
- `ADR.md`
- `ROADMAP.md`
- `DESIGN.md`
- `SECURITY.md`
- `.gitignore`
- `.env.example`
- License decision
- Basic CI
- Secret scanning and push protection review

### Exit Criteria

- Repository can be safely cloned publicly.
- No runtime-generated personal data is stored under the repository.
- Local data-directory conventions are documented.

---

## Phase 1: Minimal Voice Loop

### Scope

```text
Wake Word
 -> Record utterance
 -> STT
 -> Local LLM
 -> VOICEVOX
 -> Playback
 -> Idle
```

### Deliverables

- Microphone capture
- Wake-word detection
- End-of-speech detection or timeout
- Local STT adapter
- Ollama-compatible LLM adapter
- VOICEVOX adapter
- Audio playback
- State machine
- Basic structured logging
- Graceful error recovery

### Non-Goals

- Discord
- Gmail
- Calendar
- Home Assistant
- Long-term memory
- Multi-user support
- Mobile clients
- Cloud deployment

### Exit Criteria

- "ぽんず" activates the assistant consistently.
- One-turn conversation works end to end.
- The system returns to idle automatically.
- No audio or transcript is persisted unless explicitly enabled.

---

## Phase 2: Conversation Quality

### Goals

- Improve natural turn-taking
- Reduce latency
- Support short conversational context

### Candidate Work

- Voice activity detection
- Streaming STT
- Streaming LLM output
- Sentence-level TTS
- Barge-in and cancellation
- Conversation timeout
- Short-term memory
- Persona configuration
- Response-length control
- Benchmark harness

### Exit Criteria

- Typical response begins quickly enough for daily use.
- The user can interrupt speech.
- Multi-turn conversation remains coherent for a limited session.

---

## Phase 3: Daily Assistant Functions

### Goals

Introduce explicit tools while preserving local control.

### Candidate Skills

- Clock and timers
- Local notes
- Weather through an optional API adapter
- Calendar read access
- Gmail summaries
- GitHub notifications
- Utsuroi updates
- HomeLab status

### Design Constraints

- Every skill declares required permissions.
- Read-only access is preferred initially.
- External actions require explicit confirmation.
- Credentials remain in the user data directory or OS keychain.

---

## Phase 4: Memory and Personalization

### Candidate Features

- User preferences
- Entity memory
- Routine awareness
- Local semantic search
- Conversation summaries
- Configurable retention
- Memory inspection and deletion
- Per-skill memory boundaries

### Privacy Requirements

- Memory is opt-in.
- Users can inspect and delete stored memory.
- Sensitive data classes can be excluded.
- Raw recordings are disabled by default.

---

## Phase 5: Home Assistant / Home OS

### Candidate Features

- Home Assistant integration
- Device control
- HomeLab operations
- Presence awareness
- Scheduled routines
- Proactive notifications
- Multi-device audio endpoints

### Safety Requirements

- Dangerous actions require confirmation.
- Network, security, financial, and account-management actions are isolated.
- Audit logs are available locally.
- Skills use least-privilege credentials.

---

## Phase 6: Additional Interfaces

### Candidate Interfaces

- Menu bar application
- Web UI
- CLI
- Discord adapter
- Remote microphone/speaker nodes
- Mobile companion
- Optional cloud fallback

These interfaces remain adapters around the same core orchestration layer.
