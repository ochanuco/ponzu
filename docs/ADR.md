# Architecture Decision Records

## ADR-001: Project Name

- **Status:** Accepted
- **Decision:** The project and assistant name will be **Ponzu / ぽんず**.

### Context

The assistant is intended to feel like a familiar household companion rather than a formal chatbot, anime character, or theatrical AI butler.

The name must also work as:

- a wake word,
- a repository name,
- a CLI command,
- a persistent identity independent of the selected voice engine.

### Consequences

- Repository name: `ponzu`
- CLI examples: `ponzu start`, `ponzu chat`, `ponzu config`
- The assistant's identity remains "ぽんず" even if its voice changes later.

---

## ADR-002: Local-First Architecture

- **Status:** Accepted
- **Decision:** The MVP will run locally without paid cloud inference APIs.

### Context

The target machine is a MacBook Pro with an M1 Max and 64 GB of memory. This provides sufficient local compute for speech recognition, local LLM inference, and speech synthesis.

### Decision

The initial processing pipeline will be:

```text
Microphone
  -> Wake Word
  -> Speech-to-Text
  -> Local LLM
  -> VOICEVOX
  -> Speaker
  -> Idle
```

### Consequences

- No recurring LLM API cost for the MVP.
- User audio and conversation data remain local by default.
- Performance and model selection are constrained by the local machine.
- Cloud services may be added later only as optional adapters.

---

## ADR-003: Discord Is Excluded from the MVP

- **Status:** Accepted
- **Decision:** Discord integration will not be part of the initial implementation.

### Context

Discord was initially considered as a voice transport layer, but it adds bot lifecycle, connection management, audio encoding, permissions, and operational complexity.

### Consequences

The MVP is limited to direct local interaction:

1. Detect wake word
2. Capture speech
3. Transcribe
4. Generate response
5. Speak response
6. Return to idle

Discord may be introduced later as an optional input/output adapter.

---

## ADR-004: VOICEVOX as the Initial TTS Engine

- **Status:** Accepted
- **Decision:** The MVP will use VOICEVOX for speech synthesis.

### Context

A.I.VOICE and 結月ゆかり remain desirable future options, but their Windows dependency complicates an Apple Silicon-first MVP.

### Consequences

- VOICEVOX Engine is the initial TTS backend.
- TTS must be behind an adapter interface.
- Voice identity and assistant identity are kept separate.
- Future engines may include A.I.VOICE, Style-Bert-VITS2, or other local engines.

---

## ADR-005: Local LLM via Ollama-Compatible Interface

- **Status:** Accepted
- **Decision:** The initial LLM integration will use a local Ollama-compatible HTTP interface.

### Context

The implementation should avoid coupling the application core to a single model.

### Consequences

- The LLM layer is accessed through an adapter.
- Model name and endpoint are configuration values.
- Prompt construction and conversation state remain outside the adapter.
- Other runtimes may be added later.

---

## ADR-006: Public Repository with Local Private Data

- **Status:** Accepted
- **Decision:** The source repository may be public, while all user-specific and sensitive data remains outside the repository.

### Public Content

- Source code
- Default configuration
- Example configuration
- Documentation
- Generic prompts
- Test fixtures containing no personal data

### Private Content

- API tokens and credentials
- User configuration
- Audio recordings
- Speech transcripts
- Conversation history
- Long-term memory
- Email and calendar data
- Home network addresses and device names
- Wake-word training recordings
- Debug logs containing user content

### Storage Principle

```text
Git repository
  -> source, examples, docs

User application data directory
  -> secrets, configuration, logs, databases, audio, memory
```

On macOS, the default data location should be:

```text
~/Library/Application Support/Ponzu/
```

### Consequences

- `.env.example` may be committed.
- `.env`, local databases, recordings, and runtime data must be ignored.
- Logging must avoid raw user content by default.
- Public release assumes committed content is permanently public.

---

## ADR-007: Modular Pipeline and Adapter Boundaries

- **Status:** Accepted
- **Decision:** Wake word, STT, LLM, TTS, audio I/O, and future skills will be independently replaceable components.

### Proposed Core Interfaces

```text
WakeWordDetector
SpeechRecognizer
LanguageModel
SpeechSynthesizer
AudioInput
AudioOutput
Skill
MemoryStore
```

### Consequences

- MVP implementation can remain simple.
- Components can be benchmarked independently.
- Platform-specific engines do not leak into the application core.
- Future cloud and remote adapters remain possible without redesigning the whole system.

---

## ADR-008: Conversation State Machine

- **Status:** Accepted
- **Decision:** The MVP will use an explicit state machine rather than an implicit event chain.

### States

```text
IDLE
LISTENING
TRANSCRIBING
THINKING
SPEAKING
ERROR
```

### Primary Transition

```text
IDLE
 -> LISTENING
 -> TRANSCRIBING
 -> THINKING
 -> SPEAKING
 -> IDLE
```

### Text Transition

`ponzu chat` (ADR-011) takes typed input, so no audio is captured and nothing is
transcribed:

```text
IDLE
 -> THINKING
 -> SPEAKING   (only when speech output is requested)
 -> IDLE
```

`IDLE -> THINKING` and `THINKING -> IDLE` are therefore legal transitions.

The alternative — routing text turns through `LISTENING` and `TRANSCRIBING`
anyway — was rejected. Those states would be a lie in the log trace, and the
state trace is the primary evidence for diagnosing a stuck assistant.

### Recovery Transitions

- Any state may transition to `ERROR`.
- `ERROR` may only transition to `IDLE`.

### Consequences

- Cancellation and timeout behavior can be implemented consistently.
- UI or status indicators can be added later.
- Failures can return safely to `IDLE`.
- The transition table is explicit and closed: a transition not listed above is
  a programming error and raises, rather than silently corrupting the trace.

---

## ADR-009: Python with `uv`, Targeting Apple Silicon

- **Status:** Accepted
- **Decision:** The implementation language is **Python**, with dependencies and
  the virtual environment managed by **`uv`**. The supported target is
  **macOS on Apple Silicon**.

### Context

DESIGN section 11 previously left the language open, to be chosen on audio
ecosystem maturity, Apple Silicon support, packaging, and maintainability.

Python has the most direct bindings for the components this project already
committed to: `faster-whisper` / `whisper.cpp` for STT, `sounddevice`
(PortAudio) for capture and playback, and plain HTTP for both Ollama and
VOICEVOX. `uv` provides reproducible resolution, a lockfile, and fast
environment creation without a separate tool for packaging.

### Consequences

- `pyproject.toml` plus `uv.lock` are the single source of dependency truth.
- Heavy or hardware-bound dependencies (`sounddevice`, `faster-whisper`,
  `numpy`) are declared as **optional extras**, so a clone can install, import,
  and run the test suite on a machine with no audio stack.
- Python 3.11+ is required.
- Linux and Windows are explicitly out of scope for the MVP; nothing should
  hard-code Apple-only behavior beyond the data directory resolution.
- The STT backend is **`faster-whisper`**, resolving that entry in section 11.
  DESIGN section 4.4 listed `whisper.cpp` and `faster-whisper` as candidates,
  and the original example config named a `whisper_cpp` provider with a
  `models/ggml-small.bin` path. Those values are incompatible with the backend
  that was actually implemented: `faster-whisper` takes either a size name
  (`small`) or a directory holding a CTranslate2 model, and interprets anything
  else as a Hugging Face repository id. Left unchanged, the first voice turn
  failed with `RepositoryNotFoundError`. The config now names the Python
  backend (`faster_whisper`) and a size (`small`), and `whisper_cpp` is not
  accepted — nothing implements it.

---

## ADR-010: Substitute Wake Word Engine for the Initial Implementation

- **Status:** Accepted
- **Decision:** The first implementation may ship a **substitute** wake-word
  trigger instead of true acoustic "ぽんず" detection, provided it sits behind
  the `WakeWordDetector` interface defined in ADR-007.

### Context

Acoustic wake-word detection for a Japanese phrase requires either a custom
trained model or a licensed engine. Blocking the entire voice loop on that work
would prevent the rest of the pipeline from being exercised end to end.

### Interface

```text
start()
stop()
on_detected(callback)
is_running -> bool
```

`is_running` is an addition to the three methods listed in DESIGN section 4.2.
Without it the orchestrator cannot distinguish "idle, waiting for a wake word"
from "the detector has died", and those two states look identical from the
outside: the loop simply never fires again.

This is not hypothetical. The keyboard substitute reaches end-of-stream when
stdin is not interactive — piped input, a service manager with no controlling
terminal — and its reader thread exits immediately. A loop that only watches
its own stop flag then spins forever without ever being able to accept a turn,
which DESIGN section 8 forbids ("the assistant must not remain stuck").

DESIGN section 8 also lists "wake-word engine restart" as a recoverable
failure, which likewise requires the loop to be able to observe that the engine
is no longer running.

### Consequences

- The interface (`start` / `stop` / `on_detected` / `is_running`) is fixed now;
  the engine behind it is not.
- `run_forever` exits when the detector stops on its own, instead of spinning.
- `ponzu start` refuses to start the keyboard substitute on a non-interactive
  stdin, rather than appearing to run while being unable to ever respond.
- The MVP ships a keyboard-triggered detector as the default so `ponzu start`
  is runnable, and an always-on detector for testing.
- A real acoustic engine is a drop-in replacement selected by configuration —
  no orchestrator change may be required to adopt it.
- DESIGN section 4.2's requirement "Detect ぽんず" is **not** satisfied by the
  MVP. This is a known, recorded gap rather than an oversight.

---

## ADR-011: Three-Command MVP Surface

- **Status:** Accepted
- **Decision:** The first milestone is complete when `ponzu doctor`,
  `ponzu chat`, and `ponzu start` work. The MVP is CLI-only.

### Context

The voice loop depends on four external things — a microphone, an STT model,
Ollama, and VOICEVOX — any of which may be missing or misconfigured. A voice
interface is also a poor channel for diagnosing its own failures.

### Consequences

- `ponzu doctor` probes every adapter independently and reports status without
  starting the loop. It is the supported way to diagnose setup problems.
- `ponzu chat` exercises orchestration, prompting, and the LLM adapter with no
  audio hardware, which also makes those layers testable in CI.
- `ponzu start` is the full loop and is the last of the three to be reachable.
- The menu bar application from ROADMAP Phase 6 is confirmed out of MVP scope.
- `ponzu start` runs the same probes as `doctor` before entering the loop and
  refuses to start if any component fails. Every turn uses the same adapters,
  so a component already known to be unavailable yields a loop that can only
  fail — identically, on every wake — which is the stuck behaviour DESIGN
  section 8 forbids. The probes are cheap: no model is loaded and no stream is
  opened.
- The loop prints each turn's outcome to the terminal. DESIGN section 8 step 2
  asks for a message on failure, and the structured log deliberately records
  only an error *category* (section 7), which is not enough for a user to know
  what to fix.

---

## ADR-012: `qwen3:30b` as the Default Local LLM

- **Status:** Accepted
- **Decision:** The default local model stays **`qwen3:30b`**. Conversational
  latency is addressed by streaming the response, not by choosing a smaller
  model.

### Context

DESIGN section 11 left the default local LLM open. Measured end to end on an
M1 Max / 64 GB, the LLM is the largest stage of a turn at 4.0-4.8 s, out of
4.9-6.4 s from the user finishing speaking to the reply beginning.

The obvious reading — "the model is too big, use a smaller one" — does not
survive looking at what the model actually is:

```text
family            : qwen3moe
parameter_size    : 30.5B
expert_count      : 128
expert_used_count : 8
```

`qwen3:30b` is Qwen3-30B-A3B: a mixture-of-experts model with roughly 3B
active parameters per token. It already delivers 30B-class quality at
3B-class speed, so it sits on the efficiency frontier rather than above it.
A dense model of comparable quality would be several times slower, and a
smaller dense model would give up quality without a matching speed win
against 3B active.

An assistant that answers quickly but poorly is not a usable assistant. The
conversational requirement in section 2 is about how long the user waits
before hearing *something*, which is not the same as how long generation takes.

### Consequences

- `llm.model` stays `qwen3:30b`; `llm.timeout_s` stays sized for the ~27 s
  cold load rather than the ~4-5 s steady state.
- Perceived latency is a **streaming** problem (ROADMAP Phase 2: streaming LLM
  output, sentence-level TTS). Emitting the first sentence to VOICEVOX while
  the rest is still generating should cut time-to-first-audio to 1-2 s without
  touching model choice.
- Reasoning stays enabled. ADR-009's note holds: Ollama's `think: false` makes
  this model leak its reasoning into `message.content`.
- The persona's response-length constraint (DESIGN section 4.6) is
  load-bearing for latency, not only for tone. Without it the same model
  produced ~6,800 characters of reasoning and took ~35 s. Loosening it has a
  measurable latency cost.
- Machines much smaller than 64 GB are out of scope for this default.
