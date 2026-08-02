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
- CLI examples: `ponzu start`, `ponzu chat` (ADR-011 fixes the MVP surface
  to `doctor`, `chat`, and `start`; there is no `ponzu config`)
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
MemoryStore   -- split in two by ADR-017: persona memory and agent memory
              -- have opposite lifetimes and cannot share one store
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
- **`stop()` must not return while the detector thread is still inside its
  audio backend.** Python runs `atexit` handlers while daemon threads are
  still executing, and `sounddevice` terminates PortAudio from one. A detector
  thread still tearing its stream down then deadlocks against that teardown on
  a CoreAudio HAL mutex, and the process hangs on exit with both threads in
  `__psynch_mutexwait`. Observed and captured with `sample(1)`:

  ```text
  main    Py_Exit -> atexit -> Pa_Terminate -> AudioOutputUnitStop
                  -> std::recursive_mutex::lock()   [blocked]
  gate    FinishStoppingStream -> AudioDeviceStop_mac_imp
                  -> HALB_Mutex::Lock()             [blocked]
  ```

  The join in `stop()` therefore has to be long enough to cover a slow
  CoreAudio close, not merely long enough for the loop to notice the stop
  flag.
- The MVP ships a keyboard-triggered detector as the default so `ponzu start`
  is runnable, and an always-on detector for testing.
- A real acoustic engine is a drop-in replacement selected by configuration —
  no orchestrator change may be required to adopt it.
- DESIGN section 4.2's requirement "Detect ぽんず" was **not** satisfied by the
  MVP. ADR-013 closes that gap; the keyboard detector remains available and is
  still what `ponzu chat`-style non-audio use relies on.

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
- ~~Perceived latency is a **streaming** problem.~~ **Superseded by ADR-014's
  measurements: this was wrong.** Streaming was implemented and it moved
  time-to-first-audio by 0.16 s, because `qwen3:30b` emits its entire reasoning
  trace before the first character of `content`. Nothing downstream can start
  earlier than reasoning finishes, so the latency is a *model* property after
  all — the opposite of what this ADR concluded. See ADR-014.
- Reasoning stays enabled. ADR-009's note holds: Ollama's `think: false` makes
  this model leak its reasoning into `message.content`.
- The persona's response-length constraint (DESIGN section 4.6) is
  load-bearing for latency, not only for tone. Without it the same model
  produced ~6,800 characters of reasoning and took ~35 s. Loosening it has a
  measurable latency cost.
- Machines much smaller than 64 GB are out of scope for this default.

---

## ADR-013: Acoustic Wake Word via a Whisper Gate

- **Status:** Accepted
- **Decision:** Acoustic detection of "ぽんず" is implemented by reusing the
  existing `faster-whisper` STT behind an energy gate, rather than adding a
  dedicated wake-word engine. It becomes the default `wake_word.provider`.

### Context

ADR-010 accepted a keyboard substitute and left the real engine open. Three
dedicated options were evaluated against the constraint that the phrase is
Japanese and the machine is Apple Silicon:

| Option | Japanese | New dependency | Credential | Effort |
| --- | --- | --- | --- | --- |
| openWakeWord | pretrained models are English only | ONNX runtime | none | trains a custom model from synthetic speech |
| Porcupine | supported, inference is local | `pvporcupine` | free AccessKey | small |
| sherpa-onnx KWS | no practical Japanese model | `sherpa-onnx` | none | large |

Porcupine is the strongest on accuracy and CPU, but its AccessKey is exactly
what SECURITY.md classifies as a credential that must never be committed, and
it makes a core MVP function depend on a proprietary service registration.
openWakeWord and sherpa-onnx both require producing a Japanese model that does
not currently exist.

Meanwhile the project already ships a Japanese speech recogniser.

### Decision

The gate runs on its own thread:

```text
mic stream -> RMS above threshold? -> accumulate until silence
           -> transcribe the window with a small model
           -> phrase match -> fire the callback
```

Transcription only runs when the energy gate has already detected speech, so
silence costs almost nothing beyond reading the stream.

`wake_word.sensitivity` finally has a meaning: it is the minimum transcript
confidence accepted for a match. ADR-010 reserved the field for exactly this.

### Consequences

- No new dependency and no credential. The `stt` extra, already required for
  the voice loop, is the only thing needed.
- The gate uses its own `wake_word.model`, defaulting to `base`, kept separate
  from `stt.model` so the gate stays cheap while transcription stays accurate.
  The measurements below are why it is `base` and not `tiny`.
- Matching normalises the transcript (NFKC, drop punctuation, fold katakana to
  hiragana) and then accepts an **edit distance of 1** against the phrase, not
  an exact match. Measurements below show why exact matching does not work.
- The gate model is **`base`**, not `tiny`.
- Accuracy is worse than a purpose-built detector. Expect false negatives on a
  quiet or distant utterance and occasional false positives on similar-sounding
  speech. This is the accepted trade for shipping without a credential.
- CPU cost is proportional to how much speech is in the room, not to elapsed
  time — but a noisy room does mean continuous transcription.
- The detector owns the microphone while idling and must release it before the
  turn runs, since `voice_turn` opens its own capture stream.
- Porcupine remains a drop-in future provider if the false-positive rate proves
  unacceptable. ADR-010's interface is unchanged, so adopting it is a
  configuration change plus one new module.

### Measurements

Synthesised through VOICEVOX at 16 kHz and fed to the recogniser. **No model
size transcribes the isolated phrase correctly:**

| Model | "ぽんず" | "ねえぽんず" | warm latency |
| --- | --- | --- | --- |
| `tiny` | コンゼ | メイコンゼ | ~130 ms |
| `base` | コンズ | メイポンズ | ~280 ms |
| `small` | コンズ | メーコンズ | ~780 ms |

A two-mora word in isolation gives the model almost no context, and it lands on
コンズ/コンゼ every time. Listing observed spellings as variants would be
overfitting to one recogniser's quirks, so matching is by edit distance on the
normalised form instead — `こんず` is distance 1 from `ぽんず`.

Distance was chosen by measuring both directions against the transcriptions
above and 24 everyday phrases:

| Max distance | Detected | False positives |
| --- | --- | --- |
| 1 | 5/7 | 1/24 (`ポーズ`) |
| 2 | 7/7 | 11/24 (`こんにちは`, `こんばんは`, `そんな`, `ほんと`, …) |

Distance 2 is unusable. Distance 1 is the setting, and it is what forces the
gate model up to `base`: `tiny`'s コンゼ is distance 2 away and would never
fire.

`sensitivity` compares against `exp(mean(avg_logprob))`, which measured
~0.42 for every utterance tried — correct and incorrect alike. It is therefore
a floor against garbage, not a discriminating threshold, and the default is set
below the observed band rather than at the 0.6 the original DESIGN example
suggested. Do not read it as a probability.

---

## ADR-014: Streaming Response with Sentence-Level Synthesis

- **Status:** Accepted
- **Decision:** `ponzu start` streams the model's answer and synthesises it one
  sentence at a time, so speech begins before generation finishes.

### Context

Measured on a live turn, wake word to the reply becoming audible was ~19 s:

| Stage | Time |
| --- | --- |
| User speaking | 3.9 s |
| Trailing silence before capture ends | 1.2 s |
| STT | 1.3 s |
| **LLM** | **13.6 s** |
| Synthesis | 1.1 s |

The LLM is roughly 70% of the wait, and essentially all of it is spent waiting
for the *last* token of an answer whose first sentence was ready far earlier.
ADR-012 already concluded that this is a streaming problem rather than a
model-size problem; this is that work.

### Decision

- `LanguageModel` gains `generate_stream`. It is **additive**: `generate`
  stays, and `ponzu chat` keeps using it. Only the voice loop streams.
  Collapsing both into one iterator-returning method would have rewritten every
  existing call site, adapter, and test to buy nothing for the text path.
- The orchestrator accumulates fragments and cuts a sentence on `。`, `！`, `？`
  or a newline. Simple and predictable, and it fits the persona, which already
  instructs replies of two to three sentences (DESIGN section 4.6).
- Sentences are synthesised and played in order while generation continues.
- A failure part-way through **lets queued audio finish playing** before
  transitioning to `ERROR`. Cutting the assistant off mid-word to report a
  backend failure is worse than letting it complete the sentence it already
  committed to.

### Consequences

- Time to first audio becomes a function of the first sentence, not the whole
  answer.
- `SPEAKING` is now entered while the model is still generating, so the state
  no longer means "generation finished". ADR-008's table is unchanged —
  `THINKING -> SPEAKING` was already legal — but the trace reads differently.
- `AudioOutput` must play queued clips back to back (DESIGN section 4.8).
  Overlapping assistant speech was already forbidden; now the sequencing is
  load-bearing rather than incidental.
- `TurnMetrics.tts_ms` becomes the sum of per-sentence synthesis, and a new
  first-audio measurement is what actually reflects the improvement.
- Barge-in remains out of scope (ROADMAP Phase 2, separate).

### Measured after implementing it

Streaming shipped, and then measured against `qwen3:30b`:

```text
first content character : 28.86 s
full answer             : 29.02 s   (difference: 0.16 s)
```

Broken down by field, on a warm model:

| | Starts | Ends | Output |
| --- | --- | --- | --- |
| `thinking` | 0.22 s | 5.37 s | 1,184 chars |
| `content` | 5.44 s | — | 27 chars |

Reasoning is ~99% of the wait and it **completes before the first character of
`content` exists**. Streaming `content` therefore cannot start speech any
earlier — there is nothing to stream until the model has already finished
thinking. The answer itself is 27 characters, so there is no meaningful
generation time left to overlap with.

This refutes the premise this ADR was written on, and ADR-012's conclusion that
latency was a streaming problem rather than a model-size problem. Both were
reasoned from the shape of the pipeline instead of measured.

What follows:

- The streaming implementation is kept. It is correct, tested, and becomes the
  win it was meant to be the moment the model emits `content` progressively.
  It is simply not sufficient on its own.
- The lever that remains is the **model**. A non-reasoning model would start
  emitting `content` immediately, at which point streaming does what ADR-014
  claimed. Disabling reasoning on `qwen3:30b` is not that option: ADR-009
  records that `think: false` makes it leak reasoning into `content`, so the
  assistant would speak "Okay, the user said…" aloud.
- DESIGN section 11's default-LLM entry, closed by ADR-012, is reopened.

---

## ADR-015: Follow-Up Window and Visible Reasoning

- **Status:** Accepted
- **Decision:** After speaking, the assistant keeps listening briefly so a
  follow-up needs no second wake word. Separately, the model's reasoning is
  shown on the terminal while it is being produced.

### Context — follow-up

Every turn required saying "ぽんず" again, because `SPEAKING` returned
unconditionally to `IDLE` and only the wake gate could start a turn. That is
wrong for conversation: the natural thing after an answer is to keep talking.

### Context — visible reasoning

ADR-014 measured that `qwen3:30b` spends ~99% of a turn producing `thinking`
before the first character of `content` exists. The adapter deliberately drops
those fragments so the assistant never speaks its reasoning aloud, but dropping
them is why roughly ten seconds of every turn shows nothing but `…thinking`.

The material to fill that silence already arrives; it was simply discarded.

### Decision

- `SPEAKING -> LISTENING` becomes a legal transition, taken when a follow-up
  window is configured. If speech starts within the window a turn runs with no
  wake word; if it does not, the assistant returns to `IDLE`.
- The window reuses the existing `speech_start_timeout_ms` machinery — "wait
  this long for speech to begin, then give up" is exactly the same question.
- `generate_stream` gains an optional `on_thinking` callback. Reasoning
  fragments go there; the iterator still yields only speakable `content`. The
  adapter stays transport-only (ADR-005): it forwards what the backend sent and
  interprets nothing.
- The CLI prints reasoning to the terminal. It is **not** logged: DESIGN
  section 7 excludes model output from logs, and reasoning is model output.
  Printing to a terminal the user is already watching is a different act from
  writing it to a file.

### Consequences

- ADR-008's transition table gains one edge. `SPEAKING -> LISTENING` is only
  taken when the follow-up window is enabled, so the trace still distinguishes
  a wake-word turn from a follow-up.
- The follow-up window is a **false-trigger risk**: room noise measured above
  the RMS threshold in practice, and during the window there is no wake word
  standing between that noise and a turn. It is therefore short by default and
  can be disabled with `0`.
- A follow-up turn skips the gate entirely, so it does not pay the
  transcription the gate would have done — follow-ups are cheaper, not just
  more convenient.
- Reasoning on screen is verbose (over 1,000 characters is normal). It is on by
  default because an assistant that looks frozen for ten seconds is the worse
  failure, and it is configurable.

---

## ADR-016: `qwen3:30b-instruct` — Switching Models Instead of Switching Modes

- **Status:** Accepted
- **Supersedes:** ADR-012's model choice
- **Decision:** The default becomes **`qwen3:30b-instruct`**, the non-thinking
  variant of the same 30B-A3B model.

### Context

ADR-014 measured that the reasoning variant produces its entire thinking trace
before the first character of an answer exists, so streaming cannot start speech
any earlier. Qwen3 is designed to switch between thinking and non-thinking modes
within one model, so the obvious fix was to switch modes per turn — greetings,
the time, a status check do not need deliberation.

That does not work here, and the reason is Ollama, not Qwen3. Every documented
placement was measured against a greeting:

| | Time | Thinking | Answer |
| --- | --- | --- | --- |
| baseline | 3.9 s | 814 chars | clean |
| `/no_think` in the user message | 27.4 s | 5,196 chars | clean |
| `/no_think` at the end of system | 15.7 s | 2,932 chars | clean |
| `/no_think` at the start of system | 35.3 s | 6,499 chars | clean |
| Ollama's `think: false` | 19.5 s | 0 chars | **leaks reasoning** |

Reading Ollama's chat template for this model explains all of it: there is no
`enable_thinking` handling and no `/no_think` handling — only the assembly of a
`<think>` block. So `/no_think` reaches the model as ordinary text with nothing
to act on, and `think: false` merely stops Ollama *parsing* the thinking; the
model still generates it, and with no `<think>` block to land in it appears in
`content`. That is why the assistant said "Okay, the user said…" out loud.

The switch Qwen3 provides is not reachable through this runtime, so the tag is
the switch instead.

### Measured

Same persona, same prompts, warm model:

| | Reasoning variant | `-instruct` |
| --- | --- | --- |
| Time to first character (median) | 13.12 s | **0.23 s** |
| Thinking emitted | 843-7,330 chars | 0 |
| "今日の天気を教えて" fabricated | 0/5 | **0/5** |

57x faster to first audio with no loss of honesty — asked for the time or the
weather it still declines rather than inventing one, which is what ruled out
`qwen2.5:14b` (4 of 5 fabricated a forecast).

### Consequences

- `llm.timeout_s` stays at 120 s. Warm turns are now sub-second, but the ~27 s
  cold load is unchanged and is what the timeout has to cover.
- ADR-014's streaming becomes what it was meant to be: with `content` arriving
  immediately, sentence-level synthesis starts speech almost at once.
- Visible reasoning (ADR-015) has nothing to display on this model. The feature
  stays — it costs nothing when no fragments arrive, and it is what makes a
  reasoning model tolerable if one is ever configured again.
- Deliberation is gone as well as the wait. If a task later needs it, the
  reasoning variant is one config line away, and that is the trade being made
  knowingly rather than by default.

---

## ADR-017: Persona Memory Is Not Agent Memory

- **Status:** Accepted (design; only the layer split is implementable today)
- **Decision:** ADR-007's single `MemoryStore` is split in two. What makes
  ぽんず *ぽんず* is stored, budgeted and compiled differently from what the
  agent looks things up in. Both use the Open Knowledge Format; only one of
  them is a graph that reaches the prompt.

### Context

ADR-001 fixed the principle: "the assistant's identity remains ぽんず even if
its voice changes later." ADR-004 restated it for the voice engine. It has
never been applied to memory, and ADR-007 lists one `MemoryStore` for both.

This stopped being hypothetical. The default model changed from `qwen3:30b` to
`qwen3:30b-instruct` (ADR-016). Had embeddings or conversation summaries
existed, that swap would have invalidated them — while ADR-001 says ぽんず must
come through it unchanged. One store cannot honour both.

### The distinction

|  | Persona memory | Agent memory |
| --- | --- | --- |
| Answers | who ぽんず is | what ぽんず can look up |
| Voice | first person | third person |
| Lifetime | outlives the model and the backend | dies with the model that produced it |
| Losing it | ぽんず becomes someone else | accuracy drops; rebuildable |
| Reaches the prompt | **the compiled projection, every turn** | only the retrieved slice |

The last row is the whole engineering difference. Agent memory can grow without
bound because only a retrieved slice is ever loaded. Persona memory's
*projection* is the system message — it is in every single turn. The store
behind it is never loaded whole either; the difference is that a projection has
to stand in for the entire store, where a retrieval only has to answer one
question.

That is not a style preference, it is measured. ADR-012 records that the
persona's response-length constraint is load-bearing for latency, not only for
tone: without it the same model spent ~35 s reasoning. A persona that
accumulates freely would make every future turn slower and every instruction in
it weaker.

### Store and projection

An earlier draft of this decision concluded "persona memory is one file, not a
graph", reasoning from the prompt budget. That was wrong, and the error is
worth recording because it is easy to repeat: it conflated **what is stored**
with **what is loaded**.

The budget applies to what reaches the prompt. It says nothing about what may
be kept.

```text
store       OKF graph. Grows. Never loaded whole.

              A ──┐
                  ├──▶ C          C links back to A and B
              B ──┘

projection  Compiled from the store. Hard character cap. This is the
            system message.
```

Source and build artifact. The projection stays small because it is a
projection, not because the store is small.

Collapsing A and B into C destructively — which is what the one-file design
forced — throws away three things:

- **provenance.** Nobody can tell why ぽんず behaves that way.
- **revisability.** If A turns out to be wrong, nothing points at C.
- **inspectability.** ROADMAP Phase 4 requires memory be inspectable. For a
  persona that cannot mean only the conclusions: how ぽんず came to understand
  itself *is* the character.

Keeping the lineage also changes what forgetting means, for the better:

- **forget** = drop from the projection. The store keeps it, still reachable as
  provenance. Reversible.
- **delete** = remove from the store. A privacy operation, and the one ROADMAP
  Phase 4's deletion requirement is about.

### Format: OKF

Both stores use the [Open Knowledge Format](https://okf.md/spec/) — a directory
of markdown files with YAML frontmatter, where links between documents form the
graph.

For persona memory it satisfies the requirements that follow from ADR-001:

1. **Survives implementation changes.** A file of sentences outlives a model
   swap, a backend swap and a rebuild. An embedding does not.
2. **Readable and hand-editable.** If ぽんず comes to believe something wrong
   about itself, fixing it must not require a conversation.
3. **Links express derivation**, which is exactly the provenance above.

For agent memory OKF is what it was designed for, and it buys something extra:
`oolong` is already OKF v0.1. A shared substrate means ぽんず can read the
user's notes without a translation layer, which is what ROADMAP Phase 3's
"Local notes" needs.

### Layer 3 is not version controlled

Git is the obvious reflex for something you do not want to lose, and it is the
wrong tool here. Listing what layer 3 actually needs:

| Requirement | Needs git? |
| --- | --- |
| Readable and hand-editable | no — they are just files |
| Provenance | no — the OKF links carry it |
| Revisable | no — same |
| Survives a rebuild | no — that is *backup*, not version control |
| Deletion actually deletes | **git actively prevents this** |

The only thing git adds is a chronological history, and that is precisely what
conflicts with ROADMAP Phase 4's deletion requirement: a "deleted" memory stays
in the history.

It is not even the better history. Git records *when* something changed; the
lineage links record *what it came from*. To understand how ぽんず developed,
following `C → A, B` beats reading a commit log — the graph is a semantic
history rather than a chronological one.

Where a timeline genuinely helps, OKF already has the convention: an optional
`log.md` in the bundle. Unlike git history, it can be deleted.

So layer 3 lives in the data directory as a plain OKF bundle with no repository
of its own. Losing it on a rebuild is a backup problem, and backup is the right
tool for it.

Layers 1 and 2 stay in the public repository, where git is doing its actual job
on code and shipped defaults.

### Layers

Persona memory is three layers, separated by who writes them:

```text
1. identity     code    immutable  the name, and the honesty constraints
2. character    config  human      tone, speech habits, response limits
3. relationship store   ぽんず      accumulated, with lineage, projected
```

Concretely:

| Layer | Lives in | Tracked by git |
| --- | --- | --- |
| 1 identity | `src/ponzu/core/prompt.py` | yes, public repo |
| 2 character | `config/default.example.yaml`, overridden in the user's `config.yaml` | the default yes; the override no |
| 3 relationship | `persona/` bundle in the data directory | no |

Layer 1 is what ADR-001 protects; it is not a preference, so it stays in code.
Layer 2 ships a public default and is overridden per user, so it can be changed
without a code change. Layer 3 is the OKF graph above.

**Today both layers 1 and 2 are one `DEFAULT_PERSONA` constant in code.** That
is a transitional state, not the design: layer 2 has not been extracted yet.

### Cap behaviour

When the projection reaches its cap, the compile step drops in this order:

1. layer 3 traits, least recently reinforced first
2. nothing else — layers 1 and 2 are never dropped

Layer 1 carries the honesty constraints, and silently shedding those to make
room for accumulated character is the one failure mode this must not have.

### Consequences

- `MemoryStore` becomes two interfaces. The test for which side something
  belongs to: *must it survive a model swap?*
- Layer 3 is written by the assistant, so it needs ROADMAP Phase 3's skill
  framework first — DESIGN section 10 makes `local.write` require
  confirmation. Layers 1 and 2 can be separated without it.
- The projection needs a compile step, and a cap enforced there rather than at
  write time.
- **`oolong` is read-only to ぽんず.** Reading it is the point of sharing the
  format — agent memory can consult the user's notes with no translation layer.
  Writing to it is what is forbidden, for *either* kind of memory: `oolong` is a
  git repository, so anything ぽんず put there would survive its own deletion,
  which Phase 4 does not permit. ぽんず writes only to its own `persona/` and
  `memory/` bundles under the data directory (ADR-006). A human committing
  something to `oolong` themselves is a separate act and unaffected.
- The vocabulary in `stt.initial_prompt` is **neither** kind of memory. It
  tunes how ぽんず hears, not what it is or knows, and DESIGN section 4.2
  already files per-room calibration apart from memory.
