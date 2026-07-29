# Ponzu / ぽんず

A local-first voice assistant for macOS on Apple Silicon.

Ponzu listens for a wake word, transcribes what you say, answers with a locally
running LLM, and speaks the reply through VOICEVOX. Audio, transcripts, and
conversations stay on the machine — the MVP requires no cloud API and no paid
inference service.

> **Status: early development.** The voice loop runs end to end, but acoustic
> wake-word detection is not implemented yet — see [Known gaps](#known-gaps).

## Design documents

The documents under [`docs/`](docs/) are the specification, not background
reading. Implementation follows them, and when the two diverge the document is
updated first.

| Document | Contents |
| --- | --- |
| [`docs/ADR.md`](docs/ADR.md) | Architecture decisions and their rationale |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Component design, configuration, error handling |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased development plan |
| [`docs/SECURITY.md`](docs/SECURITY.md) | What may and may not be committed |

## Requirements

- macOS on Apple Silicon (ADR-009)
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally, for inference
- [VOICEVOX Engine](https://voicevox.hiroshiba.jp/) running locally, for speech

Ollama and VOICEVOX should stay bound to loopback. DESIGN section 9 requires
they not be exposed to the LAN by default.

## Install

```sh
git clone https://github.com/ochanuco/ponzu.git
cd ponzu
uv sync --extra audio --extra stt
```

The `audio` and `stt` extras pull in `sounddevice`, `numpy`, and
`faster-whisper`. They are optional on purpose: `uv sync` alone is enough to
install the package and run the test suite on a machine with no audio stack.

## Usage

```sh
uv run ponzu doctor   # check configuration and every dependency
uv run ponzu chat     # text-only conversation, no audio hardware needed
uv run ponzu start    # the full voice loop
```

Start with `doctor`. It probes the config file, microphone, speaker, STT model,
Ollama, and VOICEVOX independently and reports what is missing, without
starting the loop.

`chat` exercises the same orchestration and prompt layer as `start` over stdin
and stdout, which makes it the fastest way to check that the model and persona
behave before involving audio. It also takes a single utterance for scripting,
and can speak its replies while still taking typed input:

```sh
uv run ponzu chat "こんにちは"   # one turn, then exit
uv run ponzu chat --speak       # typed input, spoken reply
```

## Configuration

Ponzu runs with built-in defaults and no configuration file. To customise it:

```sh
uv run ponzu doctor --write-config   # writes a starting config.yaml
```

It will not overwrite an existing file. Configuration lives in the user data
directory, never in the repository (ADR-006):

```
~/Library/Application Support/Ponzu/
├── config.yaml
├── logs/
├── cache/
└── models/
```

Set `PONZU_DATA_DIR` to relocate it, or `PONZU_CONFIG` to point at a specific
file. [`config/default.example.yaml`](config/default.example.yaml) documents
every key.

## Privacy

Defaults, from DESIGN section 5.1 and SECURITY.md:

- Audio is not written to disk.
- Transcripts are not persisted.
- Conversations are not persisted; history is an in-memory window that ends
  with the process.
- Logs record timings, model names, and character counts — never transcripts,
  prompts, model output, or credentials. A redaction filter enforces this
  rather than relying on call sites being careful.

## Known gaps

- **Wake word.** `ponzu start` currently triggers on Enter, not on hearing
  "ぽんず". ADR-010 accepts this for the MVP on the condition that the
  substitute sits behind the same `WakeWordDetector` interface a real acoustic
  engine will use.
- Streaming LLM output, barge-in, and voice activity detection are ROADMAP
  Phase 2 and not implemented.
- Skills, memory, and Discord are out of MVP scope by ADR-003 and ROADMAP.

## License

Not yet chosen (DESIGN section 11). Until one is added, no license is granted.
