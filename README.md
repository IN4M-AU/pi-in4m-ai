# pi-in4m-ai

A [pi](https://github.com/earendil-works/pi-mono) extension that registers the
**IN4M AI** server (`https://ai.in4m.au`) as a model provider.

The server speaks the **OpenAI Chat Completions API** (`/v1/chat/completions`,
`/v1/models`), so this extension uses pi's built-in `openAICompletionsApi()`
streaming implementation. Tool calling, streaming, usage accounting, and
context-window handling flow through pi's standard OpenAI-completions path —
no custom streaming code.

## Features

- **Provider `in4m-ai`** in pi's model list, with display name "IN4M AI".
- **`/login in4m-ai`** — interactive API-key login (prompted for a secret).
- **`IN4M_AI_API_KEY` env-var fallback** for headless / CI use.
- **Dynamic model discovery** from `GET /v1/models` after the key is
  configured. The catalog is persisted by pi and is available to
  `pi --list-models` and `/model`.
- **`/in4m-ai`** slash command — shows auth source + discovered model count.

## Install

**As a pi package (recommended — auto-loads, updatable via `pi update`):**

```bash
pi install git:github.com/IN4M-AU/pi-in4m-ai
# or: pi install https://github.com/IN4M-AU/pi-in4m-ai
```

The extension loads on the next pi start. Update later with:

```bash
pi update --extension git:github.com/IN4M-AU/pi-in4m-ai
```

**Or the single-file copy (no package manager):**

```bash
git clone https://github.com/IN4M-AU/pi-in4m-ai
cp -r pi-in4m-ai/extensions ~/.pi/agent/extensions/in4m-ai
```

No build step — pi loads TypeScript via [jiti](https://github.com/unjs/jiti).
No runtime dependencies — only pi's own packages (declared as peers).

## Usage

Inside pi:

```
/login in4m-ai      # paste your IN4M AI API key (from https://ai.in4m.au/console)
/in4m-ai            # show auth source + available models
/model              # pick in4m-ai/<model>
```

Or, without `/login`:

```bash
export IN4M_AI_API_KEY=sk-...
pi
```

## How it works

The extension calls `pi.registerProvider(createProvider({ … }))` with:

- `baseUrl: "https://ai.in4m.au/v1"`
- `api: openAICompletionsApi()` — pi's built-in OpenAI Chat Completions streamer
- `auth.apiKey` with `login` (interactive secret prompt), `resolve`
  (stored key → `IN4M_AI_API_KEY` env fallback), and `check`
- `fetchModels` — `GET /v1/models` with the resolved bearer key, mapped to
  pi `Model<"openai-completions">` entries with conservative `compat` flags
  (`supportsDeveloperRole: false`, `maxTokensField: "max_tokens"`,
  `supportsStrictMode: false`, `supportsReasoningEffort: false`) so the
  gateway gets the request shape most OpenAI-compatible servers expect.

Models default to a 128K context window and 8192 max output tokens when the
server does not report `context_window` / `max_tokens` per model. Nemotron
models are recognized and named accordingly.

## License

MIT