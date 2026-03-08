# chi

`chi` is a compact, dependency-free C coding-agent CLI runtime:

- queued user messages
- resumable sessions via `--session`
- tool-call loop (`assistant -> tool -> assistant`)
- switchable backends (`openai`, `chatgpt`)
- built-in `bash` + freeform `apply_patch` tools
- optional model reasoning summaries printed as `[thinking]` (when provided by the API)

It is intentionally small.

## Build

```bash
cd chi
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L chi.c apply_patch.c -o chi
```

## Run Live Agent

```bash
OPENAI_API_KEY=$OPENAI_API_KEY ./chi \
  --model gpt-5.2-codex \
  "Use bash to create hello.py and run it with uv run hello.py" \
  .
```

Use `--system-prompt-file` (or `CHI_SYSTEM_PROMPT_FILE`) to load a custom system prompt from a file. `chi` injects it as a literal `system` message in the request `input`:

```bash
./chi --system-prompt-file prompts/system/codex-lite-ed.txt \
  "Edit hello.py to print hi and run it" \
  .
```

`chi` uses `OPENAI_API_KEY` from your shell env and requires `curl`.
It forces `curl` retries off (`--retry 0`) and ignores user/global curl configs (`-q`)
so provider failures fail fast instead of looping.

Every completed model response now prints the session id after `[final]`.
`chi` stores session state in `.chi-sessions/` by default, or `CHI_SESSION_DIR` if set.

Resume an existing thread by passing the session id back in:

```bash
./chi --session session-abc123 \
  "Continue and add tests for the change" \
  .
```

You can combine that with the existing queueing support:

```bash
./chi --session session-abc123 \
  --queue "Then summarize what changed" \
  "Fix the failing test first" \
  .
```

## Backends And Auth

- `openai` backend (default): set `OPENAI_API_KEY`
- `chatgpt` backend: set `CHATGPT_ACCESS_TOKEN` or `CHATGPT_SESSION_TOKEN`
- optional network tuning:
  - `CHI_MODEL` (same as passing `--model`, default `gpt-5.2-codex`)
  - `CHI_REASONING_EFFORT` (same as passing `--reasoning`, default `high`)
  - `CHI_SESSION_DIR` (session state dir, default `.chi-sessions`)
  - `CHI_SYSTEM_PROMPT_FILE` (custom system prompt text file)
  - `CHI_HTTP_CONNECT_TIMEOUT` (default `5`)
  - `CHI_HTTP_MAX_TIME` (default `120`)

Example:

```bash
CHI_BACKEND=chatgpt CHATGPT_ACCESS_TOKEN=$CHATGPT_ACCESS_TOKEN ./chi "List files" .
```
