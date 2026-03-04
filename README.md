# chi

`chi` is a compact, dependency-free C coding-agent CLI runtime:

- queued user messages
- tool-call loop (`assistant -> bash -> assistant`)
- switchable backends (`openai`, `chatgpt`)
- built-in `bash` tool
- optional model reasoning summaries printed as `[thinking]` (when provided by the API)

It is intentionally small.

## Build

```bash
cd chi
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L chi.c -o chi
```

## Run Live Agent

```bash
OPENAI_API_KEY=$OPENAI_API_KEY ./chi \
  "Use bash to create hello.py and run it with uv run hello.py" \
  ./agent_playground
```

`chi` uses `OPENAI_API_KEY` from your shell env and requires `curl`.
It forces `curl` retries off (`--retry 0`) and ignores user/global curl configs (`-q`)
so provider failures fail fast instead of looping.

## Backends And Auth

- `openai` backend (default): set `OPENAI_API_KEY`
- `chatgpt` backend: set `CHATGPT_ACCESS_TOKEN` or `CHATGPT_SESSION_TOKEN`
- optional network tuning:
  - `CHI_HTTP_CONNECT_TIMEOUT` (default `5`)
  - `CHI_HTTP_MAX_TIME` (default `120`)

Example:

```bash
CHI_BACKEND=chatgpt CHATGPT_ACCESS_TOKEN=$CHATGPT_ACCESS_TOKEN ./chi "List files" .
```
