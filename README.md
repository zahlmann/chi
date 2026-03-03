# chi

`chi` is a compact, dependency-free C coding-agent CLI runtime:

- queued user messages
- tool-call loop (`assistant -> bash -> assistant`)
- switchable backends (`openai`, `chatgpt`)
- built-in `bash` tool

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

## Backends And Auth

- `openai` backend (default): set `OPENAI_API_KEY`
- `chatgpt` backend: set `CHATGPT_ACCESS_TOKEN` or `CHATGPT_SESSION_TOKEN`

Example:

```bash
CHI_BACKEND=chatgpt CHATGPT_ACCESS_TOKEN=$CHATGPT_ACCESS_TOKEN ./chi "List files" .
```
