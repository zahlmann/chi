# chi

`chi` is a compact, dependency-free C coding-agent runtime:

- session runtime with queued user messages
- tool-call loop (`assistant -> tool -> assistant`)
- event stream callbacks (`tool_call_started`, `tool_call_finished`, `final_message`)
- built-in `bash` tool

It is intentionally small and now ships as one executable source file: `chi.c`.

## Build

```bash
cd /home/johann/code/chi
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L chi.c -o chi
```

## Run Live Agent

```bash
OPENAI_API_KEY=$OPENAI_API_KEY ./chi \
  "Use bash to create hello.py and run it with uv run hello.py" \
  /home/johann/code/chi/agent_playground
```

`chi` uses `OPENAI_API_KEY` from your shell env and requires `curl`.
