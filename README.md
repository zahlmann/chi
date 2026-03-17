# chi

`chi` is a compact C coding-agent CLI runtime:

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
make
```

If the build fails with `fatal error: curl/curl.h: No such file or directory`, install the libcurl development package first:

```bash
# Debian/Ubuntu
sudo apt install libcurl4-openssl-dev pkg-config

# Fedora/RHEL
sudo dnf install libcurl-devel pkgconf-pkg-config
```

If `make` says `pkg-config not found`, install `pkg-config` first or pass the flags manually:

```bash
make CURL_CFLAGS="..." CURL_LIBS="..."
```

You can still compile manually if needed:

```bash
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L \
  $(pkg-config --cflags libcurl) \
  chi.c apply_patch.c \
  $(pkg-config --libs libcurl) \
  -o chi
```

## Run Live Agent

```bash
./chi \
  --model gpt-5.4 \
  "Use bash to create hello.py and run it with uv run hello.py" \
  .
```

Use `--system-prompt-file` (or `CHI_SYSTEM_PROMPT_FILE`) to load a custom system prompt from a file. `chi` injects it as a literal `system` message in the request `input`:

```bash
./chi --system-prompt-file prompts/system/codex-lite-ed.txt \
  "Edit hello.py to print hi and run it" \
  .
```

By default `chi` uses the `chatgpt` backend. It resolves ChatGPT auth in this order: `CHATGPT_ACCESS_TOKEN`, `~/.chi/auth.json`, then `~/.codex/auth.json` `tokens.access_token`. If no usable credentials are found, `chi` starts a device-style ChatGPT login: it prints `https://auth.openai.com/codex/device`, shows a one-time code, waits for approval, and then saves refreshable credentials in `~/.chi/auth.json`. Use `CHI_BACKEND=openai` or `--backend openai` when you want the OpenAI API path with `OPENAI_API_KEY`. `chi` requires `libcurl` at build/runtime.

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

- `chatgpt` backend (default): resolution order is `CHATGPT_ACCESS_TOKEN`, then `~/.chi/auth.json`, then `~/.codex/auth.json`
- if no usable ChatGPT credentials are found and `chi` has a terminal, it prints `https://auth.openai.com/codex/device`, shows a one-time sign-in code, waits for approval, and then saves `access_token`, `refresh_token`, `account_id`, and expiry in `~/.chi/auth.json`
- `openai` backend: set `OPENAI_API_KEY` and pass `--backend openai` or `CHI_BACKEND=openai`
- optional network tuning:
  - `CHI_MODEL` (same as passing `--model`, default `gpt-5.4`)
  - `CHI_REASONING_EFFORT` (same as passing `--reasoning`, default `high`)
  - `CHI_SESSION_DIR` (session state dir, default `.chi-sessions`)
  - `CHI_SYSTEM_PROMPT_FILE` (custom system prompt text file)
  - `CHI_HTTP_CONNECT_TIMEOUT` (default `5`)
  - `CHI_HTTP_MAX_TIME` (default `120`)

Example:

```bash
./chi "List files" .
```

OpenAI example:

```bash
CHI_BACKEND=openai OPENAI_API_KEY=$OPENAI_API_KEY ./chi "List files" .
```
