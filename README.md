# chi

`chi` is a compact, dependency-free C reimplementation of `phi`'s core logic:

- session runtime with queued user messages
- tool-call loop (`assistant -> tool -> assistant`)
- event stream callbacks (`tool_call_started`, `tool_call_finished`, `final_message`)
- built-in `bash` tool

It is intentionally small and uses only `chi.h` + `chi.c` for the SDK.

## Build

```bash
cd /home/johann/code/chi
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L \
  -I. chi.c examples/demo.c -o chi_demo
```

## Run Demo

```bash
./chi_demo
```

## Run Tests

```bash
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L \
  -I. chi.c tests/test_chi.c -o test_chi
./test_chi
```

## API Shape

```c
chi_runtime_options opts = chi_runtime_options_default();
opts.provider = my_provider;
opts.system_prompt = "You are concise.";

chi_runtime *rt = chi_runtime_create(&opts);

size_t sub_id = 0;
chi_runtime_subscribe(rt, on_event, NULL, &sub_id);

char session_id[CHI_SESSION_ID_MAX];
chi_runtime_start_session(rt, "list files and summarize", session_id, sizeof(session_id));
chi_runtime_queue_message(rt, session_id, "now include hidden files too");

chi_runtime_destroy(rt);
```

## Provider Contract

Your provider callback receives:

- full conversation history
- system prompt, model id, reasoning effort
- available tool definitions

It returns one assistant response:

- final text (`CHI_STOP_STOP`), or
- tool calls (`CHI_STOP_TOOL_USE`)

`chi` then executes tool calls, appends tool-result messages, injects one queued user message at each tool boundary, and continues rounds until a final assistant response or max rounds reached.
