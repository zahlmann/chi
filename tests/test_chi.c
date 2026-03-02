#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chi.h"

#define CHECK(cond, msg)                  \
  do {                                    \
    if (!(cond)) {                        \
      fprintf(stderr, "FAIL: %s\n", msg); \
      return 1;                           \
    }                                     \
  } while (0)

typedef struct {
  int rounds;
  int final_count;
  int saw_injected;
  int queued_once;
  int tool_started;
  char final_text[256];
  char final_error[256];
  chi_runtime *runtime;
} test_state;

static int conversation_has_user_text(const chi_provider_request *request, const char *text) {
  size_t i;
  for (i = 0; i < request->message_count; i++) {
    const chi_message *m = &request->messages[i];
    if (m->role == NULL || strcmp(m->role, "user") != 0) {
      continue;
    }
    if (m->text != NULL && strcmp(m->text, text) == 0) {
      return 1;
    }
  }
  return 0;
}

static chi_status provider_simple_text(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  test_state *state = (test_state *)user_data;
  (void)request;
  memset(response, 0, sizeof(*response));
  state->rounds += 1;
  response->text = "ok";
  response->stop_reason = CHI_STOP_STOP;
  return CHI_OK;
}

static chi_status provider_tool_then_text(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  test_state *state = (test_state *)user_data;
  memset(response, 0, sizeof(*response));
  state->rounds += 1;

  if (state->rounds == 1) {
    static const chi_tool_call tool_calls[] = {
        {.id = "call_1", .name = "bash", .arguments_json = "{\"command\":\"echo first\"}"},
    };
    response->tool_calls = tool_calls;
    response->tool_call_count = 1;
    response->stop_reason = CHI_STOP_TOOL_USE;
    return CHI_OK;
  }

  if (conversation_has_user_text(request, "second message")) {
    state->saw_injected = 1;
  }

  response->text = "done";
  response->stop_reason = CHI_STOP_STOP;
  return CHI_OK;
}

static chi_status loop_tool_execute(
    const chi_tool_request *request,
    chi_tool_response *response,
    void *user_data) {
  (void)request;
  (void)user_data;
  memset(response, 0, sizeof(*response));
  response->text = "ok";
  return CHI_OK;
}

static chi_status provider_tool_forever(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  test_state *state = (test_state *)user_data;
  (void)request;
  memset(response, 0, sizeof(*response));
  state->rounds += 1;

  static const chi_tool_call calls[] = {
      {.id = "call_loop", .name = "loop_tool", .arguments_json = "{\"n\":1}"},
  };

  response->tool_calls = calls;
  response->tool_call_count = 1;
  response->stop_reason = CHI_STOP_TOOL_USE;
  return CHI_OK;
}

static void on_event(const chi_event *event, void *user_data) {
  test_state *state = (test_state *)user_data;

  if (event->type == CHI_EVENT_TOOL_CALL_STARTED) {
    state->tool_started = 1;
    if (!state->queued_once && state->runtime != NULL && event->session_id != NULL) {
      chi_runtime_queue_message(state->runtime, event->session_id, "second message");
      state->queued_once = 1;
    }
    return;
  }

  if (event->type == CHI_EVENT_FINAL_MESSAGE) {
    state->final_count += 1;
    state->final_text[0] = '\0';
    state->final_error[0] = '\0';
    if (event->assistant_message != NULL && event->assistant_message->text != NULL) {
      snprintf(state->final_text, sizeof(state->final_text), "%s", event->assistant_message->text);
    }
    if (event->error != NULL) {
      snprintf(state->final_error, sizeof(state->final_error), "%s", event->error);
    }
  }
}

static int test_start_session_final_event(void) {
  chi_runtime_options options = chi_runtime_options_default();
  chi_runtime *runtime;
  test_state state;
  char session_id[CHI_SESSION_ID_MAX];
  chi_status status;

  memset(&state, 0, sizeof(state));
  options.provider = provider_simple_text;
  options.provider_user_data = &state;

  runtime = chi_runtime_create(&options);
  CHECK(runtime != NULL, "runtime create");
  state.runtime = runtime;

  status = chi_runtime_subscribe(runtime, on_event, &state, NULL);
  CHECK(status == CHI_OK, "subscribe");

  status = chi_runtime_start_session(runtime, "hello", session_id, sizeof(session_id));
  CHECK(status == CHI_OK, "start session");
  CHECK(session_id[0] != '\0', "session id generated");
  CHECK(state.final_count == 1, "received final event");
  CHECK(strcmp(state.final_text, "ok") == 0, "final assistant text");

  chi_runtime_destroy(runtime);
  return 0;
}

static int test_queue_injected_after_tool_boundary(void) {
  chi_runtime_options options = chi_runtime_options_default();
  chi_runtime *runtime;
  test_state state;
  char session_id[CHI_SESSION_ID_MAX];
  chi_status status;

  memset(&state, 0, sizeof(state));
  options.provider = provider_tool_then_text;
  options.provider_user_data = &state;

  runtime = chi_runtime_create(&options);
  CHECK(runtime != NULL, "runtime create");
  state.runtime = runtime;

  status = chi_runtime_subscribe(runtime, on_event, &state, NULL);
  CHECK(status == CHI_OK, "subscribe");

  status = chi_runtime_start_session(runtime, "first message", session_id, sizeof(session_id));
  CHECK(status == CHI_OK, "start session");
  CHECK(state.tool_started == 1, "tool start event fired");
  CHECK(state.saw_injected == 1, "queued message injected before round 2");
  CHECK(state.rounds == 2, "exactly 2 rounds");
  CHECK(state.final_count == 1, "final event emitted");

  chi_runtime_destroy(runtime);
  return 0;
}

static int test_max_tool_rounds_error_path(void) {
  chi_runtime_options options = chi_runtime_options_default();
  chi_runtime *runtime;
  test_state state;
  char session_id[CHI_SESSION_ID_MAX];
  chi_status status;
  chi_tool_def loop_tool;

  memset(&state, 0, sizeof(state));
  options.provider = provider_tool_forever;
  options.provider_user_data = &state;
  options.max_tool_rounds = 2;
  options.include_bash_tool = 0;

  runtime = chi_runtime_create(&options);
  CHECK(runtime != NULL, "runtime create");
  state.runtime = runtime;

  memset(&loop_tool, 0, sizeof(loop_tool));
  loop_tool.name = "loop_tool";
  loop_tool.description = "loop";
  loop_tool.execute = loop_tool_execute;
  status = chi_runtime_add_tool(runtime, loop_tool);
  CHECK(status == CHI_OK, "add loop tool");

  status = chi_runtime_subscribe(runtime, on_event, &state, NULL);
  CHECK(status == CHI_OK, "subscribe");

  status = chi_runtime_start_session(runtime, "go", session_id, sizeof(session_id));
  CHECK(status == CHI_OK, "start session");
  CHECK(state.final_count == 1, "final event emitted");
  CHECK(state.final_error[0] != '\0', "final event carries error");
  CHECK(strstr(state.final_error, "max tool rounds") != NULL, "max rounds error text");

  chi_runtime_destroy(runtime);
  return 0;
}

int main(void) {
  if (test_start_session_final_event() != 0) {
    return 1;
  }
  if (test_queue_injected_after_tool_boundary() != 0) {
    return 1;
  }
  if (test_max_tool_rounds_error_path() != 0) {
    return 1;
  }

  printf("PASS: all chi tests\n");
  return 0;
}
