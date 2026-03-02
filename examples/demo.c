#include <stdio.h>
#include <string.h>

#include "chi.h"

typedef struct {
  int round;
} demo_provider_state;

static chi_status demo_provider(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  demo_provider_state *state = (demo_provider_state *)user_data;
  (void)request;

  memset(response, 0, sizeof(*response));
  state->round += 1;

  if (state->round == 1) {
    static const chi_tool_call calls[] = {
        {.id = "call_1", .name = "bash", .arguments_json = "{\"command\":\"echo hello from chi\"}"},
    };
    response->tool_calls = calls;
    response->tool_call_count = 1;
    response->stop_reason = CHI_STOP_TOOL_USE;
    return CHI_OK;
  }

  response->text = "done";
  response->stop_reason = CHI_STOP_STOP;
  return CHI_OK;
}

static void print_event(const chi_event *event, void *user_data) {
  (void)user_data;
  if (event->type == CHI_EVENT_TOOL_CALL_STARTED) {
    printf("[event] tool started: %s (%s)\n", event->tool_name, event->tool_call_id);
    return;
  }
  if (event->type == CHI_EVENT_TOOL_CALL_FINISHED) {
    printf("[event] tool finished: %s (%s), error=%d\n",
           event->tool_name,
           event->tool_call_id,
           event->is_error);
    if (event->tool_result != NULL && event->tool_result->text != NULL) {
      printf("[event] tool output:\n%s\n", event->tool_result->text);
    }
    return;
  }
  if (event->type == CHI_EVENT_FINAL_MESSAGE) {
    if (event->assistant_message != NULL && event->assistant_message->text != NULL) {
      printf("[event] final assistant: %s\n", event->assistant_message->text);
    }
    if (event->error != NULL) {
      printf("[event] final error: %s\n", event->error);
    }
  }
}

int main(void) {
  chi_runtime_options options = chi_runtime_options_default();
  chi_runtime *runtime;
  demo_provider_state provider_state = {0};
  char session_id[CHI_SESSION_ID_MAX];
  size_t i;
  size_t message_count;

  options.provider = demo_provider;
  options.provider_user_data = &provider_state;
  options.system_prompt = "You are concise.";
  options.working_dir = ".";

  runtime = chi_runtime_create(&options);
  if (runtime == NULL) {
    fprintf(stderr, "failed to create runtime\n");
    return 1;
  }

  if (chi_runtime_subscribe(runtime, print_event, NULL, NULL) != CHI_OK) {
    fprintf(stderr, "failed to subscribe\n");
    chi_runtime_destroy(runtime);
    return 1;
  }

  if (chi_runtime_start_session(runtime, "run a command", session_id, sizeof(session_id)) != CHI_OK) {
    fprintf(stderr, "failed to start session\n");
    chi_runtime_destroy(runtime);
    return 1;
  }

  message_count = chi_runtime_session_message_count(runtime, session_id);
  printf("\nconversation (%zu messages):\n", message_count);
  for (i = 0; i < message_count; i++) {
    const chi_message *m = chi_runtime_session_message_at(runtime, session_id, i);
    if (m == NULL) {
      continue;
    }
    printf("%zu. %s: %s\n", i + 1, m->role == NULL ? "?" : m->role, m->text == NULL ? "" : m->text);
  }

  chi_runtime_destroy(runtime);
  return 0;
}
