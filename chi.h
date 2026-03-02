#ifndef CHI_H
#define CHI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHI_SESSION_ID_MAX 64

typedef enum {
  CHI_OK = 0,
  CHI_ERR_INVALID_ARGUMENT,
  CHI_ERR_NOT_FOUND,
  CHI_ERR_QUEUE_FULL,
  CHI_ERR_PROVIDER_FAILED,
  CHI_ERR_TOOL_FAILED,
  CHI_ERR_MAX_TOOL_ROUNDS,
  CHI_ERR_OOM,
  CHI_ERR_INTERNAL
} chi_status;

const char *chi_status_string(chi_status status);

typedef enum {
  CHI_STOP_STOP = 0,
  CHI_STOP_LENGTH,
  CHI_STOP_TOOL_USE,
  CHI_STOP_ERROR
} chi_stop_reason;

typedef struct {
  const char *id;
  const char *name;
  const char *arguments_json;
} chi_tool_call;

typedef struct chi_message {
  const char *role;
  const char *text;
  const char *tool_call_id;
  const char *tool_name;
  const char *arguments_json;
  const char *details_json;
  chi_stop_reason stop_reason;
  const chi_tool_call *tool_calls;
  size_t tool_call_count;
  long long timestamp_ms;
} chi_message;

typedef struct chi_provider_response {
  const char *text;
  chi_stop_reason stop_reason;
  const chi_tool_call *tool_calls;
  size_t tool_call_count;
  const char *error_message;
} chi_provider_response;

struct chi_tool_def;

typedef struct {
  const char *session_id;
  const char *system_prompt;
  const char *model_id;
  const char *reasoning_effort;
  const chi_message *messages;
  size_t message_count;
  const struct chi_tool_def *tools;
  size_t tool_count;
} chi_provider_request;

typedef struct {
  const char *session_id;
  const char *tool_call_id;
  const char *tool_name;
  const char *arguments_json;
} chi_tool_request;

typedef struct {
  const char *text;
  const char *details_json;
  const char *error;
} chi_tool_response;

typedef chi_status (*chi_provider_fn)(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data);

typedef chi_status (*chi_tool_execute_fn)(
    const chi_tool_request *request,
    chi_tool_response *response,
    void *user_data);

typedef void (*chi_tool_destroy_fn)(void *user_data);

typedef struct chi_tool_def {
  const char *name;
  const char *description;
  chi_tool_execute_fn execute;
  void *user_data;
  chi_tool_destroy_fn destroy_user_data;
} chi_tool_def;

typedef enum {
  CHI_EVENT_TOOL_CALL_STARTED = 0,
  CHI_EVENT_TOOL_CALL_FINISHED,
  CHI_EVENT_FINAL_MESSAGE
} chi_event_type;

typedef struct {
  chi_event_type type;
  const char *session_id;
  const char *tool_name;
  const char *tool_call_id;
  int is_error;
  const chi_message *tool_result;
  const chi_message *assistant_message;
  const char *error;
} chi_event;

typedef void (*chi_event_handler)(const chi_event *event, void *user_data);

typedef struct {
  chi_provider_fn provider;
  void *provider_user_data;
  const char *model_id;
  const char *system_prompt;
  const char *reasoning_effort;
  const char *working_dir;
  int max_tool_rounds;
  size_t queue_capacity;
  int include_bash_tool;
  double bash_timeout_seconds;
} chi_runtime_options;

typedef struct chi_runtime chi_runtime;

chi_runtime_options chi_runtime_options_default(void);
chi_runtime *chi_runtime_create(const chi_runtime_options *options);
void chi_runtime_destroy(chi_runtime *runtime);

chi_status chi_runtime_add_tool(chi_runtime *runtime, chi_tool_def tool);

chi_status chi_runtime_subscribe(
    chi_runtime *runtime,
    chi_event_handler handler,
    void *user_data,
    size_t *subscription_id_out);

chi_status chi_runtime_unsubscribe(chi_runtime *runtime, size_t subscription_id);

chi_status chi_runtime_start_session(
    chi_runtime *runtime,
    const char *prompt,
    char *session_id_out,
    size_t session_id_out_size);

chi_status chi_runtime_queue_message(
    chi_runtime *runtime,
    const char *session_id,
    const char *prompt);

size_t chi_runtime_session_message_count(
    const chi_runtime *runtime,
    const char *session_id);

const chi_message *chi_runtime_session_message_at(
    const chi_runtime *runtime,
    const char *session_id,
    size_t index);

chi_tool_def chi_make_bash_tool(const char *working_dir, double default_timeout_seconds);

#ifdef __cplusplus
}
#endif

#endif
