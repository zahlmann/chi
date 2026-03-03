#include <stddef.h>

typedef enum {
  CHI_OK = 0,
  CHI_ERR_INVALID_ARGUMENT,
  CHI_ERR_NOT_FOUND,
  CHI_ERR_QUEUE_FULL,
  CHI_ERR_PROVIDER_FAILED,
  CHI_ERR_TOOL_FAILED,
  CHI_ERR_OOM,
  CHI_ERR_INTERNAL
} chi_status;

typedef enum {
  CHI_STOP_STOP = 0,
  CHI_STOP_LENGTH,
  CHI_STOP_TOOL_USE,
  CHI_STOP_ERROR
} chi_stop_reason;

#define CHI_SESSION_ID_MAX 64

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
  size_t queue_capacity;
  int include_bash_tool;
  double bash_timeout_seconds;
} chi_runtime_options;

typedef struct chi_runtime chi_runtime;

const char *chi_status_string(chi_status status);
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

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHI_DEFAULT_QUEUE_CAPACITY 256
#define CHI_BASH_MAX_LINES 2000
#define CHI_BASH_MAX_BYTES (50 * 1024)

typedef struct {
  size_t id;
  chi_event_handler handler;
  void *user_data;
  int active;
} chi_subscriber;

typedef struct {
  char id[CHI_SESSION_ID_MAX];
  chi_message *messages;
  size_t message_count;
  size_t message_capacity;
  char **pending_prompts;
  size_t pending_count;
  size_t pending_capacity;
  int processing;
} chi_session;

struct chi_runtime {
  chi_provider_fn provider;
  void *provider_user_data;
  char *model_id;
  char *system_prompt;
  char *reasoning_effort;
  size_t queue_capacity;

  chi_tool_def *tools;
  size_t tool_count;
  size_t tool_capacity;

  chi_session *sessions;
  size_t session_count;
  size_t session_capacity;

  chi_subscriber *subscribers;
  size_t subscriber_count;
  size_t subscriber_capacity;
  size_t next_subscription_id;
};

typedef struct {
  int truncated;
  const char *truncated_by;
  int total_lines;
  int total_bytes;
  int output_lines;
  int output_bytes;
} chi_truncation_info;

typedef struct {
  char *output;
  int exit_code;
  int timed_out;
} chi_shell_result;

typedef struct {
  char *cwd;
  double default_timeout_seconds;
  char *scratch_text;
  char *scratch_details;
  char *scratch_error;
} chi_bash_tool_state;

static const char *k_role_user = "user";
static const char *k_role_assistant = "assistant";
static const char *k_role_tool_result = "toolResult";

static char *chi_strdup_local(const char *s) {
  size_t len;
  char *out;

  if (s == NULL) {
    return NULL;
  }
  len = strlen(s) + 1;
  out = (char *)malloc(len);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out, s, len);
  return out;
}

static char *chi_format(const char *fmt, ...) {
  va_list args;
  va_list copy;
  int needed;
  char *buf;

  va_start(args, fmt);
  va_copy(copy, args);
  needed = vsnprintf(NULL, 0, fmt, copy);
  va_end(copy);
  if (needed < 0) {
    va_end(args);
    return NULL;
  }

  buf = (char *)malloc((size_t)needed + 1);
  if (buf == NULL) {
    va_end(args);
    return NULL;
  }

  vsnprintf(buf, (size_t)needed + 1, fmt, args);
  va_end(args);
  return buf;
}

static int ensure_capacity(void **items, size_t *capacity, size_t needed, size_t item_size) {
  void *new_items;
  size_t new_capacity;

  if (needed <= *capacity) {
    return 1;
  }

  new_capacity = (*capacity == 0) ? 8 : *capacity;
  while (new_capacity < needed) {
    if (new_capacity > SIZE_MAX / 2) {
      return 0;
    }
    new_capacity *= 2;
  }

  if (item_size != 0 && new_capacity > SIZE_MAX / item_size) {
    return 0;
  }

  new_items = realloc(*items, new_capacity * item_size);
  if (new_items == NULL) {
    return 0;
  }

  *items = new_items;
  *capacity = new_capacity;
  return 1;
}

static long long now_ms(void) {
  struct timespec ts;

  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    return 0;
  }
  return (long long)ts.tv_sec * 1000LL + (long long)(ts.tv_nsec / 1000000LL);
}

static int is_blank(const char *s) {
  if (s == NULL) {
    return 1;
  }
  while (*s != '\0') {
    if (!isspace((unsigned char)*s)) {
      return 0;
    }
    s++;
  }
  return 1;
}

static void random_hex(char *dst, size_t dst_size, size_t bytes_needed) {
  static const char hex[] = "0123456789abcdef";
  unsigned char buf[32];
  size_t i;

  if (bytes_needed > sizeof(buf)) {
    bytes_needed = sizeof(buf);
  }

  memset(buf, 0, sizeof(buf));

  {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
      ssize_t got = read(fd, buf, bytes_needed);
      close(fd);
      if (got != (ssize_t)bytes_needed) {
        size_t j;
        for (j = 0; j < bytes_needed; j++) {
          buf[j] = (unsigned char)(rand() & 0xff);
        }
      }
    } else {
      size_t j;
      for (j = 0; j < bytes_needed; j++) {
        buf[j] = (unsigned char)(rand() & 0xff);
      }
    }
  }

  if (dst_size == 0) {
    return;
  }

  for (i = 0; i < bytes_needed && (i * 2 + 1) < dst_size; i++) {
    dst[i * 2] = hex[(buf[i] >> 4) & 0x0f];
    dst[i * 2 + 1] = hex[buf[i] & 0x0f];
  }

  if (i * 2 < dst_size) {
    dst[i * 2] = '\0';
  } else {
    dst[dst_size - 1] = '\0';
  }
}

static void generate_session_id(char out[CHI_SESSION_ID_MAX]) {
  char hex[33];
  random_hex(hex, sizeof(hex), 16);
  if (hex[0] == '\0') {
    snprintf(out, CHI_SESSION_ID_MAX, "session-%lld-%d", now_ms(), (int)getpid());
    return;
  }
  snprintf(out, CHI_SESSION_ID_MAX, "session-%s", hex);
}

static chi_session *find_session(chi_runtime *runtime, const char *session_id) {
  size_t i;

  if (runtime == NULL || session_id == NULL) {
    return NULL;
  }

  for (i = 0; i < runtime->session_count; i++) {
    if (strcmp(runtime->sessions[i].id, session_id) == 0) {
      return &runtime->sessions[i];
    }
  }
  return NULL;
}

static const chi_session *find_session_const(const chi_runtime *runtime, const char *session_id) {
  size_t i;

  if (runtime == NULL || session_id == NULL) {
    return NULL;
  }

  for (i = 0; i < runtime->session_count; i++) {
    if (strcmp(runtime->sessions[i].id, session_id) == 0) {
      return &runtime->sessions[i];
    }
  }
  return NULL;
}

static void free_message(chi_message *msg) {
  size_t i;

  if (msg == NULL) {
    return;
  }

  free((char *)msg->text);
  free((char *)msg->tool_call_id);
  free((char *)msg->tool_name);
  free((char *)msg->arguments_json);
  free((char *)msg->details_json);

  if (msg->tool_calls != NULL) {
    for (i = 0; i < msg->tool_call_count; i++) {
      free((char *)msg->tool_calls[i].id);
      free((char *)msg->tool_calls[i].name);
      free((char *)msg->tool_calls[i].arguments_json);
    }
    free((chi_tool_call *)msg->tool_calls);
  }

  memset(msg, 0, sizeof(*msg));
}

static chi_status clone_tool_calls(const chi_tool_call *in, size_t count, chi_tool_call **out_calls) {
  chi_tool_call *out;
  size_t i;

  *out_calls = NULL;
  if (count == 0 || in == NULL) {
    return CHI_OK;
  }

  out = (chi_tool_call *)calloc(count, sizeof(chi_tool_call));
  if (out == NULL) {
    return CHI_ERR_OOM;
  }

  for (i = 0; i < count; i++) {
    char generated_id[32];
    const char *id = in[i].id;
    const char *name = in[i].name;
    const char *args = in[i].arguments_json;

    if (is_blank(id)) {
      snprintf(generated_id, sizeof(generated_id), "call_%zu", i + 1);
      id = generated_id;
    }
    if (is_blank(name)) {
      name = "tool";
    }
    if (args == NULL) {
      args = "{}";
    }

    out[i].id = chi_strdup_local(id);
    out[i].name = chi_strdup_local(name);
    out[i].arguments_json = chi_strdup_local(args);

    if (out[i].id == NULL || out[i].name == NULL || out[i].arguments_json == NULL) {
      size_t j;
      for (j = 0; j <= i; j++) {
        free((char *)out[j].id);
        free((char *)out[j].name);
        free((char *)out[j].arguments_json);
      }
      free(out);
      return CHI_ERR_OOM;
    }
  }

  *out_calls = out;
  return CHI_OK;
}

static chi_status append_message(chi_session *session, const chi_message *src) {
  chi_message msg;

  memset(&msg, 0, sizeof(msg));

  msg.role = src->role;
  msg.text = chi_strdup_local(src->text);
  msg.tool_call_id = chi_strdup_local(src->tool_call_id);
  msg.tool_name = chi_strdup_local(src->tool_name);
  msg.arguments_json = chi_strdup_local(src->arguments_json);
  msg.details_json = chi_strdup_local(src->details_json);
  msg.stop_reason = src->stop_reason;
  msg.timestamp_ms = src->timestamp_ms;

  if (clone_tool_calls(src->tool_calls, src->tool_call_count, (chi_tool_call **)&msg.tool_calls) != CHI_OK) {
    free_message(&msg);
    return CHI_ERR_OOM;
  }
  msg.tool_call_count = src->tool_call_count;

  if ((src->text != NULL && msg.text == NULL) ||
      (src->tool_call_id != NULL && msg.tool_call_id == NULL) ||
      (src->tool_name != NULL && msg.tool_name == NULL) ||
      (src->arguments_json != NULL && msg.arguments_json == NULL) ||
      (src->details_json != NULL && msg.details_json == NULL)) {
    free_message(&msg);
    return CHI_ERR_OOM;
  }

  if (!ensure_capacity((void **)&session->messages, &session->message_capacity,
                       session->message_count + 1, sizeof(chi_message))) {
    free_message(&msg);
    return CHI_ERR_OOM;
  }

  session->messages[session->message_count] = msg;
  session->message_count++;
  return CHI_OK;
}

static chi_status append_user_message(chi_session *session, const char *prompt) {
  chi_message msg;
  memset(&msg, 0, sizeof(msg));
  msg.role = k_role_user;
  msg.text = prompt;
  msg.stop_reason = CHI_STOP_STOP;
  msg.timestamp_ms = now_ms();
  return append_message(session, &msg);
}

static void emit_event(chi_runtime *runtime, const chi_event *event) {
  size_t i;

  if (runtime == NULL || event == NULL) {
    return;
  }

  for (i = 0; i < runtime->subscriber_count; i++) {
    chi_subscriber *sub = &runtime->subscribers[i];
    if (!sub->active || sub->handler == NULL) {
      continue;
    }
    sub->handler(event, sub->user_data);
  }
}

static char *pop_pending_prompt(chi_session *session) {
  char *prompt;

  if (session->pending_count == 0) {
    return NULL;
  }

  prompt = session->pending_prompts[0];
  if (session->pending_count > 1) {
    memmove(session->pending_prompts,
            session->pending_prompts + 1,
            (session->pending_count - 1) * sizeof(char *));
  }
  session->pending_count--;
  return prompt;
}

static chi_tool_def *find_tool(chi_runtime *runtime, const char *tool_name) {
  size_t i;
  if (runtime == NULL || tool_name == NULL) {
    return NULL;
  }
  for (i = 0; i < runtime->tool_count; i++) {
    if (runtime->tools[i].name != NULL && strcmp(runtime->tools[i].name, tool_name) == 0) {
      return &runtime->tools[i];
    }
  }
  return NULL;
}

static const char *skip_ws(const char *s) {
  while (s != NULL && *s != '\0' && isspace((unsigned char)*s)) {
    s++;
  }
  return s;
}

static const char *find_json_key(const char *json, const char *key) {
  char *needle;
  size_t key_len;
  const char *p;

  if (json == NULL || key == NULL) {
    return NULL;
  }

  key_len = strlen(key);
  needle = (char *)malloc(key_len + 3);
  if (needle == NULL) {
    return NULL;
  }
  snprintf(needle, key_len + 3, "\"%s\"", key);

  p = strstr(json, needle);
  free(needle);
  if (p == NULL) {
    return NULL;
  }

  p += key_len + 2;
  p = skip_ws(p);
  if (p == NULL || *p != ':') {
    return NULL;
  }
  p++;
  return skip_ws(p);
}

static char *json_extract_string(const char *json, const char *key) {
  const char *p;
  char *out;
  size_t out_len = 0;
  size_t out_cap = 32;

  p = find_json_key(json, key);
  if (p == NULL || *p != '"') {
    return NULL;
  }
  p++;

  out = (char *)malloc(out_cap);
  if (out == NULL) {
    return NULL;
  }

  while (*p != '\0' && *p != '"') {
    char c = *p;
    if (c == '\\') {
      p++;
      if (*p == '\0') {
        break;
      }
      switch (*p) {
        case 'n':
          c = '\n';
          break;
        case 'r':
          c = '\r';
          break;
        case 't':
          c = '\t';
          break;
        case '"':
          c = '"';
          break;
        case '\\':
          c = '\\';
          break;
        default:
          c = *p;
          break;
      }
    }

    if (out_len + 2 > out_cap) {
      char *grown;
      out_cap *= 2;
      grown = (char *)realloc(out, out_cap);
      if (grown == NULL) {
        free(out);
        return NULL;
      }
      out = grown;
    }

    out[out_len++] = c;
    p++;
  }

  out[out_len] = '\0';
  return out;
}

static int json_extract_number(const char *json, const char *key, double *out_value) {
  const char *p;
  char *end;
  double value;

  if (out_value == NULL) {
    return 0;
  }

  p = find_json_key(json, key);
  if (p == NULL) {
    return 0;
  }

  errno = 0;
  value = strtod(p, &end);
  if (p == end || errno != 0) {
    return 0;
  }

  *out_value = value;
  return 1;
}

static int append_bytes(char **buffer, size_t *len, size_t *cap, const char *data, size_t data_len) {
  size_t needed;

  if (data_len == 0) {
    return 1;
  }

  needed = *len + data_len + 1;
  if (needed > *cap) {
    char *new_buf;
    size_t new_cap = (*cap == 0) ? 4096 : *cap;
    while (new_cap < needed) {
      if (new_cap > SIZE_MAX / 2) {
        return 0;
      }
      new_cap *= 2;
    }
    new_buf = (char *)realloc(*buffer, new_cap);
    if (new_buf == NULL) {
      return 0;
    }
    *buffer = new_buf;
    *cap = new_cap;
  }

  memcpy(*buffer + *len, data, data_len);
  *len += data_len;
  (*buffer)[*len] = '\0';
  return 1;
}

static char *normalize_newlines(const char *input) {
  size_t i;
  size_t len;
  char *out;
  size_t j = 0;

  if (input == NULL) {
    return chi_strdup_local("");
  }

  len = strlen(input);
  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }

  for (i = 0; i < len; i++) {
    if (input[i] == '\r') {
      if (i + 1 < len && input[i + 1] == '\n') {
        continue;
      }
      out[j++] = '\n';
    } else {
      out[j++] = input[i];
    }
  }

  out[j] = '\0';
  return out;
}

static int count_lines(const char *text) {
  int lines = 1;
  const char *p;

  if (text == NULL || *text == '\0') {
    return 1;
  }

  for (p = text; *p != '\0'; p++) {
    if (*p == '\n') {
      lines++;
    }
  }
  return lines;
}

static char *truncate_tail(
    const char *full,
    int max_lines,
    int max_bytes,
    chi_truncation_info *info) {
  size_t full_len;
  int total_lines;
  int truncated = 0;
  size_t start_byte = 0;
  size_t start_line_byte = 0;
  size_t final_start;
  const char *truncated_by = "";
  const char *p;
  int seen_lines;
  char *out;

  memset(info, 0, sizeof(*info));

  if (full == NULL) {
    full = "";
  }

  full_len = strlen(full);
  total_lines = count_lines(full);

  info->total_lines = total_lines;
  info->total_bytes = (int)full_len;

  if ((int)full_len > max_bytes) {
    truncated = 1;
    start_byte = full_len - (size_t)max_bytes;
    truncated_by = "bytes";
  }

  if (total_lines > max_lines) {
    truncated = 1;
    seen_lines = 1;
    for (p = full + full_len - 1; p >= full; p--) {
      if (*p == '\n') {
        seen_lines++;
        if (seen_lines > max_lines) {
          start_line_byte = (size_t)(p - full + 1);
          break;
        }
      }
      if (p == full) {
        break;
      }
    }
    if (strcmp(truncated_by, "") == 0) {
      truncated_by = "lines";
    }
  }

  final_start = start_byte;
  if (start_line_byte > final_start) {
    final_start = start_line_byte;
  }

  if (!truncated || final_start >= full_len) {
    out = chi_strdup_local(full);
    if (out == NULL) {
      return NULL;
    }
    info->truncated = 0;
    info->truncated_by = "";
    info->output_lines = count_lines(out);
    info->output_bytes = (int)strlen(out);
    return out;
  }

  out = chi_strdup_local(full + final_start);
  if (out == NULL) {
    return NULL;
  }

  info->truncated = 1;
  info->truncated_by = truncated_by;
  info->output_lines = count_lines(out);
  info->output_bytes = (int)strlen(out);
  return out;
}

static char *format_size(int bytes) {
  if (bytes < 1024) {
    return chi_format("%dB", bytes);
  }
  if (bytes < 1024 * 1024) {
    return chi_format("%.1fKB", (double)bytes / 1024.0);
  }
  return chi_format("%.1fMB", (double)bytes / (1024.0 * 1024.0));
}

static char *create_temp_output_file(const char *content) {
  char path[] = "/tmp/chi-bash-XXXXXX";
  int fd;
  size_t len;

  fd = mkstemp(path);
  if (fd < 0) {
    return NULL;
  }

  len = strlen(content);
  if (len > 0) {
    ssize_t wrote = write(fd, content, len);
    (void)wrote;
  }
  close(fd);

  return chi_strdup_local(path);
}

static char *json_escape(const char *input) {
  size_t i;
  size_t len;
  size_t cap;
  size_t out_len = 0;
  char *out;

  if (input == NULL) {
    return chi_strdup_local("");
  }

  len = strlen(input);
  cap = len * 2 + 16;
  out = (char *)malloc(cap);
  if (out == NULL) {
    return NULL;
  }

  for (i = 0; i < len; i++) {
    char c = input[i];
    const char *escaped = NULL;

    switch (c) {
      case '\\':
        escaped = "\\\\";
        break;
      case '"':
        escaped = "\\\"";
        break;
      case '\n':
        escaped = "\\n";
        break;
      case '\r':
        escaped = "\\r";
        break;
      case '\t':
        escaped = "\\t";
        break;
      default:
        break;
    }

    if (escaped != NULL) {
      size_t esc_len = strlen(escaped);
      if (out_len + esc_len + 1 > cap) {
        cap *= 2;
        out = (char *)realloc(out, cap);
        if (out == NULL) {
          return NULL;
        }
      }
      memcpy(out + out_len, escaped, esc_len);
      out_len += esc_len;
    } else {
      if (out_len + 2 > cap) {
        cap *= 2;
        out = (char *)realloc(out, cap);
        if (out == NULL) {
          return NULL;
        }
      }
      out[out_len++] = c;
    }
  }

  out[out_len] = '\0';
  return out;
}

static chi_status run_shell_command(
    const char *cwd,
    const char *command,
    double timeout_seconds,
    chi_shell_result *result) {
  int pipefd[2];
  pid_t pid;
  int status = 0;
  int child_done = 0;
  int read_eof = 0;
  int timed_out = 0;
  char *buffer = NULL;
  size_t buffer_len = 0;
  size_t buffer_cap = 0;
  long long start_time = now_ms();

  memset(result, 0, sizeof(*result));

  if (pipe(pipefd) != 0) {
    return CHI_ERR_INTERNAL;
  }

  pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return CHI_ERR_INTERNAL;
  }

  if (pid == 0) {
    close(pipefd[0]);

    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);

    if (cwd != NULL && *cwd != '\0') {
      if (chdir(cwd) != 0) {
        dprintf(STDERR_FILENO, "failed to chdir to %s: %s\n", cwd, strerror(errno));
        _exit(127);
      }
    }

    execl("/bin/bash", "bash", "-lc", command, (char *)NULL);
    dprintf(STDERR_FILENO, "failed to launch bash: %s\n", strerror(errno));
    _exit(127);
  }

  close(pipefd[1]);
  {
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    if (flags >= 0) {
      fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    }
  }

  while (!read_eof || !child_done) {
    fd_set rfds;
    struct timeval tv;
    int sel;

    if (timeout_seconds > 0 && !timed_out) {
      double elapsed = (double)(now_ms() - start_time) / 1000.0;
      if (elapsed > timeout_seconds) {
        kill(pid, SIGKILL);
        timed_out = 1;
      }
    }

    FD_ZERO(&rfds);
    FD_SET(pipefd[0], &rfds);
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    sel = select(pipefd[0] + 1, &rfds, NULL, NULL, &tv);

    if (sel > 0 && FD_ISSET(pipefd[0], &rfds)) {
      char chunk[4096];
      ssize_t n = read(pipefd[0], chunk, sizeof(chunk));
      if (n > 0) {
        if (!append_bytes(&buffer, &buffer_len, &buffer_cap, chunk, (size_t)n)) {
          close(pipefd[0]);
          kill(pid, SIGKILL);
          waitpid(pid, NULL, 0);
          free(buffer);
          return CHI_ERR_OOM;
        }
      } else if (n == 0) {
        read_eof = 1;
      }
    }

    if (!child_done) {
      pid_t w = waitpid(pid, &status, WNOHANG);
      if (w == pid) {
        child_done = 1;
      }
    }
  }

  close(pipefd[0]);

  if (!child_done) {
    waitpid(pid, &status, 0);
  }

  if (buffer == NULL) {
    buffer = chi_strdup_local("");
    if (buffer == NULL) {
      return CHI_ERR_OOM;
    }
  }

  result->output = buffer;
  result->timed_out = timed_out;
  if (WIFEXITED(status)) {
    result->exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result->exit_code = 128 + WTERMSIG(status);
  } else {
    result->exit_code = 1;
  }

  return CHI_OK;
}

static void bash_state_reset(chi_bash_tool_state *state) {
  if (state == NULL) {
    return;
  }
  free(state->scratch_text);
  free(state->scratch_details);
  free(state->scratch_error);
  state->scratch_text = NULL;
  state->scratch_details = NULL;
  state->scratch_error = NULL;
}

static void chi_bash_tool_destroy(void *user_data) {
  chi_bash_tool_state *state = (chi_bash_tool_state *)user_data;
  if (state == NULL) {
    return;
  }
  bash_state_reset(state);
  free(state->cwd);
  free(state);
}

static chi_status chi_bash_tool_execute(
    const chi_tool_request *request,
    chi_tool_response *response,
    void *user_data) {
  chi_bash_tool_state *state = (chi_bash_tool_state *)user_data;
  chi_shell_result shell;
  chi_truncation_info trunc;
  char *command = NULL;
  double timeout = 0;
  char *normalized = NULL;
  char *tail = NULL;
  char *full_output_path = NULL;
  char *display = NULL;
  char *escaped_command = NULL;
  char *escaped_cwd = NULL;
  char *escaped_path = NULL;
  chi_status status;

  if (response == NULL || request == NULL || state == NULL) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  memset(response, 0, sizeof(*response));
  bash_state_reset(state);
  memset(&shell, 0, sizeof(shell));

  command = json_extract_string(request->arguments_json, "command");
  if (command == NULL || is_blank(command)) {
    free(command);
    state->scratch_error = chi_strdup_local("missing required argument: command");
    state->scratch_text = chi_strdup_local("(no output)");
    response->text = state->scratch_text;
    response->error = state->scratch_error;
    return CHI_ERR_TOOL_FAILED;
  }

  timeout = state->default_timeout_seconds;
  {
    double parsed_timeout = 0;
    if (json_extract_number(request->arguments_json, "timeout", &parsed_timeout) && parsed_timeout > 0) {
      timeout = parsed_timeout;
    }
  }

  status = run_shell_command(state->cwd, command, timeout, &shell);
  if (status != CHI_OK) {
    free(command);
    state->scratch_error = chi_strdup_local("failed to execute shell command");
    state->scratch_text = chi_strdup_local("(no output)");
    response->text = state->scratch_text;
    response->error = state->scratch_error;
    return CHI_ERR_TOOL_FAILED;
  }

  normalized = normalize_newlines(shell.output);
  if (normalized == NULL) {
    free(command);
    free(shell.output);
    return CHI_ERR_OOM;
  }

  tail = truncate_tail(normalized, CHI_BASH_MAX_LINES, CHI_BASH_MAX_BYTES, &trunc);
  if (tail == NULL) {
    free(command);
    free(shell.output);
    free(normalized);
    return CHI_ERR_OOM;
  }

  if (is_blank(tail)) {
    free(tail);
    tail = chi_strdup_local("(no output)");
    if (tail == NULL) {
      free(command);
      free(shell.output);
      free(normalized);
      return CHI_ERR_OOM;
    }
  }

  display = tail;
  if (trunc.truncated) {
    char *size_label = format_size(CHI_BASH_MAX_BYTES);
    full_output_path = create_temp_output_file(normalized);
    if (size_label == NULL) {
      size_label = chi_strdup_local("50.0KB");
    }

    if (full_output_path != NULL) {
      char *note = chi_format(
          "\n\n[Showing tail (%s/%d lines limit). Full output: %s]",
          size_label,
          CHI_BASH_MAX_LINES,
          full_output_path);
      if (note != NULL) {
        char *joined = chi_format("%s%s", display, note);
        free(note);
        if (joined != NULL) {
          free(display);
          display = joined;
        }
      }
    } else {
      char *note = chi_format(
          "\n\n[Showing tail (%s/%d lines limit). Full output file could not be created]",
          size_label,
          CHI_BASH_MAX_LINES);
      if (note != NULL) {
        char *joined = chi_format("%s%s", display, note);
        free(note);
        if (joined != NULL) {
          free(display);
          display = joined;
        }
      }
    }

    free(size_label);
  }

  if (shell.timed_out) {
    char *note = chi_format("\n\nCommand timed out after %.1f seconds", timeout);
    if (note != NULL) {
      char *joined = chi_format("%s%s", display, note);
      free(note);
      if (joined != NULL) {
        free(display);
        display = joined;
      }
    }
  }

  escaped_command = json_escape(command);
  escaped_cwd = json_escape(state->cwd);
  escaped_path = json_escape(full_output_path == NULL ? "" : full_output_path);
  if (escaped_command != NULL && escaped_cwd != NULL && escaped_path != NULL) {
    state->scratch_details = chi_format(
        "{\"command\":\"%s\",\"cwd\":\"%s\",\"fullOutputPath\":\"%s\","
        "\"truncated\":%s,\"truncatedBy\":\"%s\",\"totalLines\":%d,\"totalBytes\":%d}",
        escaped_command,
        escaped_cwd,
        escaped_path,
        trunc.truncated ? "true" : "false",
        trunc.truncated_by == NULL ? "" : trunc.truncated_by,
        trunc.total_lines,
        trunc.total_bytes);
  }

  state->scratch_text = display;
  response->text = state->scratch_text;
  response->details_json = state->scratch_details;

  if (shell.timed_out) {
    state->scratch_error = chi_strdup_local("command timed out");
    response->error = state->scratch_error;
    status = CHI_ERR_TOOL_FAILED;
  } else if (shell.exit_code != 0) {
    state->scratch_error = chi_format("Command exited with code %d", shell.exit_code);
    response->error = state->scratch_error;
    status = CHI_ERR_TOOL_FAILED;
  } else {
    status = CHI_OK;
  }

  free(command);
  free(shell.output);
  free(normalized);
  free(full_output_path);
  free(escaped_command);
  free(escaped_cwd);
  free(escaped_path);

  return status;
}

chi_tool_def chi_make_bash_tool(const char *working_dir, double default_timeout_seconds) {
  chi_tool_def out;
  chi_bash_tool_state *state;
  char cwd_buf[4096];

  memset(&out, 0, sizeof(out));

  state = (chi_bash_tool_state *)calloc(1, sizeof(chi_bash_tool_state));
  if (state == NULL) {
    return out;
  }

  if (working_dir != NULL && *working_dir != '\0') {
    state->cwd = chi_strdup_local(working_dir);
  } else if (getcwd(cwd_buf, sizeof(cwd_buf)) != NULL) {
    state->cwd = chi_strdup_local(cwd_buf);
  } else {
    state->cwd = chi_strdup_local(".");
  }

  if (state->cwd == NULL) {
    free(state);
    return out;
  }

  state->default_timeout_seconds = default_timeout_seconds;

  out.name = "bash";
  out.description = "Execute a bash command in the working directory and return stdout/stderr.";
  out.execute = chi_bash_tool_execute;
  out.user_data = state;
  out.destroy_user_data = chi_bash_tool_destroy;
  return out;
}

static chi_status append_tool_result_message(
    chi_session *session,
    const char *tool_call_id,
    const char *tool_name,
    const char *arguments_json,
    const char *text,
    const char *details_json,
    int is_error,
    chi_message **out_message_ptr) {
  chi_message msg;

  memset(&msg, 0, sizeof(msg));
  msg.role = k_role_tool_result;
  msg.tool_call_id = tool_call_id;
  msg.tool_name = tool_name;
  msg.arguments_json = arguments_json;
  msg.text = text;
  msg.details_json = details_json;
  msg.stop_reason = is_error ? CHI_STOP_ERROR : CHI_STOP_STOP;
  msg.timestamp_ms = now_ms();

  if (append_message(session, &msg) != CHI_OK) {
    return CHI_ERR_OOM;
  }

  if (out_message_ptr != NULL) {
    *out_message_ptr = &session->messages[session->message_count - 1];
  }
  return CHI_OK;
}

static chi_status execute_single_tool_call(
    chi_runtime *runtime,
    chi_session *session,
    const chi_tool_call *call,
    int *is_error_out,
    chi_message **result_out) {
  chi_tool_def *tool;
  chi_tool_response tool_response;
  chi_status tool_status;
  chi_status append_status;
  const char *result_text;
  const char *details_json = NULL;
  char *owned_text = NULL;

  tool = find_tool(runtime, call->name);
  memset(&tool_response, 0, sizeof(tool_response));

  if (tool == NULL) {
    owned_text = chi_format("Tool not found: %s", call->name);
    if (owned_text == NULL) {
      return CHI_ERR_OOM;
    }
    append_status = append_tool_result_message(
        session,
        call->id,
        call->name,
        call->arguments_json,
        owned_text,
        NULL,
        1,
        result_out);
    free(owned_text);
    if (append_status != CHI_OK) {
      return append_status;
    }
    *is_error_out = 1;
    return CHI_OK;
  }

  {
    chi_tool_request request;
    memset(&request, 0, sizeof(request));
    request.session_id = session->id;
    request.tool_call_id = call->id;
    request.tool_name = call->name;
    request.arguments_json = call->arguments_json;

    tool_status = tool->execute(&request, &tool_response, tool->user_data);
  }

  result_text = tool_response.text;
  if (is_blank(result_text)) {
    result_text = "(tool returned no output)";
  }
  details_json = tool_response.details_json;

  if (tool_status != CHI_OK) {
    const char *err_text = tool_response.error;
    if (is_blank(err_text)) {
      err_text = chi_status_string(tool_status);
    }
    owned_text = chi_format("Tool execution error: %s", err_text);
    if (owned_text == NULL) {
      return CHI_ERR_OOM;
    }

    append_status = append_tool_result_message(
        session,
        call->id,
        call->name,
        call->arguments_json,
        owned_text,
        details_json,
        1,
        result_out);
    free(owned_text);
    if (append_status != CHI_OK) {
      return append_status;
    }
    *is_error_out = 1;
    return CHI_OK;
  }

  append_status = append_tool_result_message(
      session,
      call->id,
      call->name,
      call->arguments_json,
      result_text,
      details_json,
      0,
      result_out);
  if (append_status != CHI_OK) {
    return append_status;
  }

  *is_error_out = 0;
  return CHI_OK;
}

static chi_status run_turn(
    chi_runtime *runtime,
    chi_session *session,
    chi_message **assistant_out,
    char **error_out) {
  chi_message *last_assistant = NULL;

  if (assistant_out != NULL) {
    *assistant_out = NULL;
  }
  if (error_out != NULL) {
    *error_out = NULL;
  }

  if (runtime->provider == NULL) {
    if (assistant_out != NULL) {
      *assistant_out = NULL;
    }
    return CHI_OK;
  }

  for (;;) {
    chi_provider_request request;
    chi_provider_response response;
    chi_status provider_status;
    chi_message assistant_msg;
    chi_status append_status;

    memset(&request, 0, sizeof(request));
    memset(&response, 0, sizeof(response));
    memset(&assistant_msg, 0, sizeof(assistant_msg));

    request.session_id = session->id;
    request.system_prompt = runtime->system_prompt;
    request.model_id = runtime->model_id;
    request.reasoning_effort = runtime->reasoning_effort;
    request.messages = session->messages;
    request.message_count = session->message_count;
    request.tools = runtime->tools;
    request.tool_count = runtime->tool_count;

    provider_status = runtime->provider(&request, &response, runtime->provider_user_data);
    if (provider_status != CHI_OK) {
      if (error_out != NULL) {
        *error_out = chi_format("provider call failed: %s", chi_status_string(provider_status));
      }
      return CHI_ERR_PROVIDER_FAILED;
    }

    assistant_msg.role = k_role_assistant;
    assistant_msg.text = response.text;
    assistant_msg.stop_reason = response.stop_reason;
    assistant_msg.details_json = response.error_message;
    assistant_msg.tool_calls = response.tool_calls;
    assistant_msg.tool_call_count = response.tool_call_count;
    assistant_msg.timestamp_ms = now_ms();

    append_status = append_message(session, &assistant_msg);
    if (append_status != CHI_OK) {
      if (error_out != NULL) {
        *error_out = chi_strdup_local("out of memory while appending assistant message");
      }
      return append_status;
    }

    last_assistant = &session->messages[session->message_count - 1];

    if (last_assistant->tool_call_count == 0 || response.stop_reason != CHI_STOP_TOOL_USE) {
      if (assistant_out != NULL) {
        *assistant_out = last_assistant;
      }
      return CHI_OK;
    }

    {
      size_t i;
      for (i = 0; i < last_assistant->tool_call_count; i++) {
        const chi_tool_call *call = &last_assistant->tool_calls[i];
        chi_event start_event;
        chi_event end_event;
        chi_message *tool_result = NULL;
        int is_error = 0;
        chi_status exec_status;

        memset(&start_event, 0, sizeof(start_event));
        start_event.type = CHI_EVENT_TOOL_CALL_STARTED;
        start_event.session_id = session->id;
        start_event.tool_name = call->name;
        start_event.tool_call_id = call->id;
        emit_event(runtime, &start_event);

        exec_status = execute_single_tool_call(runtime, session, call, &is_error, &tool_result);
        if (exec_status != CHI_OK) {
          if (error_out != NULL) {
            *error_out = chi_strdup_local("failed to execute tool call");
          }
          return exec_status;
        }

        memset(&end_event, 0, sizeof(end_event));
        end_event.type = CHI_EVENT_TOOL_CALL_FINISHED;
        end_event.session_id = session->id;
        end_event.tool_name = call->name;
        end_event.tool_call_id = call->id;
        end_event.is_error = is_error;
        end_event.tool_result = tool_result;
        emit_event(runtime, &end_event);

        if (session->pending_count > 0) {
          char *queued = pop_pending_prompt(session);
          if (queued != NULL) {
            append_user_message(session, queued);
            free(queued);
          }
        }
      }
    }
  }
}

static void drain_session(chi_runtime *runtime, chi_session *session) {
  if (session->processing) {
    return;
  }

  session->processing = 1;

  while (session->pending_count > 0) {
    char *prompt = pop_pending_prompt(session);
    chi_status status;
    chi_message *assistant = NULL;
    char *turn_error = NULL;

    if (prompt == NULL) {
      break;
    }

    if (append_user_message(session, prompt) != CHI_OK) {
      free(prompt);
      continue;
    }
    free(prompt);

    status = run_turn(runtime, session, &assistant, &turn_error);

    if (status != CHI_OK) {
      chi_event event;
      chi_message failed;
      chi_message *failed_ptr = NULL;
      char *error_text;
      char *assistant_text;

      memset(&event, 0, sizeof(event));
      memset(&failed, 0, sizeof(failed));

      error_text = turn_error;
      if (error_text == NULL) {
        error_text = chi_strdup_local(chi_status_string(status));
      }
      assistant_text = chi_format("Assistant turn failed: %s", error_text == NULL ? "unknown error" : error_text);

      failed.role = k_role_assistant;
      failed.text = assistant_text;
      failed.stop_reason = CHI_STOP_ERROR;
      failed.details_json = error_text;
      failed.timestamp_ms = now_ms();

      if (append_message(session, &failed) == CHI_OK) {
        failed_ptr = &session->messages[session->message_count - 1];
      }

      event.type = CHI_EVENT_FINAL_MESSAGE;
      event.session_id = session->id;
      event.assistant_message = failed_ptr;
      event.error = error_text;
      emit_event(runtime, &event);

      free(assistant_text);
      free(error_text);
      continue;
    }

    if (assistant != NULL) {
      chi_event event;
      memset(&event, 0, sizeof(event));
      event.type = CHI_EVENT_FINAL_MESSAGE;
      event.session_id = session->id;
      event.assistant_message = assistant;
      emit_event(runtime, &event);
    }

    free(turn_error);
  }

  session->processing = 0;
}

static void free_session(chi_session *session) {
  size_t i;

  if (session == NULL) {
    return;
  }

  for (i = 0; i < session->message_count; i++) {
    free_message(&session->messages[i]);
  }
  free(session->messages);

  for (i = 0; i < session->pending_count; i++) {
    free(session->pending_prompts[i]);
  }
  free(session->pending_prompts);

  memset(session, 0, sizeof(*session));
}

const char *chi_status_string(chi_status status) {
  switch (status) {
    case CHI_OK:
      return "ok";
    case CHI_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case CHI_ERR_NOT_FOUND:
      return "not found";
    case CHI_ERR_QUEUE_FULL:
      return "queue full";
    case CHI_ERR_PROVIDER_FAILED:
      return "provider failed";
    case CHI_ERR_TOOL_FAILED:
      return "tool failed";
    case CHI_ERR_OOM:
      return "out of memory";
    case CHI_ERR_INTERNAL:
      return "internal error";
    default:
      return "unknown";
  }
}

chi_runtime_options chi_runtime_options_default(void) {
  chi_runtime_options options;
  memset(&options, 0, sizeof(options));
  options.model_id = "gpt-5.2-codex";
  options.reasoning_effort = "xhigh";
  options.queue_capacity = CHI_DEFAULT_QUEUE_CAPACITY;
  options.include_bash_tool = 1;
  options.bash_timeout_seconds = 0.0;
  return options;
}

chi_runtime *chi_runtime_create(const chi_runtime_options *options) {
  chi_runtime_options defaults;
  const chi_runtime_options *cfg = options;
  chi_runtime *runtime;

  defaults = chi_runtime_options_default();
  if (cfg == NULL) {
    cfg = &defaults;
  }

  runtime = (chi_runtime *)calloc(1, sizeof(chi_runtime));
  if (runtime == NULL) {
    return NULL;
  }

  runtime->provider = cfg->provider;
  runtime->provider_user_data = cfg->provider_user_data;
  runtime->queue_capacity = cfg->queue_capacity > 0 ? cfg->queue_capacity : CHI_DEFAULT_QUEUE_CAPACITY;
  runtime->next_subscription_id = 1;

  runtime->model_id = chi_strdup_local(cfg->model_id != NULL ? cfg->model_id : defaults.model_id);
  runtime->system_prompt = chi_strdup_local(cfg->system_prompt != NULL ? cfg->system_prompt : "");
  runtime->reasoning_effort = chi_strdup_local(cfg->reasoning_effort != NULL ? cfg->reasoning_effort : defaults.reasoning_effort);

  if (runtime->model_id == NULL || runtime->system_prompt == NULL || runtime->reasoning_effort == NULL) {
    chi_runtime_destroy(runtime);
    return NULL;
  }

  if (cfg->include_bash_tool) {
    chi_tool_def bash_tool = chi_make_bash_tool(cfg->working_dir, cfg->bash_timeout_seconds);
    if (bash_tool.execute != NULL) {
      chi_runtime_add_tool(runtime, bash_tool);
    }
  }

  return runtime;
}

void chi_runtime_destroy(chi_runtime *runtime) {
  size_t i;

  if (runtime == NULL) {
    return;
  }

  for (i = 0; i < runtime->tool_count; i++) {
    if (runtime->tools[i].destroy_user_data != NULL && runtime->tools[i].user_data != NULL) {
      runtime->tools[i].destroy_user_data(runtime->tools[i].user_data);
    }
    free((char *)runtime->tools[i].name);
    free((char *)runtime->tools[i].description);
  }
  free(runtime->tools);

  for (i = 0; i < runtime->session_count; i++) {
    free_session(&runtime->sessions[i]);
  }
  free(runtime->sessions);

  free(runtime->subscribers);
  free(runtime->model_id);
  free(runtime->system_prompt);
  free(runtime->reasoning_effort);

  free(runtime);
}

chi_status chi_runtime_add_tool(chi_runtime *runtime, chi_tool_def tool) {
  chi_tool_def copy;

  if (runtime == NULL || tool.execute == NULL || is_blank(tool.name)) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  if (!ensure_capacity((void **)&runtime->tools, &runtime->tool_capacity,
                       runtime->tool_count + 1, sizeof(chi_tool_def))) {
    return CHI_ERR_OOM;
  }

  memset(&copy, 0, sizeof(copy));
  copy.name = chi_strdup_local(tool.name);
  copy.description = chi_strdup_local(tool.description != NULL ? tool.description : "");
  copy.execute = tool.execute;
  copy.user_data = tool.user_data;
  copy.destroy_user_data = tool.destroy_user_data;

  if (copy.name == NULL || copy.description == NULL) {
    free((char *)copy.name);
    free((char *)copy.description);
    return CHI_ERR_OOM;
  }

  runtime->tools[runtime->tool_count] = copy;
  runtime->tool_count++;
  return CHI_OK;
}

chi_status chi_runtime_subscribe(
    chi_runtime *runtime,
    chi_event_handler handler,
    void *user_data,
    size_t *subscription_id_out) {
  chi_subscriber sub;

  if (runtime == NULL || handler == NULL) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  if (!ensure_capacity((void **)&runtime->subscribers, &runtime->subscriber_capacity,
                       runtime->subscriber_count + 1, sizeof(chi_subscriber))) {
    return CHI_ERR_OOM;
  }

  memset(&sub, 0, sizeof(sub));
  sub.id = runtime->next_subscription_id++;
  sub.handler = handler;
  sub.user_data = user_data;
  sub.active = 1;

  runtime->subscribers[runtime->subscriber_count] = sub;
  runtime->subscriber_count++;

  if (subscription_id_out != NULL) {
    *subscription_id_out = sub.id;
  }

  return CHI_OK;
}

chi_status chi_runtime_unsubscribe(chi_runtime *runtime, size_t subscription_id) {
  size_t i;

  if (runtime == NULL || subscription_id == 0) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  for (i = 0; i < runtime->subscriber_count; i++) {
    if (runtime->subscribers[i].id == subscription_id) {
      runtime->subscribers[i].active = 0;
      runtime->subscribers[i].handler = NULL;
      runtime->subscribers[i].user_data = NULL;
      return CHI_OK;
    }
  }

  return CHI_ERR_NOT_FOUND;
}

chi_status chi_runtime_start_session(
    chi_runtime *runtime,
    const char *prompt,
    char *session_id_out,
    size_t session_id_out_size) {
  chi_session session;
  int attempts;

  if (runtime == NULL || is_blank(prompt)) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  memset(&session, 0, sizeof(session));

  for (attempts = 0; attempts < 8; attempts++) {
    generate_session_id(session.id);
    if (find_session(runtime, session.id) == NULL) {
      break;
    }
  }

  if (find_session(runtime, session.id) != NULL) {
    return CHI_ERR_INTERNAL;
  }

  if (!ensure_capacity((void **)&runtime->sessions, &runtime->session_capacity,
                       runtime->session_count + 1, sizeof(chi_session))) {
    return CHI_ERR_OOM;
  }

  runtime->sessions[runtime->session_count] = session;
  runtime->session_count++;

  if (session_id_out != NULL && session_id_out_size > 0) {
    snprintf(session_id_out, session_id_out_size, "%s", session.id);
  }

  return chi_runtime_queue_message(runtime, session.id, prompt);
}

chi_status chi_runtime_queue_message(
    chi_runtime *runtime,
    const char *session_id,
    const char *prompt) {
  chi_session *session;
  char *prompt_copy;

  if (runtime == NULL || is_blank(session_id) || is_blank(prompt)) {
    return CHI_ERR_INVALID_ARGUMENT;
  }

  session = find_session(runtime, session_id);
  if (session == NULL) {
    return CHI_ERR_NOT_FOUND;
  }

  if (session->pending_count >= runtime->queue_capacity) {
    return CHI_ERR_QUEUE_FULL;
  }

  prompt_copy = chi_strdup_local(prompt);
  if (prompt_copy == NULL) {
    return CHI_ERR_OOM;
  }

  if (!ensure_capacity((void **)&session->pending_prompts, &session->pending_capacity,
                       session->pending_count + 1, sizeof(char *))) {
    free(prompt_copy);
    return CHI_ERR_OOM;
  }

  session->pending_prompts[session->pending_count] = prompt_copy;
  session->pending_count++;

  drain_session(runtime, session);
  return CHI_OK;
}

size_t chi_runtime_session_message_count(
    const chi_runtime *runtime,
    const char *session_id) {
  const chi_session *session = find_session_const(runtime, session_id);
  if (session == NULL) {
    return 0;
  }
  return session->message_count;
}

const chi_message *chi_runtime_session_message_at(
    const chi_runtime *runtime,
    const char *session_id,
    size_t index) {
  const chi_session *session = find_session_const(runtime, session_id);
  if (session == NULL || index >= session->message_count) {
    return NULL;
  }
  return &session->messages[index];
}

typedef struct {
  char *response_json;
  char *output_text;
  char *action_json;
  char *tool_args_json;
  char *tool_call_id;
  char *final_text;
  char *error_text;
  chi_tool_call tool_call;
  unsigned long long call_seq;
} chi_openai_state;

static int chi_cli_debug_enabled(void) {
  const char *v = getenv("CHI_DEBUG");
  return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

static void chi_openai_state_reset(chi_openai_state *state) {
  if (state == NULL) {
    return;
  }
  free((char *)state->tool_call.name);
  free(state->response_json);
  free(state->output_text);
  free(state->action_json);
  free(state->tool_args_json);
  free(state->tool_call_id);
  free(state->final_text);
  free(state->error_text);
  memset(state, 0, sizeof(*state));
}

static int chi_cli_append(char **buf, size_t *len, size_t *cap, const char *text) {
  size_t n;
  size_t needed;
  char *grown;

  if (text == NULL) {
    text = "";
  }
  n = strlen(text);
  needed = *len + n + 1;
  if (needed > *cap) {
    size_t new_cap = (*cap == 0) ? 512 : *cap;
    while (new_cap < needed) {
      if (new_cap > (size_t)-1 / 2) {
        return 0;
      }
      new_cap *= 2;
    }
    grown = (char *)realloc(*buf, new_cap);
    if (grown == NULL) {
      return 0;
    }
    *buf = grown;
    *cap = new_cap;
  }
  memcpy(*buf + *len, text, n);
  *len += n;
  (*buf)[*len] = '\0';
  return 1;
}

static char *chi_cli_parse_json_string(const char *p) {
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (p == NULL || *p != '"') {
    return NULL;
  }
  p++;

  while (*p != '\0' && *p != '"') {
    char c = *p;
    if (c == '\\') {
      p++;
      if (*p == '\0') {
        break;
      }
      switch (*p) {
        case 'n':
          c = '\n';
          break;
        case 'r':
          c = '\r';
          break;
        case 't':
          c = '\t';
          break;
        case '"':
          c = '"';
          break;
        case '\\':
          c = '\\';
          break;
        default:
          c = *p;
          break;
      }
    }

    {
      char one[2];
      one[0] = c;
      one[1] = '\0';
      if (!chi_cli_append(&out, &len, &cap, one)) {
        free(out);
        return NULL;
      }
    }

    p++;
  }

  if (out == NULL) {
    out = chi_strdup_local("");
  }
  return out;
}

static int chi_cli_write_file(const char *path, const char *content) {
  FILE *f;
  size_t n;

  f = fopen(path, "wb");
  if (f == NULL) {
    return 0;
  }
  n = strlen(content);
  if (n > 0 && fwrite(content, 1, n, f) != n) {
    fclose(f);
    return 0;
  }
  fclose(f);
  return 1;
}

static char *chi_cli_read_file(const char *path) {
  FILE *f;
  char *buf;
  long size;
  size_t got;

  f = fopen(path, "rb");
  if (f == NULL) {
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  size = ftell(f);
  if (size < 0) {
    fclose(f);
    return NULL;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }

  buf = (char *)malloc((size_t)size + 1);
  if (buf == NULL) {
    fclose(f);
    return NULL;
  }

  got = fread(buf, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) {
    free(buf);
    return NULL;
  }
  buf[size] = '\0';
  return buf;
}

static int chi_cli_curl_responses(const char *request_json, char **response_body, int *http_code, char **err_out) {
  char req_path[] = "/tmp/chi-openai-req-XXXXXX";
  char resp_path[] = "/tmp/chi-openai-resp-XXXXXX";
  int req_fd;
  int resp_fd;
  FILE *pipe;
  int status;
  char status_buf[64];
  char *cmd;

  *response_body = NULL;
  *http_code = 0;
  *err_out = NULL;

  req_fd = mkstemp(req_path);
  if (req_fd < 0) {
    *err_out = chi_strdup_local("failed to create request temp file");
    return 0;
  }
  close(req_fd);

  resp_fd = mkstemp(resp_path);
  if (resp_fd < 0) {
    unlink(req_path);
    *err_out = chi_strdup_local("failed to create response temp file");
    return 0;
  }
  close(resp_fd);

  if (!chi_cli_write_file(req_path, request_json)) {
    unlink(req_path);
    unlink(resp_path);
    *err_out = chi_strdup_local("failed to write request body");
    return 0;
  }

  cmd = (char *)malloc(strlen(req_path) + strlen(resp_path) + 512);
  if (cmd == NULL) {
    unlink(req_path);
    unlink(resp_path);
    *err_out = chi_strdup_local("out of memory building curl command");
    return 0;
  }

  snprintf(
      cmd,
      strlen(req_path) + strlen(resp_path) + 512,
      "curl -sS -o '%s' -w '%%{http_code}' https://api.openai.com/v1/responses "
      "-H \"Authorization: Bearer $OPENAI_API_KEY\" "
      "-H 'Content-Type: application/json' --data-binary '@%s'",
      resp_path,
      req_path);

  pipe = popen(cmd, "r");
  if (pipe == NULL) {
    free(cmd);
    unlink(req_path);
    unlink(resp_path);
    *err_out = chi_strdup_local("failed to run curl");
    return 0;
  }

  memset(status_buf, 0, sizeof(status_buf));
  if (fgets(status_buf, sizeof(status_buf), pipe) == NULL) {
    status_buf[0] = '\0';
  }

  status = pclose(pipe);
  free(cmd);

  *response_body = chi_cli_read_file(resp_path);
  unlink(req_path);
  unlink(resp_path);

  if (*response_body == NULL) {
    *err_out = chi_strdup_local("failed to read curl response body");
    return 0;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    *err_out = chi_strdup_local("curl request failed");
    free(*response_body);
    *response_body = NULL;
    return 0;
  }

  *http_code = atoi(status_buf);
  return 1;
}

static char *chi_cli_extract_output_text(const char *json) {
  const char *p;
  const char *next_text;
  const char *v;
  char *text;

  text = json_extract_string(json, "output_text");
  if (text != NULL && text[0] != '\0') {
    return text;
  }
  free(text);

  p = json;
  while (p != NULL) {
    p = strstr(p, "\"output_text\"");
    if (p == NULL) {
      break;
    }
    next_text = strstr(p, "\"text\"");
    if (next_text != NULL) {
      v = strchr(next_text, ':');
      if (v != NULL) {
        v = skip_ws(v + 1);
        if (v != NULL && *v == '"') {
          char *parsed = chi_cli_parse_json_string(v);
          if (parsed != NULL && parsed[0] != '\0') {
            return parsed;
          }
          free(parsed);
        }
      }
    }
    p += 12;
  }

  return NULL;
}

static char *chi_cli_extract_json_object(const char *text) {
  const char *start;
  const char *end;
  size_t len;
  char *out;

  if (text == NULL) {
    return NULL;
  }

  start = strchr(text, '{');
  end = strrchr(text, '}');
  if (start == NULL || end == NULL || end < start) {
    return NULL;
  }

  len = (size_t)(end - start + 1);
  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out, start, len);
  out[len] = '\0';
  return out;
}

static char *chi_cli_serialize_conversation(const chi_provider_request *request) {
  size_t i;
  char *buf = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (!chi_cli_append(&buf, &len, &cap, "Conversation so far:\n")) {
    free(buf);
    return NULL;
  }

  for (i = 0; i < request->message_count; i++) {
    const chi_message *m = &request->messages[i];
    const char *role = (m->role == NULL) ? "unknown" : m->role;
    const char *text = (m->text == NULL) ? "" : m->text;

    if (!chi_cli_append(&buf, &len, &cap, "- ") ||
        !chi_cli_append(&buf, &len, &cap, role) ||
        !chi_cli_append(&buf, &len, &cap, ": ") ||
        !chi_cli_append(&buf, &len, &cap, text) ||
        !chi_cli_append(&buf, &len, &cap, "\n")) {
      free(buf);
      return NULL;
    }

    if (m->arguments_json != NULL && m->arguments_json[0] != '\0') {
      if (!chi_cli_append(&buf, &len, &cap, "  args=") ||
          !chi_cli_append(&buf, &len, &cap, m->arguments_json) ||
          !chi_cli_append(&buf, &len, &cap, "\n")) {
        free(buf);
        return NULL;
      }
    }
  }

  if (!chi_cli_append(&buf, &len, &cap, "\nDecide next step with JSON only.\n")) {
    free(buf);
    return NULL;
  }

  return buf;
}

static chi_status chi_openai_provider(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  chi_openai_state *state = (chi_openai_state *)user_data;
  const char *model;
  const char *reasoning;
  const char *instructions =
      "You are a coding agent controller with one tool: bash. "
      "Return only JSON with no markdown.\n"
      "Schema:\n"
      "- tool: {\"kind\":\"tool\",\"name\":\"bash\",\"arguments\":{\"command\":\"...\",\"timeout\":number(optional)}}\n"
      "- final: {\"kind\":\"final\",\"text\":\"...\"}\n"
      "Rules:\n"
      "- Use bash to write/run files.\n"
      "- Use uv run for python.\n"
      "- Once task is complete, return final.\n";
  char *conversation = NULL;
  char *esc_model = NULL;
  char *esc_reasoning = NULL;
  char *esc_instructions = NULL;
  char *esc_conversation = NULL;
  char *request_json = NULL;
  char *resp_body = NULL;
  char *curl_err = NULL;
  char *kind = NULL;
  int http_code = 0;

  memset(response, 0, sizeof(*response));
  chi_openai_state_reset(state);

  if (getenv("OPENAI_API_KEY") == NULL || getenv("OPENAI_API_KEY")[0] == '\0') {
    state->error_text = chi_strdup_local("OPENAI_API_KEY is not set");
    return CHI_ERR_PROVIDER_FAILED;
  }

  model = (request->model_id != NULL && request->model_id[0] != '\0') ? request->model_id : "gpt-5.2-codex";
  reasoning = (request->reasoning_effort != NULL && request->reasoning_effort[0] != '\0') ? request->reasoning_effort : "high";

  conversation = chi_cli_serialize_conversation(request);
  esc_model = json_escape(model);
  esc_reasoning = json_escape(reasoning);
  esc_instructions = json_escape(instructions);
  esc_conversation = json_escape(conversation);

  if (conversation == NULL || esc_model == NULL || esc_reasoning == NULL ||
      esc_instructions == NULL || esc_conversation == NULL) {
    free(conversation);
    free(esc_model);
    free(esc_reasoning);
    free(esc_instructions);
    free(esc_conversation);
    return CHI_ERR_OOM;
  }

  request_json = (char *)malloc(strlen(esc_model) + strlen(esc_reasoning) + strlen(esc_instructions) +
                                strlen(esc_conversation) + 512);
  if (request_json == NULL) {
    free(conversation);
    free(esc_model);
    free(esc_reasoning);
    free(esc_instructions);
    free(esc_conversation);
    return CHI_ERR_OOM;
  }

  snprintf(request_json,
           strlen(esc_model) + strlen(esc_reasoning) + strlen(esc_instructions) + strlen(esc_conversation) + 512,
           "{"
           "\"model\":\"%s\","
           "\"input\":["
           "{\"role\":\"system\",\"content\":[{\"type\":\"input_text\",\"text\":\"%s\"}]},"
           "{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"%s\"}]}"
           "],"
           "\"reasoning\":{\"effort\":\"%s\"},"
           "\"max_output_tokens\":900"
           "}",
           esc_model,
           esc_instructions,
           esc_conversation,
           esc_reasoning);

  free(conversation);
  free(esc_model);
  free(esc_reasoning);
  free(esc_instructions);
  free(esc_conversation);

  if (!chi_cli_curl_responses(request_json, &resp_body, &http_code, &curl_err)) {
    state->error_text = curl_err != NULL ? curl_err : chi_strdup_local("curl call failed");
    if (chi_cli_debug_enabled()) {
      fprintf(stderr, "[chi debug] curl failed: %s\n", state->error_text == NULL ? "(unknown)" : state->error_text);
    }
    free(request_json);
    return CHI_ERR_PROVIDER_FAILED;
  }
  free(request_json);

  state->response_json = resp_body;

  if (http_code < 200 || http_code >= 300) {
    state->error_text = chi_format("openai http %d: %s", http_code, resp_body);
    if (chi_cli_debug_enabled()) {
      fprintf(stderr, "[chi debug] http_code=%d body=%s\n", http_code, resp_body);
    }
    return CHI_ERR_PROVIDER_FAILED;
  }

  state->output_text = chi_cli_extract_output_text(resp_body);
  if (state->output_text == NULL) {
    state->error_text = chi_strdup_local("could not parse output text from openai response");
    if (chi_cli_debug_enabled()) {
      fprintf(stderr, "[chi debug] parse output_text failed; body=%s\n", resp_body);
    }
    return CHI_ERR_PROVIDER_FAILED;
  }

  state->action_json = chi_cli_extract_json_object(state->output_text);
  if (state->action_json == NULL) {
    state->final_text = chi_strdup_local(state->output_text);
    if (state->final_text == NULL) {
      return CHI_ERR_OOM;
    }
    response->text = state->final_text;
    response->stop_reason = CHI_STOP_STOP;
    return CHI_OK;
  }

  kind = json_extract_string(state->action_json, "kind");
  if (kind != NULL && strcmp(kind, "tool") == 0) {
    char *tool_name = json_extract_string(state->action_json, "name");
    char *command = json_extract_string(state->action_json, "command");
    char *escaped = NULL;
    double timeout = 0;
    int has_timeout = json_extract_number(state->action_json, "timeout", &timeout);

    if (tool_name == NULL || tool_name[0] == '\0') {
      free(tool_name);
      tool_name = chi_strdup_local("bash");
    }
    if (command == NULL || command[0] == '\0') {
      free(tool_name);
      free(command);
      free(kind);
      state->final_text = chi_strdup_local("tool call missing command");
      if (state->final_text == NULL) {
        return CHI_ERR_OOM;
      }
      response->text = state->final_text;
      response->stop_reason = CHI_STOP_STOP;
      return CHI_OK;
    }

    escaped = json_escape(command);
    if (escaped == NULL) {
      free(tool_name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }

    if (has_timeout && timeout > 0) {
      state->tool_args_json = chi_format("{\"command\":\"%s\",\"timeout\":%.3f}", escaped, timeout);
    } else {
      state->tool_args_json = chi_format("{\"command\":\"%s\"}", escaped);
    }
    free(escaped);

    if (state->tool_args_json == NULL) {
      free(tool_name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }

    state->call_seq++;
    state->tool_call_id = chi_format("call_%llu", (unsigned long long)state->call_seq);
    if (state->tool_call_id == NULL) {
      free(tool_name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }

    state->tool_call.id = state->tool_call_id;
    state->tool_call.name = tool_name;
    state->tool_call.arguments_json = state->tool_args_json;

    response->tool_calls = &state->tool_call;
    response->tool_call_count = 1;
    response->stop_reason = CHI_STOP_TOOL_USE;

    free(command);
    free(kind);
    return CHI_OK;
  }

  {
    char *final_text = json_extract_string(state->action_json, "text");
    if (final_text == NULL || final_text[0] == '\0') {
      free(final_text);
      state->final_text = chi_strdup_local(state->output_text);
    } else {
      state->final_text = final_text;
    }
  }

  free(kind);

  if (state->final_text == NULL) {
    return CHI_ERR_OOM;
  }

  response->text = state->final_text;
  response->stop_reason = CHI_STOP_STOP;
  return CHI_OK;
}

static void chi_cli_on_event(const chi_event *event, void *user_data) {
  (void)user_data;

  if (event->type == CHI_EVENT_TOOL_CALL_STARTED) {
    printf("[tool start] %s (%s)\n",
           event->tool_name == NULL ? "?" : event->tool_name,
           event->tool_call_id == NULL ? "?" : event->tool_call_id);
    return;
  }

  if (event->type == CHI_EVENT_TOOL_CALL_FINISHED) {
    printf("[tool done] %s (%s) error=%d\n",
           event->tool_name == NULL ? "?" : event->tool_name,
           event->tool_call_id == NULL ? "?" : event->tool_call_id,
           event->is_error);
    if (event->tool_result != NULL && event->tool_result->text != NULL) {
      printf("%s\n", event->tool_result->text);
    }
    return;
  }

  if (event->type == CHI_EVENT_FINAL_MESSAGE) {
    if (event->error != NULL && event->error[0] != '\0') {
      printf("[final error] %s\n", event->error);
    }
    if (event->assistant_message != NULL && event->assistant_message->text != NULL) {
      printf("[final]\n%s\n", event->assistant_message->text);
    }
  }
}

int main(int argc, char **argv) {
  chi_runtime_options options;
  chi_runtime *runtime;
  chi_openai_state provider;
  const char *prompt;
  const char *working_dir;
  chi_status status;
  char session_id[CHI_SESSION_ID_MAX];

  if (argc < 2) {
    fprintf(stderr,
            "usage: %s \"prompt\" [working_dir]\n"
            "example: %s \"write hello.py and run it with uv run hello.py\" ./agent_playground\n",
            argv[0],
            argv[0]);
    return 2;
  }

  if (getenv("OPENAI_API_KEY") == NULL || getenv("OPENAI_API_KEY")[0] == '\0') {
    fprintf(stderr, "OPENAI_API_KEY is not set\n");
    return 2;
  }

  prompt = argv[1];
  working_dir = (argc >= 3) ? argv[2] : ".";

  memset(&provider, 0, sizeof(provider));

  options = chi_runtime_options_default();
  options.provider = chi_openai_provider;
  options.provider_user_data = &provider;
  options.system_prompt = "You are a concise coding assistant.";
  options.working_dir = working_dir;

  runtime = chi_runtime_create(&options);
  if (runtime == NULL) {
    fprintf(stderr, "failed to create runtime\n");
    chi_openai_state_reset(&provider);
    return 1;
  }

  status = chi_runtime_subscribe(runtime, chi_cli_on_event, NULL, NULL);
  if (status != CHI_OK) {
    fprintf(stderr, "failed to subscribe: %s\n", chi_status_string(status));
    chi_runtime_destroy(runtime);
    chi_openai_state_reset(&provider);
    return 1;
  }

  status = chi_runtime_start_session(runtime, prompt, session_id, sizeof(session_id));
  if (status != CHI_OK) {
    fprintf(stderr, "start_session failed: %s\n", chi_status_string(status));
    chi_runtime_destroy(runtime);
    chi_openai_state_reset(&provider);
    return 1;
  }

  printf("session: %s\n", session_id);

  chi_runtime_destroy(runtime);
  chi_openai_state_reset(&provider);
  return 0;
}
