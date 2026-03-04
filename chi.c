#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHI_MAX_TURNS 256
#define CHI_MAX_OUTPUT_LINES 300
#define CHI_MAX_OUTPUT_BYTES (24 * 1024)
#define CHI_SESSION_ID_MAX 64
#define CHI_HTTP_CONNECT_TIMEOUT_DEFAULT 5
#define CHI_HTTP_MAX_TIME_DEFAULT 45

typedef enum {
  CHI_BACKEND_OPENAI = 0,
  CHI_BACKEND_CHATGPT = 1
} chi_backend;

typedef struct {
  char *role;
  char *text;
  char *tool_call_id;
  char *tool_name;
  char *arguments_json;
} chi_message;

typedef struct {
  chi_message *items;
  size_t count;
  size_t cap;
} chi_conversation;

typedef struct {
  char **items;
  size_t count;
  size_t cap;
} chi_prompt_queue;

typedef struct {
  char *output;
  int exit_code;
  int timed_out;
} chi_shell_result;

typedef struct {
  int is_tool;
  char *assistant_text;
  char *final_text;
  char *tool_call_id;
  char *tool_name;
  char *tool_arguments_json;
  char *tool_command;
  double timeout_seconds;
} chi_action;

typedef struct {
  chi_backend backend;
  const char *model;
  const char *reasoning_effort;
  const char *working_dir;
  int debug;
  char session_id[CHI_SESSION_ID_MAX];
  char *chatgpt_access_token;
} chi_config;

static char *chi_strdup(const char *s) {
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

static int chi_is_blank(const char *s) {
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

static int chi_equals_ignore_case(const char *a, const char *b) {
  unsigned char ca;
  unsigned char cb;

  if (a == NULL || b == NULL) {
    return 0;
  }
  while (*a != '\0' && *b != '\0') {
    ca = (unsigned char)*a++;
    cb = (unsigned char)*b++;
    if (tolower(ca) != tolower(cb)) {
      return 0;
    }
  }
  return *a == '\0' && *b == '\0';
}

static char *chi_format(const char *fmt, ...) {
  va_list ap;
  va_list ap2;
  int needed;
  char *out;

  va_start(ap, fmt);
  va_copy(ap2, ap);
  needed = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);

  if (needed < 0) {
    va_end(ap2);
    return NULL;
  }

  out = (char *)malloc((size_t)needed + 1);
  if (out == NULL) {
    va_end(ap2);
    return NULL;
  }

  vsnprintf(out, (size_t)needed + 1, fmt, ap2);
  va_end(ap2);
  return out;
}

static int chi_ensure_cap(void **ptr, size_t *cap, size_t need, size_t item_size) {
  size_t next;
  void *grown;

  if (need <= *cap) {
    return 1;
  }

  next = (*cap == 0) ? 8 : *cap;
  while (next < need) {
    if (next > (size_t)-1 / 2) {
      return 0;
    }
    next *= 2;
  }

  grown = realloc(*ptr, next * item_size);
  if (grown == NULL) {
    return 0;
  }

  *ptr = grown;
  *cap = next;
  return 1;
}

static int chi_append_n(char **buf, size_t *len, size_t *cap, const char *data, size_t n) {
  size_t need;

  if (n == 0) {
    return 1;
  }

  need = *len + n + 1;
  if (need > *cap) {
    size_t next = (*cap == 0) ? 256 : *cap;
    char *grown;
    while (next < need) {
      if (next > (size_t)-1 / 2) {
        return 0;
      }
      next *= 2;
    }
    grown = (char *)realloc(*buf, next);
    if (grown == NULL) {
      return 0;
    }
    *buf = grown;
    *cap = next;
  }

  memcpy(*buf + *len, data, n);
  *len += n;
  (*buf)[*len] = '\0';
  return 1;
}

static int chi_append(char **buf, size_t *len, size_t *cap, const char *text) {
  if (text == NULL) {
    text = "";
  }
  return chi_append_n(buf, len, cap, text, strlen(text));
}

static long long chi_now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    return 0;
  }
  return (long long)ts.tv_sec * 1000LL + (long long)(ts.tv_nsec / 1000000L);
}

static int chi_env_positive_int(const char *name, int fallback) {
  const char *raw;
  char *end = NULL;
  long value;

  if (name == NULL || fallback <= 0) {
    return fallback;
  }

  raw = getenv(name);
  if (chi_is_blank(raw)) {
    return fallback;
  }

  errno = 0;
  value = strtol(raw, &end, 10);
  if (errno != 0 || end == raw) {
    return fallback;
  }

  while (*end != '\0') {
    if (!isspace((unsigned char)*end)) {
      return fallback;
    }
    end++;
  }

  if (value <= 0 || value > 86400) {
    return fallback;
  }
  return (int)value;
}

static void chi_random_hex(char *out, size_t out_size, size_t bytes_needed) {
  static const char *hex = "0123456789abcdef";
  size_t i;
  int fd;
  unsigned char *bytes;

  if (out == NULL || out_size == 0) {
    return;
  }

  if (out_size < bytes_needed * 2 + 1) {
    out[0] = '\0';
    return;
  }

  bytes = (unsigned char *)malloc(bytes_needed);
  if (bytes == NULL) {
    out[0] = '\0';
    return;
  }

  fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0 || (size_t)read(fd, bytes, bytes_needed) != bytes_needed) {
    size_t seed = (size_t)time(NULL) ^ (size_t)getpid();
    for (i = 0; i < bytes_needed; i++) {
      seed = seed * 1103515245u + 12345u;
      bytes[i] = (unsigned char)((seed >> 16) & 0xff);
    }
  }
  if (fd >= 0) {
    close(fd);
  }

  for (i = 0; i < bytes_needed; i++) {
    out[i * 2] = hex[(bytes[i] >> 4) & 0xf];
    out[i * 2 + 1] = hex[bytes[i] & 0xf];
  }
  out[bytes_needed * 2] = '\0';
  free(bytes);
}

static void chi_make_session_id(char out[CHI_SESSION_ID_MAX]) {
  char hex[33];
  chi_random_hex(hex, sizeof(hex), 16);
  if (hex[0] == '\0') {
    snprintf(out, CHI_SESSION_ID_MAX, "session-%lld-%d", chi_now_ms(), (int)getpid());
    return;
  }
  snprintf(out, CHI_SESSION_ID_MAX, "session-%s", hex);
}

static void chi_free_message(chi_message *m) {
  if (m == NULL) {
    return;
  }
  free(m->role);
  free(m->text);
  free(m->tool_call_id);
  free(m->tool_name);
  free(m->arguments_json);
  memset(m, 0, sizeof(*m));
}

static void chi_conversation_destroy(chi_conversation *c) {
  size_t i;
  if (c == NULL) {
    return;
  }
  for (i = 0; i < c->count; i++) {
    chi_free_message(&c->items[i]);
  }
  free(c->items);
  memset(c, 0, sizeof(*c));
}

static int chi_conversation_add(
    chi_conversation *c,
    const char *role,
    const char *text,
    const char *tool_call_id,
    const char *tool_name,
    const char *arguments_json) {
  chi_message m;

  if (c == NULL || chi_is_blank(role)) {
    return 0;
  }

  if (!chi_ensure_cap((void **)&c->items, &c->cap, c->count + 1, sizeof(chi_message))) {
    return 0;
  }

  memset(&m, 0, sizeof(m));
  m.role = chi_strdup(role);
  m.text = chi_strdup(text == NULL ? "" : text);
  m.tool_call_id = chi_strdup(tool_call_id);
  m.tool_name = chi_strdup(tool_name);
  m.arguments_json = chi_strdup(arguments_json);

  if (m.role == NULL || m.text == NULL ||
      (tool_call_id != NULL && m.tool_call_id == NULL) ||
      (tool_name != NULL && m.tool_name == NULL) ||
      (arguments_json != NULL && m.arguments_json == NULL)) {
    chi_free_message(&m);
    return 0;
  }

  c->items[c->count++] = m;
  return 1;
}

static void chi_prompt_queue_destroy(chi_prompt_queue *q) {
  size_t i;
  if (q == NULL) {
    return;
  }
  for (i = 0; i < q->count; i++) {
    free(q->items[i]);
  }
  free(q->items);
  memset(q, 0, sizeof(*q));
}

static int chi_prompt_queue_insert(chi_prompt_queue *q, const char *prompt, int front) {
  char *copy;
  if (q == NULL || chi_is_blank(prompt)) {
    return 0;
  }
  if (!chi_ensure_cap((void **)&q->items, &q->cap, q->count + 1, sizeof(char *))) {
    return 0;
  }
  copy = chi_strdup(prompt);
  if (copy == NULL) {
    return 0;
  }
  if (front && q->count > 0) {
    memmove(q->items + 1, q->items, q->count * sizeof(char *));
    q->items[0] = copy;
  } else {
    q->items[q->count] = copy;
  }
  q->count++;
  return 1;
}

static int chi_prompt_queue_push(chi_prompt_queue *q, const char *prompt) {
  return chi_prompt_queue_insert(q, prompt, 0);
}

static int chi_prompt_queue_push_front(chi_prompt_queue *q, const char *prompt) {
  return chi_prompt_queue_insert(q, prompt, 1);
}

static char *chi_prompt_queue_pop(chi_prompt_queue *q) {
  char *first;
  if (q == NULL || q->count == 0) {
    return NULL;
  }
  first = q->items[0];
  if (q->count > 1) {
    memmove(q->items, q->items + 1, (q->count - 1) * sizeof(char *));
  }
  q->count--;
  return first;
}

static const char *chi_skip_ws(const char *s) {
  while (s != NULL && *s != '\0' && isspace((unsigned char)*s)) {
    s++;
  }
  return s;
}

static const char *chi_find_json_key(const char *json, const char *key) {
  size_t key_len;
  char *needle;
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
  p = chi_skip_ws(p);
  if (p == NULL || *p != ':') {
    return NULL;
  }
  p++;
  return chi_skip_ws(p);
}

static char *chi_json_get_string(const char *json, const char *key) {
  const char *p;
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  p = chi_find_json_key(json, key);
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
        case '\\':
          c = '\\';
          break;
        case '"':
          c = '"';
          break;
        default:
          c = *p;
          break;
      }
    }

    if (!chi_append_n(&out, &len, &cap, &c, 1)) {
      free(out);
      return NULL;
    }

    p++;
  }

  if (out == NULL) {
    return chi_strdup("");
  }
  return out;
}

static int chi_json_get_number(const char *json, const char *key, double *out) {
  const char *p;
  char *end;

  if (out == NULL) {
    return 0;
  }

  p = chi_find_json_key(json, key);
  if (p == NULL) {
    return 0;
  }

  errno = 0;
  *out = strtod(p, &end);
  return (errno == 0 && end != p);
}

static char *chi_json_escape(const char *input) {
  size_t i;
  size_t len;
  size_t cap;
  size_t out_len = 0;
  char *out;

  if (input == NULL) {
    return chi_strdup("");
  }

  len = strlen(input);
  cap = len * 2 + 16;
  out = (char *)malloc(cap);
  if (out == NULL) {
    return NULL;
  }

  for (i = 0; i < len; i++) {
    const char *esc = NULL;
    switch (input[i]) {
      case '\\':
        esc = "\\\\";
        break;
      case '"':
        esc = "\\\"";
        break;
      case '\n':
        esc = "\\n";
        break;
      case '\r':
        esc = "\\r";
        break;
      case '\t':
        esc = "\\t";
        break;
      default:
        break;
    }

    if (esc != NULL) {
      size_t n = strlen(esc);
      if (!chi_append_n(&out, &out_len, &cap, esc, n)) {
        free(out);
        return NULL;
      }
      continue;
    }

    if (!chi_append_n(&out, &out_len, &cap, &input[i], 1)) {
      free(out);
      return NULL;
    }
  }

  return out;
}

static char *chi_extract_first_json_object(const char *text) {
  const char *start;
  const char *end;
  size_t n;
  char *out;

  if (text == NULL) {
    return NULL;
  }

  start = strchr(text, '{');
  end = strrchr(text, '}');
  if (start == NULL || end == NULL || end < start) {
    return NULL;
  }

  n = (size_t)(end - start + 1);
  out = (char *)malloc(n + 1);
  if (out == NULL) {
    return NULL;
  }

  memcpy(out, start, n);
  out[n] = '\0';
  return out;
}

static char *chi_normalize_newlines(const char *text) {
  size_t i;
  size_t j = 0;
  size_t len;
  char *out;

  if (text == NULL) {
    return chi_strdup("");
  }

  len = strlen(text);
  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }

  for (i = 0; i < len; i++) {
    if (text[i] == '\r') {
      if (i + 1 < len && text[i + 1] == '\n') {
        continue;
      }
      out[j++] = '\n';
      continue;
    }
    out[j++] = text[i];
  }

  out[j] = '\0';
  return out;
}

static char *chi_truncate_tail(const char *text, int max_lines, size_t max_bytes, int *was_truncated) {
  size_t len;
  size_t start = 0;
  int lines = 1;
  size_t i;
  char *out;

  if (was_truncated != NULL) {
    *was_truncated = 0;
  }

  if (text == NULL) {
    return chi_strdup("");
  }

  len = strlen(text);
  if (len > max_bytes) {
    start = len - max_bytes;
    if (was_truncated != NULL) {
      *was_truncated = 1;
    }
  }

  if (max_lines > 0) {
    i = len;
    while (i > start) {
      i--;
      if (text[i] == '\n') {
        lines++;
        if (lines > max_lines) {
          start = i + 1;
          if (was_truncated != NULL) {
            *was_truncated = 1;
          }
          break;
        }
      }
    }
  }

  out = chi_strdup(text + start);
  if (out == NULL) {
    return NULL;
  }

  if (chi_is_blank(out)) {
    free(out);
    return chi_strdup("(no output)");
  }

  return out;
}

static int chi_write_file(const char *path, const char *content) {
  FILE *f;
  size_t n;

  f = fopen(path, "wb");
  if (f == NULL) {
    return 0;
  }

  if (content == NULL) {
    content = "";
  }

  n = strlen(content);
  if (n > 0 && fwrite(content, 1, n, f) != n) {
    fclose(f);
    return 0;
  }

  fclose(f);
  return 1;
}

static char *chi_read_file(const char *path) {
  FILE *f;
  long size;
  size_t got;
  char *buf;

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

static char *chi_shell_quote(const char *text) {
  size_t i;
  size_t len = 2;
  size_t out_i = 0;
  char *out;

  if (text == NULL) {
    text = "";
  }

  for (i = 0; text[i] != '\0'; i++) {
    len += (text[i] == '\'') ? 4 : 1;
  }

  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }

  out[out_i++] = '\'';
  for (i = 0; text[i] != '\0'; i++) {
    if (text[i] == '\'') {
      memcpy(out + out_i, "'\\''", 4);
      out_i += 4;
      continue;
    }
    out[out_i++] = text[i];
  }
  out[out_i++] = '\'';
  out[out_i] = '\0';
  return out;
}

static int chi_cmd_append_checked(
    char **cmd,
    size_t *cmd_len,
    size_t *cmd_cap,
    const char *text,
    const char **fail_msg) {
  if (!chi_append(cmd, cmd_len, cmd_cap, text)) {
    *fail_msg = "out of memory building curl command";
    return 0;
  }
  return 1;
}

static int chi_cmd_append_quoted(
    char **cmd,
    size_t *cmd_len,
    size_t *cmd_cap,
    const char *prefix,
    const char *raw_value,
    const char **fail_msg) {
  char *quoted;
  int ok;

  quoted = chi_shell_quote(raw_value);
  if (quoted == NULL) {
    *fail_msg = "out of memory building curl command";
    return 0;
  }

  ok = 1;
  if (prefix != NULL && !chi_cmd_append_checked(cmd, cmd_len, cmd_cap, prefix, fail_msg)) {
    ok = 0;
  }
  if (ok && !chi_cmd_append_checked(cmd, cmd_len, cmd_cap, quoted, fail_msg)) {
    ok = 0;
  }

  free(quoted);
  return ok;
}

static int chi_prepare_temp_file(char path_template[], int *fd, int *created) {
  *fd = mkstemp(path_template);
  if (*fd < 0) {
    return 0;
  }
  *created = 1;
  close(*fd);
  *fd = -1;
  return 1;
}

static int chi_curl_request(
    const char *method,
    const char *url,
    const char **headers,
    size_t header_count,
    const char *request_body,
    char **response_body,
    int *http_code,
    char **err_out) {
  char req_path[] = "/tmp/chi-req-XXXXXX";
  char resp_path[] = "/tmp/chi-resp-XXXXXX";
  char err_path[] = "/tmp/chi-err-XXXXXX";
  int req_fd = -1;
  int resp_fd = -1;
  int err_fd = -1;
  char *cmd = NULL;
  size_t cmd_len = 0;
  size_t cmd_cap = 0;
  FILE *pipe = NULL;
  char status_buf[64];
  char curl_flags[128];
  int status = 0;
  size_t i;
  int ok = 0;
  int req_created = 0;
  int resp_created = 0;
  int err_created = 0;
  int connect_timeout = 0;
  int max_time = 0;
  const char *fail_msg = NULL;
  char *fail_detail = NULL;

  *response_body = NULL;
  *http_code = 0;
  *err_out = NULL;

  connect_timeout = chi_env_positive_int("CHI_HTTP_CONNECT_TIMEOUT", CHI_HTTP_CONNECT_TIMEOUT_DEFAULT);
  max_time = chi_env_positive_int("CHI_HTTP_MAX_TIME", CHI_HTTP_MAX_TIME_DEFAULT);
  if (max_time < connect_timeout) {
    max_time = connect_timeout;
  }

  if (request_body != NULL) {
    if (!chi_prepare_temp_file(req_path, &req_fd, &req_created)) {
      fail_msg = "failed to create request temp file";
      goto cleanup;
    }

    if (!chi_write_file(req_path, request_body)) {
      fail_msg = "failed to write request body";
      goto cleanup;
    }
  }

  if (!chi_prepare_temp_file(resp_path, &resp_fd, &resp_created)) {
    fail_msg = "failed to create response temp file";
    goto cleanup;
  }

  if (!chi_prepare_temp_file(err_path, &err_fd, &err_created)) {
    fail_msg = "failed to create curl stderr temp file";
    goto cleanup;
  }

  snprintf(
      curl_flags,
      sizeof(curl_flags),
      "curl -q -sS --retry 0 --connect-timeout %d --max-time %d -o ",
      connect_timeout,
      max_time);
  if (!chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, curl_flags, &fail_msg) ||
      !chi_cmd_append_quoted(&cmd, &cmd_len, &cmd_cap, NULL, resp_path, &fail_msg) ||
      !chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, " -w '%{http_code}'", &fail_msg)) {
    goto cleanup;
  }

  if (method != NULL && strcmp(method, "POST") == 0) {
    if (!chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, " -X POST", &fail_msg)) {
      goto cleanup;
    }
  }

  for (i = 0; i < header_count; i++) {
    if (!chi_cmd_append_quoted(&cmd, &cmd_len, &cmd_cap, " -H ", headers[i], &fail_msg)) {
      goto cleanup;
    }
  }

  if (request_body != NULL) {
    if (!chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, " --data-binary @", &fail_msg) ||
        !chi_cmd_append_quoted(&cmd, &cmd_len, &cmd_cap, NULL, req_path, &fail_msg)) {
      goto cleanup;
    }
  }

  if (!chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, " ", &fail_msg) ||
      !chi_cmd_append_quoted(&cmd, &cmd_len, &cmd_cap, NULL, url, &fail_msg)) {
    goto cleanup;
  }
  if (!chi_cmd_append_checked(&cmd, &cmd_len, &cmd_cap, " 2>", &fail_msg) ||
      !chi_cmd_append_quoted(&cmd, &cmd_len, &cmd_cap, NULL, err_path, &fail_msg)) {
    goto cleanup;
  }

  pipe = popen(cmd, "r");
  if (pipe == NULL) {
    fail_msg = "failed to run curl";
    goto cleanup;
  }

  memset(status_buf, 0, sizeof(status_buf));
  if (fgets(status_buf, sizeof(status_buf), pipe) == NULL) {
    status_buf[0] = '\0';
  }

  status = pclose(pipe);
  pipe = NULL;

  *response_body = chi_read_file(resp_path);
  if (*response_body == NULL) {
    fail_msg = "failed to read curl response body";
    goto cleanup;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    char *curl_stderr = chi_read_file(err_path);
    int exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (WIFSIGNALED(status)) {
      fail_detail = chi_format("curl request terminated by signal %d", WTERMSIG(status));
    } else if (!chi_is_blank(curl_stderr)) {
      fail_detail = chi_format("curl request failed (exit %d): %s", exit_status, curl_stderr);
    } else if (exit_status >= 0) {
      fail_detail = chi_format("curl request failed (exit %d)", exit_status);
    }
    free(curl_stderr);
    fail_msg = fail_detail != NULL ? fail_detail : "curl request failed";
    free(*response_body);
    *response_body = NULL;
    goto cleanup;
  }

  *http_code = atoi(status_buf);
  ok = 1;

cleanup:
  if (req_fd >= 0) {
    close(req_fd);
  }
  if (resp_fd >= 0) {
    close(resp_fd);
  }
  if (err_fd >= 0) {
    close(err_fd);
  }
  if (pipe != NULL) {
    pclose(pipe);
  }
  free(cmd);
  if (resp_created) {
    unlink(resp_path);
  }
  if (req_created) {
    unlink(req_path);
  }
  if (err_created) {
    unlink(err_path);
  }
  if (!ok && fail_msg != NULL) {
    *err_out = chi_strdup(fail_msg);
  }
  free(fail_detail);
  return ok;
}

static int chi_run_shell_command(const char *cwd, const char *command, double timeout_s, chi_shell_result *out) {
  int pipefd[2];
  pid_t pid;
  int status = 0;
  int child_done = 0;
  int read_eof = 0;
  int timed_out = 0;
  char *buffer = NULL;
  size_t buffer_len = 0;
  size_t buffer_cap = 0;
  long long started_ms;

  if (out == NULL || chi_is_blank(command)) {
    return 0;
  }

  memset(out, 0, sizeof(*out));
  started_ms = chi_now_ms();

  if (pipe(pipefd) != 0) {
    return 0;
  }

  pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return 0;
  }

  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);

    if (!chi_is_blank(cwd) && chdir(cwd) != 0) {
      dprintf(STDERR_FILENO, "failed to chdir to %s: %s\n", cwd, strerror(errno));
      _exit(127);
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

    if (timeout_s > 0 && !timed_out) {
      double elapsed = (double)(chi_now_ms() - started_ms) / 1000.0;
      if (elapsed > timeout_s) {
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
        if (!chi_append_n(&buffer, &buffer_len, &buffer_cap, chunk, (size_t)n)) {
          close(pipefd[0]);
          kill(pid, SIGKILL);
          waitpid(pid, NULL, 0);
          free(buffer);
          return 0;
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
    buffer = chi_strdup("");
    if (buffer == NULL) {
      return 0;
    }
  }

  out->output = buffer;
  out->timed_out = timed_out;
  if (WIFEXITED(status)) {
    out->exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    out->exit_code = 128 + WTERMSIG(status);
  } else {
    out->exit_code = 1;
  }

  return 1;
}

static char *chi_extract_responses_output_text(const char *json) {
  const char *p;
  char *direct = NULL;
  char *type = NULL;

  if (json == NULL) {
    return NULL;
  }

  type = chi_json_get_string(json, "type");
  if (type != NULL && strcmp(type, "response.output_text.done") == 0) {
    direct = chi_json_get_string(json, "text");
    if (direct != NULL && direct[0] != '\0') {
      free(type);
      return direct;
    }
    free(direct);
  }
  free(type);

  p = json;
  while (p != NULL) {
    char *type;
    char *text;

    p = strstr(p, "\"type\"");
    if (p == NULL) {
      break;
    }

    type = chi_json_get_string(p, "type");
    if (type != NULL && strcmp(type, "output_text") == 0) {
      text = chi_json_get_string(p, "text");
      if (text != NULL && text[0] != '\0') {
        free(type);
        return text;
      }
      free(text);
    }
    free(type);
    p += 6;
  }

  direct = chi_json_get_string(json, "output_text");
  if (direct != NULL && direct[0] != '\0') {
    return direct;
  }
  free(direct);

  return NULL;
}

static int chi_extract_responses_function_call(
    const char *json,
    char **call_id_out,
    char **tool_name_out,
    char **arguments_json_out) {
  const char *p;

  *call_id_out = NULL;
  *tool_name_out = NULL;
  *arguments_json_out = NULL;

  if (json == NULL) {
    return 0;
  }

  p = json;
  while (p != NULL) {
    char *type;

    p = strstr(p, "\"type\"");
    if (p == NULL) {
      break;
    }

    type = chi_json_get_string(p, "type");
    if (type != NULL && strcmp(type, "function_call") == 0) {
      char *call_id = chi_json_get_string(p, "call_id");
      char *tool_name = chi_json_get_string(p, "name");
      char *arguments = chi_json_get_string(p, "arguments");
      free(type);

      if (chi_is_blank(arguments)) {
        free(arguments);
        arguments = chi_strdup("{}");
      }
      if (arguments == NULL) {
        free(call_id);
        free(tool_name);
        return 0;
      }

      *call_id_out = call_id;
      *tool_name_out = tool_name;
      *arguments_json_out = arguments;
      return 1;
    }
    free(type);
    p += 6;
  }

  return 0;
}

static int chi_sse_type_matches(const char *payload, const char *event_type) {
  char *type;
  int ok;

  if (payload == NULL || event_type == NULL) {
    return 0;
  }
  type = chi_json_get_string(payload, "type");
  ok = (type != NULL && strcmp(type, event_type) == 0);
  free(type);
  return ok;
}

static char *chi_extract_sse_payload(const char *body, const char *event_type) {
  const char *line = body;
  char *last = NULL;

  if (body == NULL || strstr(body, "data:") == NULL) {
    if (event_type == NULL) {
      return chi_strdup(body == NULL ? "" : body);
    }
    return NULL;
  }

  while (*line != '\0') {
    const char *end = strchr(line, '\n');
    size_t n = end == NULL ? strlen(line) : (size_t)(end - line);

    if (n > 5 && strncmp(line, "data:", 5) == 0) {
      const char *payload = chi_skip_ws(line + 5);
      if (strncmp(payload, "[DONE]", 6) != 0) {
        size_t payload_len = n - (size_t)(payload - line);
        char *candidate = (char *)malloc(payload_len + 1);
        if (candidate == NULL) {
          free(last);
          return NULL;
        }
        memcpy(candidate, payload, payload_len);
        candidate[payload_len] = '\0';
        if (event_type == NULL || chi_sse_type_matches(candidate, event_type)) {
          free(last);
          last = candidate;
          candidate = NULL;
        }
        free(candidate);
      }
    }

    if (end == NULL) {
      break;
    }
    line = end + 1;
  }

  if (last != NULL) {
    return last;
  }
  if (event_type == NULL) {
    return chi_strdup(body);
  }
  return NULL;
}

static char *chi_extract_provider_payload(const char *body) {
  char *payload;

  payload = chi_extract_sse_payload(body, "response.completed");
  if (payload != NULL) {
    return payload;
  }

  payload = chi_extract_sse_payload(body, "response.output_text.done");
  if (payload != NULL) {
    return payload;
  }

  return chi_extract_sse_payload(body, NULL);
}

static char *chi_extract_sse_output_text_deltas(const char *body) {
  const char *line = body;
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (body == NULL || strstr(body, "data:") == NULL) {
    return NULL;
  }

  while (*line != '\0') {
    const char *end = strchr(line, '\n');
    size_t n = end == NULL ? strlen(line) : (size_t)(end - line);

    if (n > 5 && strncmp(line, "data:", 5) == 0) {
      const char *payload = chi_skip_ws(line + 5);
      if (strncmp(payload, "[DONE]", 6) != 0) {
        size_t payload_len = n - (size_t)(payload - line);
        char *candidate = (char *)malloc(payload_len + 1);
        if (candidate == NULL) {
          free(out);
          return NULL;
        }
        memcpy(candidate, payload, payload_len);
        candidate[payload_len] = '\0';

        if (chi_sse_type_matches(candidate, "response.output_text.delta")) {
          char *delta = chi_json_get_string(candidate, "delta");
          if (!chi_is_blank(delta)) {
            if (!chi_append(&out, &len, &cap, delta)) {
              free(delta);
              free(candidate);
              free(out);
              return NULL;
            }
          }
          free(delta);
        }
        free(candidate);
      }
    }

    if (end == NULL) {
      break;
    }
    line = end + 1;
  }

  if (chi_is_blank(out)) {
    free(out);
    return NULL;
  }
  return out;
}

static char *chi_extract_incomplete_reason(const char *json) {
  const char *details;
  if (json == NULL) {
    return NULL;
  }
  details = strstr(json, "\"incomplete_details\"");
  if (details == NULL) {
    return NULL;
  }
  return chi_json_get_string(details, "reason");
}

static int chi_parse_action(const char *raw_text, chi_action *out, char **err_out) {
  char *json = NULL;
  char *kind = NULL;

  memset(out, 0, sizeof(*out));
  *err_out = NULL;

  if (chi_is_blank(raw_text)) {
    *err_out = chi_strdup("provider returned empty output");
    return 0;
  }

  out->assistant_text = chi_strdup(raw_text);
  if (out->assistant_text == NULL) {
    *err_out = chi_strdup("out of memory");
    return 0;
  }

  json = chi_extract_first_json_object(raw_text);
  if (json == NULL) {
    out->final_text = chi_strdup(raw_text);
    return out->final_text != NULL;
  }

  kind = chi_json_get_string(json, "kind");
  if (kind != NULL && strcmp(kind, "tool") == 0) {
    double timeout = 0;
    out->tool_name = chi_json_get_string(json, "name");
    out->tool_command = chi_json_get_string(json, "command");
    if (chi_is_blank(out->tool_name)) {
      free(out->tool_name);
      out->tool_name = chi_strdup("bash");
    }
    if (chi_is_blank(out->tool_command)) {
      free(out->tool_command);
      out->tool_command = NULL;
      out->final_text = chi_strdup("tool call missing command");
      out->is_tool = 0;
      free(kind);
      free(json);
      return out->final_text != NULL;
    }
    if (chi_json_get_number(json, "timeout", &timeout) && timeout > 0) {
      out->timeout_seconds = timeout;
    }
    out->is_tool = 1;
    free(kind);
    free(json);
    return 1;
  }

  out->final_text = chi_json_get_string(json, "text");
  if (chi_is_blank(out->final_text)) {
    free(out->final_text);
    out->final_text = chi_strdup(raw_text);
  }

  free(kind);
  free(json);
  return out->final_text != NULL;
}

static void chi_action_reset(chi_action *a) {
  if (a == NULL) {
    return;
  }
  free(a->assistant_text);
  free(a->final_text);
  free(a->tool_call_id);
  free(a->tool_name);
  free(a->tool_arguments_json);
  free(a->tool_command);
  memset(a, 0, sizeof(*a));
}

static int chi_append_json_quoted(char **buf, size_t *len, size_t *cap, const char *value) {
  char *escaped = chi_json_escape(value == NULL ? "" : value);
  int ok;
  if (escaped == NULL) {
    return 0;
  }
  ok = chi_append(buf, len, cap, "\"") &&
       chi_append(buf, len, cap, escaped) &&
       chi_append(buf, len, cap, "\"");
  free(escaped);
  return ok;
}

static const char *chi_normalize_reasoning_effort(const char *raw) {
  if (chi_equals_ignore_case(raw, "none")) {
    return "none";
  }
  if (chi_equals_ignore_case(raw, "minimal")) {
    return "minimal";
  }
  if (chi_equals_ignore_case(raw, "low")) {
    return "low";
  }
  if (chi_equals_ignore_case(raw, "medium")) {
    return "medium";
  }
  if (chi_equals_ignore_case(raw, "high")) {
    return "high";
  }
  if (chi_equals_ignore_case(raw, "xhigh")) {
    return "xhigh";
  }
  return NULL;
}

static int chi_append_input_item_start(
    char **buf,
    size_t *len,
    size_t *cap,
    int *first_item) {
  if (!*first_item) {
    if (!chi_append(buf, len, cap, ",")) {
      return 0;
    }
  }
  *first_item = 0;
  return 1;
}

static int chi_append_responses_user_message(
    const chi_message *m,
    char **buf,
    size_t *len,
    size_t *cap,
    int *first_item) {
  if (chi_is_blank(m->text)) {
    return 1;
  }
  if (!chi_append_input_item_start(buf, len, cap, first_item) ||
      !chi_append(buf, len, cap, "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":") ||
      !chi_append_json_quoted(buf, len, cap, m->text) ||
      !chi_append(buf, len, cap, "}]}")) {
    return 0;
  }
  return 1;
}

static int chi_append_responses_assistant_message(
    const chi_message *m,
    size_t index,
    char **buf,
    size_t *len,
    size_t *cap,
    int *first_item) {
  if (!chi_is_blank(m->text)) {
    if (!chi_append_input_item_start(buf, len, cap, first_item) ||
        !chi_append(buf, len, cap, "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":") ||
        !chi_append_json_quoted(buf, len, cap, m->text) ||
        !chi_append(buf, len, cap, "}]}")) {
      return 0;
    }
  }

  if (!chi_is_blank(m->tool_name)) {
    char fallback_call_id[64];
    const char *call_id = m->tool_call_id;
    const char *tool_name = m->tool_name;
    const char *arguments = m->arguments_json;
    if (chi_is_blank(call_id)) {
      snprintf(fallback_call_id, sizeof(fallback_call_id), "call_%zu", index + 1);
      call_id = fallback_call_id;
    }
    if (chi_is_blank(tool_name)) {
      tool_name = "tool";
    }
    if (chi_is_blank(arguments)) {
      arguments = "{}";
    }

    if (!chi_append_input_item_start(buf, len, cap, first_item) ||
        !chi_append(buf, len, cap, "{\"type\":\"function_call\",\"call_id\":") ||
        !chi_append_json_quoted(buf, len, cap, call_id) ||
        !chi_append(buf, len, cap, ",\"name\":") ||
        !chi_append_json_quoted(buf, len, cap, tool_name) ||
        !chi_append(buf, len, cap, ",\"arguments\":") ||
        !chi_append_json_quoted(buf, len, cap, arguments) ||
        !chi_append(buf, len, cap, "}")) {
      return 0;
    }
  }

  return 1;
}

static int chi_append_responses_tool_result(
    const chi_message *m,
    char **buf,
    size_t *len,
    size_t *cap,
    int *first_item) {
  const char *output;
  if (chi_is_blank(m->tool_call_id)) {
    return 1;
  }
  output = chi_is_blank(m->text) ? "(no content)" : m->text;

  if (!chi_append_input_item_start(buf, len, cap, first_item) ||
      !chi_append(buf, len, cap, "{\"type\":\"function_call_output\",\"call_id\":") ||
      !chi_append_json_quoted(buf, len, cap, m->tool_call_id) ||
      !chi_append(buf, len, cap, ",\"output\":") ||
      !chi_append_json_quoted(buf, len, cap, output) ||
      !chi_append(buf, len, cap, "}")) {
    return 0;
  }
  return 1;
}

static int chi_append_responses_input(
    const chi_conversation *conversation,
    char **buf,
    size_t *len,
    size_t *cap) {
  size_t i;
  int first_item = 1;

  if (!chi_append(buf, len, cap, "\"input\":[")) {
    return 0;
  }

  for (i = 0; i < conversation->count; i++) {
    const chi_message *m = &conversation->items[i];
    if (strcmp(m->role, "user") == 0) {
      if (!chi_append_responses_user_message(m, buf, len, cap, &first_item)) {
        return 0;
      }
      continue;
    }
    if (strcmp(m->role, "assistant") == 0) {
      if (!chi_append_responses_assistant_message(m, i, buf, len, cap, &first_item)) {
        return 0;
      }
      continue;
    }
    if (strcmp(m->role, "toolResult") == 0) {
      if (!chi_append_responses_tool_result(m, buf, len, cap, &first_item)) {
        return 0;
      }
      continue;
    }
  }

  if (!chi_append(buf, len, cap, "]")) {
    return 0;
  }
  return 1;
}

static char *chi_build_request_json(const chi_config *cfg, const chi_conversation *conversation, char **err_out) {
  const char *instructions =
      "You are a coding agent controller with one tool: bash. "
      "Use bash to create/edit/run files. Use uv run for python.";
  const char *reasoning_effort = chi_normalize_reasoning_effort(cfg->reasoning_effort);
  const char *bash_tool_json =
      "{\"type\":\"function\","
      "\"name\":\"bash\","
      "\"description\":\"Execute a bash command in the working directory and return stdout/stderr.\","
      "\"parameters\":{"
      "\"type\":\"object\","
      "\"properties\":{"
      "\"command\":{\"type\":\"string\",\"description\":\"Bash command to execute\"},"
      "\"timeout\":{\"type\":\"number\",\"description\":\"Timeout in seconds (optional, no default timeout)\"}"
      "},"
      "\"required\":[\"command\"]"
      "}"
      "}";
  char *json = NULL;
  size_t len = 0;
  size_t cap = 0;

  *err_out = NULL;

  if (!chi_append(&json, &len, &cap, "{") ||
      !chi_append(&json, &len, &cap, "\"model\":") ||
      !chi_append_json_quoted(&json, &len, &cap, cfg->model) ||
      !chi_append(&json, &len, &cap, ",\"instructions\":") ||
      !chi_append_json_quoted(&json, &len, &cap, instructions) ||
      !chi_append(&json, &len, &cap, ",") ||
      !chi_append_responses_input(conversation, &json, &len, &cap) ||
      !chi_append(&json, &len, &cap, ",\"tools\":[") ||
      !chi_append(&json, &len, &cap, bash_tool_json) ||
      !chi_append(&json, &len, &cap, "]") ||
      !chi_append(&json, &len, &cap, ",\"tool_choice\":\"auto\",\"parallel_tool_calls\":true")) {
    free(json);
    *err_out = chi_strdup("out of memory while building provider request");
    return NULL;
  }

  if (reasoning_effort != NULL) {
    if (!chi_append(&json, &len, &cap, ",\"reasoning\":{\"effort\":") ||
        !chi_append_json_quoted(&json, &len, &cap, reasoning_effort) ||
        !chi_append(&json, &len, &cap, "}")) {
      free(json);
      *err_out = chi_strdup("out of memory while building provider request");
      return NULL;
    }
  }

  if (!chi_append(&json, &len, &cap, ",\"store\":false,\"stream\":true}")) {
    free(json);
    *err_out = chi_strdup("out of memory while building provider request");
    return NULL;
  }
  return json;
}

static int chi_extract_provider_action(const char *response_body, chi_action *action, char **err_out) {
  char *payload = NULL;
  char *tool_payload = NULL;
  char *output_done_payload = NULL;
  char *output_text = NULL;
  char *status = NULL;
  char *reason = NULL;
  char *call_id = NULL;
  char *tool_name = NULL;
  char *arguments_json = NULL;
  double timeout = 0;
  int has_tool = 0;
  int ok;

  *err_out = NULL;
  memset(action, 0, sizeof(*action));

  payload = chi_extract_provider_payload(response_body);
  if (payload == NULL) {
    *err_out = chi_strdup("failed to parse provider payload");
    return 0;
  }

  tool_payload = chi_extract_sse_payload(response_body, "response.output_item.done");
  if (tool_payload != NULL) {
    has_tool = chi_extract_responses_function_call(tool_payload, &call_id, &tool_name, &arguments_json);
  }
  if (!has_tool) {
    has_tool = chi_extract_responses_function_call(payload, &call_id, &tool_name, &arguments_json);
  }

  output_text = chi_extract_responses_output_text(payload);
  if (output_text == NULL) {
    output_done_payload = chi_extract_sse_payload(response_body, "response.output_text.done");
    if (output_done_payload != NULL) {
      output_text = chi_json_get_string(output_done_payload, "text");
    }
  }
  if (output_text == NULL) {
    output_text = chi_extract_sse_output_text_deltas(response_body);
  }

  if (has_tool) {
    action->assistant_text = output_text != NULL ? output_text : chi_strdup("");
    output_text = NULL;
    action->tool_call_id = call_id;
    action->tool_name = tool_name;
    action->tool_arguments_json = arguments_json;
    call_id = NULL;
    tool_name = NULL;
    arguments_json = NULL;

    if (action->assistant_text == NULL || action->tool_arguments_json == NULL) {
      chi_action_reset(action);
      *err_out = chi_strdup("out of memory while parsing tool call");
      free(status);
      free(reason);
      free(payload);
      free(tool_payload);
      free(output_done_payload);
      free(call_id);
      free(tool_name);
      free(arguments_json);
      return 0;
    }

    if (chi_is_blank(action->tool_name)) {
      free(action->tool_name);
      action->tool_name = chi_strdup("bash");
      if (action->tool_name == NULL) {
        chi_action_reset(action);
        *err_out = chi_strdup("out of memory while parsing tool call");
        free(status);
        free(reason);
        free(payload);
        free(tool_payload);
        free(output_done_payload);
        free(call_id);
        free(tool_name);
        free(arguments_json);
        return 0;
      }
    }

    action->tool_command = chi_json_get_string(action->tool_arguments_json, "command");
    if (chi_json_get_number(action->tool_arguments_json, "timeout", &timeout) && timeout > 0) {
      action->timeout_seconds = timeout;
    }
    if (chi_is_blank(action->tool_command)) {
      free(action->tool_command);
      action->tool_command = NULL;
      action->final_text = chi_strdup("tool call missing command");
      action->is_tool = 0;
      ok = action->final_text != NULL;
    } else {
      action->is_tool = 1;
      ok = 1;
    }

    free(status);
    free(reason);
    free(payload);
    free(tool_payload);
    free(output_done_payload);
    free(call_id);
    free(tool_name);
    free(arguments_json);
    return ok;
  }

  if (output_text == NULL) {
    status = chi_json_get_string(payload, "status");
    reason = chi_extract_incomplete_reason(payload);
    if (status != NULL && strcmp(status, "incomplete") == 0) {
      if (reason != NULL && strcmp(reason, "max_output_tokens") == 0) {
        *err_out = chi_strdup(
            "provider response incomplete (max_output_tokens reached); increase token budget or lower reasoning effort");
      } else if (!chi_is_blank(reason)) {
        *err_out = chi_format("provider response incomplete: %s", reason);
      } else {
        *err_out = chi_strdup("provider response incomplete");
      }
    }
  }

  if (output_text == NULL) {
    free(status);
    free(reason);
    free(payload);
    free(tool_payload);
    free(output_done_payload);
    free(call_id);
    free(tool_name);
    free(arguments_json);
    if (*err_out == NULL) {
      *err_out = chi_format("could not parse output text from provider response: %s", response_body);
    }
    return 0;
  }

  free(status);
  free(reason);
  ok = chi_parse_action(output_text, action, err_out);
  free(payload);
  free(tool_payload);
  free(output_done_payload);
  free(output_text);
  free(call_id);
  free(tool_name);
  free(arguments_json);
  return ok;
}

static int chi_provider_request_with_auth(
    const chi_config *cfg,
    const chi_conversation *conversation,
    const char *auth_token,
    const char *url_env,
    const char *default_url,
    const char *http_label,
    chi_action *action,
    char **err_out) {
  const char *url;
  char *req_json = NULL;
  char *resp_json = NULL;
  char *tmp_err = NULL;
  char *auth = NULL;
  const char *headers[3];
  int http_code = 0;
  int ok = 0;

  *err_out = NULL;

  req_json = chi_build_request_json(cfg, conversation, &tmp_err);
  if (req_json == NULL) {
    *err_out = tmp_err;
    return 0;
  }

  auth = chi_format("Authorization: Bearer %s", auth_token);
  if (auth == NULL) {
    free(req_json);
    *err_out = chi_strdup("out of memory while building auth header");
    return 0;
  }

  headers[0] = auth;
  headers[1] = "Content-Type: application/json";
  headers[2] = "Accept: text/event-stream";
  url = getenv(url_env);
  if (chi_is_blank(url)) {
    url = default_url;
  }

  if (!chi_curl_request("POST", url, headers, 3, req_json, &resp_json, &http_code, &tmp_err)) {
    *err_out = tmp_err;
    if (*err_out == NULL) {
      *err_out = chi_strdup("provider request failed");
    }
    free(req_json);
    free(auth);
    return 0;
  }

  if (http_code < 200 || http_code >= 300) {
    *err_out = chi_format("%s http %d: %s", http_label, http_code, resp_json);
    free(req_json);
    free(auth);
    free(resp_json);
    return 0;
  }

  ok = chi_extract_provider_action(resp_json, action, &tmp_err);
  if (!ok) {
    *err_out = tmp_err;
  }

  free(req_json);
  free(auth);
  free(resp_json);
  return ok;
}

static int chi_provider_openai(
    const chi_config *cfg,
    const chi_conversation *conversation,
    chi_action *action,
    char **err_out) {
  const char *api_key = getenv("OPENAI_API_KEY");
  if (chi_is_blank(api_key)) {
    *err_out = chi_strdup("OPENAI_API_KEY is not set");
    return 0;
  }
  return chi_provider_request_with_auth(
      cfg,
      conversation,
      api_key,
      "OPENAI_API_URL",
      "https://api.openai.com/v1/responses",
      "openai",
      action,
      err_out);
}

static int chi_resolve_chatgpt_access_token(chi_config *cfg, char **token_out, char **err_out) {
  const char *direct;
  const char *session;
  char *auth_resp = NULL;
  char *cookie = NULL;
  char *tmp_err = NULL;
  char *token = NULL;
  const char *headers[2];
  int http_code = 0;

  *token_out = NULL;
  *err_out = NULL;

  if (!chi_is_blank(cfg->chatgpt_access_token)) {
    *token_out = chi_strdup(cfg->chatgpt_access_token);
    return *token_out != NULL;
  }

  direct = getenv("CHATGPT_ACCESS_TOKEN");
  if (!chi_is_blank(direct)) {
    cfg->chatgpt_access_token = chi_strdup(direct);
    *token_out = chi_strdup(direct);
    return cfg->chatgpt_access_token != NULL && *token_out != NULL;
  }

  session = getenv("CHATGPT_SESSION_TOKEN");
  if (chi_is_blank(session)) {
    *err_out = chi_strdup("set CHATGPT_ACCESS_TOKEN or CHATGPT_SESSION_TOKEN for chatgpt backend");
    return 0;
  }

  cookie = chi_format("Cookie: __Secure-next-auth.session-token=%s", session);
  if (cookie == NULL) {
    *err_out = chi_strdup("out of memory while building session cookie header");
    return 0;
  }

  headers[0] = cookie;
  headers[1] = "Content-Type: application/json";

  if (!chi_curl_request(
          "GET",
          "https://chatgpt.com/api/auth/session",
          headers,
          2,
          NULL,
          &auth_resp,
          &http_code,
          &tmp_err)) {
    *err_out = tmp_err;
    free(cookie);
    return 0;
  }

  free(cookie);

  if (http_code < 200 || http_code >= 300) {
    *err_out = chi_format("chatgpt auth http %d: %s", http_code, auth_resp);
    free(auth_resp);
    return 0;
  }

  token = chi_json_get_string(auth_resp, "accessToken");
  free(auth_resp);

  if (chi_is_blank(token)) {
    free(token);
    *err_out = chi_strdup("failed to parse accessToken from chatgpt auth session response");
    return 0;
  }

  cfg->chatgpt_access_token = chi_strdup(token);
  *token_out = token;
  if (cfg->chatgpt_access_token == NULL) {
    free(*token_out);
    *token_out = NULL;
    *err_out = chi_strdup("out of memory while caching access token");
    return 0;
  }

  return 1;
}

static int chi_provider_chatgpt(
    chi_config *cfg,
    const chi_conversation *conversation,
    chi_action *action,
    char **err_out) {
  char *token = NULL;
  int ok;

  *err_out = NULL;

  if (!chi_resolve_chatgpt_access_token(cfg, &token, err_out)) {
    return 0;
  }
  ok = chi_provider_request_with_auth(
      cfg,
      conversation,
      token,
      "CHATGPT_API_URL",
      "https://chatgpt.com/backend-api/codex/responses",
      "chatgpt",
      action,
      err_out);
  free(token);
  return ok;
}

static int chi_provider_next_action(
    chi_config *cfg,
    const chi_conversation *conversation,
    chi_action *action,
    char **err_out) {
  if (cfg->backend == CHI_BACKEND_CHATGPT) {
    return chi_provider_chatgpt(cfg, conversation, action, err_out);
  }
  return chi_provider_openai(cfg, conversation, action, err_out);
}

static int chi_run_bash(
    const char *cwd,
    const char *command,
    double timeout,
    char **display_out,
    int *is_error_out,
    char **error_text_out) {
  chi_shell_result shell;
  char *normalized;
  char *tail;
  char *final;
  int truncated = 0;

  *display_out = NULL;
  *is_error_out = 1;
  *error_text_out = NULL;

  memset(&shell, 0, sizeof(shell));
  if (!chi_run_shell_command(cwd, command, timeout, &shell)) {
    *error_text_out = chi_strdup("failed to execute shell command");
    *display_out = chi_strdup("(no output)");
    return *display_out != NULL && *error_text_out != NULL;
  }

  normalized = chi_normalize_newlines(shell.output);
  free(shell.output);
  if (normalized == NULL) {
    *error_text_out = chi_strdup("out of memory normalizing command output");
    return 0;
  }

  tail = chi_truncate_tail(normalized, CHI_MAX_OUTPUT_LINES, CHI_MAX_OUTPUT_BYTES, &truncated);
  free(normalized);
  if (tail == NULL) {
    *error_text_out = chi_strdup("out of memory truncating command output");
    return 0;
  }

  if (truncated) {
    final = chi_format("%s\n\n[output truncated to tail: %d lines / %d bytes]",
                       tail,
                       CHI_MAX_OUTPUT_LINES,
                       CHI_MAX_OUTPUT_BYTES);
    free(tail);
    if (final == NULL) {
      *error_text_out = chi_strdup("out of memory building truncated output note");
      return 0;
    }
    tail = final;
  }

  if (shell.timed_out) {
    *is_error_out = 1;
    *error_text_out = chi_format("command timed out after %.1f seconds", timeout);
  } else if (shell.exit_code != 0) {
    *is_error_out = 1;
    *error_text_out = chi_format("command exited with code %d", shell.exit_code);
  } else {
    *is_error_out = 0;
    *error_text_out = NULL;
  }

  *display_out = tail;
  return 1;
}

static int chi_parse_backend(const char *value, chi_backend *out) {
  if (value == NULL || strcmp(value, "openai") == 0 || strcmp(value, "openai_api") == 0) {
    *out = CHI_BACKEND_OPENAI;
    return 1;
  }
  if (strcmp(value, "chatgpt") == 0) {
    *out = CHI_BACKEND_CHATGPT;
    return 1;
  }
  return 0;
}

static void chi_usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s [--backend openai|chatgpt] [--model MODEL] [--reasoning EFFORT] [--queue \"prompt\"] \"prompt\" [working_dir]\n"
          "example: %s \"Use bash to create hello.py and run it with uv run hello.py\" ./agent_playground\n"
          "\n"
          "env:\n"
          "  OPENAI_API_KEY               auth for openai backend\n"
          "  CHATGPT_ACCESS_TOKEN         direct auth for chatgpt backend\n"
          "  CHATGPT_SESSION_TOKEN        alternate auth for chatgpt backend\n"
          "  CHI_BACKEND                  default backend (openai|chatgpt)\n"
          "  CHI_HTTP_CONNECT_TIMEOUT     curl connect timeout seconds (default: 5)\n"
          "  CHI_HTTP_MAX_TIME            curl total timeout seconds (default: 45)\n"
          "  CHI_DEBUG                    set to non-zero for debug logs\n",
          argv0,
          argv0);
}

static int chi_arg_fail(chi_prompt_queue *queue, const char *msg) {
  fprintf(stderr, "%s\n", msg);
  chi_prompt_queue_destroy(queue);
  return 2;
}

static int chi_arg_usage_fail(chi_prompt_queue *queue, const char *argv0) {
  chi_usage(argv0);
  chi_prompt_queue_destroy(queue);
  return 2;
}

static int chi_queue_pop_to_conversation(chi_prompt_queue *queue, chi_conversation *convo) {
  char *prompt = chi_prompt_queue_pop(queue);
  int ok;
  if (prompt == NULL) {
    return 0;
  }
  ok = chi_conversation_add(convo, "user", prompt, NULL, NULL, NULL);
  free(prompt);
  return ok;
}

int main(int argc, char **argv) {
  chi_config cfg;
  chi_conversation convo;
  chi_prompt_queue queue;
  const char *prompt = NULL;
  const char *working_dir = ".";
  int max_turns = CHI_MAX_TURNS;
  int i;
  int exit_code = 0;
  unsigned long long call_seq = 0;

  memset(&cfg, 0, sizeof(cfg));
  memset(&convo, 0, sizeof(convo));
  memset(&queue, 0, sizeof(queue));

  {
    const char *env_backend = getenv("CHI_BACKEND");
    if (!chi_parse_backend(env_backend, &cfg.backend)) {
      fprintf(stderr, "invalid CHI_BACKEND: %s\n", env_backend);
      return 2;
    }
  }

  cfg.model = getenv("CHI_MODEL");
  if (chi_is_blank(cfg.model)) {
    cfg.model = "gpt-5.2-codex";
  }
  cfg.reasoning_effort = getenv("CHI_REASONING_EFFORT");
  if (chi_is_blank(cfg.reasoning_effort)) {
    cfg.reasoning_effort = "high";
  }
  cfg.debug = !chi_is_blank(getenv("CHI_DEBUG")) && strcmp(getenv("CHI_DEBUG"), "0") != 0;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      chi_usage(argv[0]);
      chi_prompt_queue_destroy(&queue);
      return 0;
    }

    if (strcmp(argv[i], "--backend") == 0) {
      if (i + 1 >= argc || !chi_parse_backend(argv[i + 1], &cfg.backend)) {
        return chi_arg_fail(&queue, "invalid --backend value");
      }
      i++;
      continue;
    }

    if (strcmp(argv[i], "--model") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --model value");
      }
      cfg.model = argv[++i];
      continue;
    }

    if (strcmp(argv[i], "--reasoning") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --reasoning value");
      }
      cfg.reasoning_effort = argv[++i];
      continue;
    }

    if (strcmp(argv[i], "--max-turns") == 0) {
      if (i + 1 >= argc) {
        return chi_arg_fail(&queue, "missing --max-turns value");
      }
      max_turns = atoi(argv[++i]);
      if (max_turns <= 0) {
        return chi_arg_fail(&queue, "--max-turns must be > 0");
      }
      continue;
    }

    if (strcmp(argv[i], "--queue") == 0) {
      if (i + 1 >= argc || !chi_prompt_queue_push(&queue, argv[i + 1])) {
        return chi_arg_fail(&queue, "missing or invalid --queue value");
      }
      i++;
      continue;
    }

    if (prompt == NULL) {
      prompt = argv[i];
      continue;
    }

    if (working_dir == NULL || strcmp(working_dir, ".") == 0) {
      working_dir = argv[i];
      continue;
    }

    fprintf(stderr, "unexpected argument: %s\n", argv[i]);
    return chi_arg_usage_fail(&queue, argv[0]);
  }

  if (chi_is_blank(prompt)) {
    return chi_arg_usage_fail(&queue, argv[0]);
  }

  if (!chi_prompt_queue_push_front(&queue, prompt)) {
    fprintf(stderr, "failed to queue initial prompt\n");
    chi_prompt_queue_destroy(&queue);
    return 1;
  }

  cfg.working_dir = working_dir;
  chi_make_session_id(cfg.session_id);
  printf("session: %s\n", cfg.session_id);

  while (queue.count > 0) {
    int finalized = 0;
    int turn;

    if (!chi_queue_pop_to_conversation(&queue, &convo)) {
      fprintf(stderr, "failed to append queued user message\n");
      exit_code = 1;
      goto cleanup;
    }

    for (turn = 0; turn < max_turns; turn++) {
      chi_action action;
      char *provider_err = NULL;

      memset(&action, 0, sizeof(action));
      if (!chi_provider_next_action(&cfg, &convo, &action, &provider_err)) {
        fprintf(stderr, "[final error] %s\n", provider_err == NULL ? "provider failed" : provider_err);
        free(provider_err);
        chi_action_reset(&action);
        exit_code = 1;
        goto cleanup;
      }

      if (action.is_tool && chi_is_blank(action.tool_call_id)) {
        call_seq++;
        action.tool_call_id = chi_format("call_%llu", call_seq);
        if (action.tool_call_id == NULL) {
          fprintf(stderr, "out of memory generating tool call id\n");
          chi_action_reset(&action);
          exit_code = 1;
          goto cleanup;
        }
      }

      if (!chi_conversation_add(
              &convo,
              "assistant",
              action.assistant_text,
              action.is_tool ? action.tool_call_id : NULL,
              action.is_tool ? action.tool_name : NULL,
              action.is_tool ? action.tool_arguments_json : NULL)) {
        fprintf(stderr, "failed to append assistant message\n");
        chi_action_reset(&action);
        exit_code = 1;
        goto cleanup;
      }

      if (!action.is_tool) {
        printf("[final]\n%s\n", action.final_text == NULL ? "" : action.final_text);
        chi_action_reset(&action);
        finalized = 1;
        break;
      }

      {
        char *call_id = NULL;
        char *tool_output = NULL;
        char *tool_error = NULL;
        char *tool_record = NULL;
        int is_error = 0;
        int tool_step_ok = 0;
        double timeout = action.timeout_seconds > 0 ? action.timeout_seconds : 0;
        const char *tool_name = chi_is_blank(action.tool_name) ? "bash" : action.tool_name;

        call_id = chi_strdup(action.tool_call_id);
        if (call_id == NULL) {
          fprintf(stderr, "out of memory preparing tool call id\n");
          goto tool_cleanup;
        }

        printf("[tool start] %s (%s)\n", tool_name, call_id);
        printf("[tool command]\n%s\n", action.tool_command == NULL ? "" : action.tool_command);
        if (!chi_run_bash(cfg.working_dir, action.tool_command, timeout, &tool_output, &is_error, &tool_error)) {
          fprintf(stderr, "failed to run tool\n");
          goto tool_cleanup;
        }

        printf("[tool done] %s (%s) error=%d\n", tool_name, call_id, is_error);
        if (!chi_is_blank(tool_output)) {
          printf("%s\n", tool_output);
        }

        if (tool_error != NULL) {
          tool_record = chi_format("%s\n\n[tool error] %s", tool_output == NULL ? "" : tool_output, tool_error);
        } else {
          tool_record = chi_strdup(tool_output == NULL ? "" : tool_output);
        }

        if (tool_record == NULL ||
            !chi_conversation_add(
                &convo,
                "toolResult",
                tool_record,
                call_id,
                NULL,
                NULL)) {
          fprintf(stderr, "failed to append tool result\n");
          goto tool_cleanup;
        }

        if (queue.count > 0) {
          if (!chi_queue_pop_to_conversation(&queue, &convo)) {
            fprintf(stderr, "failed to inject queued user message\n");
            goto tool_cleanup;
          }
        }

        tool_step_ok = 1;

      tool_cleanup:
        free(call_id);
        free(tool_output);
        free(tool_error);
        free(tool_record);
        if (!tool_step_ok) {
          chi_action_reset(&action);
          exit_code = 1;
          goto cleanup;
        }
      }

      chi_action_reset(&action);
    }

    if (!finalized) {
      fprintf(stderr, "[final error] reached max turns (%d)\n", max_turns);
      exit_code = 1;
      goto cleanup;
    }
  }

cleanup:
  chi_prompt_queue_destroy(&queue);
  chi_conversation_destroy(&convo);
  free(cfg.chatgpt_access_token);
  return exit_code;
}
