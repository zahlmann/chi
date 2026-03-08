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
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <curl/curl.h>

#include "apply_patch.h"

#define CHI_MAX_TURNS 256
#define CHI_MAX_OUTPUT_LINES 300
#define CHI_MAX_OUTPUT_BYTES (24 * 1024)
#define CHI_SESSION_ID_MAX 64
#define CHI_HTTP_CONNECT_TIMEOUT_DEFAULT 5
#define CHI_HTTP_MAX_TIME_DEFAULT 120

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
  int tool_is_custom;
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
  int tool_is_custom;
  char *assistant_text;
  char *reasoning_summary;
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
  const char *system_prompt_file;
  const char *session_dir;
  const char *working_dir;
  int debug;
  char session_id[CHI_SESSION_ID_MAX];
  char *system_prompt_text;
  char *session_path;
  char *loaded_model;
  char *loaded_reasoning_effort;
  char *loaded_working_dir;
  char *chatgpt_access_token;
} chi_config;

static int chi_parse_backend(const char *value, chi_backend *out);

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

static char *chi_read_named_text_file(const char *path, const char *label, char **err_out) {
  FILE *fp;
  char chunk[4096];
  size_t n;
  char *buf = NULL;
  size_t len = 0;
  size_t cap = 0;

  *err_out = NULL;

  if (chi_is_blank(path)) {
    *err_out = chi_format("%s path is blank", label == NULL ? "file" : label);
    return NULL;
  }

  fp = fopen(path, "rb");
  if (fp == NULL) {
    *err_out = chi_format("failed to open %s '%s': %s", label == NULL ? "file" : label, path, strerror(errno));
    return NULL;
  }

  while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) {
    if (!chi_append_n(&buf, &len, &cap, chunk, n)) {
      free(buf);
      fclose(fp);
      *err_out = chi_format("out of memory while loading %s", label == NULL ? "file" : label);
      return NULL;
    }
  }

  if (ferror(fp)) {
    free(buf);
    fclose(fp);
    *err_out = chi_format("failed to read %s '%s': %s", label == NULL ? "file" : label, path, strerror(errno));
    return NULL;
  }

  if (fclose(fp) != 0) {
    free(buf);
    *err_out = chi_format("failed to close %s '%s': %s", label == NULL ? "file" : label, path, strerror(errno));
    return NULL;
  }

  if (chi_is_blank(buf)) {
    free(buf);
    *err_out = chi_format("%s '%s' is empty", label == NULL ? "file" : label, path);
    return NULL;
  }

  return buf;
}

static char *chi_read_text_file(const char *path, char **err_out) {
  return chi_read_named_text_file(path, "system prompt file", err_out);
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
    const char *arguments_json,
    int tool_is_custom) {
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
  m.tool_is_custom = tool_is_custom ? 1 : 0;

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

static char *chi_json_get_object(const char *json, const char *key) {
  const char *p;
  const char *start;
  const char *end = NULL;
  int depth = 0;
  int in_string = 0;
  int escaped = 0;
  size_t n;
  char *out;

  p = chi_find_json_key(json, key);
  if (p == NULL || *p != '{') {
    return NULL;
  }

  start = p;
  while (*p != '\0') {
    char c = *p;

    if (in_string) {
      if (escaped) {
        escaped = 0;
      } else if (c == '\\') {
        escaped = 1;
      } else if (c == '"') {
        in_string = 0;
      }
      p++;
      continue;
    }

    if (c == '"') {
      in_string = 1;
      p++;
      continue;
    }

    if (c == '{') {
      depth++;
    } else if (c == '}') {
      depth--;
      if (depth == 0) {
        end = p;
        break;
      }
      if (depth < 0) {
        return NULL;
      }
    }

    p++;
  }

  if (end == NULL) {
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

static void chi_trim_trailing_newlines(char *text) {
  size_t len;

  if (text == NULL) {
    return;
  }

  len = strlen(text);
  while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == '\r')) {
    text[--len] = '\0';
  }
}

static const char *chi_backend_name(chi_backend backend) {
  if (backend == CHI_BACKEND_CHATGPT) {
    return "chatgpt";
  }
  return "openai";
}

static int chi_ensure_directory(const char *path, char **err_out) {
  struct stat st;

  *err_out = NULL;

  if (chi_is_blank(path)) {
    *err_out = chi_strdup("session directory path is blank");
    return 0;
  }

  if (stat(path, &st) == 0) {
    if (S_ISDIR(st.st_mode)) {
      return 1;
    }
    *err_out = chi_format("session path '%s' exists and is not a directory", path);
    return 0;
  }

  if (mkdir(path, 0700) == 0) {
    return 1;
  }

  if (errno == EEXIST) {
    return 1;
  }

  *err_out = chi_format("failed to create session directory '%s': %s", path, strerror(errno));
  return 0;
}

static char *chi_build_session_path(const char *session_dir, const char *session_id) {
  if (chi_is_blank(session_dir) || chi_is_blank(session_id)) {
    return NULL;
  }
  return chi_format("%s/%s.jsonl", session_dir, session_id);
}

static char *chi_strdup_nonblank_or_null(const char *text) {
  char *copy;

  if (chi_is_blank(text)) {
    return NULL;
  }

  copy = chi_strdup(text);
  return copy;
}

static int chi_session_write_meta(FILE *fp, const chi_config *cfg) {
  char *session_id = NULL;
  char *backend = NULL;
  char *model = NULL;
  char *reasoning = NULL;
  char *working_dir = NULL;
  char *system_prompt = NULL;
  int ok = 0;

  session_id = chi_json_escape(cfg->session_id);
  backend = chi_json_escape(chi_backend_name(cfg->backend));
  model = chi_json_escape(cfg->model == NULL ? "" : cfg->model);
  reasoning = chi_json_escape(cfg->reasoning_effort == NULL ? "" : cfg->reasoning_effort);
  working_dir = chi_json_escape(cfg->working_dir == NULL ? "" : cfg->working_dir);
  system_prompt = chi_json_escape(cfg->system_prompt_text == NULL ? "" : cfg->system_prompt_text);
  if (session_id == NULL || backend == NULL || model == NULL || reasoning == NULL || working_dir == NULL ||
      system_prompt == NULL) {
    goto done;
  }

  if (fprintf(fp,
              "{\"type\":\"meta\",\"version\":1,\"session_id\":\"%s\",\"backend\":\"%s\","
              "\"model\":\"%s\",\"reasoning_effort\":\"%s\",\"working_dir\":\"%s\",\"system_prompt_text\":\"%s\"}\n",
              session_id,
              backend,
              model,
              reasoning,
              working_dir,
              system_prompt) < 0) {
    goto done;
  }

  ok = 1;

done:
  free(session_id);
  free(backend);
  free(model);
  free(reasoning);
  free(working_dir);
  free(system_prompt);
  return ok;
}

static int chi_session_write_message(FILE *fp, const chi_message *m) {
  char *role = NULL;
  char *text = NULL;
  char *tool_call_id = NULL;
  char *tool_name = NULL;
  char *arguments_json = NULL;
  int ok = 0;

  role = chi_json_escape(m->role == NULL ? "" : m->role);
  text = chi_json_escape(m->text == NULL ? "" : m->text);
  tool_call_id = chi_json_escape(m->tool_call_id == NULL ? "" : m->tool_call_id);
  tool_name = chi_json_escape(m->tool_name == NULL ? "" : m->tool_name);
  arguments_json = chi_json_escape(m->arguments_json == NULL ? "" : m->arguments_json);
  if (role == NULL || text == NULL || tool_call_id == NULL || tool_name == NULL || arguments_json == NULL) {
    goto done;
  }

  if (fprintf(fp,
              "{\"type\":\"message\",\"role\":\"%s\",\"text\":\"%s\",\"tool_call_id\":\"%s\","
              "\"tool_name\":\"%s\",\"arguments_json\":\"%s\",\"tool_is_custom\":%d}\n",
              role,
              text,
              tool_call_id,
              tool_name,
              arguments_json,
              m->tool_is_custom ? 1 : 0) < 0) {
    goto done;
  }

  ok = 1;

done:
  free(role);
  free(text);
  free(tool_call_id);
  free(tool_name);
  free(arguments_json);
  return ok;
}

static int chi_session_save(const chi_config *cfg, const chi_conversation *conversation, char **err_out) {
  FILE *fp = NULL;
  size_t i;

  *err_out = NULL;

  if (chi_is_blank(cfg->session_dir) || chi_is_blank(cfg->session_id) || chi_is_blank(cfg->session_path)) {
    *err_out = chi_strdup("session config is incomplete");
    return 0;
  }

  if (!chi_ensure_directory(cfg->session_dir, err_out)) {
    return 0;
  }

  fp = fopen(cfg->session_path, "wb");
  if (fp == NULL) {
    *err_out = chi_format("failed to open session file '%s': %s", cfg->session_path, strerror(errno));
    return 0;
  }

  if (!chi_session_write_meta(fp, cfg)) {
    *err_out = chi_strdup("failed to write session metadata");
    fclose(fp);
    return 0;
  }

  for (i = 0; i < conversation->count; i++) {
    if (!chi_session_write_message(fp, &conversation->items[i])) {
      *err_out = chi_strdup("failed to write session message");
      fclose(fp);
      return 0;
    }
  }

  if (fclose(fp) != 0) {
    *err_out = chi_format("failed to close session file '%s': %s", cfg->session_path, strerror(errno));
    return 0;
  }

  return 1;
}

static int chi_session_load(
    chi_config *cfg,
    chi_conversation *conversation,
    const char *session_id,
    int keep_backend,
    int keep_model,
    int keep_reasoning,
    int keep_system_prompt,
    int keep_working_dir,
    char **err_out) {
  FILE *fp = NULL;
  char *line = NULL;
  size_t line_cap = 0;
  ssize_t line_len;
  int saw_meta = 0;

  *err_out = NULL;

  if (chi_is_blank(cfg->session_path) || chi_is_blank(session_id)) {
    *err_out = chi_strdup("session config is incomplete");
    return 0;
  }

  fp = fopen(cfg->session_path, "rb");
  if (fp == NULL) {
    *err_out = chi_format("failed to open session file '%s': %s", cfg->session_path, strerror(errno));
    return 0;
  }

  while ((line_len = getline(&line, &line_cap, fp)) >= 0) {
    char *type = NULL;

    (void)line_len;
    chi_trim_trailing_newlines(line);
    if (chi_is_blank(line)) {
      continue;
    }

    type = chi_json_get_string(line, "type");
    if (type == NULL) {
      *err_out = chi_format("invalid session record in '%s'", cfg->session_path);
      free(type);
      goto fail;
    }

    if (strcmp(type, "meta") == 0) {
      char *stored_id = chi_json_get_string(line, "session_id");
      char *backend = chi_json_get_string(line, "backend");
      char *model = chi_json_get_string(line, "model");
      char *reasoning = chi_json_get_string(line, "reasoning_effort");
      char *working_dir = chi_json_get_string(line, "working_dir");
      char *system_prompt = chi_json_get_string(line, "system_prompt_text");

      if (stored_id == NULL || strcmp(stored_id, session_id) != 0) {
        *err_out = chi_format("session file '%s' does not match session id '%s'", cfg->session_path, session_id);
        free(stored_id);
        free(backend);
        free(model);
        free(reasoning);
        free(working_dir);
        free(system_prompt);
        free(type);
        goto fail;
      }

      if (!keep_backend && !chi_is_blank(backend) && !chi_parse_backend(backend, &cfg->backend)) {
        *err_out = chi_format("session '%s' contains unsupported backend '%s'", session_id, backend);
        free(stored_id);
        free(backend);
        free(model);
        free(reasoning);
        free(working_dir);
        free(system_prompt);
        free(type);
        goto fail;
      }

      if (!keep_model && !chi_is_blank(model)) {
        free(cfg->loaded_model);
        cfg->loaded_model = model;
        cfg->model = cfg->loaded_model;
        model = NULL;
      }

      if (!keep_reasoning && !chi_is_blank(reasoning)) {
        free(cfg->loaded_reasoning_effort);
        cfg->loaded_reasoning_effort = reasoning;
        cfg->reasoning_effort = cfg->loaded_reasoning_effort;
        reasoning = NULL;
      }

      if (!keep_working_dir && !chi_is_blank(working_dir)) {
        free(cfg->loaded_working_dir);
        cfg->loaded_working_dir = working_dir;
        cfg->working_dir = cfg->loaded_working_dir;
        working_dir = NULL;
      }

      if (!keep_system_prompt && !chi_is_blank(system_prompt)) {
        free(cfg->system_prompt_text);
        cfg->system_prompt_text = system_prompt;
        system_prompt = NULL;
      }

      saw_meta = 1;
      free(stored_id);
      free(backend);
      free(model);
      free(reasoning);
      free(working_dir);
      free(system_prompt);
      free(type);
      continue;
    }

    if (strcmp(type, "message") == 0) {
      char *role = chi_json_get_string(line, "role");
      char *text = chi_json_get_string(line, "text");
      char *tool_call_id = chi_json_get_string(line, "tool_call_id");
      char *tool_name = chi_json_get_string(line, "tool_name");
      char *arguments_json = chi_json_get_string(line, "arguments_json");
      double tool_is_custom_num = 0;
      int tool_is_custom = chi_json_get_number(line, "tool_is_custom", &tool_is_custom_num) && tool_is_custom_num > 0;
      char *tool_call_id_opt = NULL;
      char *tool_name_opt = NULL;
      char *arguments_json_opt = NULL;

      if (role == NULL || text == NULL) {
        *err_out = chi_format("invalid session message in '%s'", cfg->session_path);
        free(role);
        free(text);
        free(tool_call_id);
        free(tool_name);
        free(arguments_json);
        free(type);
        goto fail;
      }

      tool_call_id_opt = chi_strdup_nonblank_or_null(tool_call_id);
      tool_name_opt = chi_strdup_nonblank_or_null(tool_name);
      arguments_json_opt = chi_strdup_nonblank_or_null(arguments_json);

      if ((tool_call_id_opt == NULL && !chi_is_blank(tool_call_id)) ||
          (tool_name_opt == NULL && !chi_is_blank(tool_name)) ||
          (arguments_json_opt == NULL && !chi_is_blank(arguments_json)) ||
          !chi_conversation_add(
              conversation, role, text, tool_call_id_opt, tool_name_opt, arguments_json_opt, tool_is_custom)) {
        *err_out = chi_strdup("failed to restore session message");
        free(role);
        free(text);
        free(tool_call_id);
        free(tool_name);
        free(arguments_json);
        free(tool_call_id_opt);
        free(tool_name_opt);
        free(arguments_json_opt);
        free(type);
        goto fail;
      }

      free(role);
      free(text);
      free(tool_call_id);
      free(tool_name);
      free(arguments_json);
      free(tool_call_id_opt);
      free(tool_name_opt);
      free(arguments_json_opt);
      free(type);
      continue;
    }

    *err_out = chi_format("unsupported session record type '%s' in '%s'", type, cfg->session_path);
    free(type);
    goto fail;
  }

  if (ferror(fp)) {
    *err_out = chi_format("failed to read session file '%s': %s", cfg->session_path, strerror(errno));
    goto fail;
  }

  if (!saw_meta) {
    *err_out = chi_format("session file '%s' is missing metadata", cfg->session_path);
    goto fail;
  }

  free(line);
  if (fclose(fp) != 0) {
    *err_out = chi_format("failed to close session file '%s': %s", cfg->session_path, strerror(errno));
    return 0;
  }
  return 1;

fail:
  free(line);
  fclose(fp);
  return 0;
}

static unsigned long long chi_parse_generated_call_seq(const char *tool_call_id) {
  unsigned long long value = 0;
  const char *p;

  if (tool_call_id == NULL || strncmp(tool_call_id, "call_", 5) != 0) {
    return 0;
  }

  p = tool_call_id + 5;
  if (*p == '\0') {
    return 0;
  }

  while (*p != '\0') {
    if (!isdigit((unsigned char)*p)) {
      return 0;
    }
    value = value * 10ULL + (unsigned long long)(*p - '0');
    p++;
  }

  return value;
}

static unsigned long long chi_conversation_max_generated_call_seq(const chi_conversation *conversation) {
  size_t i;
  unsigned long long max_value = 0;

  if (conversation == NULL) {
    return 0;
  }

  for (i = 0; i < conversation->count; i++) {
    unsigned long long current = chi_parse_generated_call_seq(conversation->items[i].tool_call_id);
    if (current > max_value) {
      max_value = current;
    }
  }

  return max_value;
}

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} chi_http_buffer;

static size_t chi_curl_write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
  size_t total = size * nmemb;
  chi_http_buffer *buffer = (chi_http_buffer *)userdata;

  if (total == 0) {
    return 0;
  }

  if (!chi_append_n(&buffer->data, &buffer->len, &buffer->cap, (const char *)ptr, total)) {
    return 0;
  }

  return total;
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
  static int curl_global_ready = 0;
  CURL *curl = NULL;
  struct curl_slist *curl_headers = NULL;
  chi_http_buffer body;
  int ok = 0;
  int connect_timeout = 0;
  int max_time = 0;
  const char *fail_msg = "curl request failed";
  char *fail_detail = NULL;
  char errbuf[CURL_ERROR_SIZE];
  size_t i;
  CURLcode curl_code;
  long status_code = 0;

  *response_body = NULL;
  *http_code = 0;
  *err_out = NULL;

  body.data = NULL;
  body.len = 0;
  body.cap = 0;
  memset(errbuf, 0, sizeof(errbuf));

  connect_timeout = chi_env_positive_int("CHI_HTTP_CONNECT_TIMEOUT", CHI_HTTP_CONNECT_TIMEOUT_DEFAULT);
  max_time = chi_env_positive_int("CHI_HTTP_MAX_TIME", CHI_HTTP_MAX_TIME_DEFAULT);
  if (max_time < connect_timeout) {
    max_time = connect_timeout;
  }

  if (!curl_global_ready) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
      fail_msg = "failed to initialize libcurl";
      goto cleanup;
    }
    curl_global_ready = 1;
  }

  curl = curl_easy_init();
  if (curl == NULL) {
    fail_msg = "failed to initialize curl request";
    goto cleanup;
  }

  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, (long)connect_timeout);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)max_time);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, chi_curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "chi/1");

  if (method != NULL && strcmp(method, "POST") == 0) {
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
  } else {
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  }

  for (i = 0; i < header_count; i++) {
    curl_headers = curl_slist_append(curl_headers, headers[i]);
    if (curl_headers == NULL) {
      fail_msg = "out of memory while building request headers";
      goto cleanup;
    }
  }
  if (curl_headers != NULL) {
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_headers);
  }

  if (request_body != NULL) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(request_body));
  }

  curl_code = curl_easy_perform(curl);
  if (curl_code != CURLE_OK) {
    if (!chi_is_blank(errbuf)) {
      fail_detail = chi_format("curl request failed: %s", errbuf);
    } else {
      fail_detail = chi_format("curl request failed: %s", curl_easy_strerror(curl_code));
    }
    fail_msg = fail_detail != NULL ? fail_detail : "curl request failed";
    goto cleanup;
  }

  if (curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code) != CURLE_OK) {
    fail_msg = "failed to read HTTP status code";
    goto cleanup;
  }

  if (status_code < 0 || status_code > INT32_MAX) {
    fail_msg = "invalid HTTP status code from curl";
    goto cleanup;
  }

  if (body.data == NULL) {
    body.data = chi_strdup("");
    if (body.data == NULL) {
      fail_msg = "out of memory storing response body";
      goto cleanup;
    }
  }

  *response_body = body.data;
  body.data = NULL;
  *http_code = (int)status_code;
  ok = 1;

cleanup:
  if (curl_headers != NULL) {
    curl_slist_free_all(curl_headers);
  }
  if (curl != NULL) {
    curl_easy_cleanup(curl);
  }
  free(body.data);
  if (!ok) {
    *err_out = chi_strdup(fail_msg);
    if (*err_out == NULL) {
      *err_out = chi_strdup("request failed (and out of memory while reporting error)");
    }
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

static int chi_extract_responses_tool_call(
    const char *json,
    char **call_id_out,
    char **tool_name_out,
    char **tool_payload_out,
    int *is_custom_out) {
  const char *p;

  *call_id_out = NULL;
  *tool_name_out = NULL;
  *tool_payload_out = NULL;
  *is_custom_out = 0;

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
    if (type != NULL && strcmp(type, "custom_tool_call") == 0) {
      char *call_id = chi_json_get_string(p, "call_id");
      char *tool_name = chi_json_get_string(p, "name");
      char *tool_input = chi_json_get_string(p, "input");
      free(type);

      if (tool_input == NULL) {
        free(call_id);
        free(tool_name);
        return 0;
      }

      *call_id_out = call_id;
      *tool_name_out = tool_name;
      *tool_payload_out = tool_input;
      *is_custom_out = 1;
      return 1;
    }
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
      *tool_payload_out = arguments;
      *is_custom_out = 0;
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

static char *chi_extract_responses_reasoning_summary_text(const char *json) {
  const char *p;
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (json == NULL) {
    return NULL;
  }

  p = json;
  while (p != NULL) {
    char *type;
    char *text;

    p = strstr(p, "\"type\"");
    if (p == NULL) {
      break;
    }

    type = chi_json_get_string(p, "type");
    if (type != NULL && strcmp(type, "summary_text") == 0) {
      text = chi_json_get_string(p, "text");
      if (!chi_is_blank(text)) {
        if (len > 0 && !chi_append(&out, &len, &cap, "\n")) {
          free(type);
          free(text);
          free(out);
          return NULL;
        }
        if (!chi_append(&out, &len, &cap, text)) {
          free(type);
          free(text);
          free(out);
          return NULL;
        }
      }
      free(text);
    }
    free(type);
    p += 6;
  }

  if (chi_is_blank(out)) {
    free(out);
    return NULL;
  }
  return out;
}

static char *chi_extract_sse_reasoning_summary_done(const char *body) {
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

        if (chi_sse_type_matches(candidate, "response.reasoning_summary_text.done")) {
          char *text = chi_json_get_string(candidate, "text");
          if (!chi_is_blank(text)) {
            if (len > 0 && !chi_append(&out, &len, &cap, "\n")) {
              free(text);
              free(candidate);
              free(out);
              return NULL;
            }
            if (!chi_append(&out, &len, &cap, text)) {
              free(text);
              free(candidate);
              free(out);
              return NULL;
            }
          }
          free(text);
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

static char *chi_extract_sse_reasoning_summary_deltas(const char *body) {
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

        if (chi_sse_type_matches(candidate, "response.reasoning_summary_text.delta")) {
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
  free(a->reasoning_summary);
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

static int chi_append_responses_system_message(
    const char *text,
    char **buf,
    size_t *len,
    size_t *cap,
    int *first_item) {
  if (chi_is_blank(text)) {
    return 1;
  }
  if (!chi_append_input_item_start(buf, len, cap, first_item) ||
      !chi_append(buf, len, cap, "{\"type\":\"message\",\"role\":\"system\",\"content\":[{\"type\":\"input_text\",\"text\":") ||
      !chi_append_json_quoted(buf, len, cap, text) ||
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
    const char *tool_payload = m->arguments_json;
    if (chi_is_blank(call_id)) {
      snprintf(fallback_call_id, sizeof(fallback_call_id), "call_%zu", index + 1);
      call_id = fallback_call_id;
    }
    if (chi_is_blank(tool_name)) {
      tool_name = "tool";
    }
    if (chi_is_blank(tool_payload)) {
      tool_payload = m->tool_is_custom ? "" : "{}";
    }

    if (m->tool_is_custom) {
      if (!chi_append_input_item_start(buf, len, cap, first_item) ||
          !chi_append(buf, len, cap, "{\"type\":\"custom_tool_call\",\"call_id\":") ||
          !chi_append_json_quoted(buf, len, cap, call_id) ||
          !chi_append(buf, len, cap, ",\"name\":") ||
          !chi_append_json_quoted(buf, len, cap, tool_name) ||
          !chi_append(buf, len, cap, ",\"input\":") ||
          !chi_append_json_quoted(buf, len, cap, tool_payload) ||
          !chi_append(buf, len, cap, "}")) {
        return 0;
      }
    } else {
      if (!chi_append_input_item_start(buf, len, cap, first_item) ||
          !chi_append(buf, len, cap, "{\"type\":\"function_call\",\"call_id\":") ||
          !chi_append_json_quoted(buf, len, cap, call_id) ||
          !chi_append(buf, len, cap, ",\"name\":") ||
          !chi_append_json_quoted(buf, len, cap, tool_name) ||
          !chi_append(buf, len, cap, ",\"arguments\":") ||
          !chi_append_json_quoted(buf, len, cap, tool_payload) ||
          !chi_append(buf, len, cap, "}")) {
        return 0;
      }
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

  if (m->tool_is_custom) {
    if (!chi_append_input_item_start(buf, len, cap, first_item) ||
        !chi_append(buf, len, cap, "{\"type\":\"custom_tool_call_output\",\"call_id\":") ||
        !chi_append_json_quoted(buf, len, cap, m->tool_call_id) ||
        !chi_append(buf, len, cap, ",\"output\":") ||
        !chi_append_json_quoted(buf, len, cap, output) ||
        !chi_append(buf, len, cap, "}")) {
      return 0;
    }
  } else {
    if (!chi_append_input_item_start(buf, len, cap, first_item) ||
        !chi_append(buf, len, cap, "{\"type\":\"function_call_output\",\"call_id\":") ||
        !chi_append_json_quoted(buf, len, cap, m->tool_call_id) ||
        !chi_append(buf, len, cap, ",\"output\":") ||
        !chi_append_json_quoted(buf, len, cap, output) ||
        !chi_append(buf, len, cap, "}")) {
      return 0;
    }
  }
  return 1;
}

static int chi_append_responses_input(
    const chi_conversation *conversation,
    const char *system_text,
    char **buf,
    size_t *len,
    size_t *cap) {
  size_t i;
  int first_item = 1;

  if (!chi_append(buf, len, cap, "\"input\":[")) {
    return 0;
  }

  if (!chi_append_responses_system_message(system_text, buf, len, cap, &first_item)) {
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
  const char *system_prompt = cfg->system_prompt_text;
  const char *input_system_prompt = cfg->backend == CHI_BACKEND_CHATGPT ? NULL : system_prompt;
  const char *reasoning_effort = chi_normalize_reasoning_effort(cfg->reasoning_effort);
  const char *apply_patch_description =
      "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.";
  const char *apply_patch_grammar =
      "start: begin_patch hunk+ end_patch\n"
      "begin_patch: \"*** Begin Patch\" LF\n"
      "end_patch: \"*** End Patch\" LF?\n"
      "hunk: add_hunk | delete_hunk | update_hunk\n"
      "add_hunk: \"*** Add File: \" filename LF add_line+\n"
      "delete_hunk: \"*** Delete File: \" filename LF\n"
      "update_hunk: \"*** Update File: \" filename LF change_move? change?\n"
      "filename: /(.+)/\n"
      "add_line: \"+\" /(.*)/ LF -> line\n"
      "change_move: \"*** Move to: \" filename LF\n"
      "change: (change_context | change_line)+ eof_line?\n"
      "change_context: (\"@@\" | \"@@ \" /(.+)/) LF\n"
      "change_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\n"
      "eof_line: \"*** End of File\" LF\n"
      "%import common.LF\n";
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
      !chi_append(&json, &len, &cap, ",") ||
      !chi_append_responses_input(conversation, input_system_prompt, &json, &len, &cap) ||
      !chi_append(&json, &len, &cap, ",\"tools\":[") ||
      !chi_append(&json, &len, &cap, bash_tool_json) ||
      !chi_append(&json, &len, &cap, ",{\"type\":\"custom\",\"name\":\"apply_patch\",\"description\":") ||
      !chi_append_json_quoted(&json, &len, &cap, apply_patch_description) ||
      !chi_append(&json, &len, &cap, ",\"format\":{\"type\":\"grammar\",\"syntax\":\"lark\",\"definition\":") ||
      !chi_append_json_quoted(&json, &len, &cap, apply_patch_grammar) ||
      !chi_append(&json, &len, &cap, "}}") ||
      !chi_append(&json, &len, &cap, "]") ||
      !chi_append(&json, &len, &cap, ",\"tool_choice\":\"auto\",\"parallel_tool_calls\":true")) {
    free(json);
    *err_out = chi_strdup("out of memory while building provider request");
    return NULL;
  }

  if (reasoning_effort != NULL) {
    if (!chi_append(&json, &len, &cap, ",\"reasoning\":{\"effort\":") ||
        !chi_append_json_quoted(&json, &len, &cap, reasoning_effort) ||
        (!chi_equals_ignore_case(reasoning_effort, "none") &&
         (!chi_append(&json, &len, &cap, ",\"summary\":\"auto\""))) ||
        !chi_append(&json, &len, &cap, "}")) {
      free(json);
      *err_out = chi_strdup("out of memory while building provider request");
      return NULL;
    }
  }

  if (cfg->backend == CHI_BACKEND_CHATGPT) {
    if (!chi_append(&json, &len, &cap, ",\"instructions\":") ||
        !chi_append_json_quoted(&json, &len, &cap, system_prompt == NULL ? "" : system_prompt)) {
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
  char *reasoning_summary = NULL;
  char *status = NULL;
  char *reason = NULL;
  char *call_id = NULL;
  char *tool_name = NULL;
  char *tool_call_payload = NULL;
  double timeout = 0;
  int has_tool = 0;
  int tool_is_custom = 0;
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
    has_tool = chi_extract_responses_tool_call(
        tool_payload, &call_id, &tool_name, &tool_call_payload, &tool_is_custom);
  }
  if (!has_tool) {
    has_tool = chi_extract_responses_tool_call(
        payload, &call_id, &tool_name, &tool_call_payload, &tool_is_custom);
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
  reasoning_summary = chi_extract_responses_reasoning_summary_text(payload);
  if (reasoning_summary == NULL) {
    reasoning_summary = chi_extract_sse_reasoning_summary_done(response_body);
  }
  if (reasoning_summary == NULL) {
    reasoning_summary = chi_extract_sse_reasoning_summary_deltas(response_body);
  }

  if (has_tool) {
    action->assistant_text = output_text != NULL ? output_text : chi_strdup("");
    action->reasoning_summary = reasoning_summary;
    reasoning_summary = NULL;
    output_text = NULL;
    action->tool_call_id = call_id;
    action->tool_name = tool_name;
    action->tool_arguments_json = tool_call_payload;
    action->tool_is_custom = tool_is_custom;
    call_id = NULL;
    tool_name = NULL;
    tool_call_payload = NULL;

    if (action->assistant_text == NULL || action->tool_arguments_json == NULL) {
      chi_action_reset(action);
      *err_out = chi_strdup("out of memory while parsing tool call");
      free(status);
      free(reason);
      free(reasoning_summary);
      free(payload);
      free(tool_payload);
      free(output_done_payload);
      free(call_id);
      free(tool_name);
      free(tool_call_payload);
      return 0;
    }

    if (chi_is_blank(action->tool_name)) {
      free(action->tool_name);
      action->tool_name = chi_strdup(action->tool_is_custom ? "tool" : "bash");
      if (action->tool_name == NULL) {
        chi_action_reset(action);
        *err_out = chi_strdup("out of memory while parsing tool call");
        free(status);
        free(reason);
        free(reasoning_summary);
        free(payload);
        free(tool_payload);
        free(output_done_payload);
        free(call_id);
        free(tool_name);
        free(tool_call_payload);
        return 0;
      }
    }

    if (action->tool_is_custom) {
      action->tool_command = chi_strdup(action->tool_arguments_json);
      if (chi_is_blank(action->tool_command)) {
        free(action->tool_command);
        action->tool_command = NULL;
        action->final_text = chi_strdup("tool call missing input");
        action->is_tool = 0;
        ok = action->final_text != NULL;
      } else {
        action->is_tool = 1;
        ok = 1;
      }
    } else {
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
    }

    free(status);
    free(reason);
    free(reasoning_summary);
    free(payload);
    free(tool_payload);
    free(output_done_payload);
    free(call_id);
    free(tool_name);
    free(tool_call_payload);
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
    free(tool_call_payload);
    if (*err_out == NULL) {
      *err_out = chi_format("could not parse output text from provider response: %s", response_body);
    }
    return 0;
  }

  free(status);
  free(reason);
  ok = chi_parse_action(output_text, action, err_out);
  if (ok) {
    action->reasoning_summary = reasoning_summary;
    reasoning_summary = NULL;
  }
  free(payload);
  free(tool_payload);
  free(output_done_payload);
  free(output_text);
  free(reasoning_summary);
  free(call_id);
  free(tool_name);
  free(tool_call_payload);
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
  char *tmp_err = NULL;
  char *auth_path = NULL;
  char *auth_json = NULL;
  char *tokens_json = NULL;
  char *token = NULL;
  const char *home;

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

  home = getenv("HOME");
  if (chi_is_blank(home)) {
    *err_out = chi_strdup("set CHATGPT_ACCESS_TOKEN or ensure ~/.codex/auth.json contains tokens.access_token");
    return 0;
  }

  auth_path = chi_format("%s/.codex/auth.json", home);
  if (auth_path == NULL) {
    *err_out = chi_strdup("out of memory while building ~/.codex/auth.json path");
    return 0;
  }

  auth_json = chi_read_named_text_file(auth_path, "codex auth file", &tmp_err);
  if (auth_json == NULL) {
    free(auth_path);
    free(tmp_err);
    *err_out = chi_strdup("set CHATGPT_ACCESS_TOKEN or ensure ~/.codex/auth.json contains tokens.access_token");
    return 0;
  }

  tokens_json = chi_json_get_object(auth_json, "tokens");
  if (tokens_json == NULL) {
    free(auth_json);
    free(auth_path);
    *err_out = chi_strdup("set CHATGPT_ACCESS_TOKEN or ensure ~/.codex/auth.json contains tokens.access_token");
    return 0;
  }

  token = chi_json_get_string(tokens_json, "access_token");
  free(tokens_json);
  free(auth_json);
  free(auth_path);

  if (chi_is_blank(token)) {
    free(token);
    *err_out = chi_strdup("set CHATGPT_ACCESS_TOKEN or ensure ~/.codex/auth.json contains tokens.access_token");
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

static int chi_run_apply_patch_tool(
    const char *cwd,
    const char *patch_input,
    char **display_out,
    int *is_error_out,
    char **error_text_out) {
  char *summary = NULL;
  char *patch_error = NULL;

  *display_out = NULL;
  *is_error_out = 1;
  *error_text_out = NULL;

  if (chi_is_blank(patch_input)) {
    *display_out = chi_strdup("(no output)");
    *error_text_out = chi_strdup("tool call missing input");
    return *display_out != NULL && *error_text_out != NULL;
  }

  if (chi_apply_patch(cwd, patch_input, &summary, &patch_error)) {
    *display_out = summary == NULL ? chi_strdup("") : summary;
    *is_error_out = 0;
    *error_text_out = NULL;
    free(patch_error);
    return *display_out != NULL;
  }

  *display_out = chi_strdup("");
  *is_error_out = 1;
  *error_text_out = patch_error == NULL ? chi_strdup("apply_patch failed") : patch_error;
  free(summary);
  return *display_out != NULL && *error_text_out != NULL;
}

static int chi_parse_backend(const char *value, chi_backend *out) {
  if (value == NULL || strcmp(value, "chatgpt") == 0) {
    *out = CHI_BACKEND_CHATGPT;
    return 1;
  }
  if (strcmp(value, "openai") == 0 || strcmp(value, "openai_api") == 0) {
    *out = CHI_BACKEND_OPENAI;
    return 1;
  }
  return 0;
}

static void chi_usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s [--backend openai|chatgpt] [--model MODEL] [--reasoning EFFORT] [--system-prompt-file PATH] [--session SESSION_ID] [--queue \"prompt\"] \"prompt\" [working_dir]\n"
          "example: %s \"Edit hello.py to print hi and run it\" .\n"
          "resume:  %s --session session-abc123 \"continue\" .\n"
          "\n"
          "env:\n"
          "  OPENAI_API_KEY               auth for openai backend when selected\n"
          "  CHATGPT_ACCESS_TOKEN         direct auth for default chatgpt backend (else ~/.codex/auth.json)\n"
          "  CHI_BACKEND                  backend override (default: chatgpt; openai|chatgpt)\n"
          "  CHI_MODEL                    default model (default: gpt-5.2-codex)\n"
          "  CHI_REASONING_EFFORT         default reasoning effort (default: high)\n"
          "  CHI_SESSION_DIR              session state dir (default: .chi-sessions)\n"
          "  CHI_SYSTEM_PROMPT_FILE       path to custom system prompt text file\n"
          "  CHI_HTTP_CONNECT_TIMEOUT     curl connect timeout seconds (default: 5)\n"
          "  CHI_HTTP_MAX_TIME            curl total timeout seconds (default: 120)\n"
          "  CHI_DEBUG                    set to non-zero for debug logs\n",
          argv0,
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
  ok = chi_conversation_add(convo, "user", prompt, NULL, NULL, NULL, 0);
  free(prompt);
  return ok;
}

int main(int argc, char **argv) {
  chi_config cfg;
  chi_conversation convo;
  chi_prompt_queue queue;
  const char *prompt = NULL;
  const char *working_dir = ".";
  const char *resume_session_id = NULL;
  int max_turns = CHI_MAX_TURNS;
  int i;
  int exit_code = 0;
  unsigned long long call_seq = 0;
  int backend_explicit = 0;
  int model_explicit = 0;
  int reasoning_explicit = 0;
  int system_prompt_explicit = 0;
  int working_dir_explicit = 0;

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
  cfg.session_dir = getenv("CHI_SESSION_DIR");
  if (chi_is_blank(cfg.session_dir)) {
    cfg.session_dir = ".chi-sessions";
  }
  cfg.system_prompt_file = getenv("CHI_SYSTEM_PROMPT_FILE");
  if (chi_is_blank(cfg.system_prompt_file)) {
    cfg.system_prompt_file = NULL;
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
      backend_explicit = 1;
      i++;
      continue;
    }

    if (strcmp(argv[i], "--model") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --model value");
      }
      cfg.model = argv[++i];
      model_explicit = 1;
      continue;
    }

    if (strcmp(argv[i], "--reasoning") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --reasoning value");
      }
      cfg.reasoning_effort = argv[++i];
      reasoning_explicit = 1;
      continue;
    }

    if (strcmp(argv[i], "--system-prompt-file") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --system-prompt-file value");
      }
      cfg.system_prompt_file = argv[++i];
      system_prompt_explicit = 1;
      free(cfg.system_prompt_text);
      cfg.system_prompt_text = NULL;
      continue;
    }

    if (strcmp(argv[i], "--session") == 0) {
      if (i + 1 >= argc || chi_is_blank(argv[i + 1])) {
        return chi_arg_fail(&queue, "missing --session value");
      }
      resume_session_id = argv[++i];
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
      working_dir_explicit = 1;
      continue;
    }

    fprintf(stderr, "unexpected argument: %s\n", argv[i]);
    return chi_arg_usage_fail(&queue, argv[0]);
  }

  if (chi_is_blank(prompt)) {
    return chi_arg_usage_fail(&queue, argv[0]);
  }

  if (!chi_is_blank(resume_session_id)) {
    if (strlen(resume_session_id) >= sizeof(cfg.session_id)) {
      fprintf(stderr, "session id is too long\n");
      chi_prompt_queue_destroy(&queue);
      return 2;
    }
    memcpy(cfg.session_id, resume_session_id, strlen(resume_session_id) + 1);
  } else {
    chi_make_session_id(cfg.session_id);
  }

  cfg.session_path = chi_build_session_path(cfg.session_dir, cfg.session_id);
  if (cfg.session_path == NULL) {
    fprintf(stderr, "failed to build session path\n");
    chi_prompt_queue_destroy(&queue);
    return 1;
  }

  if (!chi_is_blank(resume_session_id)) {
    char *load_err = NULL;
    cfg.working_dir = working_dir;
    if (!chi_session_load(
            &cfg,
            &convo,
            resume_session_id,
            backend_explicit,
            model_explicit,
            reasoning_explicit,
            system_prompt_explicit,
            working_dir_explicit,
            &load_err)) {
      fprintf(stderr, "%s\n", load_err == NULL ? "failed to load session" : load_err);
      free(load_err);
      chi_prompt_queue_destroy(&queue);
      chi_conversation_destroy(&convo);
      free(cfg.system_prompt_text);
      free(cfg.session_path);
      free(cfg.loaded_model);
      free(cfg.loaded_reasoning_effort);
      free(cfg.loaded_working_dir);
      free(cfg.chatgpt_access_token);
      return 1;
    }
    call_seq = chi_conversation_max_generated_call_seq(&convo);
  }

  if (!chi_is_blank(cfg.system_prompt_file)) {
    char *load_err = NULL;
    free(cfg.system_prompt_text);
    cfg.system_prompt_text = chi_read_text_file(cfg.system_prompt_file, &load_err);
    if (cfg.system_prompt_text == NULL) {
      fprintf(stderr, "%s\n", load_err == NULL ? "failed to load system prompt file" : load_err);
      free(load_err);
      chi_prompt_queue_destroy(&queue);
      chi_conversation_destroy(&convo);
      free(cfg.system_prompt_text);
      free(cfg.session_path);
      free(cfg.loaded_model);
      free(cfg.loaded_reasoning_effort);
      free(cfg.loaded_working_dir);
      free(cfg.chatgpt_access_token);
      return 2;
    }
  }

  if (!chi_prompt_queue_push_front(&queue, prompt)) {
    fprintf(stderr, "failed to queue initial prompt\n");
    chi_prompt_queue_destroy(&queue);
    chi_conversation_destroy(&convo);
    free(cfg.system_prompt_text);
    free(cfg.session_path);
    free(cfg.loaded_model);
    free(cfg.loaded_reasoning_effort);
    free(cfg.loaded_working_dir);
    free(cfg.chatgpt_access_token);
    return 1;
  }

  cfg.working_dir = working_dir_explicit ? working_dir : (cfg.working_dir == NULL ? working_dir : cfg.working_dir);

  while (queue.count > 0) {
    int finalized = 0;
    int turn;

    if (!chi_queue_pop_to_conversation(&queue, &convo)) {
      fprintf(stderr, "failed to append queued user message\n");
      exit_code = 1;
      goto cleanup;
    }

    {
      char *save_err = NULL;
      if (!chi_session_save(&cfg, &convo, &save_err)) {
        fprintf(stderr, "%s\n", save_err == NULL ? "failed to save session" : save_err);
        free(save_err);
        exit_code = 1;
        goto cleanup;
      }
      free(save_err);
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
              action.is_tool ? action.tool_arguments_json : NULL,
              action.is_tool ? action.tool_is_custom : 0)) {
        fprintf(stderr, "failed to append assistant message\n");
        chi_action_reset(&action);
        exit_code = 1;
        goto cleanup;
      }

      {
        char *save_err = NULL;
        if (!chi_session_save(&cfg, &convo, &save_err)) {
          fprintf(stderr, "%s\n", save_err == NULL ? "failed to save session" : save_err);
          free(save_err);
          chi_action_reset(&action);
          exit_code = 1;
          goto cleanup;
        }
        free(save_err);
      }

      if (!chi_is_blank(action.reasoning_summary)) {
        printf("[thinking]\n%s\n", action.reasoning_summary);
      }

      if (!action.is_tool) {
        printf("[final]\n%s\n", action.final_text == NULL ? "" : action.final_text);
        printf("session: %s\n", cfg.session_id);
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
        if (action.tool_is_custom) {
          printf("[tool input]\n%s\n", action.tool_command == NULL ? "" : action.tool_command);
        } else {
          printf("[tool command]\n%s\n", action.tool_command == NULL ? "" : action.tool_command);
        }

        if (action.tool_is_custom) {
          if (strcmp(tool_name, "apply_patch") == 0) {
            if (!chi_run_apply_patch_tool(
                    cfg.working_dir, action.tool_command, &tool_output, &is_error, &tool_error)) {
              fprintf(stderr, "failed to run apply_patch\n");
              goto tool_cleanup;
            }
          } else {
            tool_output = chi_strdup("");
            tool_error = chi_format("unsupported custom tool: %s", tool_name);
            is_error = 1;
            if (tool_output == NULL || tool_error == NULL) {
              fprintf(stderr, "out of memory handling unsupported custom tool\n");
              goto tool_cleanup;
            }
          }
        } else if (strcmp(tool_name, "bash") == 0) {
          if (!chi_run_bash(cfg.working_dir, action.tool_command, timeout, &tool_output, &is_error, &tool_error)) {
            fprintf(stderr, "failed to run bash tool\n");
            goto tool_cleanup;
          }
        } else if (strcmp(tool_name, "apply_patch") == 0) {
          tool_output = chi_strdup("");
          tool_error = chi_strdup("apply_patch must be called as a freeform custom tool");
          is_error = 1;
          if (tool_output == NULL || tool_error == NULL) {
            fprintf(stderr, "out of memory handling unsupported apply_patch invocation\n");
            goto tool_cleanup;
          }
        } else {
          tool_output = chi_strdup("");
          tool_error = chi_format("unsupported function tool: %s", tool_name);
          is_error = 1;
          if (tool_output == NULL || tool_error == NULL) {
            fprintf(stderr, "out of memory handling unsupported function tool\n");
            goto tool_cleanup;
          }
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
                NULL,
                action.tool_is_custom)) {
          fprintf(stderr, "failed to append tool result\n");
          goto tool_cleanup;
        }

        if (queue.count > 0) {
          if (!chi_queue_pop_to_conversation(&queue, &convo)) {
            fprintf(stderr, "failed to inject queued user message\n");
            goto tool_cleanup;
          }
        }

        {
          char *save_err = NULL;
          if (!chi_session_save(&cfg, &convo, &save_err)) {
            fprintf(stderr, "%s\n", save_err == NULL ? "failed to save session" : save_err);
            free(save_err);
            goto tool_cleanup;
          }
          free(save_err);
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
  free(cfg.system_prompt_text);
  free(cfg.session_path);
  free(cfg.loaded_model);
  free(cfg.loaded_reasoning_effort);
  free(cfg.loaded_working_dir);
  free(cfg.chatgpt_access_token);
  return exit_code;
}
