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
#define CHI_CHATGPT_CLIENT_ID "app_EMoamEEZ73f0CkXaXp7hrann"
#define CHI_CHATGPT_AUTHORIZE_URL "https://auth.openai.com/oauth/authorize"
#define CHI_CHATGPT_TOKEN_URL "https://auth.openai.com/oauth/token"
#define CHI_CHATGPT_REDIRECT_URI "http://localhost:1455/auth/callback"
#define CHI_CHATGPT_SCOPE "openid profile email offline_access"
#define CHI_CHATGPT_JWT_CLAIM_PATH "https://api.openai.com/auth"
#define CHI_CHATGPT_AUTH_SOURCE_NONE 0
#define CHI_CHATGPT_AUTH_SOURCE_ENV 1
#define CHI_CHATGPT_AUTH_SOURCE_CHI_FILE 2
#define CHI_CHATGPT_AUTH_SOURCE_LEGACY_CODEX 3

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
  char *chatgpt_refresh_token;
  char *chatgpt_account_id;
  long long chatgpt_expires_at_ms;
  int chatgpt_auth_source;
  int chatgpt_login_attempted;
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

typedef struct {
  uint32_t state[8];
  uint64_t bitlen;
  unsigned char data[64];
  size_t datalen;
} chi_sha256_ctx;

static uint32_t chi_sha256_rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

static void chi_sha256_transform(chi_sha256_ctx *ctx, const unsigned char data[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
      0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
      0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
      0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
      0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
      0xc67178f2u};
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;
  uint32_t f;
  uint32_t g;
  uint32_t h;
  uint32_t m[64];
  uint32_t s0;
  uint32_t s1;
  uint32_t ch;
  uint32_t maj;
  uint32_t temp1;
  uint32_t temp2;
  int i;

  for (i = 0; i < 16; i++) {
    m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) | ((uint32_t)data[i * 4 + 2] << 8) |
           (uint32_t)data[i * 4 + 3];
  }
  for (i = 16; i < 64; i++) {
    s0 = chi_sha256_rotr(m[i - 15], 7) ^ chi_sha256_rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
    s1 = chi_sha256_rotr(m[i - 2], 17) ^ chi_sha256_rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
    m[i] = m[i - 16] + s0 + m[i - 7] + s1;
  }

  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];

  for (i = 0; i < 64; i++) {
    s1 = chi_sha256_rotr(e, 6) ^ chi_sha256_rotr(e, 11) ^ chi_sha256_rotr(e, 25);
    ch = (e & f) ^ ((~e) & g);
    temp1 = h + s1 + ch + k[i] + m[i];
    s0 = chi_sha256_rotr(a, 2) ^ chi_sha256_rotr(a, 13) ^ chi_sha256_rotr(a, 22);
    maj = (a & b) ^ (a & c) ^ (b & c);
    temp2 = s0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void chi_sha256_init(chi_sha256_ctx *ctx) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->state[0] = 0x6a09e667u;
  ctx->state[1] = 0xbb67ae85u;
  ctx->state[2] = 0x3c6ef372u;
  ctx->state[3] = 0xa54ff53au;
  ctx->state[4] = 0x510e527fu;
  ctx->state[5] = 0x9b05688cu;
  ctx->state[6] = 0x1f83d9abu;
  ctx->state[7] = 0x5be0cd19u;
}

static void chi_sha256_update(chi_sha256_ctx *ctx, const unsigned char *data, size_t len) {
  size_t i;

  for (i = 0; i < len; i++) {
    ctx->data[ctx->datalen++] = data[i];
    if (ctx->datalen == sizeof(ctx->data)) {
      chi_sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      ctx->datalen = 0;
    }
  }
}

static void chi_sha256_final(chi_sha256_ctx *ctx, unsigned char out[32]) {
  size_t i;

  ctx->bitlen += (uint64_t)ctx->datalen * 8u;
  ctx->data[ctx->datalen++] = 0x80u;

  if (ctx->datalen > 56) {
    while (ctx->datalen < 64) {
      ctx->data[ctx->datalen++] = 0;
    }
    chi_sha256_transform(ctx, ctx->data);
    ctx->datalen = 0;
  }

  while (ctx->datalen < 56) {
    ctx->data[ctx->datalen++] = 0;
  }

  for (i = 0; i < 8; i++) {
    ctx->data[63 - i] = (unsigned char)((ctx->bitlen >> (i * 8)) & 0xffu);
  }
  chi_sha256_transform(ctx, ctx->data);

  for (i = 0; i < 8; i++) {
    out[i * 4] = (unsigned char)((ctx->state[i] >> 24) & 0xffu);
    out[i * 4 + 1] = (unsigned char)((ctx->state[i] >> 16) & 0xffu);
    out[i * 4 + 2] = (unsigned char)((ctx->state[i] >> 8) & 0xffu);
    out[i * 4 + 3] = (unsigned char)(ctx->state[i] & 0xffu);
  }
}

static void chi_sha256_sum(const unsigned char *data, size_t len, unsigned char out[32]) {
  chi_sha256_ctx ctx;
  chi_sha256_init(&ctx);
  chi_sha256_update(&ctx, data, len);
  chi_sha256_final(&ctx, out);
}

static char *chi_base64url_encode(const unsigned char *data, size_t len) {
  static const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  char *out;
  size_t out_len;
  size_t i;
  size_t j = 0;

  out_len = (len / 3u) * 4u;
  if (len % 3u == 1u) {
    out_len += 2u;
  } else if (len % 3u == 2u) {
    out_len += 3u;
  }

  out = (char *)malloc(out_len + 1u);
  if (out == NULL) {
    return NULL;
  }

  for (i = 0; i + 3u <= len; i += 3u) {
    uint32_t v = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8) | (uint32_t)data[i + 2];
    out[j++] = alphabet[(v >> 18) & 63u];
    out[j++] = alphabet[(v >> 12) & 63u];
    out[j++] = alphabet[(v >> 6) & 63u];
    out[j++] = alphabet[v & 63u];
  }

  if (len % 3u == 1u) {
    uint32_t v = (uint32_t)data[i] << 16;
    out[j++] = alphabet[(v >> 18) & 63u];
    out[j++] = alphabet[(v >> 12) & 63u];
  } else if (len % 3u == 2u) {
    uint32_t v = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8);
    out[j++] = alphabet[(v >> 18) & 63u];
    out[j++] = alphabet[(v >> 12) & 63u];
    out[j++] = alphabet[(v >> 6) & 63u];
  }

  out[j] = '\0';
  return out;
}

static int chi_base64url_value(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c - 'A';
  }
  if (c >= 'a' && c <= 'z') {
    return c - 'a' + 26;
  }
  if (c >= '0' && c <= '9') {
    return c - '0' + 52;
  }
  if (c == '-' || c == '+') {
    return 62;
  }
  if (c == '_' || c == '/') {
    return 63;
  }
  return -1;
}

static unsigned char *chi_base64url_decode(const char *input, size_t *out_len) {
  size_t len;
  size_t padded_len;
  char *padded = NULL;
  unsigned char *out = NULL;
  size_t i;
  size_t j = 0;

  if (out_len != NULL) {
    *out_len = 0;
  }
  if (input == NULL) {
    return NULL;
  }

  len = strlen(input);
  padded_len = ((len + 3u) / 4u) * 4u;
  padded = (char *)malloc(padded_len + 1u);
  out = (unsigned char *)malloc((padded_len / 4u) * 3u + 1u);
  if (padded == NULL || out == NULL) {
    free(padded);
    free(out);
    return NULL;
  }

  memcpy(padded, input, len);
  for (i = len; i < padded_len; i++) {
    padded[i] = '=';
  }
  padded[padded_len] = '\0';

  for (i = 0; i < padded_len; i += 4u) {
    int a = chi_base64url_value(padded[i]);
    int b = chi_base64url_value(padded[i + 1]);
    int c = padded[i + 2] == '=' ? -2 : chi_base64url_value(padded[i + 2]);
    int d = padded[i + 3] == '=' ? -2 : chi_base64url_value(padded[i + 3]);
    uint32_t v;

    if (a < 0 || b < 0 || c == -1 || d == -1) {
      free(padded);
      free(out);
      return NULL;
    }

    v = ((uint32_t)a << 18) | ((uint32_t)b << 12);
    if (c >= 0) {
      v |= (uint32_t)c << 6;
    }
    if (d >= 0) {
      v |= (uint32_t)d;
    }

    out[j++] = (unsigned char)((v >> 16) & 0xffu);
    if (c >= 0) {
      out[j++] = (unsigned char)((v >> 8) & 0xffu);
    }
    if (d >= 0) {
      out[j++] = (unsigned char)(v & 0xffu);
    }
  }

  free(padded);
  if (out_len != NULL) {
    *out_len = j;
  }
  return out;
}

static char *chi_url_encode_dup(const char *text) {
  CURL *curl;
  char *encoded;
  char *copy;

  if (text == NULL) {
    return chi_strdup("");
  }

  curl = curl_easy_init();
  if (curl == NULL) {
    return NULL;
  }

  encoded = curl_easy_escape(curl, text, 0);
  curl_easy_cleanup(curl);
  if (encoded == NULL) {
    return NULL;
  }

  copy = chi_strdup(encoded);
  curl_free(encoded);
  return copy;
}

static char *chi_url_decode_dup(const char *text) {
  CURL *curl;
  char *decoded;
  char *copy;
  int out_len = 0;

  if (text == NULL) {
    return NULL;
  }

  curl = curl_easy_init();
  if (curl == NULL) {
    return NULL;
  }

  decoded = curl_easy_unescape(curl, text, 0, &out_len);
  curl_easy_cleanup(curl);
  if (decoded == NULL) {
    return NULL;
  }

  copy = (char *)malloc((size_t)out_len + 1u);
  if (copy != NULL) {
    memcpy(copy, decoded, (size_t)out_len);
    copy[out_len] = '\0';
  }
  curl_free(decoded);
  return copy;
}

static int chi_can_prompt_user(void) {
  FILE *tty = fopen("/dev/tty", "r+");
  if (tty != NULL) {
    fclose(tty);
    return 1;
  }
  return isatty(STDIN_FILENO) != 0;
}

static char *chi_prompt_line(const char *prompt, char **err_out) {
  FILE *tty = NULL;
  char *line = NULL;
  size_t cap = 0;
  ssize_t nread;

  *err_out = NULL;

  tty = fopen("/dev/tty", "r+");
  if (tty == NULL) {
    if (!isatty(STDIN_FILENO)) {
      *err_out = chi_strdup("interactive login requires a terminal");
      return NULL;
    }
    tty = stdin;
  }

  fprintf(tty == stdin ? stderr : tty, "%s", prompt == NULL ? "" : prompt);
  fflush(tty == stdin ? stderr : tty);

  nread = getline(&line, &cap, tty == stdin ? stdin : tty);
  if (tty != stdin) {
    fclose(tty);
  }
  if (nread < 0) {
    free(line);
    *err_out = chi_strdup("failed to read interactive input");
    return NULL;
  }

  chi_trim_trailing_newlines(line);
  return line;
}

static void chi_try_open_url(const char *url) {
  pid_t pid;

  if (chi_is_blank(url)) {
    return;
  }

  pid = fork();
  if (pid != 0) {
    return;
  }

#if defined(__APPLE__)
  execlp("open", "open", url, (char *)NULL);
#else
  execlp("xdg-open", "xdg-open", url, (char *)NULL);
#endif
  _exit(127);
}

static char *chi_home_join(const char *suffix, char **err_out) {
  const char *home = getenv("HOME");

  *err_out = NULL;
  if (chi_is_blank(home)) {
    *err_out = chi_strdup("HOME is not set");
    return NULL;
  }

  return chi_format("%s/%s", home, suffix);
}

static void chi_clear_chatgpt_auth_cache(chi_config *cfg) {
  if (cfg == NULL) {
    return;
  }
  free(cfg->chatgpt_access_token);
  free(cfg->chatgpt_refresh_token);
  free(cfg->chatgpt_account_id);
  cfg->chatgpt_access_token = NULL;
  cfg->chatgpt_refresh_token = NULL;
  cfg->chatgpt_account_id = NULL;
  cfg->chatgpt_expires_at_ms = 0;
  cfg->chatgpt_auth_source = CHI_CHATGPT_AUTH_SOURCE_NONE;
}

static int chi_set_chatgpt_auth_cache(
    chi_config *cfg,
    const char *access_token,
    const char *refresh_token,
    const char *account_id,
    long long expires_at_ms,
    int auth_source) {
  char *access_copy = NULL;
  char *refresh_copy = NULL;
  char *account_copy = NULL;

  if (cfg == NULL || chi_is_blank(access_token) || chi_is_blank(account_id)) {
    return 0;
  }

  access_copy = chi_strdup(access_token);
  if (access_copy == NULL) {
    return 0;
  }
  if (!chi_is_blank(refresh_token)) {
    refresh_copy = chi_strdup(refresh_token);
    if (refresh_copy == NULL) {
      free(access_copy);
      return 0;
    }
  }
  account_copy = chi_strdup(account_id);
  if (account_copy == NULL) {
    free(access_copy);
    free(refresh_copy);
    return 0;
  }

  chi_clear_chatgpt_auth_cache(cfg);
  cfg->chatgpt_access_token = access_copy;
  cfg->chatgpt_refresh_token = refresh_copy;
  cfg->chatgpt_account_id = account_copy;
  cfg->chatgpt_expires_at_ms = expires_at_ms;
  cfg->chatgpt_auth_source = auth_source;
  return 1;
}

static char *chi_build_chi_auth_path(char **err_out) {
  return chi_home_join(".chi/auth.json", err_out);
}

static char *chi_build_legacy_codex_auth_path(char **err_out) {
  return chi_home_join(".codex/auth.json", err_out);
}

static int chi_write_private_text_file(const char *path, const char *text, char **err_out) {
  FILE *fp;
  int fd;
  char *dir_path = NULL;
  char *slash;

  *err_out = NULL;
  if (chi_is_blank(path)) {
    *err_out = chi_strdup("auth file path is blank");
    return 0;
  }

  dir_path = chi_strdup(path);
  if (dir_path == NULL) {
    *err_out = chi_strdup("out of memory while preparing auth directory");
    return 0;
  }
  slash = strrchr(dir_path, '/');
  if (slash != NULL) {
    *slash = '\0';
    if (!chi_ensure_directory(dir_path, err_out)) {
      free(dir_path);
      return 0;
    }
  }
  free(dir_path);

  fp = fopen(path, "wb");
  if (fp == NULL) {
    *err_out = chi_format("failed to open auth file '%s': %s", path, strerror(errno));
    return 0;
  }

  if (fwrite(text, 1, strlen(text), fp) != strlen(text)) {
    *err_out = chi_format("failed to write auth file '%s': %s", path, strerror(errno));
    fclose(fp);
    return 0;
  }

  if (fflush(fp) != 0) {
    *err_out = chi_format("failed to flush auth file '%s': %s", path, strerror(errno));
    fclose(fp);
    return 0;
  }

  fd = fileno(fp);
  if (fd >= 0) {
    (void)fchmod(fd, 0600);
  }
  if (fclose(fp) != 0) {
    *err_out = chi_format("failed to close auth file '%s': %s", path, strerror(errno));
    return 0;
  }

  return 1;
}

static char *chi_extract_param_value(const char *text, const char *name) {
  char *needle;
  const char *p;
  char *raw = NULL;
  char *decoded = NULL;
  size_t name_len;
  const char *end;

  if (chi_is_blank(text) || chi_is_blank(name)) {
    return NULL;
  }

  name_len = strlen(name);
  needle = chi_format("%s=", name);
  if (needle == NULL) {
    return NULL;
  }

  p = text;
  while ((p = strstr(p, needle)) != NULL) {
    if (p == text || p[-1] == '?' || p[-1] == '&' || p[-1] == '#') {
      p += name_len + 1;
      end = p;
      while (*end != '\0' && *end != '&' && *end != '#' && !isspace((unsigned char)*end)) {
        end++;
      }
      raw = (char *)malloc((size_t)(end - p) + 1);
      if (raw == NULL) {
        break;
      }
      memcpy(raw, p, (size_t)(end - p));
      raw[end - p] = '\0';
      decoded = chi_url_decode_dup(raw);
      free(raw);
      free(needle);
      return decoded;
    }
    p += name_len + 1;
  }

  free(needle);
  return NULL;
}

static int chi_parse_chatgpt_authorization_input(
    const char *input,
    char **code_out,
    char **state_out,
    char **err_out) {
  char *trimmed = NULL;
  char *code = NULL;
  char *state = NULL;
  char *hash = NULL;

  *code_out = NULL;
  *state_out = NULL;
  *err_out = NULL;

  if (chi_is_blank(input)) {
    *err_out = chi_strdup("authorization input is blank");
    return 0;
  }

  trimmed = chi_strdup(input);
  if (trimmed == NULL) {
    *err_out = chi_strdup("out of memory while parsing authorization input");
    return 0;
  }
  while (*trimmed != '\0' && isspace((unsigned char)*trimmed)) {
    memmove(trimmed, trimmed + 1, strlen(trimmed));
  }
  chi_trim_trailing_newlines(trimmed);

  code = chi_extract_param_value(trimmed, "code");
  state = chi_extract_param_value(trimmed, "state");
  if (code == NULL) {
    hash = strchr(trimmed, '#');
    if (hash != NULL) {
      *hash = '\0';
      hash++;
      code = chi_strdup(trimmed);
      state = chi_strdup_nonblank_or_null(hash);
    } else {
      code = chi_strdup(trimmed);
    }
  }
  free(trimmed);

  if (chi_is_blank(code)) {
    free(code);
    free(state);
    *err_out = chi_strdup("missing authorization code");
    return 0;
  }

  *code_out = code;
  *state_out = state;
  return 1;
}

static char *chi_extract_chatgpt_account_id(const char *access_token) {
  const char *first_dot;
  const char *second_dot;
  char *payload_part = NULL;
  unsigned char *decoded = NULL;
  size_t decoded_len = 0;
  char *payload_json = NULL;
  char *auth_json = NULL;
  char *account_id = NULL;

  if (chi_is_blank(access_token)) {
    return NULL;
  }

  first_dot = strchr(access_token, '.');
  if (first_dot == NULL) {
    return NULL;
  }
  second_dot = strchr(first_dot + 1, '.');
  if (second_dot == NULL || second_dot <= first_dot + 1) {
    return NULL;
  }

  payload_part = (char *)malloc((size_t)(second_dot - first_dot - 1) + 1);
  if (payload_part == NULL) {
    return NULL;
  }
  memcpy(payload_part, first_dot + 1, (size_t)(second_dot - first_dot - 1));
  payload_part[second_dot - first_dot - 1] = '\0';

  decoded = chi_base64url_decode(payload_part, &decoded_len);
  free(payload_part);
  if (decoded == NULL) {
    return NULL;
  }

  payload_json = (char *)malloc(decoded_len + 1);
  if (payload_json == NULL) {
    free(decoded);
    return NULL;
  }
  memcpy(payload_json, decoded, decoded_len);
  payload_json[decoded_len] = '\0';
  free(decoded);

  auth_json = chi_json_get_object(payload_json, CHI_CHATGPT_JWT_CLAIM_PATH);
  if (auth_json != NULL) {
    account_id = chi_json_get_string(auth_json, "chatgpt_account_id");
  }

  free(payload_json);
  free(auth_json);
  return chi_is_blank(account_id) ? (free(account_id), (char *)NULL) : account_id;
}

static int chi_build_chatgpt_pkce_pair(char **verifier_out, char **challenge_out, char **err_out) {
  unsigned char random_bytes[32];
  unsigned char digest[32];
  char *verifier = NULL;
  char *challenge = NULL;
  int fd;
  ssize_t nread;

  *verifier_out = NULL;
  *challenge_out = NULL;
  *err_out = NULL;

  fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0) {
    *err_out = chi_strdup("failed to open /dev/urandom for PKCE");
    return 0;
  }
  nread = read(fd, random_bytes, sizeof(random_bytes));
  close(fd);
  if (nread != (ssize_t)sizeof(random_bytes)) {
    *err_out = chi_strdup("failed to read random bytes for PKCE");
    return 0;
  }

  verifier = chi_base64url_encode(random_bytes, sizeof(random_bytes));
  if (verifier == NULL) {
    *err_out = chi_strdup("out of memory while building PKCE verifier");
    return 0;
  }

  chi_sha256_sum((const unsigned char *)verifier, strlen(verifier), digest);
  challenge = chi_base64url_encode(digest, sizeof(digest));
  if (challenge == NULL) {
    free(verifier);
    *err_out = chi_strdup("out of memory while building PKCE challenge");
    return 0;
  }

  *verifier_out = verifier;
  *challenge_out = challenge;
  return 1;
}

static char *chi_build_chatgpt_authorize_url(const char *challenge, const char *state, char **err_out) {
  char *redirect = NULL;
  char *scope = NULL;
  char *encoded_challenge = NULL;
  char *encoded_state = NULL;
  char *url = NULL;

  *err_out = NULL;
  redirect = chi_url_encode_dup(CHI_CHATGPT_REDIRECT_URI);
  scope = chi_url_encode_dup(CHI_CHATGPT_SCOPE);
  encoded_challenge = chi_url_encode_dup(challenge);
  encoded_state = chi_url_encode_dup(state);
  if (redirect == NULL || scope == NULL || encoded_challenge == NULL || encoded_state == NULL) {
    *err_out = chi_strdup("out of memory while building authorization url");
    goto cleanup;
  }

  url = chi_format(
      CHI_CHATGPT_AUTHORIZE_URL
      "?response_type=code&client_id=%s&redirect_uri=%s&scope=%s&code_challenge=%s&code_challenge_method=S256&"
      "state=%s&id_token_add_organizations=true&codex_cli_simplified_flow=true&originator=pi",
      CHI_CHATGPT_CLIENT_ID,
      redirect,
      scope,
      encoded_challenge,
      encoded_state);

cleanup:
  free(redirect);
  free(scope);
  free(encoded_challenge);
  free(encoded_state);
  return url;
}

static int chi_request_chatgpt_tokens(
    const char *form_body,
    char **access_token_out,
    char **refresh_token_out,
    long long *expires_at_ms_out,
    char **err_out) {
  const char *headers[2];
  char *response_body = NULL;
  char *tmp_err = NULL;
  char *access_token = NULL;
  char *refresh_token = NULL;
  double expires_in = 0;
  int http_code = 0;

  *access_token_out = NULL;
  *refresh_token_out = NULL;
  *expires_at_ms_out = 0;
  *err_out = NULL;

  headers[0] = "Content-Type: application/x-www-form-urlencoded";
  headers[1] = "Accept: application/json";

  if (!chi_curl_request(
          "POST", CHI_CHATGPT_TOKEN_URL, headers, 2, form_body, &response_body, &http_code, &tmp_err)) {
    *err_out = tmp_err;
    return 0;
  }

  if (http_code < 200 || http_code >= 300) {
    *err_out = chi_format("chatgpt token endpoint http %d: %s", http_code, response_body);
    free(response_body);
    return 0;
  }

  access_token = chi_json_get_string(response_body, "access_token");
  refresh_token = chi_json_get_string(response_body, "refresh_token");
  if (!chi_json_get_number(response_body, "expires_in", &expires_in) || chi_is_blank(access_token) ||
      chi_is_blank(refresh_token) || expires_in <= 0) {
    free(access_token);
    free(refresh_token);
    free(response_body);
    *err_out = chi_strdup("chatgpt token response is missing required fields");
    return 0;
  }

  *access_token_out = access_token;
  *refresh_token_out = refresh_token;
  *expires_at_ms_out = chi_now_ms() + (long long)(expires_in * 1000.0);
  free(response_body);
  return 1;
}

static int chi_exchange_chatgpt_authorization_code(
    const char *code,
    const char *verifier,
    char **access_token_out,
    char **refresh_token_out,
    long long *expires_at_ms_out,
    char **err_out) {
  char *encoded_code = NULL;
  char *encoded_verifier = NULL;
  char *encoded_redirect = NULL;
  char *form_body = NULL;
  int ok = 0;

  *err_out = NULL;
  encoded_code = chi_url_encode_dup(code);
  encoded_verifier = chi_url_encode_dup(verifier);
  encoded_redirect = chi_url_encode_dup(CHI_CHATGPT_REDIRECT_URI);
  if (encoded_code == NULL || encoded_verifier == NULL || encoded_redirect == NULL) {
    *err_out = chi_strdup("out of memory while preparing token exchange");
    goto cleanup;
  }

  form_body = chi_format(
      "grant_type=authorization_code&client_id=%s&code=%s&code_verifier=%s&redirect_uri=%s",
      CHI_CHATGPT_CLIENT_ID,
      encoded_code,
      encoded_verifier,
      encoded_redirect);
  if (form_body == NULL) {
    *err_out = chi_strdup("out of memory while preparing token exchange");
    goto cleanup;
  }

  ok = chi_request_chatgpt_tokens(form_body, access_token_out, refresh_token_out, expires_at_ms_out, err_out);

cleanup:
  free(encoded_code);
  free(encoded_verifier);
  free(encoded_redirect);
  free(form_body);
  return ok;
}

static int chi_refresh_chatgpt_tokens(
    const char *refresh_token,
    char **access_token_out,
    char **refresh_token_out,
    long long *expires_at_ms_out,
    char **err_out) {
  char *encoded_refresh = NULL;
  char *form_body = NULL;
  int ok = 0;

  *err_out = NULL;
  encoded_refresh = chi_url_encode_dup(refresh_token);
  if (encoded_refresh == NULL) {
    *err_out = chi_strdup("out of memory while preparing token refresh");
    return 0;
  }

  form_body = chi_format(
      "grant_type=refresh_token&refresh_token=%s&client_id=%s", encoded_refresh, CHI_CHATGPT_CLIENT_ID);
  if (form_body == NULL) {
    free(encoded_refresh);
    *err_out = chi_strdup("out of memory while preparing token refresh");
    return 0;
  }

  ok = chi_request_chatgpt_tokens(form_body, access_token_out, refresh_token_out, expires_at_ms_out, err_out);
  free(encoded_refresh);
  free(form_body);
  return ok;
}

static int chi_save_chatgpt_auth_file(const chi_config *cfg, char **err_out) {
  char *auth_path = NULL;
  char *access = NULL;
  char *refresh = NULL;
  char *account = NULL;
  char *json = NULL;
  int ok = 0;

  *err_out = NULL;
  if (cfg == NULL || chi_is_blank(cfg->chatgpt_access_token) || chi_is_blank(cfg->chatgpt_refresh_token) ||
      chi_is_blank(cfg->chatgpt_account_id)) {
    *err_out = chi_strdup("chatgpt auth cache is incomplete");
    return 0;
  }

  auth_path = chi_build_chi_auth_path(err_out);
  if (auth_path == NULL) {
    return 0;
  }
  access = chi_json_escape(cfg->chatgpt_access_token);
  refresh = chi_json_escape(cfg->chatgpt_refresh_token);
  account = chi_json_escape(cfg->chatgpt_account_id);
  if (access == NULL || refresh == NULL || account == NULL) {
    *err_out = chi_strdup("out of memory while serializing auth file");
    goto cleanup;
  }

  json = chi_format(
      "{\n"
      "  \"version\": 1,\n"
      "  \"chatgpt\": {\n"
      "    \"access_token\": \"%s\",\n"
      "    \"refresh_token\": \"%s\",\n"
      "    \"account_id\": \"%s\",\n"
      "    \"expires_at_ms\": %lld\n"
      "  }\n"
      "}\n",
      access,
      refresh,
      account,
      cfg->chatgpt_expires_at_ms);
  if (json == NULL) {
    *err_out = chi_strdup("out of memory while serializing auth file");
    goto cleanup;
  }

  ok = chi_write_private_text_file(auth_path, json, err_out);

cleanup:
  free(auth_path);
  free(access);
  free(refresh);
  free(account);
  free(json);
  return ok;
}

static int chi_load_chatgpt_auth_file(chi_config *cfg, int *found_out, char **err_out) {
  char *auth_path = NULL;
  char *auth_json = NULL;
  char *chatgpt_json = NULL;
  char *access_token = NULL;
  char *refresh_token = NULL;
  char *account_id = NULL;
  char *tmp_err = NULL;
  double expires_num = 0;
  long long expires_at_ms = 0;

  *found_out = 0;
  *err_out = NULL;

  auth_path = chi_build_chi_auth_path(&tmp_err);
  if (auth_path == NULL) {
    *err_out = tmp_err;
    return 0;
  }

  auth_json = chi_read_named_text_file(auth_path, "chi auth file", &tmp_err);
  if (auth_json == NULL) {
    if (tmp_err != NULL && strstr(tmp_err, "failed to open") != NULL) {
      free(tmp_err);
      free(auth_path);
      return 1;
    }
    *err_out = tmp_err;
    free(auth_path);
    return 0;
  }

  *found_out = 1;
  chatgpt_json = chi_json_get_object(auth_json, "chatgpt");
  if (chatgpt_json == NULL) {
    *err_out = chi_strdup("~/.chi/auth.json is missing the chatgpt object");
    goto cleanup;
  }

  access_token = chi_json_get_string(chatgpt_json, "access_token");
  if (access_token == NULL) {
    access_token = chi_json_get_string(chatgpt_json, "access");
  }
  refresh_token = chi_json_get_string(chatgpt_json, "refresh_token");
  if (refresh_token == NULL) {
    refresh_token = chi_json_get_string(chatgpt_json, "refresh");
  }
  account_id = chi_json_get_string(chatgpt_json, "account_id");
  if (account_id == NULL) {
    account_id = chi_json_get_string(chatgpt_json, "accountId");
  }
  if (!chi_json_get_number(chatgpt_json, "expires_at_ms", &expires_num) &&
      !chi_json_get_number(chatgpt_json, "expires_at", &expires_num) &&
      !chi_json_get_number(chatgpt_json, "expires", &expires_num)) {
    expires_num = 0;
  }
  expires_at_ms = expires_num > 0 ? (long long)expires_num : 0;

  if (chi_is_blank(account_id) && !chi_is_blank(access_token)) {
    free(account_id);
    account_id = chi_extract_chatgpt_account_id(access_token);
  }

  if (!chi_set_chatgpt_auth_cache(
          cfg, access_token, refresh_token, account_id, expires_at_ms, CHI_CHATGPT_AUTH_SOURCE_CHI_FILE)) {
    *err_out = chi_strdup("failed to cache ~/.chi/auth.json credentials");
    goto cleanup;
  }

cleanup:
  free(auth_path);
  free(auth_json);
  free(chatgpt_json);
  free(access_token);
  free(refresh_token);
  free(account_id);
  return *err_out == NULL;
}

static int chi_load_legacy_codex_token(chi_config *cfg, int *found_out, char **err_out) {
  char *auth_path = NULL;
  char *auth_json = NULL;
  char *tokens_json = NULL;
  char *access_token = NULL;
  char *account_id = NULL;
  char *tmp_err = NULL;

  *found_out = 0;
  *err_out = NULL;

  auth_path = chi_build_legacy_codex_auth_path(&tmp_err);
  if (auth_path == NULL) {
    *err_out = tmp_err;
    return 0;
  }

  auth_json = chi_read_named_text_file(auth_path, "codex auth file", &tmp_err);
  if (auth_json == NULL) {
    if (tmp_err != NULL && strstr(tmp_err, "failed to open") != NULL) {
      free(tmp_err);
      free(auth_path);
      return 1;
    }
    *err_out = tmp_err;
    free(auth_path);
    return 0;
  }

  *found_out = 1;
  tokens_json = chi_json_get_object(auth_json, "tokens");
  if (tokens_json == NULL) {
    *err_out = chi_strdup("~/.codex/auth.json is missing the tokens object");
    goto cleanup;
  }

  access_token = chi_json_get_string(tokens_json, "access_token");
  account_id = chi_extract_chatgpt_account_id(access_token);
  if (!chi_set_chatgpt_auth_cache(
          cfg, access_token, NULL, account_id, 0, CHI_CHATGPT_AUTH_SOURCE_LEGACY_CODEX)) {
    *err_out = chi_strdup("failed to cache ~/.codex/auth.json credentials");
    goto cleanup;
  }

cleanup:
  free(auth_path);
  free(auth_json);
  free(tokens_json);
  free(access_token);
  free(account_id);
  return *err_out == NULL;
}

static int chi_refresh_chatgpt_auth_file(chi_config *cfg, char **err_out) {
  char *access_token = NULL;
  char *refresh_token = NULL;
  char *account_id = NULL;
  long long expires_at_ms = 0;
  int ok = 0;

  *err_out = NULL;
  if (cfg == NULL || chi_is_blank(cfg->chatgpt_refresh_token)) {
    *err_out = chi_strdup("chatgpt refresh token is not available");
    return 0;
  }

  if (!chi_refresh_chatgpt_tokens(
          cfg->chatgpt_refresh_token, &access_token, &refresh_token, &expires_at_ms, err_out)) {
    return 0;
  }

  account_id = chi_extract_chatgpt_account_id(access_token);
  if (chi_is_blank(account_id)) {
    free(access_token);
    free(refresh_token);
    free(account_id);
    *err_out = chi_strdup("failed to extract chatgpt account id from refreshed token");
    return 0;
  }

  if (!chi_set_chatgpt_auth_cache(
          cfg, access_token, refresh_token, account_id, expires_at_ms, CHI_CHATGPT_AUTH_SOURCE_CHI_FILE)) {
    *err_out = chi_strdup("failed to cache refreshed chatgpt credentials");
    goto cleanup;
  }

  ok = chi_save_chatgpt_auth_file(cfg, err_out);

cleanup:
  free(access_token);
  free(refresh_token);
  free(account_id);
  return ok;
}

static int chi_login_chatgpt_interactive(chi_config *cfg, char **err_out) {
  char state_hex[33];
  char *verifier = NULL;
  char *challenge = NULL;
  char *url = NULL;
  char *input = NULL;
  char *code = NULL;
  char *state = NULL;
  char *access_token = NULL;
  char *refresh_token = NULL;
  char *account_id = NULL;
  long long expires_at_ms = 0;
  int ok = 0;

  *err_out = NULL;
  if (!chi_can_prompt_user()) {
    *err_out = chi_strdup("interactive ChatGPT login requires a terminal");
    return 0;
  }

  chi_random_hex(state_hex, sizeof(state_hex), 16);
  if (state_hex[0] == '\0') {
    *err_out = chi_strdup("failed to generate login state");
    return 0;
  }
  if (!chi_build_chatgpt_pkce_pair(&verifier, &challenge, err_out)) {
    return 0;
  }
  url = chi_build_chatgpt_authorize_url(challenge, state_hex, err_out);
  if (url == NULL) {
    goto cleanup;
  }

  fprintf(
      stderr,
      "ChatGPT login is required for the selected backend.\n"
      "Opening your browser. After you finish signing in, the browser may land on a localhost URL that fails to "
      "load; copy the full URL and paste it here.\n\n%s\n\n",
      url);
  chi_try_open_url(url);

  input = chi_prompt_line("Paste the redirect URL or authorization code: ", err_out);
  if (input == NULL) {
    goto cleanup;
  }
  if (!chi_parse_chatgpt_authorization_input(input, &code, &state, err_out)) {
    goto cleanup;
  }
  if (!chi_is_blank(state) && strcmp(state, state_hex) != 0) {
    *err_out = chi_strdup("authorization state mismatch");
    goto cleanup;
  }

  if (!chi_exchange_chatgpt_authorization_code(
          code, verifier, &access_token, &refresh_token, &expires_at_ms, err_out)) {
    goto cleanup;
  }

  account_id = chi_extract_chatgpt_account_id(access_token);
  if (chi_is_blank(account_id)) {
    *err_out = chi_strdup("failed to extract chatgpt account id from access token");
    goto cleanup;
  }

  if (!chi_set_chatgpt_auth_cache(
          cfg, access_token, refresh_token, account_id, expires_at_ms, CHI_CHATGPT_AUTH_SOURCE_CHI_FILE)) {
    *err_out = chi_strdup("failed to cache chatgpt login credentials");
    goto cleanup;
  }

  if (!chi_save_chatgpt_auth_file(cfg, err_out)) {
    goto cleanup;
  }

  fprintf(stderr, "Saved ChatGPT credentials to ~/.chi/auth.json\n");
  ok = 1;

cleanup:
  free(verifier);
  free(challenge);
  free(url);
  free(input);
  free(code);
  free(state);
  free(access_token);
  free(refresh_token);
  free(account_id);
  return ok;
}

static int chi_resolve_chatgpt_auth(
    chi_config *cfg,
    char **token_out,
    char **account_id_out,
    char **err_out) {
  const char *direct = getenv("CHATGPT_ACCESS_TOKEN");
  char *account_id = NULL;
  char *tmp_err = NULL;
  int found = 0;

  *token_out = NULL;
  *account_id_out = NULL;
  *err_out = NULL;

  if (!chi_is_blank(cfg->chatgpt_access_token) && !chi_is_blank(cfg->chatgpt_account_id)) {
    if (cfg->chatgpt_auth_source == CHI_CHATGPT_AUTH_SOURCE_CHI_FILE && !chi_is_blank(cfg->chatgpt_refresh_token) &&
        cfg->chatgpt_expires_at_ms > 0 && chi_now_ms() + 60000 >= cfg->chatgpt_expires_at_ms) {
      if (!chi_refresh_chatgpt_auth_file(cfg, &tmp_err)) {
        if (!cfg->chatgpt_login_attempted && chi_can_prompt_user()) {
          cfg->chatgpt_login_attempted = 1;
          chi_clear_chatgpt_auth_cache(cfg);
          free(tmp_err);
          tmp_err = NULL;
          if (!chi_login_chatgpt_interactive(cfg, &tmp_err)) {
            *err_out = tmp_err;
            return 0;
          }
        } else {
          *err_out = tmp_err;
          return 0;
        }
      }
    }

    *token_out = chi_strdup(cfg->chatgpt_access_token);
    *account_id_out = chi_strdup(cfg->chatgpt_account_id);
    if (*token_out == NULL || *account_id_out == NULL) {
      free(*token_out);
      free(*account_id_out);
      *token_out = NULL;
      *account_id_out = NULL;
      *err_out = chi_strdup("out of memory while copying chatgpt auth");
      return 0;
    }
    return 1;
  }

  if (!chi_is_blank(direct)) {
    account_id = chi_extract_chatgpt_account_id(direct);
    if (chi_is_blank(account_id)) {
      free(account_id);
      *err_out = chi_strdup("CHATGPT_ACCESS_TOKEN is set but does not contain a ChatGPT account id");
      return 0;
    }
    if (!chi_set_chatgpt_auth_cache(cfg, direct, NULL, account_id, 0, CHI_CHATGPT_AUTH_SOURCE_ENV)) {
      free(account_id);
      *err_out = chi_strdup("failed to cache CHATGPT_ACCESS_TOKEN");
      return 0;
    }
    free(account_id);
    return chi_resolve_chatgpt_auth(cfg, token_out, account_id_out, err_out);
  }

  if (!chi_load_chatgpt_auth_file(cfg, &found, &tmp_err)) {
    *err_out = tmp_err;
    return 0;
  }
  if (found) {
    return chi_resolve_chatgpt_auth(cfg, token_out, account_id_out, err_out);
  }

  if (!chi_load_legacy_codex_token(cfg, &found, &tmp_err)) {
    *err_out = tmp_err;
    return 0;
  }
  if (found) {
    return chi_resolve_chatgpt_auth(cfg, token_out, account_id_out, err_out);
  }

  if (!cfg->chatgpt_login_attempted && chi_can_prompt_user()) {
    cfg->chatgpt_login_attempted = 1;
    if (!chi_login_chatgpt_interactive(cfg, &tmp_err)) {
      *err_out = tmp_err;
      return 0;
    }
    return chi_resolve_chatgpt_auth(cfg, token_out, account_id_out, err_out);
  }

  *err_out = chi_strdup(
      "set CHATGPT_ACCESS_TOKEN, create ~/.chi/auth.json via an interactive login, or ensure ~/.codex/auth.json "
      "contains tokens.access_token");
  return 0;
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

static int chi_provider_request_chatgpt(
    const chi_config *cfg,
    const chi_conversation *conversation,
    const char *access_token,
    const char *account_id,
    chi_action *action,
    char **err_out) {
  const char *url;
  char *req_json = NULL;
  char *resp_json = NULL;
  char *tmp_err = NULL;
  char *auth = NULL;
  char *account = NULL;
  char *session = NULL;
  const char *headers[7];
  int http_code = 0;
  int ok = 0;

  *err_out = NULL;

  req_json = chi_build_request_json(cfg, conversation, &tmp_err);
  if (req_json == NULL) {
    *err_out = tmp_err;
    return 0;
  }

  auth = chi_format("Authorization: Bearer %s", access_token);
  account = chi_format("chatgpt-account-id: %s", account_id);
  session = chi_format("session_id: %s", cfg->session_id);
  if (auth == NULL || account == NULL || session == NULL) {
    free(req_json);
    free(auth);
    free(account);
    free(session);
    *err_out = chi_strdup("out of memory while building chatgpt headers");
    return 0;
  }

  headers[0] = auth;
  headers[1] = account;
  headers[2] = "originator: pi";
  headers[3] = session;
  headers[4] = "OpenAI-Beta: responses=experimental";
  headers[5] = "Content-Type: application/json";
  headers[6] = "Accept: text/event-stream";
  url = getenv("CHATGPT_API_URL");
  if (chi_is_blank(url)) {
    url = "https://chatgpt.com/backend-api/codex/responses";
  }

  if (!chi_curl_request("POST", url, headers, 7, req_json, &resp_json, &http_code, &tmp_err)) {
    *err_out = tmp_err == NULL ? chi_strdup("chatgpt provider request failed") : tmp_err;
    free(req_json);
    free(auth);
    free(account);
    free(session);
    return 0;
  }

  if (http_code < 200 || http_code >= 300) {
    *err_out = chi_format("chatgpt http %d: %s", http_code, resp_json);
    free(req_json);
    free(auth);
    free(account);
    free(session);
    free(resp_json);
    return 0;
  }

  ok = chi_extract_provider_action(resp_json, action, &tmp_err);
  if (!ok) {
    *err_out = tmp_err;
  }

  free(req_json);
  free(auth);
  free(account);
  free(session);
  free(resp_json);
  return ok;
}

static int chi_provider_chatgpt(
    chi_config *cfg,
    const chi_conversation *conversation,
    chi_action *action,
    char **err_out) {
  char *token = NULL;
  char *account_id = NULL;
  int ok;

  *err_out = NULL;

  if (!chi_resolve_chatgpt_auth(cfg, &token, &account_id, err_out)) {
    return 0;
  }
  ok = chi_provider_request_chatgpt(cfg, conversation, token, account_id, action, err_out);
  free(token);
  free(account_id);
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
          "  CHATGPT_ACCESS_TOKEN         direct auth for chatgpt backend (else ~/.chi/auth.json, then ~/.codex/auth.json)\n"
          "  CHI_BACKEND                  backend override (default: chatgpt; openai|chatgpt)\n"
          "  CHI_MODEL                    default model (default: gpt-5.4)\n"
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
    cfg.model = "gpt-5.4";
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
      chi_clear_chatgpt_auth_cache(&cfg);
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
      chi_clear_chatgpt_auth_cache(&cfg);
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
    chi_clear_chatgpt_auth_cache(&cfg);
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
  chi_clear_chatgpt_auth_cache(&cfg);
  return exit_code;
}
