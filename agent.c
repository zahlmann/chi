#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "chi.h"

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
} openai_provider_state;

static int debug_enabled(void) {
  const char *v = getenv("CHI_DEBUG");
  return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

static char *dup_str(const char *s) {
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

static void provider_reset(openai_provider_state *state) {
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
  state->response_json = NULL;
  state->output_text = NULL;
  state->action_json = NULL;
  state->tool_args_json = NULL;
  state->tool_call_id = NULL;
  state->final_text = NULL;
  state->error_text = NULL;
  memset(&state->tool_call, 0, sizeof(state->tool_call));
}

static int append_str(char **buf, size_t *len, size_t *cap, const char *text) {
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

static char *json_escape(const char *input) {
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;
  const unsigned char *p;

  if (input == NULL) {
    return dup_str("");
  }

  for (p = (const unsigned char *)input; *p != '\0'; p++) {
    char esc[8];
    switch (*p) {
      case '\\':
        if (!append_str(&out, &len, &cap, "\\\\")) {
          free(out);
          return NULL;
        }
        break;
      case '"':
        if (!append_str(&out, &len, &cap, "\\\"")) {
          free(out);
          return NULL;
        }
        break;
      case '\n':
        if (!append_str(&out, &len, &cap, "\\n")) {
          free(out);
          return NULL;
        }
        break;
      case '\r':
        if (!append_str(&out, &len, &cap, "\\r")) {
          free(out);
          return NULL;
        }
        break;
      case '\t':
        if (!append_str(&out, &len, &cap, "\\t")) {
          free(out);
          return NULL;
        }
        break;
      default:
        if (*p < 0x20) {
          snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)*p);
          if (!append_str(&out, &len, &cap, esc)) {
            free(out);
            return NULL;
          }
        } else {
          char c[2];
          c[0] = (char)*p;
          c[1] = '\0';
          if (!append_str(&out, &len, &cap, c)) {
            free(out);
            return NULL;
          }
        }
        break;
    }
  }

  if (out == NULL) {
    return dup_str("");
  }
  return out;
}

static const char *skip_ws(const char *s) {
  while (s != NULL && *s != '\0' && isspace((unsigned char)*s)) {
    s++;
  }
  return s;
}

static const char *find_json_key(const char *json, const char *key) {
  char *needle;
  const char *p;
  size_t key_len;

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
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  p = find_json_key(json, key);
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
      if (!append_str(&out, &len, &cap, one)) {
        free(out);
        return NULL;
      }
    }

    p++;
  }

  if (out == NULL) {
    out = dup_str("");
  }
  return out;
}

static char *json_parse_string_value(const char *p) {
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
      if (!append_str(&out, &len, &cap, one)) {
        free(out);
        return NULL;
      }
    }

    p++;
  }

  if (out == NULL) {
    out = dup_str("");
  }
  return out;
}

static int json_extract_number(const char *json, const char *key, double *out) {
  const char *p;
  char *end;
  double value;

  if (out == NULL) {
    return 0;
  }
  p = find_json_key(json, key);
  if (p == NULL) {
    return 0;
  }

  value = strtod(p, &end);
  if (p == end) {
    return 0;
  }
  *out = value;
  return 1;
}

static char *read_file_all(const char *path) {
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

static int write_file_all(const char *path, const char *content) {
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

static int run_curl_responses(const char *request_json, char **response_body, int *http_code, char **err_out) {
  char req_path[] = "/tmp/chi-openai-req-XXXXXX";
  char resp_path[] = "/tmp/chi-openai-resp-XXXXXX";
  int req_fd;
  int resp_fd;
  char *cmd;
  FILE *pipe;
  char status_buf[64];
  int status;
  char *body;

  *response_body = NULL;
  *http_code = 0;
  *err_out = NULL;

  req_fd = mkstemp(req_path);
  if (req_fd < 0) {
    *err_out = dup_str("failed to create request temp file");
    return 0;
  }
  close(req_fd);

  resp_fd = mkstemp(resp_path);
  if (resp_fd < 0) {
    unlink(req_path);
    *err_out = dup_str("failed to create response temp file");
    return 0;
  }
  close(resp_fd);

  if (!write_file_all(req_path, request_json)) {
    unlink(req_path);
    unlink(resp_path);
    *err_out = dup_str("failed to write request body");
    return 0;
  }

  cmd = (char *)malloc(strlen(req_path) + strlen(resp_path) + 512);
  if (cmd == NULL) {
    unlink(req_path);
    unlink(resp_path);
    *err_out = dup_str("out of memory building curl command");
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
    *err_out = dup_str("failed to run curl");
    return 0;
  }

  memset(status_buf, 0, sizeof(status_buf));
  if (fgets(status_buf, sizeof(status_buf), pipe) == NULL) {
    status_buf[0] = '\0';
  }

  status = pclose(pipe);
  free(cmd);

  body = read_file_all(resp_path);
  unlink(req_path);
  unlink(resp_path);

  if (body == NULL) {
    *err_out = dup_str("failed to read curl response body");
    return 0;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    *err_out = dup_str("curl request failed");
    free(body);
    return 0;
  }

  *http_code = atoi(status_buf);
  *response_body = body;
  return 1;
}

static char *extract_first_output_text(const char *json) {
  const char *p;
  char *text;
  const char *next_text;
  const char *v;

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
          char *parsed = json_parse_string_value(v);
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

static char *extract_json_object(const char *text) {
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

static char *serialize_conversation(const chi_provider_request *request) {
  size_t i;
  char *buf = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (!append_str(&buf, &len, &cap, "Conversation so far:\n")) {
    free(buf);
    return NULL;
  }

  for (i = 0; i < request->message_count; i++) {
    const chi_message *m = &request->messages[i];
    const char *role = (m->role == NULL) ? "unknown" : m->role;
    const char *text = (m->text == NULL) ? "" : m->text;

    if (!append_str(&buf, &len, &cap, "- ")) {
      free(buf);
      return NULL;
    }
    if (!append_str(&buf, &len, &cap, role)) {
      free(buf);
      return NULL;
    }

    if (m->tool_name != NULL && m->tool_name[0] != '\0') {
      if (!append_str(&buf, &len, &cap, "[") || !append_str(&buf, &len, &cap, m->tool_name) ||
          !append_str(&buf, &len, &cap, "]")) {
        free(buf);
        return NULL;
      }
    }

    if (!append_str(&buf, &len, &cap, ": ")) {
      free(buf);
      return NULL;
    }
    if (!append_str(&buf, &len, &cap, text)) {
      free(buf);
      return NULL;
    }

    if (m->arguments_json != NULL && m->arguments_json[0] != '\0') {
      if (!append_str(&buf, &len, &cap, "\n  args=") ||
          !append_str(&buf, &len, &cap, m->arguments_json)) {
        free(buf);
        return NULL;
      }
    }

    if (!append_str(&buf, &len, &cap, "\n")) {
      free(buf);
      return NULL;
    }
  }

  if (!append_str(&buf, &len, &cap, "\nNow decide the next action.\n")) {
    free(buf);
    return NULL;
  }

  return buf;
}

static chi_status openai_provider(
    const chi_provider_request *request,
    chi_provider_response *response,
    void *user_data) {
  openai_provider_state *state = (openai_provider_state *)user_data;
  const char *api_key;
  const char *model;
  const char *reasoning;
  const char *instructions =
      "You are the policy layer for a coding agent with one tool: bash. "
      "Return JSON only with no markdown.\n"
      "Output schema:\n"
      "- Tool call: {\"kind\":\"tool\",\"name\":\"bash\",\"arguments\":{\"command\":\"...\",\"timeout\":number(optional)}}\n"
      "- Final answer: {\"kind\":\"final\",\"text\":\"...\"}\n"
      "Rules:\n"
      "- Use bash when file edits or command execution are needed.\n"
      "- Use uv run for python execution.\n"
      "- Keep commands safe and scoped to the working directory.\n"
      "- After tool results indicate success, return final.\n";
  char *conversation = NULL;
  char *esc_instructions = NULL;
  char *esc_conversation = NULL;
  char *request_json = NULL;
  char *resp_body = NULL;
  int http_code = 0;
  char *curl_err = NULL;
  char *kind = NULL;

  memset(response, 0, sizeof(*response));
  provider_reset(state);

  api_key = getenv("OPENAI_API_KEY");
  if (api_key == NULL || api_key[0] == '\0') {
    return CHI_ERR_PROVIDER_FAILED;
  }

  model = (request->model_id != NULL && request->model_id[0] != '\0') ? request->model_id : "gpt-5.2-codex";
  reasoning = (request->reasoning_effort != NULL && request->reasoning_effort[0] != '\0') ? request->reasoning_effort : "high";

  conversation = serialize_conversation(request);
  if (conversation == NULL) {
    return CHI_ERR_OOM;
  }

  esc_instructions = json_escape(instructions);
  esc_conversation = json_escape(conversation);
  if (esc_instructions == NULL || esc_conversation == NULL) {
    free(conversation);
    free(esc_instructions);
    free(esc_conversation);
    return CHI_ERR_OOM;
  }

  {
    char *esc_model = json_escape(model);
    char *esc_reasoning = json_escape(reasoning);
    if (esc_model == NULL || esc_reasoning == NULL) {
      free(conversation);
      free(esc_instructions);
      free(esc_conversation);
      free(esc_model);
      free(esc_reasoning);
      return CHI_ERR_OOM;
    }

    request_json = (char *)malloc(strlen(esc_model) + strlen(esc_reasoning) + strlen(esc_instructions) +
                                  strlen(esc_conversation) + 512);
    if (request_json == NULL) {
      free(conversation);
      free(esc_instructions);
      free(esc_conversation);
      free(esc_model);
      free(esc_reasoning);
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
             "\"max_output_tokens\":800"
             "}",
             esc_model,
             esc_instructions,
             esc_conversation,
             esc_reasoning);

    free(esc_model);
    free(esc_reasoning);
  }

  free(conversation);
  free(esc_instructions);
  free(esc_conversation);

  if (!run_curl_responses(request_json, &resp_body, &http_code, &curl_err)) {
    free(request_json);
    if (curl_err != NULL) {
      state->error_text = curl_err;
    } else {
      state->error_text = dup_str("curl call failed");
    }
    if (debug_enabled()) {
      fprintf(stderr, "[chi debug] curl failed: %s\n", state->error_text == NULL ? "(unknown)" : state->error_text);
    }
    return CHI_ERR_PROVIDER_FAILED;
  }
  free(request_json);

  state->response_json = resp_body;

  if (http_code < 200 || http_code >= 300) {
    state->error_text = (char *)malloc(128 + strlen(resp_body));
    if (state->error_text != NULL) {
      snprintf(state->error_text, 128 + strlen(resp_body), "openai http %d: %s", http_code, resp_body);
    }
    if (debug_enabled()) {
      fprintf(stderr, "[chi debug] http_code=%d body=%s\n", http_code, resp_body);
    }
    return CHI_ERR_PROVIDER_FAILED;
  }

  state->output_text = extract_first_output_text(resp_body);
  if (state->output_text == NULL) {
    state->error_text = dup_str("could not parse output text from openai response");
    if (debug_enabled()) {
      fprintf(stderr, "[chi debug] parse output_text failed; body=%s\n", resp_body);
    }
    return CHI_ERR_PROVIDER_FAILED;
  }

  state->action_json = extract_json_object(state->output_text);
  if (state->action_json == NULL) {
    state->final_text = dup_str(state->output_text);
    if (state->final_text == NULL) {
      return CHI_ERR_OOM;
    }
    response->text = state->final_text;
    response->stop_reason = CHI_STOP_STOP;
    return CHI_OK;
  }

  kind = json_extract_string(state->action_json, "kind");
  if (kind != NULL && strcmp(kind, "tool") == 0) {
    char *name = json_extract_string(state->action_json, "name");
    char *command = json_extract_string(state->action_json, "command");
    double timeout = 0;
    int has_timeout = json_extract_number(state->action_json, "timeout", &timeout);
    char *esc_command;

    if (name == NULL || name[0] == '\0') {
      free(name);
      name = dup_str("bash");
    }

    if (command == NULL || command[0] == '\0') {
      free(name);
      free(command);
      free(kind);
      state->final_text = dup_str("model requested a tool call but did not provide command");
      if (state->final_text == NULL) {
        return CHI_ERR_OOM;
      }
      response->text = state->final_text;
      response->stop_reason = CHI_STOP_STOP;
      return CHI_OK;
    }

    esc_command = json_escape(command);
    if (esc_command == NULL) {
      free(name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }

    if (has_timeout && timeout > 0) {
      state->tool_args_json = (char *)malloc(strlen(esc_command) + 96);
      if (state->tool_args_json != NULL) {
        snprintf(state->tool_args_json,
                 strlen(esc_command) + 96,
                 "{\"command\":\"%s\",\"timeout\":%.3f}",
                 esc_command,
                 timeout);
      }
    } else {
      state->tool_args_json = (char *)malloc(strlen(esc_command) + 32);
      if (state->tool_args_json != NULL) {
        snprintf(state->tool_args_json, strlen(esc_command) + 32, "{\"command\":\"%s\"}", esc_command);
      }
    }
    free(esc_command);

    if (state->tool_args_json == NULL) {
      free(name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }

    state->call_seq++;
    state->tool_call_id = (char *)malloc(64);
    if (state->tool_call_id == NULL) {
      free(name);
      free(command);
      free(kind);
      return CHI_ERR_OOM;
    }
    snprintf(state->tool_call_id, 64, "call_%llu", (unsigned long long)state->call_seq);

    state->tool_call.id = state->tool_call_id;
    state->tool_call.name = name;
    state->tool_call.arguments_json = state->tool_args_json;

    response->tool_calls = &state->tool_call;
    response->tool_call_count = 1;
    response->stop_reason = CHI_STOP_TOOL_USE;

    free(command);
    free(kind);
    return CHI_OK;
  }

  {
    char *text = json_extract_string(state->action_json, "text");
    if (text == NULL || text[0] == '\0') {
      free(text);
      state->final_text = dup_str(state->output_text);
    } else {
      state->final_text = text;
    }

    if (state->final_text == NULL) {
      free(kind);
      return CHI_ERR_OOM;
    }
    response->text = state->final_text;
    response->stop_reason = CHI_STOP_STOP;
  }

  free(kind);
  return CHI_OK;
}

static void on_event(const chi_event *event, void *user_data) {
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
  openai_provider_state provider;
  char session_id[CHI_SESSION_ID_MAX];
  const char *prompt;
  const char *working_dir = ".";
  chi_status st;

  if (argc < 2) {
    fprintf(stderr,
            "usage: %s \"prompt\" [working_dir]\n"
            "example: %s \"write hello.py and run it with uv run\" /home/johann/code/chi/agent_playground\n",
            argv[0],
            argv[0]);
    return 2;
  }

  prompt = argv[1];
  if (argc >= 3) {
    working_dir = argv[2];
  }

  if (getenv("OPENAI_API_KEY") == NULL || getenv("OPENAI_API_KEY")[0] == '\0') {
    fprintf(stderr, "OPENAI_API_KEY is not set\n");
    return 2;
  }

  memset(&provider, 0, sizeof(provider));

  options = chi_runtime_options_default();
  options.provider = openai_provider;
  options.provider_user_data = &provider;
  options.system_prompt = "You are a concise coding assistant.";
  options.working_dir = working_dir;
  options.max_tool_rounds = 12;

  runtime = chi_runtime_create(&options);
  if (runtime == NULL) {
    fprintf(stderr, "failed to create runtime\n");
    return 1;
  }

  st = chi_runtime_subscribe(runtime, on_event, NULL, NULL);
  if (st != CHI_OK) {
    fprintf(stderr, "failed to subscribe: %s\n", chi_status_string(st));
    chi_runtime_destroy(runtime);
    provider_reset(&provider);
    return 1;
  }

  st = chi_runtime_start_session(runtime, prompt, session_id, sizeof(session_id));
  if (st != CHI_OK) {
    fprintf(stderr, "start_session failed: %s\n", chi_status_string(st));
    chi_runtime_destroy(runtime);
    provider_reset(&provider);
    return 1;
  }

  printf("session: %s\n", session_id);

  chi_runtime_destroy(runtime);
  provider_reset(&provider);
  return 0;
}
