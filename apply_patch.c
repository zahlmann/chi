#include "apply_patch.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define AP_BEGIN_PATCH "*** Begin Patch"
#define AP_END_PATCH "*** End Patch"
#define AP_ADD_FILE "*** Add File: "
#define AP_DELETE_FILE "*** Delete File: "
#define AP_UPDATE_FILE "*** Update File: "
#define AP_MOVE_TO "*** Move to: "
#define AP_END_OF_FILE "*** End of File"

typedef struct {
  char **items;
  size_t count;
  size_t cap;
} ap_strvec;

typedef struct {
  char *change_context;
  ap_strvec old_lines;
  ap_strvec new_lines;
  int is_end_of_file;
} ap_update_chunk;

typedef enum {
  AP_HUNK_ADD = 1,
  AP_HUNK_DELETE = 2,
  AP_HUNK_UPDATE = 3
} ap_hunk_type;

typedef struct {
  ap_hunk_type type;
  char *path;
  char *move_to;
  ap_strvec add_lines;
  ap_update_chunk *chunks;
  size_t chunk_count;
  size_t chunk_cap;
  int line_number;
} ap_hunk;

typedef struct {
  ap_hunk *items;
  size_t count;
  size_t cap;
} ap_hunkvec;

typedef enum {
  AP_PLAN_ADD = 1,
  AP_PLAN_DELETE = 2,
  AP_PLAN_UPDATE = 3
} ap_plan_kind;

typedef struct {
  ap_plan_kind kind;
  char *src_abs;
  char *dest_abs;
  char *display_path;
  char *new_content;
} ap_plan_op;

typedef struct {
  ap_plan_op *items;
  size_t count;
  size_t cap;
} ap_planvec;

typedef struct {
  char *path;
  int known;
  int exists;
  char *content;
} ap_virtual_file;

typedef struct {
  ap_virtual_file *items;
  size_t count;
  size_t cap;
} ap_virtualfs;

typedef enum {
  AP_UNDO_RESTORE = 1,
  AP_UNDO_REMOVE = 2
} ap_undo_kind;

typedef struct {
  ap_undo_kind kind;
  char *path;
  char *content;
} ap_undo;

typedef struct {
  ap_undo *items;
  size_t count;
  size_t cap;
} ap_undovec;

typedef struct {
  size_t start;
  size_t old_len;
  ap_strvec new_lines;
} ap_replacement;

typedef struct {
  ap_replacement *items;
  size_t count;
  size_t cap;
} ap_replacement_vec;

static char *ap_strdup(const char *s) {
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

static int ap_is_blank(const char *s) {
  if (s == NULL) {
    return 1;
  }
  while (*s != '\0') {
    if (*s != ' ' && *s != '\t' && *s != '\r' && *s != '\n' && *s != '\f' && *s != '\v') {
      return 0;
    }
    s++;
  }
  return 1;
}

static char *ap_format(const char *fmt, ...) {
  va_list ap;
  va_list ap2;
  int n;
  char *out;

  va_start(ap, fmt);
  va_copy(ap2, ap);
  n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) {
    va_end(ap2);
    return NULL;
  }

  out = (char *)malloc((size_t)n + 1);
  if (out == NULL) {
    va_end(ap2);
    return NULL;
  }
  vsnprintf(out, (size_t)n + 1, fmt, ap2);
  va_end(ap2);
  return out;
}

static int ap_ensure_cap(void **ptr, size_t *cap, size_t need, size_t elem_size) {
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

  grown = realloc(*ptr, next * elem_size);
  if (grown == NULL) {
    return 0;
  }

  *ptr = grown;
  *cap = next;
  return 1;
}

static int ap_append_n(char **buf, size_t *len, size_t *cap, const char *data, size_t n) {
  size_t need;
  size_t next;
  char *grown;

  if (n == 0) {
    return 1;
  }

  need = *len + n + 1;
  if (need > *cap) {
    next = (*cap == 0) ? 256 : *cap;
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

static int ap_append(char **buf, size_t *len, size_t *cap, const char *text) {
  if (text == NULL) {
    text = "";
  }
  return ap_append_n(buf, len, cap, text, strlen(text));
}

static int ap_streq(const char *a, const char *b) {
  return a != NULL && b != NULL && strcmp(a, b) == 0;
}

static void ap_trim_view(const char *line, const char **out_start, size_t *out_len) {
  const char *start;
  const char *end;

  if (line == NULL) {
    *out_start = "";
    *out_len = 0;
    return;
  }

  start = line;
  while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n' ||
         *start == '\f' || *start == '\v') {
    start++;
  }

  end = line + strlen(line);
  while (end > start) {
    char c = end[-1];
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') {
      break;
    }
    end--;
  }

  *out_start = start;
  *out_len = (size_t)(end - start);
}

static int ap_trimmed_equals(const char *line, const char *literal) {
  const char *start;
  size_t len;
  size_t lit_len;

  if (literal == NULL) {
    return 0;
  }

  ap_trim_view(line, &start, &len);
  lit_len = strlen(literal);
  return len == lit_len && strncmp(start, literal, len) == 0;
}

static int ap_trimmed_starts_with(const char *line, const char *prefix) {
  const char *start;
  size_t len;
  size_t prefix_len;

  if (prefix == NULL) {
    return 0;
  }

  ap_trim_view(line, &start, &len);
  prefix_len = strlen(prefix);
  return len >= prefix_len && strncmp(start, prefix, prefix_len) == 0;
}

static char *ap_trimmed_substring_after(const char *line, const char *prefix) {
  const char *start;
  size_t len;
  size_t prefix_len;

  ap_trim_view(line, &start, &len);
  prefix_len = strlen(prefix);
  if (len < prefix_len || strncmp(start, prefix, prefix_len) != 0) {
    return NULL;
  }
  return ap_strdup(start + prefix_len);
}

static void ap_strvec_free(ap_strvec *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    free(v->items[i]);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static int ap_strvec_push_owned(ap_strvec *v, char *s) {
  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(char *))) {
    return 0;
  }
  v->items[v->count++] = s;
  return 1;
}

static int ap_strvec_push(ap_strvec *v, const char *s) {
  char *copy = ap_strdup(s == NULL ? "" : s);
  if (copy == NULL) {
    return 0;
  }
  if (!ap_strvec_push_owned(v, copy)) {
    free(copy);
    return 0;
  }
  return 1;
}

static void ap_update_chunk_free(ap_update_chunk *c) {
  if (c == NULL) {
    return;
  }
  free(c->change_context);
  ap_strvec_free(&c->old_lines);
  ap_strvec_free(&c->new_lines);
  memset(c, 0, sizeof(*c));
}

static void ap_hunk_free(ap_hunk *h) {
  size_t i;
  if (h == NULL) {
    return;
  }
  free(h->path);
  free(h->move_to);
  ap_strvec_free(&h->add_lines);
  for (i = 0; i < h->chunk_count; i++) {
    ap_update_chunk_free(&h->chunks[i]);
  }
  free(h->chunks);
  memset(h, 0, sizeof(*h));
}

static void ap_hunkvec_free(ap_hunkvec *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    ap_hunk_free(&v->items[i]);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static int ap_hunkvec_push_owned(ap_hunkvec *v, ap_hunk *h) {
  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(ap_hunk))) {
    return 0;
  }
  v->items[v->count++] = *h;
  memset(h, 0, sizeof(*h));
  return 1;
}

static void ap_plan_op_free(ap_plan_op *op) {
  if (op == NULL) {
    return;
  }
  free(op->src_abs);
  free(op->dest_abs);
  free(op->display_path);
  free(op->new_content);
  memset(op, 0, sizeof(*op));
}

static void ap_planvec_free(ap_planvec *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    ap_plan_op_free(&v->items[i]);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static int ap_planvec_push_owned(ap_planvec *v, ap_plan_op *op) {
  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(ap_plan_op))) {
    return 0;
  }
  v->items[v->count++] = *op;
  memset(op, 0, sizeof(*op));
  return 1;
}

static void ap_virtualfs_free(ap_virtualfs *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    free(v->items[i].path);
    free(v->items[i].content);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static ssize_t ap_virtualfs_find(const ap_virtualfs *v, const char *path) {
  size_t i;
  if (v == NULL || path == NULL) {
    return -1;
  }
  for (i = 0; i < v->count; i++) {
    if (v->items[i].path != NULL && strcmp(v->items[i].path, path) == 0) {
      return (ssize_t)i;
    }
  }
  return -1;
}

static int ap_virtualfs_set(ap_virtualfs *v, const char *path, int exists, const char *content) {
  ssize_t idx;
  char *path_copy = NULL;
  char *content_copy = NULL;

  if (v == NULL || path == NULL) {
    return 0;
  }

  idx = ap_virtualfs_find(v, path);
  if (idx >= 0) {
    ap_virtual_file *entry = &v->items[(size_t)idx];
    if (content != NULL) {
      content_copy = ap_strdup(content);
      if (content_copy == NULL) {
        return 0;
      }
    }
    free(entry->content);
    entry->content = content_copy;
    entry->known = 1;
    entry->exists = exists;
    return 1;
  }

  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(ap_virtual_file))) {
    return 0;
  }

  path_copy = ap_strdup(path);
  if (path_copy == NULL) {
    return 0;
  }

  if (content != NULL) {
    content_copy = ap_strdup(content);
    if (content_copy == NULL) {
      free(path_copy);
      return 0;
    }
  }

  v->items[v->count].path = path_copy;
  v->items[v->count].known = 1;
  v->items[v->count].exists = exists;
  v->items[v->count].content = content_copy;
  v->count++;
  return 1;
}

static int ap_virtualfs_get(const ap_virtualfs *v, const char *path, int *known, int *exists, const char **content) {
  ssize_t idx;
  if (known != NULL) {
    *known = 0;
  }
  if (exists != NULL) {
    *exists = 0;
  }
  if (content != NULL) {
    *content = NULL;
  }
  if (v == NULL || path == NULL) {
    return 0;
  }

  idx = ap_virtualfs_find(v, path);
  if (idx < 0) {
    return 1;
  }

  if (known != NULL) {
    *known = 1;
  }
  if (exists != NULL) {
    *exists = v->items[(size_t)idx].exists;
  }
  if (content != NULL) {
    *content = v->items[(size_t)idx].content;
  }
  return 1;
}

static void ap_undovec_free(ap_undovec *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    free(v->items[i].path);
    free(v->items[i].content);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static int ap_undovec_push(ap_undovec *v, ap_undo_kind kind, const char *path, const char *content) {
  char *path_copy;
  char *content_copy = NULL;

  if (v == NULL || path == NULL) {
    return 0;
  }

  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(ap_undo))) {
    return 0;
  }

  path_copy = ap_strdup(path);
  if (path_copy == NULL) {
    return 0;
  }

  if (content != NULL) {
    content_copy = ap_strdup(content);
    if (content_copy == NULL) {
      free(path_copy);
      return 0;
    }
  }

  v->items[v->count].kind = kind;
  v->items[v->count].path = path_copy;
  v->items[v->count].content = content_copy;
  v->count++;
  return 1;
}

static void ap_replacements_free(ap_replacement_vec *v) {
  size_t i;
  if (v == NULL) {
    return;
  }
  for (i = 0; i < v->count; i++) {
    ap_strvec_free(&v->items[i].new_lines);
  }
  free(v->items);
  memset(v, 0, sizeof(*v));
}

static int ap_replacements_push(ap_replacement_vec *v, size_t start, size_t old_len, const ap_strvec *new_lines, size_t from, size_t take) {
  size_t i;
  ap_replacement rep;

  memset(&rep, 0, sizeof(rep));
  rep.start = start;
  rep.old_len = old_len;

  for (i = 0; i < take; i++) {
    if (!ap_strvec_push(&rep.new_lines, new_lines->items[from + i])) {
      ap_strvec_free(&rep.new_lines);
      return 0;
    }
  }

  if (!ap_ensure_cap((void **)&v->items, &v->cap, v->count + 1, sizeof(ap_replacement))) {
    ap_strvec_free(&rep.new_lines);
    return 0;
  }

  v->items[v->count++] = rep;
  return 1;
}

static int ap_compare_replacement(const void *lhs, const void *rhs) {
  const ap_replacement *a = (const ap_replacement *)lhs;
  const ap_replacement *b = (const ap_replacement *)rhs;
  if (a->start < b->start) {
    return -1;
  }
  if (a->start > b->start) {
    return 1;
  }
  return 0;
}

static char *ap_trim_all(const char *text) {
  const char *start;
  size_t len;
  char *out;
  ap_trim_view(text, &start, &len);
  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out, start, len);
  out[len] = '\0';
  return out;
}

static int ap_split_lines(const char *text, ap_strvec *lines, char **err_out) {
  size_t i;
  size_t start;

  if (err_out != NULL) {
    *err_out = NULL;
  }

  if (text == NULL) {
    if (!ap_strvec_push(lines, "")) {
      if (err_out != NULL) {
        *err_out = ap_strdup("out of memory splitting patch");
      }
      return 0;
    }
    return 1;
  }

  start = 0;
  for (i = 0;; i++) {
    char c = text[i];
    if (c == '\n' || c == '\0') {
      size_t n = i - start;
      char *line = (char *)malloc(n + 1);
      if (line == NULL) {
        if (err_out != NULL) {
          *err_out = ap_strdup("out of memory splitting patch");
        }
        return 0;
      }
      memcpy(line, text + start, n);
      line[n] = '\0';

      if (n > 0 && line[n - 1] == '\r') {
        line[n - 1] = '\0';
      }

      if (!ap_strvec_push_owned(lines, line)) {
        free(line);
        if (err_out != NULL) {
          *err_out = ap_strdup("out of memory splitting patch");
        }
        return 0;
      }

      if (c == '\0') {
        break;
      }
      start = i + 1;
    }
  }

  return 1;
}

static int ap_join_lines_with_trailing_newline(const ap_strvec *lines, char **out) {
  size_t i;
  char *buf = NULL;
  size_t len = 0;
  size_t cap = 0;

  *out = NULL;
  for (i = 0; i < lines->count; i++) {
    if (!ap_append(&buf, &len, &cap, lines->items[i]) ||
        !ap_append_n(&buf, &len, &cap, "\n", 1)) {
      free(buf);
      return 0;
    }
  }

  if (buf == NULL) {
    buf = ap_strdup("");
    if (buf == NULL) {
      return 0;
    }
  }

  *out = buf;
  return 1;
}

static int ap_parse_update_chunk(
    const ap_strvec *lines,
    size_t start_idx,
    int allow_missing_context,
    int line_number,
    ap_update_chunk *chunk_out,
    size_t *consumed_out,
    char **err_out) {
  size_t idx;
  int parsed = 0;
  ap_update_chunk chunk;

  memset(&chunk, 0, sizeof(chunk));
  *consumed_out = 0;

  if (start_idx >= lines->count) {
    *err_out = ap_format("Invalid patch hunk on line %d: Update hunk does not contain any lines", line_number);
    return 0;
  }

  idx = start_idx;
  if (ap_streq(lines->items[idx], "@@")) {
    idx++;
  } else if (strncmp(lines->items[idx], "@@ ", 3) == 0) {
    chunk.change_context = ap_strdup(lines->items[idx] + 3);
    if (chunk.change_context == NULL) {
      *err_out = ap_strdup("out of memory parsing patch");
      return 0;
    }
    idx++;
  } else if (!allow_missing_context) {
    *err_out = ap_format(
        "Invalid patch hunk on line %d: Expected update hunk to start with a @@ context marker, got: '%s'",
        line_number,
        lines->items[idx]);
    ap_update_chunk_free(&chunk);
    return 0;
  }

  if (idx >= lines->count) {
    *err_out = ap_format("Invalid patch hunk on line %d: Update hunk does not contain any lines", line_number + 1);
    ap_update_chunk_free(&chunk);
    return 0;
  }

  while (idx < lines->count) {
    const char *line = lines->items[idx];
    if (ap_streq(line, AP_END_OF_FILE)) {
      if (parsed == 0) {
        *err_out = ap_format("Invalid patch hunk on line %d: Update hunk does not contain any lines", line_number + 1);
        ap_update_chunk_free(&chunk);
        return 0;
      }
      chunk.is_end_of_file = 1;
      idx++;
      break;
    }

    if (line[0] == '\0') {
      if (!ap_strvec_push(&chunk.old_lines, "") || !ap_strvec_push(&chunk.new_lines, "")) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_update_chunk_free(&chunk);
        return 0;
      }
      parsed++;
      idx++;
      continue;
    }

    if (line[0] == ' ') {
      if (!ap_strvec_push(&chunk.old_lines, line + 1) || !ap_strvec_push(&chunk.new_lines, line + 1)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_update_chunk_free(&chunk);
        return 0;
      }
      parsed++;
      idx++;
      continue;
    }

    if (line[0] == '+') {
      if (!ap_strvec_push(&chunk.new_lines, line + 1)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_update_chunk_free(&chunk);
        return 0;
      }
      parsed++;
      idx++;
      continue;
    }

    if (line[0] == '-') {
      if (!ap_strvec_push(&chunk.old_lines, line + 1)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_update_chunk_free(&chunk);
        return 0;
      }
      parsed++;
      idx++;
      continue;
    }

    if (parsed == 0) {
      *err_out = ap_format(
          "Invalid patch hunk on line %d: Unexpected line found in update hunk: '%s'. Every line should start with ' ' (context line), '+' (added line), or '-' (removed line)",
          line_number + 1,
          line);
      ap_update_chunk_free(&chunk);
      return 0;
    }

    break;
  }

  if (parsed == 0) {
    *err_out = ap_format("Invalid patch hunk on line %d: Update hunk does not contain any lines", line_number);
    ap_update_chunk_free(&chunk);
    return 0;
  }

  *chunk_out = chunk;
  *consumed_out = idx - start_idx;
  return 1;
}

static int ap_parse_patch(const char *patch, ap_hunkvec *hunks_out, char **err_out) {
  char *trimmed = NULL;
  ap_strvec lines;
  size_t idx;
  int line_number;

  memset(&lines, 0, sizeof(lines));
  *err_out = NULL;

  trimmed = ap_trim_all(patch);
  if (trimmed == NULL) {
    *err_out = ap_strdup("out of memory parsing patch");
    return 0;
  }

  if (!ap_split_lines(trimmed, &lines, err_out)) {
    free(trimmed);
    ap_strvec_free(&lines);
    return 0;
  }
  free(trimmed);

  if (lines.count == 0 || !ap_trimmed_equals(lines.items[0], AP_BEGIN_PATCH)) {
    *err_out = ap_strdup("Invalid patch: The first line of the patch must be '*** Begin Patch'");
    ap_strvec_free(&lines);
    return 0;
  }

  if (!ap_trimmed_equals(lines.items[lines.count - 1], AP_END_PATCH)) {
    *err_out = ap_strdup("Invalid patch: The last line of the patch must be '*** End Patch'");
    ap_strvec_free(&lines);
    return 0;
  }

  idx = 1;
  line_number = 2;

  while (idx + 1 < lines.count) {
    ap_hunk hunk;
    char *path = NULL;
    const char *line = lines.items[idx];

    memset(&hunk, 0, sizeof(hunk));
    hunk.line_number = line_number;

    if (ap_is_blank(line)) {
      idx++;
      line_number++;
      continue;
    }

    path = ap_trimmed_substring_after(line, AP_ADD_FILE);
    if (path != NULL) {
      hunk.type = AP_HUNK_ADD;
      hunk.path = path;
      idx++;
      line_number++;

      while (idx + 1 < lines.count && lines.items[idx][0] == '+') {
        if (!ap_strvec_push(&hunk.add_lines, lines.items[idx] + 1)) {
          *err_out = ap_strdup("out of memory parsing patch");
          ap_hunk_free(&hunk);
          ap_strvec_free(&lines);
          return 0;
        }
        idx++;
        line_number++;
      }

      if (!ap_hunkvec_push_owned(hunks_out, &hunk)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_hunk_free(&hunk);
        ap_strvec_free(&lines);
        return 0;
      }
      continue;
    }

    path = ap_trimmed_substring_after(line, AP_DELETE_FILE);
    if (path != NULL) {
      hunk.type = AP_HUNK_DELETE;
      hunk.path = path;
      idx++;
      line_number++;

      if (!ap_hunkvec_push_owned(hunks_out, &hunk)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_hunk_free(&hunk);
        ap_strvec_free(&lines);
        return 0;
      }
      continue;
    }

    path = ap_trimmed_substring_after(line, AP_UPDATE_FILE);
    if (path != NULL) {
      hunk.type = AP_HUNK_UPDATE;
      hunk.path = path;
      idx++;
      line_number++;

      if (idx + 1 < lines.count) {
        char *move_to = ap_trimmed_substring_after(lines.items[idx], AP_MOVE_TO);
        if (move_to != NULL) {
          hunk.move_to = move_to;
          idx++;
          line_number++;
        }
      }

      while (idx + 1 < lines.count) {
        ap_update_chunk chunk;
        size_t consumed = 0;

        if (ap_is_blank(lines.items[idx])) {
          idx++;
          line_number++;
          continue;
        }

        if (ap_trimmed_equals(lines.items[idx], AP_END_PATCH) ||
            ap_trimmed_starts_with(lines.items[idx], AP_ADD_FILE) ||
            ap_trimmed_starts_with(lines.items[idx], AP_DELETE_FILE) ||
            ap_trimmed_starts_with(lines.items[idx], AP_UPDATE_FILE)) {
          break;
        }

        memset(&chunk, 0, sizeof(chunk));
        if (!ap_parse_update_chunk(&lines, idx, hunk.chunk_count == 0, line_number, &chunk, &consumed, err_out)) {
          ap_hunk_free(&hunk);
          ap_strvec_free(&lines);
          return 0;
        }

        if (!ap_ensure_cap((void **)&hunk.chunks, &hunk.chunk_cap, hunk.chunk_count + 1, sizeof(ap_update_chunk))) {
          *err_out = ap_strdup("out of memory parsing patch");
          ap_update_chunk_free(&chunk);
          ap_hunk_free(&hunk);
          ap_strvec_free(&lines);
          return 0;
        }

        hunk.chunks[hunk.chunk_count++] = chunk;
        idx += consumed;
        line_number += (int)consumed;
      }

      if (hunk.chunk_count == 0) {
        *err_out = ap_format(
            "Invalid patch hunk on line %d: Update file hunk for path '%s' is empty",
            hunk.line_number,
            hunk.path == NULL ? "" : hunk.path);
        ap_hunk_free(&hunk);
        ap_strvec_free(&lines);
        return 0;
      }

      if (!ap_hunkvec_push_owned(hunks_out, &hunk)) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_hunk_free(&hunk);
        ap_strvec_free(&lines);
        return 0;
      }
      continue;
    }

    {
      const char *trim_start;
      size_t trim_len;
      char *first_line;

      ap_trim_view(line, &trim_start, &trim_len);
      first_line = (char *)malloc(trim_len + 1);
      if (first_line == NULL) {
        *err_out = ap_strdup("out of memory parsing patch");
        ap_strvec_free(&lines);
        return 0;
      }
      memcpy(first_line, trim_start, trim_len);
      first_line[trim_len] = '\0';
      *err_out = ap_format(
          "Invalid patch hunk on line %d: '%s' is not a valid hunk header. Valid hunk headers: '*** Add File: {path}', '*** Delete File: {path}', '*** Update File: {path}'",
          line_number,
          first_line);
      free(first_line);
      ap_strvec_free(&lines);
      return 0;
    }
  }

  ap_strvec_free(&lines);
  return 1;
}

static char *ap_join_path(const char *a, const char *b) {
  size_t a_len;
  size_t b_len;
  int need_sep;
  char *out;

  if (a == NULL || b == NULL) {
    return NULL;
  }

  a_len = strlen(a);
  b_len = strlen(b);
  need_sep = (a_len > 0 && a[a_len - 1] != '/');

  out = (char *)malloc(a_len + (need_sep ? 1 : 0) + b_len + 1);
  if (out == NULL) {
    return NULL;
  }

  memcpy(out, a, a_len);
  if (need_sep) {
    out[a_len] = '/';
    memcpy(out + a_len + 1, b, b_len + 1);
  } else {
    memcpy(out + a_len, b, b_len + 1);
  }

  return out;
}

static char *ap_resolve_path(const char *cwd, const char *path) {
  if (path == NULL) {
    return NULL;
  }
  if (path[0] == '/') {
    return ap_strdup(path);
  }
  if (cwd == NULL || cwd[0] == '\0') {
    return ap_join_path(".", path);
  }
  return ap_join_path(cwd, path);
}

static int ap_stat_file(const char *path, struct stat *st_out) {
  struct stat st;
  if (stat(path, &st) != 0) {
    return 0;
  }
  if (st_out != NULL) {
    *st_out = st;
  }
  return 1;
}

static int ap_read_file(const char *path, char **content_out, char **err_out) {
  FILE *f;
  long size;
  size_t got;
  char *buf;
  struct stat st;

  *content_out = NULL;

  if (!ap_stat_file(path, &st)) {
    if (errno == ENOENT) {
      *err_out = ap_format("Failed to read %s: No such file or directory", path);
    } else {
      *err_out = ap_format("Failed to read %s: %s", path, strerror(errno));
    }
    return 0;
  }

  if (!S_ISREG(st.st_mode)) {
    *err_out = ap_format("Failed to read %s: Not a regular file", path);
    return 0;
  }

  f = fopen(path, "rb");
  if (f == NULL) {
    *err_out = ap_format("Failed to read %s: %s", path, strerror(errno));
    return 0;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    *err_out = ap_format("Failed to read %s: %s", path, strerror(errno));
    return 0;
  }

  size = ftell(f);
  if (size < 0) {
    fclose(f);
    *err_out = ap_format("Failed to read %s: %s", path, strerror(errno));
    return 0;
  }

  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    *err_out = ap_format("Failed to read %s: %s", path, strerror(errno));
    return 0;
  }

  buf = (char *)malloc((size_t)size + 1);
  if (buf == NULL) {
    fclose(f);
    *err_out = ap_strdup("out of memory reading file");
    return 0;
  }

  got = fread(buf, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) {
    free(buf);
    *err_out = ap_format("Failed to read %s", path);
    return 0;
  }

  buf[(size_t)size] = '\0';
  *content_out = buf;
  return 1;
}

static int ap_mkdir_p(const char *path) {
  size_t i;
  char *tmp;
  size_t len;

  if (path == NULL || path[0] == '\0') {
    return 1;
  }

  len = strlen(path);
  tmp = ap_strdup(path);
  if (tmp == NULL) {
    return 0;
  }

  for (i = 1; i < len; i++) {
    if (tmp[i] == '/') {
      tmp[i] = '\0';
      if (tmp[0] != '\0' && mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        free(tmp);
        return 0;
      }
      tmp[i] = '/';
    }
  }

  if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
    free(tmp);
    return 0;
  }

  free(tmp);
  return 1;
}

static char *ap_dirname(const char *path) {
  const char *slash;
  size_t len;
  char *out;

  if (path == NULL) {
    return NULL;
  }

  slash = strrchr(path, '/');
  if (slash == NULL) {
    return ap_strdup("");
  }
  if (slash == path) {
    return ap_strdup("/");
  }

  len = (size_t)(slash - path);
  out = (char *)malloc(len + 1);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out, path, len);
  out[len] = '\0';
  return out;
}

static int ap_write_file_atomic(const char *path, const char *content, char **err_out) {
  char *dir = NULL;
  char *tmp_template = NULL;
  int fd = -1;
  FILE *f = NULL;
  size_t content_len;
  int ok = 0;

  *err_out = NULL;

  if (content == NULL) {
    content = "";
  }

  dir = ap_dirname(path);
  if (dir == NULL) {
    *err_out = ap_strdup("out of memory preparing write path");
    goto cleanup;
  }

  if (dir[0] != '\0' && !ap_streq(dir, "/") && !ap_mkdir_p(dir)) {
    *err_out = ap_format("Failed to create parent directories for %s: %s", path, strerror(errno));
    goto cleanup;
  }

  tmp_template = ap_format("%s/.chi-apply-patch-XXXXXX", dir[0] == '\0' ? "." : dir);
  if (tmp_template == NULL) {
    *err_out = ap_strdup("out of memory preparing temp file");
    goto cleanup;
  }

  fd = mkstemp(tmp_template);
  if (fd < 0) {
    *err_out = ap_format("Failed to create temp file for %s: %s", path, strerror(errno));
    goto cleanup;
  }

  f = fdopen(fd, "wb");
  if (f == NULL) {
    *err_out = ap_format("Failed to open temp file for %s: %s", path, strerror(errno));
    goto cleanup;
  }
  fd = -1;

  content_len = strlen(content);
  if (content_len > 0 && fwrite(content, 1, content_len, f) != content_len) {
    *err_out = ap_format("Failed to write file %s", path);
    goto cleanup;
  }

  if (fflush(f) != 0) {
    *err_out = ap_format("Failed to flush file %s: %s", path, strerror(errno));
    goto cleanup;
  }

  if (fclose(f) != 0) {
    f = NULL;
    *err_out = ap_format("Failed to close file %s: %s", path, strerror(errno));
    goto cleanup;
  }
  f = NULL;

  if (rename(tmp_template, path) != 0) {
    *err_out = ap_format("Failed to write file %s: %s", path, strerror(errno));
    goto cleanup;
  }

  ok = 1;

cleanup:
  if (f != NULL) {
    fclose(f);
  }
  if (fd >= 0) {
    close(fd);
  }
  if (!ok && tmp_template != NULL) {
    unlink(tmp_template);
  }
  free(dir);
  free(tmp_template);
  return ok;
}

static int ap_seek_sequence(const ap_strvec *lines, const ap_strvec *pattern, size_t pattern_len, size_t start, int end_of_file, size_t *idx_out) {
  size_t i;
  size_t j;

  if (pattern_len == 0) {
    *idx_out = start;
    return 1;
  }

  if (lines->count < pattern_len) {
    return 0;
  }

  for (i = start; i + pattern_len <= lines->count; i++) {
    if (end_of_file && (i + pattern_len != lines->count)) {
      continue;
    }

    for (j = 0; j < pattern_len; j++) {
      if (!ap_streq(lines->items[i + j], pattern->items[j])) {
        break;
      }
    }
    if (j == pattern_len) {
      *idx_out = i;
      return 1;
    }
  }

  return 0;
}

static int ap_apply_replacements(ap_strvec *lines, const ap_replacement_vec *repls, char **err_out) {
  size_t r;

  *err_out = NULL;

  if (repls->count == 0) {
    return 1;
  }

  for (r = repls->count; r > 0; r--) {
    const ap_replacement *rep = &repls->items[r - 1];
    size_t i;

    if (rep->start > lines->count) {
      *err_out = ap_strdup("internal replacement index out of range");
      return 0;
    }

    if (rep->start + rep->old_len > lines->count) {
      *err_out = ap_strdup("internal replacement length out of range");
      return 0;
    }

    for (i = 0; i < rep->old_len; i++) {
      free(lines->items[rep->start + i]);
    }

    memmove(
        lines->items + rep->start,
        lines->items + rep->start + rep->old_len,
        (lines->count - rep->start - rep->old_len) * sizeof(char *));
    lines->count -= rep->old_len;

    if (rep->new_lines.count > 0) {
      if (!ap_ensure_cap((void **)&lines->items, &lines->cap, lines->count + rep->new_lines.count, sizeof(char *))) {
        *err_out = ap_strdup("out of memory applying patch");
        return 0;
      }

      memmove(
          lines->items + rep->start + rep->new_lines.count,
          lines->items + rep->start,
          (lines->count - rep->start) * sizeof(char *));

      for (i = 0; i < rep->new_lines.count; i++) {
        lines->items[rep->start + i] = ap_strdup(rep->new_lines.items[i]);
        if (lines->items[rep->start + i] == NULL) {
          *err_out = ap_strdup("out of memory applying patch");
          return 0;
        }
      }
      lines->count += rep->new_lines.count;
    }
  }

  return 1;
}

static int ap_split_file_lines(const char *content, ap_strvec *lines) {
  if (!ap_split_lines(content, lines, NULL)) {
    return 0;
  }
  if (lines->count > 0 && ap_streq(lines->items[lines->count - 1], "")) {
    free(lines->items[lines->count - 1]);
    lines->count--;
  }
  return 1;
}

static int ap_compute_updated_content(
    const char *path,
    const char *source_content,
    const ap_update_chunk *chunks,
    size_t chunk_count,
    char **new_content_out,
    char **err_out) {
  ap_strvec lines;
  ap_replacement_vec repls;
  size_t line_index = 0;
  size_t c;

  memset(&lines, 0, sizeof(lines));
  memset(&repls, 0, sizeof(repls));
  *new_content_out = NULL;
  *err_out = NULL;

  if (!ap_split_file_lines(source_content, &lines)) {
    *err_out = ap_strdup("out of memory parsing file contents");
    return 0;
  }

  for (c = 0; c < chunk_count; c++) {
    const ap_update_chunk *chunk = &chunks[c];

    if (chunk->change_context != NULL) {
      ap_strvec ctx;
      size_t found;
      int ok;

      memset(&ctx, 0, sizeof(ctx));
      if (!ap_strvec_push(&ctx, chunk->change_context)) {
        ap_strvec_free(&ctx);
        *err_out = ap_strdup("out of memory applying patch");
        ap_strvec_free(&lines);
        ap_replacements_free(&repls);
        return 0;
      }

      ok = ap_seek_sequence(&lines, &ctx, 1, line_index, 0, &found);
      ap_strvec_free(&ctx);

      if (!ok) {
        *err_out = ap_format("Failed to find context '%s' in %s", chunk->change_context, path);
        ap_strvec_free(&lines);
        ap_replacements_free(&repls);
        return 0;
      }

      line_index = found + 1;
    }

    if (chunk->old_lines.count == 0) {
      if (!ap_replacements_push(&repls, lines.count, 0, &chunk->new_lines, 0, chunk->new_lines.count)) {
        *err_out = ap_strdup("out of memory applying patch");
        ap_strvec_free(&lines);
        ap_replacements_free(&repls);
        return 0;
      }
      continue;
    }

    {
      size_t found = 0;
      size_t pattern_len = chunk->old_lines.count;
      size_t new_take = chunk->new_lines.count;
      size_t new_from = 0;
      int ok = ap_seek_sequence(&lines, &chunk->old_lines, pattern_len, line_index, chunk->is_end_of_file, &found);

      if (!ok && pattern_len > 0 && ap_streq(chunk->old_lines.items[pattern_len - 1], "")) {
        pattern_len -= 1;
        if (new_take > 0 && ap_streq(chunk->new_lines.items[new_take - 1], "")) {
          new_take -= 1;
        }
        ok = ap_seek_sequence(&lines, &chunk->old_lines, pattern_len, line_index, chunk->is_end_of_file, &found);
      }

      if (!ok) {
        size_t i;
        char *expected = NULL;
        size_t len = 0;
        size_t cap = 0;

        for (i = 0; i < chunk->old_lines.count; i++) {
          if (i > 0 && !ap_append_n(&expected, &len, &cap, "\n", 1)) {
            free(expected);
            expected = NULL;
            break;
          }
          if (!ap_append(&expected, &len, &cap, chunk->old_lines.items[i])) {
            free(expected);
            expected = NULL;
            break;
          }
        }
        *err_out = ap_format(
            "Failed to find expected lines in %s:\n%s",
            path,
            expected == NULL ? "" : expected);
        free(expected);
        ap_strvec_free(&lines);
        ap_replacements_free(&repls);
        return 0;
      }

      if (!ap_replacements_push(&repls, found, pattern_len, &chunk->new_lines, new_from, new_take)) {
        *err_out = ap_strdup("out of memory applying patch");
        ap_strvec_free(&lines);
        ap_replacements_free(&repls);
        return 0;
      }

      line_index = found + pattern_len;
    }
  }

  if (repls.count > 1) {
    qsort(repls.items, repls.count, sizeof(ap_replacement), ap_compare_replacement);
  }

  if (!ap_apply_replacements(&lines, &repls, err_out)) {
    ap_strvec_free(&lines);
    ap_replacements_free(&repls);
    return 0;
  }

  if (lines.count == 0 || !ap_streq(lines.items[lines.count - 1], "")) {
    if (!ap_strvec_push(&lines, "")) {
      *err_out = ap_strdup("out of memory finalizing patch");
      ap_strvec_free(&lines);
      ap_replacements_free(&repls);
      return 0;
    }
  }

  {
    char *out = NULL;
    size_t i;
    size_t len = 0;
    size_t cap = 0;

    for (i = 0; i < lines.count; i++) {
      if (i > 0 && !ap_append_n(&out, &len, &cap, "\n", 1)) {
        free(out);
        out = NULL;
        break;
      }
      if (!ap_append(&out, &len, &cap, lines.items[i])) {
        free(out);
        out = NULL;
        break;
      }
    }

    if (out == NULL) {
      *err_out = ap_strdup("out of memory finalizing patch");
      ap_strvec_free(&lines);
      ap_replacements_free(&repls);
      return 0;
    }
    *new_content_out = out;
  }

  ap_strvec_free(&lines);
  ap_replacements_free(&repls);
  return 1;
}

static int ap_capture_preimage(const char *path, ap_undovec *undo, char **err_out) {
  struct stat st;

  *err_out = NULL;

  if (!ap_stat_file(path, &st)) {
    if (errno == ENOENT) {
      if (!ap_undovec_push(undo, AP_UNDO_REMOVE, path, NULL)) {
        *err_out = ap_strdup("out of memory capturing rollback state");
        return 0;
      }
      return 1;
    }
    *err_out = ap_format("Failed to inspect %s: %s", path, strerror(errno));
    return 0;
  }

  if (!S_ISREG(st.st_mode)) {
    *err_out = ap_format("Failed to modify %s: Not a regular file", path);
    return 0;
  }

  {
    char *content = NULL;
    if (!ap_read_file(path, &content, err_out)) {
      return 0;
    }
    if (!ap_undovec_push(undo, AP_UNDO_RESTORE, path, content)) {
      free(content);
      *err_out = ap_strdup("out of memory capturing rollback state");
      return 0;
    }
    free(content);
  }

  return 1;
}

static void ap_rollback(const ap_undovec *undo) {
  size_t i;
  for (i = undo->count; i > 0; i--) {
    const ap_undo *step = &undo->items[i - 1];
    if (step->kind == AP_UNDO_RESTORE) {
      char *ignored = NULL;
      ap_write_file_atomic(step->path, step->content == NULL ? "" : step->content, &ignored);
      free(ignored);
    } else if (step->kind == AP_UNDO_REMOVE) {
      unlink(step->path);
    }
  }
}

static int ap_load_effective_content(
    const ap_virtualfs *vfs,
    const char *path,
    char **content_out,
    char **err_out) {
  int known = 0;
  int exists = 0;
  const char *content = NULL;

  *content_out = NULL;
  *err_out = NULL;

  if (!ap_virtualfs_get(vfs, path, &known, &exists, &content)) {
    *err_out = ap_strdup("out of memory looking up virtual file");
    return 0;
  }

  if (known) {
    if (!exists) {
      *err_out = ap_format("Failed to read %s: No such file or directory", path);
      return 0;
    }
    *content_out = ap_strdup(content == NULL ? "" : content);
    if (*content_out == NULL) {
      *err_out = ap_strdup("out of memory loading virtual file");
      return 0;
    }
    return 1;
  }

  return ap_read_file(path, content_out, err_out);
}

static int ap_build_plan(
    const char *cwd,
    const ap_hunkvec *hunks,
    ap_planvec *plan,
    char **err_out) {
  size_t i;
  ap_virtualfs vfs;

  memset(&vfs, 0, sizeof(vfs));
  *err_out = NULL;

  if (hunks->count == 0) {
    *err_out = ap_strdup("No files were modified.");
    return 0;
  }

  for (i = 0; i < hunks->count; i++) {
    const ap_hunk *h = &hunks->items[i];
    ap_plan_op op;
    char *src_abs = NULL;

    memset(&op, 0, sizeof(op));

    src_abs = ap_resolve_path(cwd, h->path);
    if (src_abs == NULL) {
      *err_out = ap_strdup("out of memory resolving patch paths");
      ap_virtualfs_free(&vfs);
      return 0;
    }

    if (h->type == AP_HUNK_ADD) {
      char *content = NULL;

      if (!ap_join_lines_with_trailing_newline(&h->add_lines, &content)) {
        free(src_abs);
        *err_out = ap_strdup("out of memory preparing add file content");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      op.kind = AP_PLAN_ADD;
      op.src_abs = src_abs;
      op.dest_abs = ap_strdup(src_abs);
      op.display_path = ap_strdup(h->path);
      op.new_content = content;

      if (op.dest_abs == NULL || op.display_path == NULL) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      if (!ap_virtualfs_set(&vfs, src_abs, 1, content)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      if (!ap_planvec_push_owned(plan, &op)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }
      continue;
    }

    if (h->type == AP_HUNK_DELETE) {
      char *existing = NULL;

      if (!ap_load_effective_content(&vfs, src_abs, &existing, err_out)) {
        free(src_abs);
        ap_virtualfs_free(&vfs);
        return 0;
      }
      free(existing);

      op.kind = AP_PLAN_DELETE;
      op.src_abs = src_abs;
      op.display_path = ap_strdup(h->path);
      if (op.display_path == NULL) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      if (!ap_virtualfs_set(&vfs, src_abs, 0, NULL)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      if (!ap_planvec_push_owned(plan, &op)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }
      continue;
    }

    if (h->type == AP_HUNK_UPDATE) {
      char *source_content = NULL;
      char *new_content = NULL;
      char *dest_abs = NULL;
      char *display = NULL;

      if (!ap_load_effective_content(&vfs, src_abs, &source_content, err_out)) {
        free(src_abs);
        ap_virtualfs_free(&vfs);
        return 0;
      }

      if (!ap_compute_updated_content(src_abs, source_content, h->chunks, h->chunk_count, &new_content, err_out)) {
        free(source_content);
        free(src_abs);
        ap_virtualfs_free(&vfs);
        return 0;
      }
      free(source_content);

      if (h->move_to != NULL) {
        dest_abs = ap_resolve_path(cwd, h->move_to);
      } else {
        dest_abs = ap_strdup(src_abs);
      }
      if (dest_abs == NULL) {
        free(new_content);
        free(src_abs);
        *err_out = ap_strdup("out of memory resolving move path");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      display = ap_strdup(h->move_to != NULL ? h->move_to : h->path);
      if (display == NULL) {
        free(dest_abs);
        free(new_content);
        free(src_abs);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }

      op.kind = AP_PLAN_UPDATE;
      op.src_abs = src_abs;
      op.dest_abs = dest_abs;
      op.display_path = display;
      op.new_content = new_content;

      if (!ap_virtualfs_set(&vfs, dest_abs, 1, new_content)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }
      if (!ap_streq(src_abs, dest_abs)) {
        if (!ap_virtualfs_set(&vfs, src_abs, 0, NULL)) {
          ap_plan_op_free(&op);
          *err_out = ap_strdup("out of memory preparing patch plan");
          ap_virtualfs_free(&vfs);
          return 0;
        }
      }

      if (!ap_planvec_push_owned(plan, &op)) {
        ap_plan_op_free(&op);
        *err_out = ap_strdup("out of memory preparing patch plan");
        ap_virtualfs_free(&vfs);
        return 0;
      }
      continue;
    }

    free(src_abs);
    *err_out = ap_strdup("unsupported patch hunk");
    ap_virtualfs_free(&vfs);
    return 0;
  }

  ap_virtualfs_free(&vfs);
  return 1;
}

static int ap_execute_plan(const ap_planvec *plan, char **err_out) {
  ap_undovec undo;
  size_t i;

  memset(&undo, 0, sizeof(undo));
  *err_out = NULL;

  for (i = 0; i < plan->count; i++) {
    const ap_plan_op *op = &plan->items[i];

    if (op->kind == AP_PLAN_ADD) {
      if (!ap_capture_preimage(op->dest_abs, &undo, err_out)) {
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }
      if (!ap_write_file_atomic(op->dest_abs, op->new_content, err_out)) {
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }
      continue;
    }

    if (op->kind == AP_PLAN_DELETE) {
      if (!ap_capture_preimage(op->src_abs, &undo, err_out)) {
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }
      if (unlink(op->src_abs) != 0) {
        *err_out = ap_format("Failed to delete file %s: %s", op->src_abs, strerror(errno));
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }
      continue;
    }

    if (op->kind == AP_PLAN_UPDATE) {
      if (!ap_capture_preimage(op->dest_abs, &undo, err_out)) {
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }
      if (!ap_write_file_atomic(op->dest_abs, op->new_content, err_out)) {
        ap_rollback(&undo);
        ap_undovec_free(&undo);
        return 0;
      }

      if (!ap_streq(op->src_abs, op->dest_abs)) {
        if (!ap_capture_preimage(op->src_abs, &undo, err_out)) {
          ap_rollback(&undo);
          ap_undovec_free(&undo);
          return 0;
        }
        if (unlink(op->src_abs) != 0) {
          *err_out = ap_format("Failed to remove original %s: %s", op->src_abs, strerror(errno));
          ap_rollback(&undo);
          ap_undovec_free(&undo);
          return 0;
        }
      }
      continue;
    }
  }

  ap_undovec_free(&undo);
  return 1;
}

static int ap_build_summary(const ap_planvec *plan, char **summary_out) {
  size_t i;
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  if (!ap_append(&out, &len, &cap, "Success. Updated the following files:\n")) {
    free(out);
    return 0;
  }

  for (i = 0; i < plan->count; i++) {
    const ap_plan_op *op = &plan->items[i];
    const char *prefix = "";
    if (op->kind == AP_PLAN_ADD) {
      prefix = "A ";
    } else if (op->kind == AP_PLAN_DELETE) {
      prefix = "D ";
    } else {
      prefix = "M ";
    }

    if (!ap_append(&out, &len, &cap, prefix) ||
        !ap_append(&out, &len, &cap, op->display_path == NULL ? "" : op->display_path) ||
        !ap_append_n(&out, &len, &cap, "\n", 1)) {
      free(out);
      return 0;
    }
  }

  *summary_out = out;
  return 1;
}

int chi_apply_patch(const char *cwd, const char *patch, char **summary_out, char **error_out) {
  ap_hunkvec hunks;
  ap_planvec plan;
  char *err = NULL;
  int ok = 0;

  memset(&hunks, 0, sizeof(hunks));
  memset(&plan, 0, sizeof(plan));

  *summary_out = NULL;
  *error_out = NULL;

  if (ap_is_blank(cwd)) {
    cwd = ".";
  }
  if (ap_is_blank(patch)) {
    *error_out = ap_strdup("Invalid patch: patch input is empty");
    return 0;
  }

  if (!ap_parse_patch(patch, &hunks, &err)) {
    *error_out = err == NULL ? ap_strdup("Invalid patch") : err;
    ap_hunkvec_free(&hunks);
    return 0;
  }

  if (!ap_build_plan(cwd, &hunks, &plan, &err)) {
    *error_out = err == NULL ? ap_strdup("Failed to build patch plan") : err;
    ap_hunkvec_free(&hunks);
    ap_planvec_free(&plan);
    return 0;
  }

  if (!ap_execute_plan(&plan, &err)) {
    *error_out = err == NULL ? ap_strdup("Failed to apply patch") : err;
    ap_hunkvec_free(&hunks);
    ap_planvec_free(&plan);
    return 0;
  }

  if (!ap_build_summary(&plan, summary_out)) {
    *error_out = ap_strdup("out of memory building apply_patch summary");
    ap_hunkvec_free(&hunks);
    ap_planvec_free(&plan);
    return 0;
  }

  ok = 1;

  ap_hunkvec_free(&hunks);
  ap_planvec_free(&plan);
  return ok;
}
