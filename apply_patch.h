#ifndef CHI_APPLY_PATCH_H
#define CHI_APPLY_PATCH_H

/*
 * Apply a Codex-style apply_patch payload in cwd.
 *
 * Returns 1 on success and sets *summary_out to a newly allocated summary.
 * Returns 0 on failure and sets *error_out to a newly allocated error message.
 *
 * Caller owns returned strings and must free them.
 */
int chi_apply_patch(const char *cwd, const char *patch, char **summary_out, char **error_out);

#endif
