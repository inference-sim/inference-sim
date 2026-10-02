#!/usr/bin/env bash
#
# scrub_symlinks.sh <dir> — remove symlinks under <dir> whose target escapes it,
# then prove none remain (fail-closed). A checked-out PR can plant a symlink like
# `pr/x -> /etc` or `-> ../../trusted`; a reviewer's Read tool following it would
# escape the sandbox. We delete escapers before any reviewer reads the tree.
#
# NUL-separated end to end so a newline in a path cannot split one entry into two.
# Exits non-zero if the fail-closed re-scan still finds an escaper — refusing to
# continue is correct, because a surviving escaper is exactly what this guards.
set -euo pipefail

DIR="${1:?usage: scrub_symlinks.sh <dir>}"
REAL=$(readlink -f "$DIR")

# Print (NUL-separated) every symlink under DIR whose resolved target is NOT
# inside REAL. An unresolvable link resolves to empty and is treated as escaping.
scan_escaping() {
  find "$DIR" -type l -print0 | while IFS= read -r -d '' link; do
    # `printf X` then strip it: $(...) eats trailing newlines, so a target that
    # legitimately ends in newline would otherwise compare wrong.
    target=$(readlink -f -- "$link" 2>/dev/null; printf X)
    target=${target%X}
    target=${target%$'\n'}
    case "$target" in
      "$REAL"/* | "$REAL") ;;                 # inside the tree — keep
      *) printf '%s\0' "$link" ;;             # escapes — flag
    esac
  done
}

# Delete the escapers found in the first pass.
while IFS= read -r -d '' link; do
  echo "removing escaping symlink: $link" >&2
  rm -f -- "$link"
done < <(scan_escaping)

# Fail-closed: prove none remain rather than trusting the deletion loop.
remaining=$(scan_escaping | tr -cd '\0' | wc -c | tr -d ' ')
if [[ "$remaining" -ne 0 ]]; then
  echo "::error::$remaining escaping symlink(s) still present under $DIR; refusing to continue" >&2
  exit 1
fi
