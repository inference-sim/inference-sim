#!/usr/bin/env bash
#
# archon-plan-resolve.sh — resolve the archon plan a PR declares.
#
# Usage: scripts/archon-plan-resolve.sh <base-ref> <head-ref> <decl-file> <out-file>
#
# <decl-file> holds candidate declaration text: the PR body, then the body of each issue
# the PR closes, in that order. The first `archon-plan: <path>` line wins, so a PR's own
# declaration outranks one in a third-party issue it happens to close.
#
# The declared path is UNTRUSTED — anyone can write a PR or issue body — so it is validated
# before any use and dereferenced only with git plumbing against an explicit commit. Never
# through the filesystem: on an issue_comment runner the working tree holds the default
# branch, not the PR, so a filesystem read would find the wrong file and would let a
# traversal escape the repository.
#
# Prints one key=value per line on stdout:
#   status=none
#   status=resolved  plan_path=… plan_source=base|head plan_commit=…   (<out-file> written)
#   status=error     plan_path=… message=…
# Exits 0 in all three cases; exit 2 only on a usage error.
#
# `set -e` is deliberately off: a failure is a reported status, and non-zero git exits are
# expected control flow. Every state-changing command is therefore checked explicitly.

set -uo pipefail

readonly MAX_PLAN_BYTES=1048576   # 1 MiB; a compiled archon plan is orders of magnitude smaller
readonly MAX_PATH_CHARS=256       # a longer path could crowd the verdict out of the PR comment

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <base-ref> <head-ref> <decl-file> <out-file>" >&2
  exit 2
fi

BASE_REF="$1"
HEAD_REF="$2"
DECL_FILE="$3"
OUT_FILE="$4"

# Reduce an untrusted path to the allowlist, so it is safe to embed in a comment posted to
# GitHub. `:` becomes `?`, so a rejected path cannot smuggle a ::workflow:: command either.
neuter() { printf '%s' "$1" | tr -c 'A-Za-z0-9._/-' '?'; }

fail() {
  printf 'status=error\nplan_path=%s\nmessage=%s\n' "$(neuter "$1")" "$2"
  exit 0
}

# git ls-tree takes a pathspec relative to the current directory, so anchor to the root.
# Absolutise the caller's file arguments first, since they may be relative to its own cwd.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
DECL_FILE=$(abspath "$DECL_FILE")
OUT_FILE=$(abspath "$OUT_FILE")

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) \
  || fail "" "not inside a git repository; plan detection could not run"
cd "$repo_root" || fail "" "could not enter the repository root"

# --- detect -----------------------------------------------------------------

# A declaration file that was never written, or cannot be read, is an error rather than "no
# declaration": treating it as absent would drop the plan gate exactly when the collecting
# step misbehaved. Readability is checked too, since `tr` failing below is swallowed by the
# `|| true` that lets grep report "no match".
[[ -f "$DECL_FILE" ]] || fail "" "declaration file was not produced; plan detection could not run"
[[ -r "$DECL_FILE" ]] || fail "" "declaration file is not readable; plan detection could not run"

# GitHub API bodies use CRLF; an unstripped \r survives into the path and fails validation
# on every real PR while passing every LF-only test.
# The anchor admits leading whitespace and markdown list, quote, bold, and backtick markers
# but not prose, so a sentence mentioning archon-plan: in passing is not a declaration.
RAW=$(tr -d '\r' < "$DECL_FILE" | grep -m1 -E '^[^A-Za-z0-9]*archon-plan:' || true)

if [[ -z "$RAW" ]]; then
  echo "status=none"
  exit 0
fi

REST=${RAW#*archon-plan:}            # everything after the first colon
REST=${REST//[\`*]/}                 # tolerate `path` and **archon-plan:** path
read -r PLAN_PATH _ <<< "$REST"      # first whitespace-delimited token ("" if none)

# --- validate ---------------------------------------------------------------

[[ -n "$PLAN_PATH" ]] || fail "" "an archon-plan: line was declared with no path"

if (( ${#PLAN_PATH} > MAX_PATH_CHARS )); then
  fail "${PLAN_PATH:0:64}" "declared plan path is longer than ${MAX_PATH_CHARS} characters"
fi
if [[ "$PLAN_PATH" != *.json ]]; then
  fail "$PLAN_PATH" "declared plan path is not a .json file: $(neuter "$PLAN_PATH")"
fi
if [[ "$PLAN_PATH" == /* || "$PLAN_PATH" == -* || "$PLAN_PATH" == *..* \
      || ! "$PLAN_PATH" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  fail "$PLAN_PATH" "declared plan path is not a safe repository-relative path: $(neuter "$PLAN_PATH")"
fi

# --- extract: base branch tip first, then the PR head -----------------------

TMP_FILE=$(mktemp "${OUT_FILE}.XXXXXX") || fail "$PLAN_PATH" "could not create a temporary file"
trap 'rm -f "$TMP_FILE"' EXIT INT TERM

for candidate in "base:$BASE_REF" "head:$HEAD_REF"; do
  source_name=${candidate%%:*}
  source_ref=${candidate#*:}
  [[ -n "$source_ref" ]] || continue

  # An unknown commit and an absent path BOTH make git ls-tree print nothing, so check
  # reachability separately. Conflating them would let an unfetched base tip degrade
  # silently to the PR's own head copy — the thing preferring base exists to prevent.
  git rev-parse --verify --quiet "${source_ref}^{commit}" >/dev/null \
    || fail "$PLAN_PATH" "the $source_name commit ($source_ref) is not available locally"

  # `git ls-tree -l` prints: <mode> SP <type> SP <sha> SP <size> TAB <path>
  # Its exit status is checked so that empty output means only one thing: the path is absent
  # at this commit. A failure here (unreadable object store, damaged tree) would otherwise
  # also read as absent and fall through to the head copy.
  meta=$(git ls-tree -l "$source_ref" -- "$PLAN_PATH") \
    || fail "$PLAN_PATH" "could not read the tree at $source_name ($source_ref)"
  [[ -n "$meta" ]] || continue

  read -r obj_mode obj_type obj_sha obj_size _ <<< "$meta"

  # Present but unusable is TERMINAL, not a fall-through: letting a broken plan on the base
  # branch hand grading to the PR's own copy would undo the reason base is preferred.
  # Order matters — a tree's size field is `-`, so the type gate must precede the arithmetic.
  [[ "$obj_type" == "blob" ]] \
    || fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") is a $obj_type, not a file, at $source_name ($source_ref)"
  [[ "$obj_mode" == "100644" || "$obj_mode" == "100755" ]] \
    || fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") is not a regular file (mode $obj_mode) at $source_name ($source_ref)"
  (( obj_size > 0 )) \
    || fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") is empty at $source_name ($source_ref)"
  (( obj_size <= MAX_PLAN_BYTES )) \
    || fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") is $obj_size bytes at $source_name ($source_ref), over the ${MAX_PLAN_BYTES}-byte limit"

  # By object sha, not by `ref:path`: the bytes written are then provably the object whose
  # size was just checked, with no second path resolution in between.
  git cat-file blob "$obj_sha" > "$TMP_FILE" \
    || fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") could not be read at $source_name ($source_ref)"

  plan_commit=$(git rev-parse --verify "$source_ref") \
    || fail "$PLAN_PATH" "could not resolve the $source_name commit ($source_ref)"

  mv "$TMP_FILE" "$OUT_FILE" || fail "$PLAN_PATH" "could not write the extracted plan"

  printf 'status=resolved\nplan_path=%s\nplan_source=%s\nplan_commit=%s\n' \
    "$PLAN_PATH" "$source_name" "$plan_commit"
  exit 0
done

fail "$PLAN_PATH" "declared plan $(neuter "$PLAN_PATH") is not committed at the base branch tip ($BASE_REF) or the PR head ($HEAD_REF)"
