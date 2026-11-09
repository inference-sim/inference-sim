#!/usr/bin/env bash
# best_effort MESSAGE...: the one place a workflow step lets a failing command pass (X22).
#
# Use it as `cmd || best_effort "what did not happen"`. It must run directly after the
# failing command, because it reads that command's exit status from $?. It never hides the
# failure: it prints a workflow warning and adds a line to the job summary with the exit
# status and the message, then returns 0 so the step goes on. It is for the delivery loop's
# courtesy writes (a label, a comment) and reads that have a stated fallback, where stopping
# the loop would be worse than the missed write. A command whose result decides the step's
# outcome does not belong here; handle its status where it is read.
best_effort() {
  local rc=$?
  local msg="$*"
  echo "::warning::best effort failed (exit ${rc}): ${msg}"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf -- '- best effort failed (exit %s): %s\n' "${rc}" "${msg}" >> "${GITHUB_STEP_SUMMARY}"
  fi
  return 0
}
