#!/usr/bin/env bash
# Run a mike command that pushes to gh-pages, retrying if the push loses a race.
#
#   tools/docs/mike-push.sh deploy --title "dev (main)" dev
#   tools/docs/mike-push.sh set-default latest
#
# The dev and release deploys run in separate concurrency groups, so both can push at
# once. A push based on a stale gh-pages is rejected as a non-fast-forward rather than
# overwriting the other's commit; on rejection this refetches gh-pages and runs the
# command again on top of it.
set -euo pipefail

for attempt in 1 2 3 4 5; do
  # The first deploy finds no gh-pages on the remote; mike then creates the branch.
  git fetch --quiet origin "+refs/heads/gh-pages:refs/heads/gh-pages" 2>/dev/null || true
  if mike "$@" --push; then
    exit 0
  fi
  echo "mike $1: push rejected (attempt $attempt); refetching gh-pages and retrying" >&2
  sleep $((attempt * 5))
done
echo "mike $1: giving up after 5 attempts" >&2
exit 1
