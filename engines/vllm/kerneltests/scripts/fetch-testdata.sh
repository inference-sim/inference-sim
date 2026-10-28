#!/usr/bin/env bash
# Populate testdata/catalog and testdata/registry from the commits testdata/upstream.lock
# pins, so the upstream catalog and registry are read from their own repositories rather
# than redistributed in this one.
#
# Idempotent: a destination whose .upstream-commit stamp names the locked commit, and whose
# every file still matches the SHA-256 recorded in its .upstream-manifest, is left alone;
# anything else is replaced. The manifest is what makes that safe: a stamp alone survives
# its files being deleted -- as git does when a pull crosses the commit that stopped
# tracking them -- and would then vouch for an incomplete copy. Every failure is fatal -- a partial copy would
# let tests run against the wrong inputs -- and the fetched HEAD is checked against the
# lock before anything is copied.
#
# Usage: scripts/fetch-testdata.sh            (from anywhere inside the checkout)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/testdata/upstream.lock"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

while read -r dest repo tag commit paths; do
	case "$dest" in '' | '#'*) continue ;; esac
	out="$root/testdata/$dest"
	if [[ -f "$out/.upstream-commit" && "$(cat "$out/.upstream-commit")" == "$commit" &&
		-f "$out/.upstream-manifest" ]] &&
		(cd "$out" && shasum -a 256 -c --quiet .upstream-manifest >/dev/null 2>&1); then
		echo "testdata/$dest: already at $tag ($commit)"
		continue
	fi
	echo "testdata/$dest: fetching $repo $tag ($commit)"
	src="$work/$dest"
	git init -q "$src"
	git -C "$src" fetch -q --depth 1 "$repo" "$commit"
	git -C "$src" checkout -q FETCH_HEAD
	got="$(git -C "$src" rev-parse HEAD)"
	if [[ "$got" != "$commit" ]]; then
		echo "testdata/$dest: fetched $got, but the lock pins $commit" >&2
		exit 1
	fi
	rm -rf "$out.tmp"
	mkdir -p "$out.tmp"
	(
		cd "$src"
		shopt -s nullglob
		n=0
		# Unquoted on purpose: each entry is a glob.
		for p in $paths; do
			for f in $p; do
				mkdir -p "$out.tmp/$(dirname "$f")"
				cp "$f" "$out.tmp/$f"
				n=$((n + 1))
			done
		done
		if [[ $n -eq 0 ]]; then
			echo "testdata/$dest: no file in $repo matched: $paths" >&2
			exit 1
		fi
		echo "testdata/$dest: copied $n files"
	)
	(cd "$out.tmp" && find . -type f ! -name '.upstream-*' | LC_ALL=C sort |
		sed 's|^\./||' | xargs shasum -a 256 >.upstream-manifest)
	echo "$commit" >"$out.tmp/.upstream-commit"
	rm -rf "$out"
	mv "$out.tmp" "$out"
done <"$lock"
