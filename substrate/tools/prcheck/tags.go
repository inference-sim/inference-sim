package main

import (
	"fmt"
	"sort"
	"strings"
)

// The tag check (LIF-1): a tag names one commit forever. .github/tags.lock records each
// tag's commit, one "<tag> <40-hex commit>" per line, and the release bot appends a line
// when it tags. The nightly job fails when a recorded tag is missing or points elsewhere, or
// when a tag exists that the lock does not record (a tag made outside the bot and the manual
// procedure, which must be recorded in its release notes, LIF-3).

// CheckTags compares the repository's tags with the lock.
func CheckTags(lock string, tags map[string]string) []string {
	want := map[string]string{}
	var bad []string
	for i, l := range strings.Split(lock, "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		f := strings.Fields(l)
		if len(f) != 2 || !fullHash.MatchString(f[1]) || len(f[1]) != 40 {
			bad = append(bad, fmt.Sprintf("tags.lock:%d: %q is not \"<tag> <40-hex commit>\"", i+1, l))
			continue
		}
		if _, dup := want[f[0]]; dup {
			bad = append(bad, fmt.Sprintf("tags.lock:%d: %s is recorded twice", i+1, f[0]))
		}
		want[f[0]] = f[1]
	}
	for t, c := range want {
		got, ok := tags[t]
		switch {
		case !ok:
			bad = append(bad, fmt.Sprintf("tag %s is recorded at %s but no longer exists; tags are never deleted (LIF-1)", t, c))
		case got != c:
			bad = append(bad, fmt.Sprintf("tag %s points at %s; it was recorded at %s, and tags are never moved (LIF-1)", t, got, c))
		}
	}
	for t, c := range tags {
		if _, ok := want[t]; !ok {
			bad = append(bad, fmt.Sprintf("tag %s at %s is not in tags.lock; only the release bot and the manual procedure tag, and both record it (LIF-1, LIF-3)", t, c))
		}
	}
	sort.Strings(bad)
	return bad
}

func gitTags() (map[string]string, error) {
	out, err := git("for-each-ref", "--format=%(refname:short) %(objectname) %(*objectname)", "refs/tags")
	if err != nil {
		return nil, err
	}
	tags := map[string]string{}
	for _, l := range strings.Split(out, "\n") {
		f := strings.Fields(l)
		switch len(f) {
		case 2:
			tags[f[0]] = f[1]
		case 3:
			tags[f[0]] = f[2] // an annotated tag: the commit it points at
		}
	}
	return tags, nil
}
