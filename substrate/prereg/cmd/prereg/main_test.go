package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestCommands(t *testing.T) {
	t.Chdir("../../..")
	for name, c := range map[string]struct {
		args []string
		code int
		out  string
	}{
		"check":                  {[]string{"check"}, 0, "ok: 3 pre-registrations"},
		"admit a covered read":   {[]string{"admit", "refit-engine-path-vllm", "fit", "vllm-ci-perf", "fit"}, 0, ""},
		"admit an uncovered one": {[]string{"admit", "refit-engine-path-vllm", "fit", "sglang-traces", "fit"}, 1, "not covered"},
		"admit a held read":      {[]string{"admit", "refit-kernel-keyed", "select", "inferencex", "held"}, 1, "not held"},
		"admit needs arguments":  {[]string{"admit", "refit-kernel-keyed"}, 2, "usage"},
		"unknown command":        {[]string{"fit"}, 2, "unknown command"},
		"pins":                   {[]string{"pins", "vllm-ci-perf"}, 0, "vllm-ci-perf:"},
		"pins of an unknown":     {[]string{"pins", "nope"}, 1, "not in the ledger"},
	} {
		t.Run(name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(c.args, &out, &errb); code != c.code {
				t.Fatalf("exit %d, want %d: %s", code, c.code, errb.String())
			}
			if got := out.String() + errb.String(); !strings.Contains(got, c.out) {
				t.Errorf("output %q does not mention %q", got, c.out)
			}
		})
	}
}
