package main

import (
	"bytes"
	"testing"
)

// The command checks the committed ledger and lints the repository's workflows, and exits 2
// on misuse.
func TestRun(t *testing.T) {
	for _, c := range []struct {
		args []string
		want int
	}{
		{[]string{"check", "../../../catalog/ledger"}, 0},
		{[]string{"lint-workflows", "../../../.github/workflows"}, 0},
		{[]string{"check", "../../../catalog"}, 1},
		{[]string{"bogus"}, 2},
		{[]string{"check"}, 2},
		{nil, 2},
	} {
		var out, errb bytes.Buffer
		if got := run(c.args, &out, &errb); got != c.want {
			t.Errorf("ledger %v = %d, want %d: %s", c.args, got, c.want, errb.String())
		}
	}
}
