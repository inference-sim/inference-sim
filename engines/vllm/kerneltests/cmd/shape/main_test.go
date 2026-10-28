// Moved from blis-latency-kernel/cmd/shape/main_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package main

import "testing"

func TestContextComesFromTheWorkloadIdentity(t *testing.T) {
	// The snapshot states the workload as input:output. Context during decode is the prompt
	// plus half the mean output, since a request grows across its decode phase.
	cases := []struct {
		workload string
		want     int
	}{
		{"1024:1024", 1024 + 512},
		{"8192:1024", 8192 + 512},
		{"1024:8192", 1024 + 4096},
	}
	for _, tc := range cases {
		if got := contextTokens(sweep{Workload: tc.workload}); got != tc.want {
			t.Errorf("%s gave context %d, expected %d", tc.workload, got, tc.want)
		}
	}
	// An unparseable identity must give zero rather than a plausible-looking number.
	if got := contextTokens(sweep{Workload: "unknown"}); got != 0 {
		t.Errorf("an unparseable workload gave context %d, expected 0", got)
	}
}
