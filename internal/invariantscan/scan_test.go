package invariantscan

import "testing"

// TestFindConservationSums covers each evasion shape the sums this guard replaced
// actually used, plus the expressions that must NOT be flagged. A guard whose
// detection is untested is decoration: it passes on a clean tree whether or not it
// works.
func TestFindConservationSums(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want bool
	}{
		{
			name: "plain inline sum",
			src: `package p
func f() {
	_ = m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable
}`,
			want: true,
		},
		{
			name: "wrapped across lines by gofmt",
			src: `package p
func f() {
	_ = m.CompletedRequests + m.StillQueued +
		m.StillRunning + m.DroppedUnservable
}`,
			want: true,
		},
		{
			// The shape sim/cluster/cluster_tier_test.go used, one of the three sites
			// that omitted a bucket.
			name: "laundered through locals",
			src: `package p
func f() {
	completed := agg.CompletedRequests
	queued := agg.StillQueued
	running := agg.StillRunning
	_ = completed + queued + running + cs.RoutingRejections()
}`,
			want: true,
		},
		{
			name: "accumulated with +=",
			src: `package p
func f() {
	total := 0
	total += m.CompletedRequests
	total += m.StillQueued
	total += m.StillRunning
	_ = total
}`,
			want: true,
		},
		{
			name: "moved to the other side of the comparison",
			src: `package p
func f() {
	_ = m.CompletedRequests+m.StillQueued != injected-m.DroppedUnservable-cs.RoutingRejections()
}`,
			want: true,
		},
		{
			name: "reordered operands, accessors and fields mixed",
			src: `package p
func f() {
	_ = cs.GatewayExpired() + m.StillRunning + cs.RoutingRejections() + m.CompletedRequests
}`,
			want: true,
		},
		{
			// A short-horizon fixture where nothing completes can still express a real
			// ledger. This is why the rule is "CompletedRequests OR two or more
			// instance-side buckets" rather than CompletedRequests alone.
			name: "ledger without CompletedRequests",
			src: `package p
func f() {
	_ = m.StillQueued + m.StillRunning + cs.GatewayQueueDepth() + cs.GatewayQueueShed()
}`,
			want: true,
		},
		{
			name: "two buckets is not a ledger",
			src: `package p
func f() {
	_ = m.CompletedRequests + m.TimedOutRequests
}`,
			want: false,
		},
		{
			// INV-5's non-vacuity floors: several gateway counters subtracted from a
			// request count. Superficially similar, not a conservation ledger. Flagging
			// these would earn the guard an exemption list, and an exemption list is how
			// a guard stops meaning anything.
			name: "non-vacuity floor over gateway counters",
			src: `package p
func f() {
	shed := cs.GatewayQueueShed()
	_ = len(requests) - shed
}`,
			want: false,
		},
		{
			// progress_hook_test.go's real shape: the per-tier shed breakdown
			// reconciled against the shedding counters. Four cluster-only buckets, no
			// instance-side ones, so not a conservation ledger.
			name: "shed-accounting identity over cluster counters only",
			src: `package p
func f() {
	_ = sum != snap.RejectedRequests + snap.GatewayQueueShed + snap.GatewayEvicted + snap.GatewayExpired
}`,
			want: false,
		},
		{
			// The INV-5 floors' real shape: several gateway counters subtracted from a
			// request count, via locals.
			name: "non-vacuity floor subtracting four gateway counters",
			src: `package p
func f() {
	shed := cs.GatewayQueueShed()
	rejected := cs.GatewayQueueRejected()
	depth := cs.GatewayQueueDepth()
	expired := cs.GatewayExpired()
	_ = len(requests) - shed - rejected - depth - expired
}`,
			want: false,
		},
		{
			// Alias scoping: `total` accumulates a real ledger in one function while a
			// different function reuses the name for something unrelated. A file-wide
			// alias pass would let the second function's plain assignment wipe the
			// first's accumulated set, and the ledger would escape.
			name: "accumulator name reused in another function",
			src: `package p
func a() {
	total := 0
	total += m.CompletedRequests
	total += m.StillQueued
	total += m.StillRunning
	_ = total
}
func b() {
	total := somethingElse()
	_ = total
}`,
			want: true,
		},
		{
			// The reverse direction: an unrelated accumulator must not inherit buckets
			// from a same-named variable in another function.
			name: "unrelated accumulator does not inherit another function's buckets",
			src: `package p
func a() {
	total := 0
	total += m.CompletedRequests
	total += m.StillQueued
	total += m.StillRunning
	_ = total
}
func b() {
	total := 0
	total += x
	if total != 3 {
		panic("no")
	}
}`,
			// Flagged once, for function a only.
			want: true,
		},
		{
			name: "unrelated arithmetic",
			src: `package p
func f() {
	_ = a + b + c + d
}`,
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			findings, err := FindConservationSums("fixture.go", tc.src, DefaultThreshold)
			if err != nil {
				t.Fatalf("FindConservationSums: %v", err)
			}
			if got := len(findings) > 0; got != tc.want {
				t.Errorf("flagged = %v, want %v (findings: %v)", got, tc.want, findings)
			}
		})
	}
}

// TestFindConservationSums_AliasScopeIsPerFunction pins the scoping directly: the
// ledger in a() must be found exactly once, and b()'s reuse of the same variable name
// must neither suppress it nor add a second finding.
func TestFindConservationSums_AliasScopeIsPerFunction(t *testing.T) {
	src := `package p
func a() {
	total := 0
	total += m.CompletedRequests
	total += m.StillQueued
	total += m.StillRunning
	_ = total
}
func b() {
	total := 0
	total += x
	_ = total
}`
	findings, err := FindConservationSums("fixture.go", src, DefaultThreshold)
	if err != nil {
		t.Fatalf("FindConservationSums: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("got %d findings, want exactly 1 (a's ledger): %v", len(findings), findings)
	}
}

// TestFindConservationSums_ParseError verifies unparseable input is an error rather
// than a silent empty result — a caller that ignored it would report a clean scan.
func TestFindConservationSums_ParseError(t *testing.T) {
	if _, err := FindConservationSums("bad.go", "package p\nfunc f( {", DefaultThreshold); err == nil {
		t.Error("expected a parse error, got nil — a broken file would read as clean")
	}
}

// TestBucketAccessorsReturnCopies verifies a caller cannot mutate the package's bucket
// lists through the accessors, which would silently change what every guard detects.
func TestBucketAccessorsReturnCopies(t *testing.T) {
	got := InstanceBuckets()
	got[0] = "mutated"
	if again := InstanceBuckets(); again[0] != "CompletedRequests" {
		t.Errorf("InstanceBuckets() exposes package state: second call returned %q", again[0])
	}
}

// TestBucketListsCoverTheEquation pins the bucket lists against INV-1's stated term
// count: twelve terminal buckets, of which seven are cluster-only, plus
// RejectedRequests which belongs to the full-pipeline clause.
func TestBucketListsCoverTheEquation(t *testing.T) {
	if got := len(InstanceBuckets()); got != 5 {
		t.Errorf("InstanceBuckets has %d entries, want 5 (INV-1's single-instance specialisation)", got)
	}
	if got := len(ClusterOnlyBuckets()); got != 8 {
		t.Errorf("ClusterOnlyBuckets has %d entries, want 8 (seven cluster-only buckets plus RejectedRequests)", got)
	}
	if got := len(InstanceBuckets()) + len(ClusterOnlyBuckets()) - 1; got != 12 {
		t.Errorf("bucket lists cover %d of INV-1's twelve terms", got)
	}
}
