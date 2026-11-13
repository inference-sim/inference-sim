package scripts_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The sharded CI jobs split one package's tests over parallel runners (scripts/ci-shard.sh) and a
// closing job proves the shards ran every test exactly once (scripts/ci-shard-verify.sh). These
// tests run both scripts for real, on a small module made here, because only executing them shows
// whether the shards partition the tests and whether the proof fails when they do not.

// shardFixture writes a module with the named top-level tests (and one subtest each) and returns
// its directory. A test whose name is in failing calls t.Fatal.
func shardFixture(t *testing.T, names []string, failing ...string) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module shardfixture\n\ngo 1.21\n")
	write("fixture.go", "package shardfixture\n")
	var b strings.Builder
	b.WriteString("package shardfixture\n\nimport \"testing\"\n\n")
	for _, n := range names {
		fail := ""
		for _, f := range failing {
			if f == n {
				fail = "\t\tt.Fatal(\"fails on purpose\")\n"
			}
		}
		fmt.Fprintf(&b, "func %s(t *testing.T) {\n\tt.Run(\"sub\", func(t *testing.T) {\n%s\t})\n}\n\n", n, fail)
	}
	write("fixture_test.go", b.String())
	return dir
}

func fixtureNames(n int) []string {
	var out []string
	for i := 0; i < n; i++ {
		out = append(out, fmt.Sprintf("TestCase%02d", i))
	}
	return out
}

// runShard runs shard index of count in dir and returns the exit code and combined output.
func runShard(t *testing.T, dir, evidence string, index, count int) (int, string) {
	t.Helper()
	script, err := filepath.Abs("ci-shard.sh")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("bash", script, fmt.Sprint(index), fmt.Sprint(count), evidence, ".", "-count=1")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOFLAGS=", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), string(out)
	}
	if err != nil {
		t.Fatalf("running ci-shard.sh: %v", err)
	}
	return 0, string(out)
}

func runVerify(t *testing.T, want int, dir string) (int, string) {
	t.Helper()
	script, err := filepath.Abs("ci-shard-verify.sh")
	if err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("bash", script, fmt.Sprint(want), dir).CombinedOutput()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), string(out)
	}
	if err != nil {
		t.Fatalf("running ci-shard-verify.sh: %v", err)
	}
	return 0, string(out)
}

func readLines(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Fields(string(raw))
}

// Every test runs in exactly one shard, whatever the shard count, and the closing proof accepts
// the evidence.
func TestCIShardsPartitionTheTests(t *testing.T) {
	if testing.Short() {
		t.Skip("runs go test on a generated module")
	}
	names := fixtureNames(23)
	dir := shardFixture(t, names)
	for _, c := range []struct {
		count int
		salt  string
	}{{1, ""}, {2, ""}, {5, ""}, {3, "s7:"}} {
		count := c.count
		t.Setenv("CI_SHARD_SALT", c.salt)
		evidence := t.TempDir()
		owner := map[string]int{}
		for i := 1; i <= count; i++ {
			shardDir := filepath.Join(evidence, fmt.Sprintf("shard-%d", i))
			if code, out := runShard(t, dir, shardDir, i, count); code != 0 {
				t.Fatalf("shard %d of %d exited %d\n%s", i, count, code, out)
			}
			for _, n := range readLines(t, filepath.Join(shardDir, "ran.txt")) {
				if prev, dup := owner[n]; dup {
					t.Errorf("count %d: %s ran in shard %d and in shard %d", count, n, prev, i)
				}
				owner[n] = i
			}
		}
		var ran []string
		for n := range owner {
			ran = append(ran, n)
		}
		sort.Strings(ran)
		if strings.Join(ran, " ") != strings.Join(names, " ") {
			t.Errorf("count %d: the shards ran %v, want every test %v", count, ran, names)
		}
		if code, out := runVerify(t, count, evidence); code != 0 {
			t.Errorf("count %d: the proof rejected a correct run (exit %d)\n%s", count, code, out)
		}
	}
}

// A test keeps its shard when others are added, so a pull request that adds a test does not move
// the rest between runners.
func TestCIShardAssignmentIsStableUnderAdditions(t *testing.T) {
	if testing.Short() {
		t.Skip("runs go test on a generated module")
	}
	assigned := func(names []string) map[string]int {
		dir := shardFixture(t, names)
		evidence := t.TempDir()
		out := map[string]int{}
		for i := 1; i <= 3; i++ {
			shardDir := filepath.Join(evidence, fmt.Sprint(i))
			if code, o := runShard(t, dir, shardDir, i, 3); code != 0 {
				t.Fatalf("shard %d exited %d\n%s", i, code, o)
			}
			for _, n := range readLines(t, filepath.Join(shardDir, "assigned.txt")) {
				out[n] = i
			}
		}
		return out
	}
	before := assigned(fixtureNames(10))
	after := assigned(fixtureNames(16))
	for n, s := range before {
		if after[n] != s {
			t.Errorf("%s moved from shard %d to shard %d when tests were added", n, s, after[n])
		}
	}
}

// A failing test fails its shard, and a package without tests is not a pass.
func TestCIShardFailsWhenATestFailsOrThereAreNone(t *testing.T) {
	if testing.Short() {
		t.Skip("runs go test on a generated module")
	}
	names := fixtureNames(6)
	dir := shardFixture(t, names, "TestCase03")
	failed := 0
	for i := 1; i <= 2; i++ {
		if code, _ := runShard(t, dir, filepath.Join(t.TempDir(), "e"), i, 2); code != 0 {
			failed++
		}
	}
	if failed != 1 {
		t.Errorf("%d of 2 shards failed, want exactly the one that owns the failing test", failed)
	}

	empty := shardFixture(t, nil)
	if code, out := runShard(t, empty, filepath.Join(t.TempDir(), "e"), 1, 2); code == 0 {
		t.Errorf("a package with no tests passed\n%s", out)
	}
	if code, _ := runShard(t, dir, filepath.Join(t.TempDir(), "e"), 3, 2); code == 0 {
		t.Error("shard 3 of 2 passed")
	}
}

// The closing proof fails on each way the evidence can be wrong.
func TestCIShardVerifyRejectsBadEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip("runs go test on a generated module")
	}
	dir := shardFixture(t, fixtureNames(12))
	good := func() string {
		evidence := t.TempDir()
		for i := 1; i <= 3; i++ {
			if code, out := runShard(t, dir, filepath.Join(evidence, fmt.Sprint(i)), i, 3); code != 0 {
				t.Fatalf("shard %d exited %d\n%s", i, code, out)
			}
		}
		return evidence
	}
	rewrite := func(path string, edit func([]string) []string) {
		lines := readLines(t, path)
		if err := os.WriteFile(path, []byte(strings.Join(edit(lines), "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		name   string
		break_ func(evidence string)
		want   int
	}{
		{"a shard's evidence never arrived", func(e string) { _ = os.RemoveAll(filepath.Join(e, "3")) }, 3},
		{"a shard ran a test it was not assigned", func(e string) {
			rewrite(filepath.Join(e, "1", "ran.txt"), func(l []string) []string { return append(l, "TestSomethingElse") })
		}, 3},
		{"a shard skipped a test it owns", func(e string) {
			rewrite(filepath.Join(e, "2", "ran.txt"), func(l []string) []string { return l[1:] })
		}, 3},
		{"two shards own the same test", func(e string) {
			other := readLines(t, filepath.Join(e, "2", "assigned.txt"))[0]
			rewrite(filepath.Join(e, "1", "assigned.txt"), func(l []string) []string { return append(l, other) })
			rewrite(filepath.Join(e, "1", "ran.txt"), func(l []string) []string { return append(l, other) })
		}, 3},
		{"a test belongs to no shard", func(e string) {
			rewrite(filepath.Join(e, "3", "assigned.txt"), func(l []string) []string { return l[1:] })
			rewrite(filepath.Join(e, "3", "ran.txt"), func(l []string) []string { return l[1:] })
		}, 3},
		{"the workflow expects more shards than ran", func(e string) {}, 4},
	} {
		evidence := good()
		c.break_(evidence)
		if code, out := runVerify(t, c.want, evidence); code == 0 {
			t.Errorf("%s: the proof accepted it\n%s", c.name, out)
		}
	}
}
