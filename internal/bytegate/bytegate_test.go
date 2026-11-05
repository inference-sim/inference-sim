// Package bytegate is the byte gate (design SIM-4, X6, X13): a fixed set of runs whose
// standard output must not change in a refactor. Each case in testdata/bytegate/cases.txt is
// run through a blis binary built from this checkout, at GOMAXPROCS 1 and 4, and its standard
// output is compared byte for byte with testdata/bytegate/golden/<case>.json.
//
// The comparison leaves out only the Result's top-level runtime member, the one part of the
// document that depends on the host and the wall clock (contracts/spec/result). It is cut
// out at its exact byte offsets; every other byte is compared as written.
//
// A change that moves a golden file is not a refactor. Regenerate with
//
//	go test ./internal/bytegate -update
//
// and put the diff, with the reason and the price-change report, in the pull request.
package bytegate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/internal/substratedata"
)

var update = flag.Bool("update", false, "rewrite the golden files from this checkout's output")

// excluded are the top-level Result members the gate leaves out (X6).
var excluded = map[string]bool{"runtime": true}

// minCases is the fewest cases the gate accepts, so a truncated cases file cannot pass by
// running nothing (X22).
const minCases = 6

type gateCase struct {
	name string
	args []string
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller information")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// readCases reads cases.txt: one case per line, its name, a colon, and the blis arguments.
// Lines starting with # are comments.
func readCases(path string) ([]gateCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var out []gateCase
	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, args, ok := strings.Cut(line, ":")
		name = strings.TrimSpace(name)
		if !ok || name == "" || strings.ContainsAny(name, " /") {
			return nil, fmt.Errorf("%s:%d: want <name>: <arguments>", path, n)
		}
		if seen[name] {
			return nil, fmt.Errorf("%s:%d: case %q appears twice", path, n, name)
		}
		seen[name] = true
		out = append(out, gateCase{name: name, args: strings.Fields(args)})
	}
	return out, sc.Err()
}

// withoutMembers returns doc, a JSON object, with each excluded top-level member removed
// together with the comma that separates it from the member before it. It fails on anything
// but one object.
func withoutMembers(doc []byte, drop map[string]bool) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(doc))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, fmt.Errorf("standard output is not one JSON object")
	}
	type span struct{ from, to int64 }
	var cuts []span
	prevEnd := dec.InputOffset() // just past "{" or the previous member's value
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, _ := tok.(string)
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, err
		}
		end := dec.InputOffset()
		if drop[key] {
			cuts = append(cuts, span{prevEnd, end})
		}
		prevEnd = end
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("standard output holds more than one JSON value")
	}
	out := make([]byte, 0, len(doc))
	at := int64(0)
	for _, c := range cuts {
		out = append(out, doc[at:c.from]...)
		at = c.to
	}
	out = append(out, doc[at:]...)
	if len(cuts) > 0 && cuts[0].from == int64(bytes.IndexByte(doc, '{'))+1 {
		// The first member was cut: drop the comma that now leads the object.
		i := bytes.IndexByte(out, '{') + 1
		j := i
		for j < len(out) && (out[j] == ' ' || out[j] == '\n' || out[j] == '\t' || out[j] == '\r') {
			j++
		}
		if j < len(out) && out[j] == ',' {
			out = append(out[:j], out[j+1:]...)
		}
	}
	return out, nil
}

func TestWithoutMembersCutsOnlyTheMember(t *testing.T) {
	in := []byte("{\n  \"a\": 1,\n  \"runtime\": {\"x\": [1, 2]},\n  \"b\": \"runtime\"\n}\n")
	got, err := withoutMembers(in, excluded)
	if err != nil {
		t.Fatal(err)
	}
	if want := "{\n  \"a\": 1,\n  \"b\": \"runtime\"\n}\n"; string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
	last := []byte("{\n  \"a\": 1,\n  \"runtime\": {}\n}\n")
	if got, _ := withoutMembers(last, excluded); string(got) != "{\n  \"a\": 1\n}\n" {
		t.Errorf("last member: got %q", got)
	}
	first := []byte("{\"runtime\": 1, \"a\": 2}")
	if got, _ := withoutMembers(first, excluded); string(got) != "{ \"a\": 2}" {
		t.Errorf("first member: got %q", got)
	}
	if _, err := withoutMembers([]byte("{} {}"), excluded); err == nil {
		t.Error("two documents must be refused")
	}
}

// TestByteGate runs every case and compares its output with the golden file.
func TestByteGate(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and runs the blis binary")
	}
	root := repoRoot(t)
	gate := filepath.Join(root, "testdata", "bytegate")
	cases, err := readCases(filepath.Join(gate, "cases.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) < minCases {
		t.Fatalf("cases.txt holds %d cases; the gate needs at least %d", len(cases), minCases)
	}
	catalog, registry := substratedata.MustCatalog(), substratedata.MustRegistry()

	bin := filepath.Join(t.TempDir(), "blis")
	build := exec.Command("go", "build", "-buildvcs=false", "-trimpath", "-o", bin, ".")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	golden := filepath.Join(gate, "golden")
	want := map[string]bool{}
	for _, c := range cases {
		want[c.name+".json"] = true
		t.Run(c.name, func(t *testing.T) {
			var first []byte
			for _, procs := range []string{"1", "4"} {
				args := append([]string{}, c.args...)
				args = append(args, "--scenarios", filepath.Join(gate, "scenarios"),
					"--catalog", catalog, "--registry", registry)
				cmd := exec.Command(bin, args...)
				// Run outside the checkout, so no defaults.yaml in the working directory is read,
				// with only the environment the run needs (X25).
				cmd.Dir = t.TempDir()
				cmd.Env = []string{"GOMAXPROCS=" + procs, "HOME=" + cmd.Dir, "PATH=" + os.Getenv("PATH")}
				var stdout, stderr bytes.Buffer
				cmd.Stdout, cmd.Stderr = &stdout, &stderr
				if err := cmd.Run(); err != nil {
					t.Fatalf("GOMAXPROCS=%s blis %s: %v\n%s", procs, strings.Join(c.args, " "), err, tail(stderr.String()))
				}
				got, err := withoutMembers(stdout.Bytes(), excluded)
				if err != nil {
					t.Fatalf("GOMAXPROCS=%s: %v", procs, err)
				}
				if first == nil {
					first = got
				} else if !bytes.Equal(first, got) {
					t.Fatalf("output differs between GOMAXPROCS 1 and %s (X6)", procs)
				}
			}
			path := filepath.Join(golden, c.name+".json")
			if *update {
				if err := os.WriteFile(path, first, 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			old, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("%v (run go test ./internal/bytegate -update to create it)", err)
			}
			if !bytes.Equal(old, first) {
				t.Errorf("output differs from %s at byte %d: a refactor must not change it (SIM-4). "+
					"If the change is meant, regenerate with -update and report it.", path, firstDiff(old, first))
			}
		})
	}
	if *update {
		return
	}
	files, err := filepath.Glob(filepath.Join(golden, "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if !want[filepath.Base(f)] {
			t.Errorf("%s has no case in cases.txt; delete it or add the case", f)
		}
	}
}

// traceCase is the byte-gate case whose trace the gate also compares: the trace format is
// output too (trace format version 4, contracts/spec/trace), and a refactor must not move it.
const traceCase = "moe-prefix"

// TestByteGateTrace writes traceCase's trace with --trace-output, at GOMAXPROCS 1 and 4, and
// compares the header and the data byte for byte with testdata/bytegate/trace. It then
// replays the trace and checks that the replay reads back every request (the round trip).
func TestByteGateTrace(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and runs the blis binary")
	}
	root := repoRoot(t)
	gate := filepath.Join(root, "testdata", "bytegate")
	cases, err := readCases(filepath.Join(gate, "cases.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var args []string
	for _, c := range cases {
		if c.name == traceCase {
			args = c.args
		}
	}
	if args == nil {
		t.Fatalf("cases.txt has no case %q", traceCase)
	}
	catalog, registry := substratedata.MustCatalog(), substratedata.MustRegistry()
	bin := buildBlis(t, root)
	data := []string{"--scenarios", filepath.Join(gate, "scenarios"), "--catalog", catalog, "--registry", registry}
	golden := filepath.Join(gate, "trace")
	var first map[string][]byte
	var prefix string
	for _, procs := range []string{"1", "4"} {
		dir := t.TempDir()
		prefix = filepath.Join(dir, traceCase)
		runBlis(t, bin, procs, append(append(append([]string{}, args...), data...), "--trace-output", prefix)...)
		got := map[string][]byte{}
		for _, ext := range []string{".yaml", ".csv"} {
			b, err := os.ReadFile(prefix + ext)
			if err != nil {
				t.Fatal(err)
			}
			got[ext] = b
		}
		if first == nil {
			first = got
			continue
		}
		for ext := range got {
			if !bytes.Equal(first[ext], got[ext]) {
				t.Fatalf("the trace's %s differs between GOMAXPROCS 1 and %s (X6)", ext, procs)
			}
		}
	}
	for ext, b := range first {
		path := filepath.Join(golden, traceCase+ext)
		if *update {
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		old, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%v (run go test ./internal/bytegate -update to create it)", err)
		}
		if !bytes.Equal(old, b) {
			t.Errorf("the trace's %s differs from %s at byte %d: a refactor must not change it (SIM-4)", ext, path, firstDiff(old, b))
		}
	}
	if !bytes.HasPrefix(first[".yaml"], []byte("trace_version: 4\n")) {
		t.Errorf("the trace header does not start with trace_version: 4:\n%s", first[".yaml"])
	}
	// The round trip: a replay of the golden trace reads back every request it records.
	out := runBlis(t, bin, "1", append([]string{"replay", "--scenario", args[indexOf(args, "--scenario")+1],
		"--trace-header", filepath.Join(golden, traceCase+".yaml"), "--trace-data", filepath.Join(golden, traceCase+".csv")}, data...)...)
	var r struct {
		Requests struct {
			Injected int `json:"injected"`
		} `json:"requests"`
	}
	if err := json.Unmarshal(out, &r); err != nil {
		t.Fatal(err)
	}
	if rows := bytes.Count(first[".csv"], []byte("\n")) - 1; r.Requests.Injected != rows || rows == 0 {
		t.Errorf("the replay injected %d requests; the trace holds %d", r.Requests.Injected, rows)
	}
}

// buildBlis builds the blis binary from the checkout at root.
func buildBlis(t *testing.T, root string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "blis")
	build := exec.Command("go", "build", "-buildvcs=false", "-trimpath", "-o", bin, ".")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}
	return bin
}

// runBlis runs bin with args outside the checkout, with only the environment a run needs
// (X25), and returns its standard output.
func runBlis(t *testing.T, bin, procs string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command(bin, args...)
	cmd.Dir = t.TempDir()
	cmd.Env = []string{"GOMAXPROCS=" + procs, "HOME=" + cmd.Dir, "PATH=" + os.Getenv("PATH")}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("GOMAXPROCS=%s blis %s: %v\n%s", procs, strings.Join(args, " "), err, tail(stderr.String()))
	}
	return stdout.Bytes()
}

func indexOf(s []string, v string) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}

func firstDiff(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

func tail(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > 5 {
		lines = lines[len(lines)-5:]
	}
	return strings.Join(lines, "\n")
}
