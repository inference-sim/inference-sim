package cmd

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/decision"
)

const frontArtifactFixture = "../substrate/decision/testdata/artifact"

// aRealResult runs the simulator once, on a scenario of the test fixtures, and returns its Result.
func aRealResult(t *testing.T, concurrency string) (*result.Result, []byte) {
	t.Helper()
	out, errOut, err := runKernelCLI(t, "run", "--scenario", "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
		"--seed", "7", "--num-requests", "40", "--concurrency", concurrency)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, errOut)
	}
	return parseResult(t, out), []byte(out)
}

// writeFrontResult writes r to dir/<name>.json and returns the path.
func writeFrontResult(t *testing.T, dir string, r *result.Result) string {
	t.Helper()
	b, err := result.Marshal(r, result.FormatJSON)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, r.Name+".json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// variant is a copy of base with its name and objectives replaced: throughput per GPU, P90 ITL and P90 TTFT.
func variant(t *testing.T, base *result.Result, name string, x, itl, ttft float64) *result.Result {
	t.Helper()
	b, _ := result.Marshal(base, result.FormatJSON)
	var r result.Result
	if err := result.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	r.Name = name
	set := func(id string, v float64) {
		f := r.Metrics[id]
		f.Value = v
		r.Metrics[id] = f
	}
	set(frontMetricX, x)
	set(frontMetricI, itl)
	set(frontMetricT, ttft)
	return &r
}

func frontOut(t *testing.T, o frontOptions) frontReport {
	t.Helper()
	var buf bytes.Buffer
	if err := runFront(&buf, o); err != nil {
		t.Fatalf("front: %v", err)
	}
	var rep frontReport
	if err := json.Unmarshal(buf.Bytes(), &rep); err != nil {
		t.Fatalf("output is not a FrontReport: %v\n%s", err, buf.String())
	}
	return rep
}

// A case of points whose objectives differ in all directions, as files.
func aCase(t *testing.T, base *result.Result, dir string, kx, ki, kt float64) []string {
	t.Helper()
	pts := []struct {
		name         string
		x, itl, ttft float64
	}{
		{"p1", 160, 11, 120}, {"p2", 300, 25, 400}, {"p3", 90, 6, 60}, {"p4", 220, 14, 900},
		{"p5", 40, 30, 200}, {"p6", 500, 50, 1500},
	}
	var files []string
	for _, p := range pts {
		// kx scales throughput, ki scales interactivity (so divides the latency), kt scales TTFT.
		files = append(files, writeFrontResult(t, dir, variant(t, base, p.name, p.x*kx, p.itl/ki, p.ttft*kt)))
	}
	return files
}

func TestFrontReportsEachPointOnEachFrontAndAnyPairAsked(t *testing.T) {
	base, _ := aRealResult(t, "4")
	dir := t.TempDir()
	files := aCase(t, base, dir, 1, 1, 1)
	rep := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: files})
	if len(rep.Points) != 6 || rep.Epsilon != decision.DefaultEpsilon || rep.Artifact.WriterCommit == "" {
		t.Fatalf("report %+v", rep)
	}
	for _, p := range rep.Points {
		if p.Basis != basisSimulated || len(p.Fronts) != 4 {
			t.Fatalf("point %+v", p)
		}
		for f, m := range p.Fronts {
			if !m.Answered || m.Probability < 0 || m.Probability > 1 {
				t.Errorf("%s on %s: %+v", p.Name, f, m)
			}
		}
	}
	if len(rep.Pairs) != 4*15 {
		t.Fatalf("%d pairs for 6 points on 4 fronts", len(rep.Pairs))
	}
	for _, pr := range rep.Pairs {
		if pr.ADominatesB+pr.BDominatesA > 1+1e-12 {
			t.Errorf("%+v sums above one", pr)
		}
		// A point is off the front when another dominates it above one half: the output agrees with itself.
		for _, p := range rep.Points {
			if p.Name == pr.B && pr.ADominatesB > 0.5 && p.Fronts[pr.Front].OnFront {
				t.Errorf("%s dominates %s on %s with %v, and it is on the front", pr.A, pr.B, pr.Front, pr.ADominatesB)
			}
		}
	}
	// --pair in either order gives the same two probabilities, swapped.
	ab := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, Pairs: []string{"p1:p2"}, Fronts: []string{"xi"}, Files: files})
	ba := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, Pairs: []string{"p2:p1"}, Fronts: []string{"xi"}, Files: files})
	if len(ab.Pairs) != 1 || len(ba.Pairs) != 1 || ab.Pairs[0].ADominatesB != ba.Pairs[0].BDominatesA || ab.Pairs[0].BDominatesA != ba.Pairs[0].ADominatesB {
		t.Errorf("pair order changed the answer: %+v vs %+v", ab.Pairs, ba.Pairs)
	}
	if len(ab.Points[0].Fronts) != 1 {
		t.Errorf("--front xi left %d fronts", len(ab.Points[0].Fronts))
	}
}

// DEC-4 through the command: one constant factor per objective on every point of the case, and any
// order of the files, leave probabilities and membership unchanged.
func TestFrontIsUnchangedByAConstantPerObjectiveAndByTheOrderOfTheFiles(t *testing.T) {
	base, _ := aRealResult(t, "4")
	d1, d2 := t.TempDir(), t.TempDir()
	f1 := aCase(t, base, d1, 1, 1, 1)
	f2 := aCase(t, base, d2, 1.7, 0.4, 3.1)
	for i, j := 0, len(f2)-1; i < j; i, j = i+1, j-1 { // reversed order
		f2[i], f2[j] = f2[j], f2[i]
	}
	a := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: f1})
	b := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: f2})
	by := func(r frontReport) map[string]frontPoint {
		m := map[string]frontPoint{}
		for _, p := range r.Points {
			m[p.Name] = p
		}
		return m
	}
	pa, pb := by(a), by(b)
	for name, p := range pa {
		for f, m := range p.Fronts {
			o := pb[name].Fronts[f]
			if m.OnFront != o.OnFront || math.Abs(m.Probability-o.Probability) > 1e-6 {
				t.Errorf("%s on %s: %+v became %+v", name, f, m, o)
			}
		}
	}
	key := func(p frontPairProb) string { return p.Front + p.A + p.B }
	sort.Slice(a.Pairs, func(i, j int) bool { return key(a.Pairs[i]) < key(a.Pairs[j]) })
	sort.Slice(b.Pairs, func(i, j int) bool { return key(b.Pairs[i]) < key(b.Pairs[j]) })
	for i := range a.Pairs {
		if key(a.Pairs[i]) != key(b.Pairs[i]) || math.Abs(a.Pairs[i].ADominatesB-b.Pairs[i].ADominatesB) > 1e-6 {
			t.Fatalf("pair %+v became %+v", a.Pairs[i], b.Pairs[i])
		}
	}
}

// DEC-3: a point the simulator did not simulate is listed with the reason and basis not_simulated,
// is on no front, cannot be asked for in a pair, and changes no other point's answer.
func TestAPointTheSimulatorDidNotSimulateIsExcludedWithItsReason(t *testing.T) {
	base, _ := aRealResult(t, "4")
	dir := t.TempDir()
	files := aCase(t, base, dir, 1, 1, 1)
	plain := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, Files: files})

	noTTFT := variant(t, base, "no-ttft", 100, 10, 100)
	delete(noTTFT.Metrics, frontMetricT) // a run that failed an acceptance condition reports no figures
	refused := filepath.Join(dir, "refused.yaml")
	if err := os.WriteFile(refused, []byte("kind: Refusal\nname: tp16-ep16\nreason: no engine module models this layout\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	more := append(append([]string(nil), files...), writeFrontResult(t, dir, noTTFT), refused)
	rep := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, Files: more})
	if len(rep.Points) != 8 {
		t.Fatalf("%d points", len(rep.Points))
	}
	for _, p := range rep.Points[6:] {
		if p.Basis != basisNotSimulated || p.Excluded == "" || len(p.Fronts) != 0 || p.LogX != nil {
			t.Errorf("excluded point %+v", p)
		}
	}
	if !strings.Contains(rep.Points[6].Excluded, frontMetricT) || !strings.Contains(rep.Points[7].Excluded, "no engine module models this layout") {
		t.Errorf("reasons: %q, %q", rep.Points[6].Excluded, rep.Points[7].Excluded)
	}
	for i, p := range plain.Points {
		for f, m := range p.Fronts {
			if o := rep.Points[i].Fronts[f]; o != m {
				t.Errorf("%s on %s changed by an excluded point: %+v became %+v", p.Name, f, m, o)
			}
		}
	}
	var buf bytes.Buffer
	err := runFront(&buf, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, Pairs: []string{"p1:tp16-ep16"}, Files: more})
	if err == nil || !strings.Contains(err.Error(), "was not simulated") {
		t.Errorf("a pair with an unsimulated point: %v", err)
	}
}

// DEC-5 at the command: the artifact's epsilon and the command's must be one.
func TestFrontRefusesAnArtifactWhoseEpsilonDiffersFromTheCommands(t *testing.T) {
	base, _ := aRealResult(t, "4")
	files := aCase(t, base, t.TempDir(), 1, 1, 1)
	var buf bytes.Buffer
	err := runFront(&buf, frontOptions{Artifact: frontArtifactFixture, Epsilon: 0.1, Files: files})
	if err == nil || !strings.Contains(err.Error(), "epsilon") || buf.Len() != 0 {
		t.Errorf("err %v, output %q", err, buf.String())
	}
	for _, o := range []frontOptions{{Files: files}, {Artifact: frontArtifactFixture, Epsilon: 0.05}, {Artifact: filepath.Join(t.TempDir(), "nope"), Epsilon: 0.05, Files: files}} {
		if err := runFront(&buf, o); err == nil {
			t.Errorf("options %+v were accepted", o)
		}
	}
	if err := runFront(&buf, frontOptions{Artifact: frontArtifactFixture, Epsilon: 0.05, Files: []string{files[0], files[0]}}); err == nil || !strings.Contains(err.Error(), "two documents name") {
		t.Errorf("a repeated point: %v", err)
	}
}

// The command is registered on the root and reads its flags.
func TestFrontIsACommandOfBlis(t *testing.T) {
	base, _ := aRealResult(t, "4")
	files := aCase(t, base, t.TempDir(), 1, 1, 1)
	var out bytes.Buffer
	rootCmd.SetOut(&out)
	rootCmd.SetArgs(append([]string{"front", "--artifact", frontArtifactFixture, "--front", "XI"}, files...))
	defer rootCmd.SetArgs(nil)
	if err := rootCmd.Execute(); err != nil {
		t.Fatal(err)
	}
	var rep frontReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil || len(rep.Points) != 6 {
		t.Fatalf("%v\n%s", err, out.String())
	}
}

// DEC-1: blis front only reads. A run's Result is the same bytes (outside runtime) when the run is
// repeated after the stage has read it, and the stage leaves the Result file and its directory as
// they were. The points are real runs at three concurrencies, so the extraction from a real
// Result is exercised too.
func TestTheStageChangesNoSimulatedValue(t *testing.T) {
	dir := t.TempDir()
	var files []string
	stable := func(raw []byte) string {
		var r result.Result
		if err := result.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		r.Runtime = nil
		b, _ := result.Marshal(&r, result.FormatJSON)
		return string(b)
	}
	before := map[string]string{}
	for _, c := range []string{"1", "4", "16"} {
		r, raw := aRealResult(t, c)
		files = append(files, writeFrontResult(t, dir, r))
		before[c] = stable(raw)
	}
	snapshot := func() map[string]string {
		m := map[string]string{}
		ents, _ := os.ReadDir(dir)
		for _, e := range ents {
			b, _ := os.ReadFile(filepath.Join(dir, e.Name()))
			m[e.Name()] = string(b)
		}
		return m
	}
	was := snapshot()
	rep := frontOut(t, frontOptions{Artifact: frontArtifactFixture, Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: files})
	for _, p := range rep.Points {
		if p.Basis != basisSimulated || len(p.Fronts) != 4 || p.LogX == nil {
			t.Errorf("a real run was not placed: %+v", p)
		}
	}
	after := snapshot()
	if len(was) != len(after) {
		t.Fatalf("front changed the directory: %d files became %d", len(was), len(after))
	}
	for n, b := range was {
		if after[n] != b {
			t.Errorf("front rewrote %s", n)
		}
	}
	for _, c := range []string{"1", "4", "16"} {
		_, raw := aRealResult(t, c)
		if got := stable(raw); got != before[c] {
			t.Errorf("concurrency %s: the run after the stage differs from the run before it", c)
		}
	}
}
