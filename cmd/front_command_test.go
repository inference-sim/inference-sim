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
	"time"

	"gopkg.in/yaml.v3"

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

// DEC-1: blis front only reads. It leaves the directory of the Results as it found it: the same
// files, each with the same bytes and the same modification time (a rewrite with identical bytes
// changes the time), and it writes nothing else there. The points are real runs at three
// concurrencies, so the extraction from a real Result is exercised too.
func TestTheStageChangesNoSimulatedValue(t *testing.T) {
	dir := t.TempDir()
	var files []string
	for _, c := range []string{"1", "4", "16"} {
		r, _ := aRealResult(t, c)
		files = append(files, writeFrontResult(t, dir, r))
	}
	// Date the inputs an hour back, so that a later write shows in the time.
	old := time.Now().Add(-time.Hour).Truncate(time.Second)
	for _, f := range files {
		if err := os.Chtimes(f, old, old); err != nil {
			t.Fatal(err)
		}
	}
	type seen struct {
		bytes string
		mtime time.Time
	}
	snapshot := func() map[string]seen {
		m := map[string]seen{}
		ents, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			b, _ := os.ReadFile(filepath.Join(dir, e.Name()))
			fi, err := e.Info()
			if err != nil {
				t.Fatal(err)
			}
			m[e.Name()] = seen{string(b), fi.ModTime()}
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
		a, ok := after[n]
		switch {
		case !ok:
			t.Errorf("front removed %s", n)
		case a.bytes != b.bytes:
			t.Errorf("front changed the bytes of %s", n)
		case !a.mtime.Equal(b.mtime):
			t.Errorf("front rewrote %s: modified %v, was %v", n, a.mtime, b.mtime)
		}
	}
}

// refusalWithConfiguration writes a refusal document that also states the candidate's configuration, as a
// Result's engine, input and frame.
func refusalWithConfiguration(t *testing.T, dir, name, reason string, base *result.Result) string {
	t.Helper()
	b, err := yaml.Marshal(struct {
		Kind   string        `yaml:"kind"`
		Name   string        `yaml:"name"`
		Reason string        `yaml:"reason"`
		Engine result.Engine `yaml:"engine"`
		Input  result.Input  `yaml:"input"`
		Frame  result.Frame  `yaml:"frame"`
	}{"Refusal", name, reason, base.Engine, base.Input, base.Frame})
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name+".yaml")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// DEC-3 at the command: with the shipped artifact every candidate that states its configuration is
// placed. A point the simulator refused, or a Result that lacks an objective, is estimated by the fallback and
// labelled estimated, on the fronts and in the pairs it was placed on without the simulator; the points the
// simulator answered keep their basis. A point that states no configuration is listed and placed nowhere.
func TestFrontEstimatesEveryCandidateThatStatesItsConfigurationAndLabelsThem(t *testing.T) {
	base, _ := aRealResult(t, "4")
	dir := t.TempDir()
	files := aCase(t, base, dir, 1, 1, 1)
	noTTFT := variant(t, base, "no-ttft", 100, 10, 100)
	delete(noTTFT.Metrics, frontMetricT)
	refusedConfigured := refusalWithConfiguration(t, dir, "tp16-ep16", "no engine module models this layout", base)
	bare := filepath.Join(dir, "bare.yaml")
	if err := os.WriteFile(bare, []byte("kind: Refusal\nname: bare\nreason: no engine module models this layout\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	all := append(append([]string(nil), files...), writeFrontResult(t, dir, noTTFT), refusedConfigured, bare)
	rep := frontOut(t, frontOptions{Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: all})
	if rep.Artifact.Dir != "the shipped artifact" || rep.Artifact.Features != len(decision.PairFeatureNames) {
		t.Errorf("artifact %+v", rep.Artifact)
	}
	by := map[string]frontPoint{}
	for _, p := range rep.Points {
		by[p.Name] = p
	}
	for _, n := range []string{"p1", "p2", "p3", "p4", "p5", "p6"} {
		p := by[n]
		if p.Basis != basisSimulated {
			t.Fatalf("%s: %+v", n, p)
		}
		for f, m := range p.Fronts {
			if !m.Answered || m.Estimated {
				t.Errorf("%s on %s: %+v", n, f, m)
			}
		}
	}
	nt := by["no-ttft"]
	if nt.Basis != basisEstimated || !strings.Contains(nt.EstimatedBecause, frontMetricT) || nt.LogX == nil || nt.LogT != nil || len(nt.Fronts) != 4 {
		t.Fatalf("a Result without TTFT: %+v", nt)
	}
	for f, m := range nt.Fronts {
		withT := strings.Contains(f, "T")
		if m.Answered == withT || m.Estimated != withT || m.Probability < 0 || m.Probability > 1 {
			t.Errorf("no-ttft on %s: %+v (the simulator answered X and I, not T)", f, m)
		}
	}
	rf := by["tp16-ep16"]
	if rf.Basis != basisEstimated || !strings.Contains(rf.EstimatedBecause, "no engine module models this layout") || rf.LogX != nil || len(rf.Fronts) != 4 {
		t.Fatalf("a configured refusal: %+v", rf)
	}
	for f, m := range rf.Fronts {
		if m.Answered || !m.Estimated || m.Probability <= 0 || m.Probability > 1 {
			t.Errorf("refused point on %s: %+v", f, m)
		}
	}
	if b := by["bare"]; b.Basis != basisNotSimulated || len(b.Fronts) != 0 || !strings.Contains(b.Excluded, "states no configuration") {
		t.Errorf("a refusal with no configuration: %+v", b)
	}
	estimated, simulatedPairs := 0, 0
	for _, pr := range rep.Pairs {
		involves := pr.A == "tp16-ep16" || pr.B == "tp16-ep16" || (strings.Contains(pr.Front, "T") && (pr.A == "no-ttft" || pr.B == "no-ttft"))
		if pr.Estimated != involves || (pr.Basis == "fallback") != involves {
			t.Errorf("pair %+v: estimated should be %v", pr, involves)
		}
		if pr.ADominatesB+pr.BDominatesA > 1+1e-12 {
			t.Errorf("%+v sums above one", pr)
		}
		if pr.Estimated {
			estimated++
		} else {
			simulatedPairs++
		}
	}
	if estimated == 0 || simulatedPairs == 0 {
		t.Errorf("pairs: %d estimated, %d simulated", estimated, simulatedPairs)
	}
	// A pair with the unplaced point is refused, as an unplaceable point is not in any pair.
	var buf bytes.Buffer
	if err := runFront(&buf, frontOptions{Epsilon: decision.DefaultEpsilon, Pairs: []string{"p1:bare"}, Files: all}); err == nil || !strings.Contains(err.Error(), "was not simulated") {
		t.Errorf("a pair with a point no model can place: %v", err)
	}
	if err := runFront(&buf, frontOptions{Epsilon: decision.DefaultEpsilon, Pairs: []string{"p1:tp16-ep16"}, Fronts: []string{"xi"}, Files: all}); err != nil {
		t.Errorf("a pair with an estimated point: %v", err)
	}
}

// DEC-3: a point placed by the fallback changes no probability of a pair of simulated points: those pairs
// are the simulator model's, and are the same with or without the estimated point.
func TestFrontsPairsAmongSimulatedPointsAreTheSameWithAnEstimatedPointPresent(t *testing.T) {
	base, _ := aRealResult(t, "4")
	dir := t.TempDir()
	files := aCase(t, base, dir, 1, 1, 1)
	plain := frontOut(t, frontOptions{Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: files})
	with := frontOut(t, frontOptions{Epsilon: decision.DefaultEpsilon, AllPairs: true, Files: append(append([]string(nil), files...), refusalWithConfiguration(t, dir, "tp16-ep16", "no engine module models this layout", base))})
	key := func(p frontPairProb) string { return p.Front + "|" + p.A + "|" + p.B }
	got := map[string]frontPairProb{}
	for _, p := range with.Pairs {
		got[key(p)] = p
	}
	for _, p := range plain.Pairs {
		if q, ok := got[key(p)]; !ok || q.ADominatesB != p.ADominatesB || q.BDominatesA != p.BDominatesA || q.Basis != "simulator" {
			t.Errorf("pair %+v became %+v", p, q)
		}
	}
}
