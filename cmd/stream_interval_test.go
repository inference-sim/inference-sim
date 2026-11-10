package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// streamVariant is the committed capScenario with line added after its max_num_seqs line, in a
// fresh directory as variant.yaml.
func streamVariant(t *testing.T, line string) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(pdScenarios, capScenario))
	if err != nil {
		t.Fatal(err)
	}
	const anchor = "      max_num_seqs: 256\n"
	if !strings.Contains(string(body), anchor) {
		t.Fatalf("%s has no %q", capScenario, anchor)
	}
	dir := t.TempDir()
	out := strings.Replace(string(body), anchor, anchor+"      "+line+"\n", 1)
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(out), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// engine.stream_interval reaches the driver's delivery (sim.EngineDriverConfig.StreamInterval).
// Stated as 1, vLLM's default, it changes no figure; below 1 it is refused as vLLM refuses it,
// before any engine is built. The grouping itself, at the values InferenceX states, is tested
// on the driver (sim.TestEngineDriver_StreamInterval*): the engine this build pins refuses a
// value above 1, so a run at 20 does not start until the engine accepts the field.
func TestStreamInterval_DefaultChangesNothingAndNegativeIsRefused(t *testing.T) {
	args := append(burst(30, 10, 512, 64), "--num-instances", "1", "--seed", "3")
	base := capRun(t, runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", capScenario}, args...))...)
	one := streamVariant(t, "stream_interval: 1")
	got := capRun(t, runArgs(append([]string{"--scenarios", one, "--scenario", "variant.yaml"}, args...))...)
	for _, k := range []string{"ttft_mean_ms", "ttft_p99_ms", "e2e_mean_ms"} {
		if got.num(k) != base.num(k) {
			t.Errorf("stream_interval: 1 moved %s: %v, want %v", k, got.num(k), base.num(k))
		}
	}
	bad := streamVariant(t, "stream_interval: -3")
	out, stderr, err := runKernelCLI(t, runArgs(append([]string{"--scenarios", bad, "--scenario", "variant.yaml"}, args...))...)
	if err == nil || out != "" || !strings.Contains(stderr, "stream_interval: must not be negative") {
		t.Errorf("a negative stream_interval: err %v, output %d bytes, want a refusal naming it\n%s", err, len(out), lastLines(stderr, 3))
	}
}

// The pool's stated engine.stream_interval is what the driver is configured with: the value the
// scenario states, 0 when it states none. Opening the model needs no engine, so the stated 20
// that the pinned engine refuses can still be read here.
func TestStreamIntervalOf_ReadsTheScenariosValue(t *testing.T) {
	_, catalog, registry := kernelRepos(t)
	saved := [4]string{kernelScenario, kernelScenarioDir, catalogPath, registryPath}
	savedOpened, savedFor := openedData, openedFor
	t.Cleanup(func() {
		kernelScenario, kernelScenarioDir, catalogPath, registryPath = saved[0], saved[1], saved[2], saved[3]
		openedData, openedFor = savedOpened, savedFor
	})
	catalogPath, registryPath = catalog, registry
	for _, c := range []struct {
		name, dir, file string
		want            int
	}{
		{"unstated", pdScenarios, capScenario, 0},
		{"stated 32", streamVariant(t, "stream_interval: 32"), "variant.yaml", 32},
		{"stated 1", streamVariant(t, "stream_interval: 1"), "variant.yaml", 1},
	} {
		kernelScenario, kernelScenarioDir = c.file, c.dir
		m, err := openModel()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got, err := streamIntervalOf(m); err != nil || got != c.want {
			t.Errorf("%s: streamIntervalOf = %d, %v; want %d", c.name, got, err, c.want)
		}
	}
}
