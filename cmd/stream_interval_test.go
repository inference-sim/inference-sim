package cmd

import (
	"os"
	"path/filepath"
	"reflect"
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
// Stated as 1, vLLM's default, it changes no figure; a negative value is refused by the
// scenario's validation, before any engine is built. The grouping itself is tested on the driver
// (sim.TestEngineDriver_StreamInterval*), and a run at the stated values below.
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
// is read here without building an engine.
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

// At the values InferenceX states (10, 20, 32) a run starts, the engine accepts the field (it
// changes when a client sees tokens, not what the scheduler does), and the grouping reaches the
// Result through one series: the client-observed inter-token latency, itl_*. Tokens of a group
// arrive together, so the median gap falls to zero and the gap before a group grows, while the
// number of gaps and their mean (the stream's length over its tokens) stay. TTFT, TPOT, E2E, the
// engine's step intervals and every other section are those of the run that states none. A run at
// 1 (vLLM's default) is the same run, byte for byte.
func TestStreamInterval_StatedValuesReshapeTheITLSeriesAndNothingElse(t *testing.T) {
	args := append(burst(30, 10, 512, 64), "--num-instances", "1", "--seed", "3")
	run := func(dir, scenario string) map[string]any {
		stdout, stderr, err := runKernelCLI(t, runArgs(append([]string{"--scenarios", dir, "--scenario", scenario}, args...))...)
		if err != nil {
			t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
		}
		return resultDoc(t, stdout)
	}
	base := run(pdScenarios, capScenario)
	baseMetrics, _ := base["metrics"].(map[string]any)
	if len(baseMetrics) < 10 {
		t.Fatalf("non-vacuity: the Result states %d metrics", len(baseMetrics))
	}
	fig := func(m map[string]any, id string) (value, samples float64) {
		f, ok := m[id].(map[string]any)
		if !ok {
			t.Fatalf("the Result states no %s", id)
		}
		value, _ = f["value"].(float64)
		samples, _ = f["samples"].(float64)
		return value, samples
	}
	baseMean, baseN := fig(baseMetrics, "itl_mean_ms")
	baseP50, _ := fig(baseMetrics, "itl_p50_ms")
	baseP99, _ := fig(baseMetrics, "itl_p99_ms")
	if baseN < 30 || baseP50 <= 0 {
		t.Fatalf("non-vacuity: itl has %v samples and a median of %v at the default", baseN, baseP50)
	}

	one := run(streamVariant(t, "stream_interval: 1"), "variant.yaml")
	if !reflect.DeepEqual(one["metrics"], base["metrics"]) {
		t.Error("stream_interval: 1 moved a metric")
	}
	for _, v := range []string{"10", "20", "32"} {
		got := run(streamVariant(t, "stream_interval: "+v), "variant.yaml")
		for _, k := range []string{"requests", "engine"} {
			if !reflect.DeepEqual(got[k], base[k]) {
				t.Errorf("stream_interval %s moved the Result's %q section", v, k)
			}
		}
		gm, _ := got["metrics"].(map[string]any)
		for id, want := range baseMetrics {
			if strings.HasPrefix(id, "itl_") {
				continue
			}
			if !reflect.DeepEqual(gm[id], want) {
				t.Errorf("stream_interval %s moved %s: %v, want %v", v, id, gm[id], want)
			}
		}
		mean, n := fig(gm, "itl_mean_ms")
		p50, _ := fig(gm, "itl_p50_ms")
		p99, _ := fig(gm, "itl_p99_ms")
		if n != baseN {
			t.Errorf("stream_interval %s: itl has %v samples, want %v: the same tokens", v, n, baseN)
		}
		if d := mean - baseMean; d > 1e-6*baseMean || d < -1e-6*baseMean {
			t.Errorf("stream_interval %s: itl mean %v, want the default's %v", v, mean, baseMean)
		}
		if p50 != 0 {
			t.Errorf("stream_interval %s: itl p50 %v, want 0: most tokens arrive with another's chunk", v, p50)
		}
		if p99 <= baseP99 {
			t.Errorf("stream_interval %s: itl p99 %v, want above the default's %v: the gap before a group", v, p99, baseP99)
		}
	}
}
