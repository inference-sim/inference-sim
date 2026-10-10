package harness

// blisrun.go is the harness's only route to a simulation: it writes the point as files and
// runs `blis run` on them, the way any user would (#1902).

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// RequireBlis refuses a blis binary path that is empty or does not name an executable file,
// naming how to provide one. Scorers call it before their first point so a missing binary is
// one clear error rather than one failure per point.
func RequireBlis(path string) error {
	if path == "" {
		return errors.New("harness: no blis binary given. Every point is simulated by `blis run`; " +
			"build it with `go build -o blis main.go` at the repository root and pass its path " +
			"(-blis on the scorer commands, Config.Blis in code)")
	}
	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("harness: blis binary %q: %w", path, err)
	}
	if fi.IsDir() || fi.Mode()&0o111 == 0 {
		return fmt.Errorf("harness: blis binary %q is not an executable file", path)
	}
	return nil
}

// writeScenarioVariant writes the scenario file src to dst with its single pool's admission
// settings replaced by pc's: max_num_seqs, max_num_batched_tokens, block_size and
// enable_prefix_caching. Everything else -- model, hardware, parallelism, quantization, the
// remaining engine knobs -- is carried over as the committed file states it.
//
// The file is decoded and re-encoded through the blis-schemas types (via the kernel's own
// loader) rather than edited as text, so the variant is a well-formed Scenario + Deployment
// pair by construction and a field the schema does not know is refused rather than copied.
//
// A scenario stating more than one pool is refused: the corpus scores aggregated serving, and
// the resolved settings describe one engine.
func writeScenarioVariant(src, dst string, pc PointConfig) error {
	sc, dep, err := latencykernel.LoadBundle(src)
	if err != nil {
		return err
	}
	if len(dep.Pools) != 1 {
		return fmt.Errorf("%s states %d pools; the harness scores one aggregated pool", src, len(dep.Pools))
	}
	applyPointConfig(&dep.Pools[0].Engine, pc)
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(sc); err != nil {
		return fmt.Errorf("encoding scenario: %w", err)
	}
	if err := enc.Encode(dep); err != nil {
		return fmt.Errorf("encoding deployment: %w", err)
	}
	if err := enc.Close(); err != nil {
		return err
	}
	return os.WriteFile(dst, buf.Bytes(), 0o644)
}

// applyPointConfig writes the admission settings onto an engine. Prefix caching is written
// explicitly in both directions, so the variant states the point's resolved value rather than
// leaving it to the tri-state's default.
func applyPointConfig(e *deployment.Engine, pc PointConfig) {
	e.MaxNumSeqs = pc.MaxNumSeqs
	e.MaxNumBatchedTokens = pc.MaxNumBatchedTokens
	e.BlockSize = pc.BlockSize
	enabled := !pc.PrefixCachingDisabled
	e.EnablePrefixCaching = &enabled
}

// blisArgs is the `blis run` command line for one point, given the variant scenario's file
// name and directory, the workload spec and the metrics file. dp is not on it: `blis run`
// reads the scenario's data-parallel width and places that many per-rank replicas itself.
func blisArgs(cfg Config, scenarioName, scenarioDir, specPath, metricsPath string) []string {
	return []string{
		"run",
		"--scenario", scenarioName,
		"--scenarios", scenarioDir,
		"--catalog", cfg.Repos.Catalog,
		"--registry", cfg.Repos.Registry,
		"--workload-spec", specPath,
		"--seed", strconv.FormatInt(cfg.Seed, 10),
		"--metrics-path", metricsPath,
		// No client deadline: the measured benchmark's client waits for every response, so a
		// request cut off by blis run's default 300 s deadline would score a point the
		// benchmark never saw.
		"--timeout", "-1",
		"--log", "error",
	}
}

// runBlis runs one point through `blis run` in a fresh temporary directory and returns the
// --metrics-path output.
func runBlis(cfg Config, scenarioName string, pc PointConfig, spec *workload.WorkloadSpec) (*sim.MetricsOutput, error) {
	dir, err := os.MkdirTemp("", "blis-harness-")
	if err != nil {
		return nil, err
	}
	defer func() { _ = os.RemoveAll(dir) }()

	if err := writeScenarioVariant(filepath.Join(cfg.Repos.Scenarios, scenarioName),
		filepath.Join(dir, scenarioName), pc); err != nil {
		return nil, err
	}
	specYAML, err := yaml.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("encoding workload spec: %w", err)
	}
	specPath := filepath.Join(dir, "workload.yaml")
	if err := os.WriteFile(specPath, specYAML, 0o644); err != nil {
		return nil, err
	}
	metricsPath := filepath.Join(dir, "metrics.json")

	cmd := exec.Command(cfg.Blis, blisArgs(cfg, scenarioName, dir, specPath, metricsPath)...)
	// From the repository root, as a user runs it: `blis run` reads its bundled defaults.yaml
	// relative to the working directory, and scorer tests run from their package directory.
	cmd.Dir = repoRoot()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	// stdout is the run's deterministic summary; everything the harness reads is in the
	// metrics file, so stdout is discarded.
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("blis run: %w: %s", err, lastLines(stderr.String(), 10))
	}
	raw, err := os.ReadFile(metricsPath)
	if err != nil {
		return nil, fmt.Errorf("blis run wrote no metrics file: %w", err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("blis run metrics file: %w", err)
	}
	// The measured point completed every request it sent, so the simulated one must too: an
	// average over the survivors of a run that timed out, dropped or never finished requests
	// describes a different, easier point (the slowest requests are the ones lost).
	if out.CompletedRequests != out.InjectedRequests || out.TimedOutRequests > 0 ||
		out.DroppedUnservable > 0 || out.LengthCappedRequests > 0 {
		return nil, fmt.Errorf("blis run completed %d of %d request(s) (%d timed out, %d dropped "+
			"as unservable, %d length-capped, %d queued, %d running at the end)",
			out.CompletedRequests, out.InjectedRequests, out.TimedOutRequests, out.DroppedUnservable,
			out.LengthCappedRequests, out.StillQueued, out.StillRunning)
	}
	return &out, nil
}

// repoRoot is this repository's root, located from this file's compiled-in path
// (<repo>/sim/kernelmodel/harness/blisrun.go) rather than the working directory, the same way
// kernelmodel locates its vendored roots.
func repoRoot() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	return filepath.Join(filepath.Dir(self), "..", "..", "..")
}

// lastLines keeps an error message to the tail of a long stderr.
func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}
