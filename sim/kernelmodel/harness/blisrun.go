package harness

// blisrun.go is the harness's only route to a simulation: it writes the point as files and
// runs `blis replay` on them, the way any user replays a trace (#1902).
//
// Replay rather than `blis run`, because the benchmark's workload is a fixed list of requests
// with known lengths in a known order, and a TraceV2 states exactly that. `blis run` would
// re-sample the lengths with BLIS's own generator.

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

// writeScenarioVariant writes the scenario file src to dst with its single pool's run settings
// replaced by pc's: max_num_seqs, max_num_batched_tokens, block_size, enable_prefix_caching,
// max_model_len, gpu_memory_utilization and parallel.enable_expert_parallel. Everything else --
// model, hardware, parallel widths, quantization, the remaining engine knobs -- is carried
// over as the committed file states it.
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
	applyPointConfig(&dep.Pools[0], pc)
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

// applyPointConfig writes the point's run settings onto a pool. Prefix caching is written
// explicitly in both directions, so the variant states the point's resolved value rather than
// leaving it to the tri-state's default.
func applyPointConfig(pool *deployment.Pool, pc PointConfig) {
	e := &pool.Engine
	e.MaxNumSeqs = pc.MaxNumSeqs
	e.MaxNumBatchedTokens = pc.MaxNumBatchedTokens
	e.BlockSize = pc.BlockSize
	enabled := !pc.PrefixCachingDisabled
	e.EnablePrefixCaching = &enabled
	e.MaxModelLen = pc.MaxModelLen
	e.GPUMemoryUtilization = pc.GPUMemoryUtilization
	pool.Parallel.EnableExpertParallel = pc.EnableExpertParallel
}

// blisArgs is the `blis replay` command line for one point: the variant scenario's file name
// and directory, the point's trace, and the metrics file. dp is not on it: the command reads
// the scenario's data-parallel width and places that many per-rank replicas itself.
//
// The pool flags are the benchmark's concurrency gate. With --concurrent-sessions CONC every
// trace record is a one-round session, the first CONC start at t=0, and each later one is
// admitted, in trace order, when a session completes (workload.BuildSessionPool and
// SessionPoolDriver.OnComplete) -- the semaphore of CONC the client holds its 10*CONC tasks
// under. --total-sessions equals the trace length, so no session is cloned, and
// --shuffle-corpus is absent, so the order is the trace's. No --think-time-ms: a pooled
// session is admitted at the completion tick, which is the client's zero-wait behaviour at
// --request-rate inf. No per-request deadline: the trace states none (deadline_us 0), as the
// client sets no timeout on a request.
func blisArgs(cfg Config, scenarioName, scenarioDir, headerPath, dataPath, metricsPath string, requests, concurrency int) []string {
	return []string{
		"replay",
		"--scenario", scenarioName,
		"--scenarios", scenarioDir,
		"--catalog", cfg.Repos.Catalog,
		"--registry", cfg.Repos.Registry,
		"--trace-header", headerPath,
		"--trace-data", dataPath,
		"--session-mode", "closed-loop",
		"--concurrent-sessions", strconv.Itoa(concurrency),
		"--total-sessions", strconv.Itoa(requests),
		"--seed", strconv.FormatInt(cfg.Seed, 10),
		"--metrics-path", metricsPath,
		"--log", "error",
	}
}

// pointTrace is one point's main phase as a TraceV2: request i of the benchmark is record i,
// a one-round session asking for exactly the fixture's target lengths. Nothing else is
// stated -- no prefix group (the benchmark's random prompts share none), no deadline, no
// arrival time beyond 0 (the pool, not the trace, decides when each one starts).
func pointTrace(pl PointLengths) (*workload.TraceHeader, []workload.TraceRecord) {
	header := &workload.TraceHeader{
		Version:  3,
		TimeUnit: "microseconds",
		Mode:     "generated",
	}
	records := make([]workload.TraceRecord, len(pl.InputLens))
	for i := range pl.InputLens {
		records[i] = workload.TraceRecord{
			RequestID:    i,
			ClientID:     "inferencex-benchmark-serving",
			SessionID:    "main-" + strconv.Itoa(i),
			RoundIndex:   0,
			Streaming:    true,
			InputTokens:  pl.InputLens[i],
			OutputTokens: pl.OutputLens[i],
			Status:       "ok",
		}
	}
	return header, records
}

// runBlis runs one point through `blis replay` in a fresh temporary directory and returns the
// --metrics-path output.
func runBlis(cfg Config, scenarioName string, pc PointConfig, pl PointLengths) (*sim.MetricsOutput, error) {
	dir, err := os.MkdirTemp("", "blis-harness-")
	if err != nil {
		return nil, err
	}
	defer func() { _ = os.RemoveAll(dir) }()

	if err := writeScenarioVariant(filepath.Join(cfg.Repos.Scenarios, scenarioName),
		filepath.Join(dir, scenarioName), pc); err != nil {
		return nil, err
	}
	header, records := pointTrace(pl)
	headerPath, dataPath := filepath.Join(dir, "trace.yaml"), filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		return nil, err
	}
	metricsPath := filepath.Join(dir, "metrics.json")

	cmd := exec.Command(cfg.Blis, blisArgs(cfg, scenarioName, dir, headerPath, dataPath, metricsPath,
		len(records), pl.Concurrency)...)
	// From the repository root, as a user runs it: the command reads its bundled defaults.yaml
	// relative to the working directory, and scorer tests run from their package directory.
	cmd.Dir = repoRoot()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	// stdout is the run's deterministic summary; everything the harness reads is in the
	// metrics file, so stdout is discarded.
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("blis replay: %w: %s", err, lastLines(stderr.String(), 10))
	}
	raw, err := os.ReadFile(metricsPath)
	if err != nil {
		return nil, fmt.Errorf("blis replay wrote no metrics file: %w", err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("blis replay metrics file: %w", err)
	}
	// The measured point completed every request it sent, so the simulated one must too: an
	// average over the survivors of a run that timed out, dropped or never finished requests
	// describes a different, easier point (the slowest requests are the ones lost).
	if out.InjectedRequests != len(records) || out.CompletedRequests != len(records) ||
		out.TimedOutRequests > 0 || out.DroppedUnservable > 0 || out.LengthCappedRequests > 0 {
		return nil, fmt.Errorf("blis replay completed %d of %d injected request(s), %d sent (%d timed out, "+
			"%d dropped as unservable, %d length-capped, %d queued, %d running at the end)",
			out.CompletedRequests, out.InjectedRequests, len(records), out.TimedOutRequests,
			out.DroppedUnservable, out.LengthCappedRequests, out.StillQueued, out.StillRunning)
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
