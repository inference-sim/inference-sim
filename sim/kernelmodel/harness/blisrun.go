package harness

// blisrun.go is the harness's only route to a simulation: it writes the point as files and
// runs `blis replay` on them, the way any user replays a trace (#1902).
//
// Replay rather than `blis run`, because the benchmark's workload is a fixed list of requests
// with known lengths in a known order, and a TraceV2 states exactly that. `blis run` would
// re-sample the lengths with BLIS's own generator.

import (
	"bytes"
	_ "embed"
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
		return errors.New("harness: no blis binary given. Every point is simulated by `blis replay`; " +
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
// max_model_len, gpu_memory_utilization, parallel.enable_expert_parallel and, when the run
// passed one, max_cudagraph_capture_size. Everything else --
// model, hardware, parallel widths, quantization, the remaining engine knobs -- is carried
// over as the committed file states it.
//
// The file is decoded and re-encoded through the blis-schemas types (via the kernel's own
// loader) rather than edited as text, so the variant is a well-formed Scenario + Deployment
// pair by construction and a field the schema does not know is refused rather than copied.
//
// A scenario stating more than one pool is refused: the corpus scores aggregated serving, and
// the resolved settings describe one engine.
//
// A non-nil router becomes the variant's router block (its paths relative to dst's directory); a
// committed scenario that already states one is refused rather than overwritten.
func writeScenarioVariant(src, dst string, pc PointConfig, router *deployment.Router) error {
	sc, dep, err := latencykernel.LoadBundle(src)
	if err != nil {
		return err
	}
	if len(dep.Pools) != 1 {
		return fmt.Errorf("%s states %d pools; the harness scores one aggregated pool", src, len(dep.Pools))
	}
	applyPointConfig(&dep.Pools[0], pc)
	if router != nil {
		if dep.Router != nil {
			return fmt.Errorf("%s states a router block; the harness routes every point with its own", src)
		}
		dep.Router = router
	}
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
	e.MaxModelLen = deployment.MaxModelLen(pc.MaxModelLen)
	e.GPUMemoryUtilization = pc.GPUMemoryUtilization
	pool.Parallel.EnableExpertParallel = pc.EnableExpertParallel
	if pc.MaxCUDAGraphCaptureSize != nil {
		c := *pc.MaxCUDAGraphCaptureSize
		e.MaxCUDAGraphCaptureSize = &c
	}
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
func blisArgs(cfg Config, scenarioName, scenarioDir, headerPath, dataPath, metricsPath string, sessions, concurrency int) []string {
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
		"--total-sessions", strconv.Itoa(sessions),
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

// replayRun is one `blis replay` invocation: the trace and the pool that gates it.
type replayRun struct {
	header   *workload.TraceHeader
	records  []workload.TraceRecord
	sessions int // --total-sessions: the trace's session count
	pool     int // --concurrent-sessions
}

// mainPhase is the point's main phase alone, as pointTrace states it.
func mainPhase(pl PointLengths) replayRun {
	h, r := pointTrace(pl)
	return replayRun{header: h, records: r, sessions: len(r), pool: pl.Concurrency}
}

// The primed main phase -- request 0's prompt already in the prefix cache when the main phase
// starts -- written with existing trace features only, so that BLIS itself carries nothing of
// the benchmark's protocol. The exact two-phase protocol (drain-gated phases) needs a trace
// format extension, tracked as blis-schemas#56 and inference-sim#1926; until then this
// bracket is the stated uncertainty.
//
// The trace is the main phase plus `copies` warm-up sessions, listed FIRST:
//
//   - warm-up session j: round 0 is a copy of request 0 (its prompt as a shared prefix group,
//     zero further tokens, its output length) arriving at t=0; round 1 is a one-token holder
//     with think time primeHoldUs, so the session -- and its pool slot -- stays open until
//     long after the main phase ends, and its completion admits no main request early;
//   - main request 0 is in the same prefix group, so its prompt is token-for-token the
//     warm-up copy's;
//   - every main request is recorded at arrival primeMainStartUs, after the warm-up copies
//     have completed and released their blocks (Run checks this).
//
// The pool is CONC + copies, so the first CONC main requests start together at
// primeMainStartUs, in trace order, exactly as the cold main phase starts at 0, and each later
// one is admitted at a main completion. copies is the deployment's dp width: the harness's
// router places by requests in flight (vllm-dp-lb.epp.yaml), so each copy goes to a replica not
// yet holding one, every replica ends up holding request 0's prompt, and the main phase starts
// on idle replicas as in the cold run.
const (
	primeMainStartUs = int64(10_000_000_000)    // 10^4 s: after any single warm-up copy completes
	primeHoldUs      = int64(1_000_000_000_000) // 10^6 s: after any main phase completes
	primePrefixGroup = "request-0-prompt"
)

// primedPhase is pointTrace with request 0's prompt cached by `copies` warm-up copies.
func primedPhase(pl PointLengths, copies int) replayRun {
	h, main := pointTrace(pl)
	n := len(main)
	hold := primeHoldUs
	var records []workload.TraceRecord
	for j := 0; j < copies; j++ {
		sid := "warmup-" + strconv.Itoa(j)
		records = append(records,
			workload.TraceRecord{RequestID: n + 2*j, ClientID: "inferencex-benchmark-serving", SessionID: sid,
				RoundIndex: 0, PrefixGroup: primePrefixGroup, PrefixLength: pl.InputLens[0], Streaming: true,
				InputTokens: 0, OutputTokens: pl.OutputLens[0], Status: "ok"},
			workload.TraceRecord{RequestID: n + 2*j + 1, ClientID: "inferencex-benchmark-serving", SessionID: sid,
				RoundIndex: 1, Streaming: true, InputTokens: 1, OutputTokens: 1, ThinkTimeUs: &hold, Status: "ok"})
	}
	for i := range main {
		main[i].ArrivalTimeUs = primeMainStartUs
	}
	main[0].PrefixGroup, main[0].PrefixLength, main[0].InputTokens = primePrefixGroup, pl.InputLens[0], 0
	records = append(records, main...)
	return replayRun{header: h, records: records, sessions: n + copies, pool: pl.Concurrency + copies}
}

// vllmDPLoadBalancer is the router configuration of every point: vLLM's data-parallel load
// balancer sends a request to the rank with the fewest requests in flight, and BLIS simulates
// each rank as a replica behind the llm-d router, so the router places by requests in flight.
//
//go:embed vllm-dp-lb.epp.yaml
var vllmDPLoadBalancer []byte

// vllmDPLoadBalancerFile is the name the variant scenario's router block gives that configuration.
const vllmDPLoadBalancerFile = "vllm-dp-lb.epp.yaml"

// runBlis runs one point through `blis replay` in a fresh temporary directory and returns the
// --metrics-path output.
func runBlis(cfg Config, scenarioName string, pc PointConfig, run replayRun) (*sim.MetricsOutput, error) {
	dir, err := os.MkdirTemp("", "blis-harness-")
	if err != nil {
		return nil, err
	}
	defer func() { _ = os.RemoveAll(dir) }()

	// The router places like vLLM's own data-parallel balancer: the variant's router block names
	// vllm-dp-lb.epp.yaml, written beside it.
	if err := os.WriteFile(filepath.Join(dir, vllmDPLoadBalancerFile), vllmDPLoadBalancer, 0o644); err != nil {
		return nil, err
	}
	if err := writeScenarioVariant(filepath.Join(cfg.Repos.Scenarios, scenarioName),
		filepath.Join(dir, scenarioName), pc, &deployment.Router{EndpointPickerConfig: vllmDPLoadBalancerFile}); err != nil {
		return nil, err
	}
	records := run.records
	headerPath, dataPath := filepath.Join(dir, "trace.yaml"), filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(run.header, records, headerPath, dataPath); err != nil {
		return nil, err
	}
	metricsPath := filepath.Join(dir, "metrics.json")

	cmd := exec.Command(cfg.Blis, blisArgs(cfg, scenarioName, dir, headerPath, dataPath, metricsPath,
		run.sessions, run.pool)...)
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
