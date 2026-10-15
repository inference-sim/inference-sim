package cmd

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim"
	simworkload "github.com/inference-sim/inference-sim/sim/workload"
)

// `blis replay --agentic-scenario inferencex-agentx-mvp` (#1937): replay aiperf 0.12.0's
// AgentX scenario from the raw Weka v7 corpus instead of a TraceV2 file. The workload is
// described by blis-schemas' Replay descriptor (blis-schemas#71), given whole with
// --workload-spec or assembled from the flags below; the flags override the file.

var (
	agxScenarioFlag     string
	agxWorkloadSpec     string
	agxCorpusPath       string
	agxCorpusSHA256     string
	agxConcurrency      int
	agxBenchmarkDur     time.Duration
	agxGracePeriod      string
	agxMaxContextLength string
)

func registerAgentXReplayFlags(c *cobra.Command) {
	c.Flags().StringVar(&agxScenarioFlag, "agentic-scenario", "", `Replay an agentic harness scenario from its raw corpus instead of a TraceV2 file. One value: "inferencex-agentx-mvp" (aiperf 0.12.0 --scenario inferencex-agentx-mvp). Excludes --trace-header/--trace-data/--session-mode`)
	c.Flags().StringVar(&agxWorkloadSpec, "workload-spec", "", "With --agentic-scenario: a blis-schemas replay workload YAML (a Replay, or a document whose workload.replay is one); the flags below override it")
	c.Flags().StringVar(&agxCorpusPath, "corpus", "", "With --agentic-scenario: the Weka v7 corpus file (JSONL, one trace per line), read once and streamed")
	c.Flags().StringVar(&agxCorpusSHA256, "corpus-sha256", "", "With --agentic-scenario: the corpus file's sha256; the replay refuses a corpus that does not match (defaults to the descriptor's corpus.sha256)")
	c.Flags().IntVar(&agxConcurrency, "concurrency", 0, "With --agentic-scenario: session-tree lanes (live trees, not requests)")
	c.Flags().DurationVar(&agxBenchmarkDur, "benchmark-duration", 0, "With --agentic-scenario: the profiling sending window (at least 900s; scenario default 1800s)")
	c.Flags().StringVar(&agxGracePeriod, "grace-period", "", `With --agentic-scenario: how long requests in flight at the end of the window may finish before they are cancelled without a record (a duration or "inf"; default 30s)`)
	c.Flags().StringVar(&agxMaxContextLength, "max-context-length", "", `With --agentic-scenario: drop every trace whose peak prompt + max(output, 1) exceeds this ("max_model_len" = the deployment's resolved max_model_len, or a token count; default no filter)`)
}

func agentXRequested() bool { return agxScenarioFlag != "" || agxWorkloadSpec != "" }

// agentXRun is a resolved agentic replay: the driver and its corpus.
type agentXRun struct {
	spec   *workload.Replay
	corpus *simworkload.AgentXCorpus
	driver *simworkload.AgentXReplay
	limit  int
}

// loadReplaySpec reads a replay workload document: a bare Replay or a scenario-shaped
// document with workload.replay.
func loadReplaySpec(path string) (*workload.Replay, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Replay   *workload.Replay `yaml:"replay"`
		Workload *struct {
			Replay *workload.Replay `yaml:"replay"`
		} `yaml:"workload"`
	}
	if err := yaml.Unmarshal(b, &doc); err == nil {
		switch {
		case doc.Replay != nil:
			return doc.Replay, nil
		case doc.Workload != nil && doc.Workload.Replay != nil:
			return doc.Workload.Replay, nil
		}
	}
	var r workload.Replay
	if err := yaml.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &r, nil
}

// resolveAgentXSpec merges the flags over the descriptor and validates the result as
// aiperf's scenario validator would (blis-schemas Replay.Validate).
func resolveAgentXSpec(cmd *cobra.Command) (*workload.Replay, error) {
	r := &workload.Replay{}
	if agxWorkloadSpec != "" {
		var err error
		if r, err = loadReplaySpec(agxWorkloadSpec); err != nil {
			return nil, err
		}
		if r.Workload != "" {
			return nil, fmt.Errorf("--workload-spec names catalog replay workload %q; state the replay inline", r.Workload)
		}
	}
	if agxScenarioFlag != "" {
		r.Scenario = workload.ReplayScenario(agxScenarioFlag)
	}
	if r.HarnessVersion == "" {
		r.HarnessVersion = "0.12.0"
	}
	if r.Corpus.Loader == "" {
		// The corpus is given as a local file: describe it as the pinned 256k corpus the
		// scenario names, which the sha256 check holds it to.
		r.Corpus = workload.Corpus{
			Loader: "semianalysis_cc_traces_weka_062126_256k", Format: workload.CorpusWekaTraceV7, Encoding: workload.CorpusJSONL,
			Source: workload.CorpusSource{HuggingFace: "semianalysisai/cc-traces-weka-062126-256k", Revision: "local", File: agxCorpusPath},
		}
	}
	if agxCorpusSHA256 != "" {
		r.Corpus.SHA256 = strings.ToLower(agxCorpusSHA256)
	}
	if cmd.Flags().Changed("concurrency") {
		r.Concurrency = agxConcurrency
	}
	if cmd.Flags().Changed("seed") || r.RandomSeed == nil {
		s := seed
		r.RandomSeed = &s
	}
	if cmd.Flags().Changed("benchmark-duration") {
		r.BenchmarkDurationS = agxBenchmarkDur.Seconds()
	}
	if agxGracePeriod != "" {
		g := math.Inf(1)
		if agxGracePeriod != "inf" && agxGracePeriod != ".inf" {
			d, err := time.ParseDuration(agxGracePeriod)
			if err != nil {
				return nil, fmt.Errorf("--grace-period: %w", err)
			}
			g = d.Seconds()
		}
		r.GracePeriodS = &g
	}
	if agxMaxContextLength != "" {
		var n yaml.Node
		if err := yaml.Unmarshal([]byte(agxMaxContextLength), &n); err != nil || len(n.Content) == 0 {
			return nil, fmt.Errorf("--max-context-length %q", agxMaxContextLength)
		}
		if err := r.MaxContextLength.UnmarshalYAML(n.Content[0]); err != nil {
			return nil, fmt.Errorf("--max-context-length: %w", err)
		}
	}
	if p := r.Validate(); !p.OK() {
		return nil, fmt.Errorf("agentic replay:\n%s", p.Error())
	}
	if r.WarmupMode() != workload.WarmupSynthesized {
		return nil, fmt.Errorf("warmup mode %q is not implemented (only %q)", r.WarmupMode(), workload.WarmupSynthesized)
	}
	return r, nil
}

// buildAgentXRun resolves the descriptor against the deployment, loads the corpus once,
// and builds the driver. It returns the initial requests.
func buildAgentXRun(cmd *cobra.Command) (*agentXRun, []*sim.Request) {
	for _, f := range []string{"trace-header", "trace-data", "session-mode", "concurrent-sessions", "total-sessions", "shuffle-corpus", "think-time-ms", "think-time-dist", "trace-output"} {
		if cmd.Flags().Changed(f) {
			logrus.Fatalf("--%s does not apply to --agentic-scenario (the corpus and the scenario fix the workload)", f)
		}
	}
	if agxCorpusPath == "" {
		logrus.Fatalf("--agentic-scenario requires --corpus <traces.jsonl>")
	}
	spec, err := resolveAgentXSpec(cmd)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	limit, _, err := spec.MaxContextLength.Resolve(int(maxModelLen))
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	t0 := time.Now()
	corpus, err := simworkload.LoadAgentXCorpus(agxCorpusPath, spec.Corpus.SHA256, limit, spec.SplitsFlattenedAgents())
	if err != nil {
		logrus.Fatalf("agentic replay corpus: %v", err)
	}
	logrus.Infof("agentic replay: %d of %d traces kept (max_context_length %d, %d filtered), corpus sha256 %s, loaded in %.1fs",
		len(corpus.Traces), corpus.Rows, limit, corpus.FilteredByMaxCtx, corpus.SHA256, time.Since(t0).Seconds())
	lo, hi := spec.StartWindow()
	grace := spec.Grace()
	graceUs := int64(math.MaxInt64)
	if !math.IsInf(grace, 1) {
		graceUs = int64(math.Round(grace * 1e6))
	}
	drv, err := simworkload.NewAgentXReplay(corpus, simworkload.AgentXReplayConfig{
		Concurrency:      spec.Concurrency,
		Seed:             spec.Seed(),
		DurationUs:       int64(math.Round(spec.Duration() * 1e6)),
		GraceUs:          graceUs,
		IdleCapUs:        int64(math.Round(spec.IdleCap() * 1e6)),
		MinRatio:         lo,
		MaxRatio:         hi,
		RequestTimeoutUs: simworkload.AgentXRequestTimeoutUs,
	})
	if err != nil {
		logrus.Fatalf("agentic replay: %v", err)
	}
	if drv.Lanes() < spec.Concurrency {
		logrus.Warnf("agentic replay: built %d lanes for concurrency %d (too few spawnable traces)", drv.Lanes(), spec.Concurrency)
	}
	return &agentXRun{spec: spec, corpus: corpus, driver: drv, limit: limit}, drv.Start()
}

// summary is the results file's agentic block.
func (a *agentXRun) summary() simworkload.AgentXSummary {
	return a.driver.Summary(a.corpus, a.limit, a.spec.Corpus.SHA256 != "")
}

// fmtUs renders microseconds as seconds for logs.
func fmtUs(us int64) string { return strconv.FormatFloat(float64(us)/1e6, 'f', 3, 64) + "s" }
