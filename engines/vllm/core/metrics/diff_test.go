package metrics

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// eosToken is tests/v1/core/utils.py:43@5df63a72, which the GPU mock samples last.
const eosToken = 50256

type scenario struct {
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       bool   `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

// metricsTrace is oracle/run_metrics.py's output for one scenario.
type metricsTrace struct {
	Provenance struct {
		Tier           int    `json:"tier"`
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
	Steps   int                  `json:"steps"`
	Series  map[string]float64   `json:"series"`
	PerStep []map[string]float64 `json:"per_step"`
}

// checkProvenance refuses a trace that was not produced from this exact scenario file by
// the pinned vLLM release under the current GPU mock (as core/sched's diff test does).
func checkProvenance(t *testing.T, scenarioPath string, tr metricsTrace) {
	t.Helper()
	raw, err := os.ReadFile(scenarioPath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	var mocks struct {
		Version string `json:"version"`
	}
	readJSON(t, "../../oracle/mocks.json", &mocks)
	p := tr.Provenance
	for _, c := range []struct{ field, got, want string }{
		{"scenario_sha256", p.ScenarioSHA256, hex.EncodeToString(sum[:])},
		{"vllm_version", strings.SplitN(p.VLLMVersion, "+", 2)[0], "0.32.0"},
		{"vllm_commit", p.VLLMCommit, "5df63a7203cdb9328d824a27b682e77c7b9928be"},
		{"mock_version", p.MockVersion, mocks.Version},
	} {
		if c.got != c.want {
			t.Fatalf("metrics trace for %s records %s %q, want %q: regenerate it with oracle/run_metrics.py",
				filepath.Base(scenarioPath), c.field, c.got, c.want)
		}
	}
	if p.Tier != 5 {
		t.Fatalf("metrics trace for %s records tier %d, want 5", filepath.Base(scenarioPath), p.Tier)
	}
}

// manifestFamilies returns every family in the manifest by name, with whether it is exact:
// the scheduler decides its value.
func manifestFamilies(t *testing.T) map[string]bool {
	t.Helper()
	var m struct {
		Families []struct{ Name, Class string } `json:"families"`
	}
	readJSON(t, "../../docs/metrics-manifest.json", &m)
	all := map[string]bool{}
	for _, f := range m.Families {
		all[f.Name] = f.Class == "exact"
	}
	return all
}

// histogramBounds reads the bucket bounds vLLM registered (docs/metrics-buckets.json),
// for the request-token histograms recomputed for a scenario's max_model_len.
func histogramBounds(t *testing.T, maxModelLen int) []Bucketed {
	t.Helper()
	var b map[string][]float64
	readJSON(t, "../../docs/metrics-buckets.json", &b)
	var out []Bucketed
	for _, name := range []string{"vllm:iteration_tokens_total", "vllm:request_params_n", "vllm:request_num_preemptions"} {
		out = append(out, Bucketed{Family: name, Bounds: b[name]})
	}
	// request_tokens: a 1-2-5 series capped at max_model_len (vllm/v1/metrics/buckets.py:172-203,
	// 223-228@5df63a72).
	tokens := build125(maxModelLen)
	for _, name := range []string{"vllm:request_prompt_tokens", "vllm:request_generation_tokens",
		"vllm:request_max_num_generation_tokens", "vllm:request_params_max_tokens",
		"vllm:request_prefill_kv_computed_tokens"} {
		out = append(out, Bucketed{Family: name, Bounds: tokens})
	}
	return out
}

func build125(max int) []float64 {
	var out []float64
	for scale := 1; ; scale *= 10 {
		for _, m := range []int{1, 2, 5} {
			if m*scale > max {
				return out
			}
			out = append(out, float64(m*scale))
		}
	}
}

// drive runs a scenario through core/sched exactly as oracle/run_metrics.py runs vLLM, and
// feeds every step to the metrics model, calling after (when non-nil) once per step.
func drive(t *testing.T, sc scenario, after func(step int, m *Model)) *Model {
	t.Helper()
	c := sc.Config
	maxLen := c.MaxModelLen
	if maxLen == 0 {
		maxLen = c.MaxNumBatchedTokens
	}
	s, err := sched.New(sched.Config{
		MaxNumSeqs: c.MaxNumSeqs, MaxNumBatchedTokens: c.MaxNumBatchedTokens, BlockSize: c.BlockSize,
		NumBlocks: c.NumBlocks, MaxModelLen: maxLen, LongPrefillTokenThreshold: c.LongPrefillTokenThreshold,
		EnableChunkedPrefill: true, Policy: sched.Policy(c.SchedulingPolicy), EnablePrefixCaching: c.EnablePrefixCaching,
		ReserveFullSequence: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	m := New(histogramBounds(t, maxLen))
	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	outputs := map[string][]int64{}
	maxTokens := map[string]int{}
	sampled := map[string]int{}
	generated := map[string]int{}
	next := 0
	maxSteps := sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	for i := 0; i < maxSteps; i++ {
		for next < len(reqs) && reqs[next].ArrivalStep <= i {
			r := reqs[next]
			mt := r.MaxTokens
			if mt == 0 {
				mt = 1 << 20
			}
			if err := s.Add(&sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalStep),
				Prompt: r.Prompt, MaxTokens: mt, EOS: eosToken}); err != nil {
				t.Fatal(err)
			}
			outputs[r.ID] = r.Output
			maxTokens[r.ID] = mt
			next++
		}
		if !s.HasUnfinished() && next == len(reqs) {
			break
		}
		out := s.Schedule()
		step := Step{Scheduled: out.Scheduled, Preempted: out.Preempted, Emitted: map[string]int{}}
		for _, id := range out.NewIDs {
			r := s.Request(id)
			hit, looked := out.PrefixHit[id]
			if !looked {
				continue
			}
			step.Admitted = append(step.Admitted, Admission{ID: id, PromptTokens: r.NumTokens(),
				LocalHit: hit, Preempted: r.NumPreemptions > 0, CacheLookup: c.EnablePrefixCaching})
		}
		mock := map[string][]int64{}
		for id := range out.Scheduled {
			if s.Request(id).IsPrefillChunk {
				continue
			}
			k, o := sampled[id], outputs[id]
			if k == len(o)-1 {
				mock[id] = []int64{eosToken}
			} else {
				mock[id] = []int64{o[k]}
			}
			if generated[id] == 0 {
				step.FirstTokens = append(step.FirstTokens, id)
			}
			sampled[id] = k + 1
			generated[id]++
			step.Emitted[id] = 1
		}
		prompts := map[string]int{}
		preemptions := map[string]int{}
		for id := range out.Scheduled {
			r := s.Request(id)
			prompts[id] = len(r.Prompt)
			preemptions[id] = r.NumPreemptions
		}
		for _, f := range s.Update(out, mock) {
			reason := map[sched.Status]string{sched.FinishedStopped: "stop", sched.FinishedLength: "length"}[f.Status]
			step.Finished = append(step.Finished, Finish{ID: f.ID, Reason: reason, PromptTokens: prompts[f.ID],
				GenerationTokens: generated[f.ID], Preemptions: preemptions[f.ID], MaxTokens: maxTokens[f.ID]})
		}
		nr, nw := s.Counts()
		step.Running, step.Waiting, step.KVUsage = nr, nw, s.KV.Pool.Usage()
		m.Step(step)
		if after != nil {
			after(i, m)
		}
	}
	return m
}

// TestMetricsAgainstVLLM compares, for every tier-2 scenario, each sample of the exact
// families with what vLLM v0.32.0's own PrometheusStatLogger recorded for the same steps
// (testdata/golden/<name>.metrics.json, written by oracle/run_metrics.py). The target is
// zero mismatches.
func TestMetricsAgainstVLLM(t *testing.T) {
	families := manifestFamilies(t)
	goldens, err := filepath.Glob("testdata/golden/*.metrics.json")
	if err != nil || len(goldens) == 0 {
		t.Fatalf("no metric goldens (%v): generate them with oracle/run_metrics.py (docs/testing.md)", err)
	}
	compared := 0
	for _, g := range goldens {
		name := strings.TrimSuffix(filepath.Base(g), ".metrics.json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, filepath.Join("../../oracle/scenarios", name+".json"), &sc)
			var want metricsTrace
			readJSON(t, g, &want)
			checkProvenance(t, filepath.Join("../../oracle/scenarios", name+".json"), want)
			if len(want.PerStep) != want.Steps {
				t.Fatalf("the golden has %d per-step records for %d steps", len(want.PerStep), want.Steps)
			}
			// Step by step: every gauge and counter sample of an exact family, after each step,
			// with the earliest divergence reported (#2021 B8).
			steps := 0
			first := ""
			got := drive(t, sc, func(i int, m *Model) {
				steps++
				if first != "" || i >= len(want.PerStep) {
					return
				}
				now := m.Series()
				for _, sample := range sortedSamples(want.PerStep[i]) {
					fam, known := familyOf(sample, families)
					if !known || !families[fam] {
						continue
					}
					compared++
					if g := now[sample]; g != want.PerStep[i][sample] {
						first = fmt.Sprintf("step %d, %s: vLLM %v, blis-vllm %v", i, sample, want.PerStep[i][sample], g)
						return
					}
				}
			}).Series()
			if first != "" {
				t.Errorf("earliest divergence from vLLM v0.32.0: %s", first)
			}
			if steps != want.Steps {
				t.Errorf("blis-vllm ran %d steps, vLLM %d", steps, want.Steps)
			}
			for sample, w := range want.Series {
				fam, known := familyOf(sample, families)
				if !known {
					t.Errorf("%s: vLLM emits a sample of no family in the manifest", sample)
					continue
				}
				if !families[fam] {
					continue
				}
				g, ok := got[sample]
				if !ok {
					t.Errorf("%s: vLLM %v, blis-vllm emits no such sample", sample, w)
					continue
				}
				compared++
				if g != w {
					t.Errorf("%s: vLLM %v, blis-vllm %v", sample, w, g)
				}
			}
		})
	}
	t.Logf("%d exact samples compared across %d scenarios", compared, len(goldens))
}

// TestMetricsFuzzCorpusAgainstVLLM runs the seeded fuzz corpus (#2021 B8) through the
// metrics comparison: BLIS_VLLM_METRICS_FUZZ_DIR names a directory of oracle/gen_fuzz.py
// scenarios with the series vLLM v0.32.0's logger recorded for each (<name>.metrics.json,
// oracle/run_metrics.py in directory mode). CI generates both from the tier-2 seeds.
func TestMetricsFuzzCorpusAgainstVLLM(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_METRICS_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_METRICS_FUZZ_DIR is unset: the metrics fuzz corpus was NOT checked (CI's oracle job sets it)")
	}
	families := manifestFamilies(t)
	goldens, err := filepath.Glob(filepath.Join(dir, "*.metrics.json"))
	if err != nil || len(goldens) == 0 {
		t.Fatalf("no metric traces in %s (%v)", dir, err)
	}
	mismatches, compared := 0, 0
	for _, g := range goldens {
		name := strings.TrimSuffix(filepath.Base(g), ".metrics.json")
		var sc scenario
		readJSON(t, filepath.Join(dir, name+".json"), &sc)
		var want metricsTrace
		readJSON(t, g, &want)
		checkProvenance(t, filepath.Join(dir, name+".json"), want)
		first := ""
		got := drive(t, sc, func(i int, m *Model) {
			if first != "" || i >= len(want.PerStep) {
				return
			}
			now := m.Series()
			for _, sample := range sortedSamples(want.PerStep[i]) {
				if fam, known := familyOf(sample, families); !known || !families[fam] {
					continue
				}
				compared++
				if now[sample] != want.PerStep[i][sample] {
					first = fmt.Sprintf("step %d, %s: vLLM %v, blis-vllm %v", i, sample, want.PerStep[i][sample], now[sample])
					return
				}
			}
		}).Series()
		for _, sample := range sortedSamples(want.Series) {
			if first != "" {
				break
			}
			if fam, known := familyOf(sample, families); !known || !families[fam] {
				continue
			}
			compared++
			if got[sample] != want.Series[sample] {
				first = fmt.Sprintf("final %s: vLLM %v, blis-vllm %v", sample, want.Series[sample], got[sample])
			}
		}
		if first != "" {
			mismatches++
			t.Errorf("%s: earliest divergence from vLLM v0.32.0: %s", name, first)
		}
	}
	t.Logf("metrics fuzz corpus: %d scenarios, %d samples compared, %d mismatches", len(goldens), compared, mismatches)
}

// familyOf maps a sample to the family the manifest names: a histogram sample drops its
// _bucket, _count or _sum; a counter's sample carries a _total that the family name has
// only when its definition spelled it (prometheus_client strips one _total and adds one
// back).
func familyOf(sample string, families map[string]bool) (string, bool) {
	name := sample
	if i := strings.IndexByte(name, '{'); i >= 0 {
		name = name[:i]
	}
	for _, suffix := range []string{"_bucket", "_count", "_sum"} {
		if base := strings.TrimSuffix(name, suffix); base != name {
			if _, ok := families[base]; ok {
				return base, true
			}
		}
	}
	if _, ok := families[name]; ok {
		return name, true
	}
	base := strings.TrimSuffix(name, "_total")
	_, ok := families[base]
	return base, ok
}

func sortedSamples(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("%s: %v", path, fmt.Errorf("%w", err))
	}
}
