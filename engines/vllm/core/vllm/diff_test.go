package vllm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
)

// The adapter against vLLM v0.32.0 itself. Each tier-2 scenario whose requests all arrive at
// once, and whose configuration the factory builds, is driven through the engine.Engine
// contract (Add, then Schedule and Complete one batch at a time) and compared step by step
// with the trace vLLM's own Scheduler produced for it (core/sched/testdata/golden): the tokens
// each step scheduled, the preemptions and the finishes. The goldens come from
// oracle/run_scheduler.py, which samples exactly what this adapter samples: each request's
// assigned output, the last token as EOS. Step durations are fixed; only order matters here.

type oracleScenario struct {
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       bool   `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
		MaxNumActiveSeqs          int    `json:"max_num_active_seqs"`
	} `json:"config"`
	Requests []struct {
		ID          string  `json:"id"`
		ArrivalStep int     `json:"arrival_step"`
		Prompt      []int64 `json:"prompt"`
		Output      []int64 `json:"output"`
		MaxTokens   int     `json:"max_tokens"`
		Priority    int     `json:"priority"`
	} `json:"requests"`
}

type oracleTrace struct {
	Provenance struct {
		VLLMVersion    string `json:"vllm_version"`
		VLLMCommit     string `json:"vllm_commit"`
		ScenarioSHA256 string `json:"scenario_sha256"`
		MockVersion    string `json:"mock_version"`
	} `json:"provenance"`
	Steps []struct {
		Scheduled map[string]int `json:"scheduled"`
		Preempted []string       `json:"preempted"`
		Finished  []string       `json:"finished"`
	} `json:"steps"`
	Finished map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
	} `json:"finished"`
}

// adapterGoldens are the tier-2 scenarios the adapter is diff-tested on: every request arrives
// at step 0 (the adapter receives arrivals through Add, not by step), no feature the factory
// does not build (hybrid KV groups, LoRA, MTP or draft_model drafters), and the oracle's
// unlimited max_tokens is not used (an engine.Request's unset budget is vLLM's default,
// max_model_len minus the prompt, which the oracle's 1<<20 equals only when the output ends
// first: true for every scenario listed, since each request's assigned output ends with EOS
// before the window). issue-2021-b8 and preempt-fcfs are not listed: their KV pools cannot hold
// one max_model_len sequence, which vLLM refuses at startup and the factory refuses too, while
// the tier-2 oracle builds the scheduler directly and skips that check; core/sched diff-tests them.
var adapterGoldens = []string{
	"knob-max-active-seqs", "length-caps", "meta-more-kv-large", "meta-more-kv-small", "model-len-boundary",
}

func TestAdapterAgainstVLLM(t *testing.T) {
	var mocks struct {
		Version string `json:"version"`
	}
	rawMocks, err := os.ReadFile(filepath.Join("..", "..", "oracle", "mocks.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rawMocks, &mocks); err != nil || mocks.Version == "" {
		t.Fatalf("oracle/mocks.json states no version: %v", err)
	}
	mockVersion := mocks.Version
	for _, name := range adapterGoldens {
		t.Run(name, func(t *testing.T) {
			scPath := filepath.Join("..", "..", "oracle", "scenarios", name+".json")
			raw, err := os.ReadFile(scPath)
			if err != nil {
				t.Fatal(err)
			}
			var sc oracleScenario
			if err := json.Unmarshal(raw, &sc); err != nil {
				t.Fatal(err)
			}
			var want oracleTrace
			g, err := os.ReadFile(filepath.Join("..", "sched", "testdata", "golden", name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(g, &want); err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(raw)
			if want.Provenance.ScenarioSHA256 != hex.EncodeToString(sum[:]) ||
				want.Provenance.VLLMCommit != "5df63a7203cdb9328d824a27b682e77c7b9928be" ||
				strings.SplitN(want.Provenance.VLLMVersion, "+", 2)[0] != Version ||
				want.Provenance.MockVersion != mockVersion {
				t.Fatalf("golden provenance %+v does not match this scenario, vLLM %s and mock %s",
					want.Provenance, Version, mockVersion)
			}
			if d := adapterDivergence(t, sc, want); d != "" {
				t.Errorf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}

// adapterDivergence drives sc through the adapter and names the first difference from want.
func adapterDivergence(t *testing.T, sc oracleScenario, want oracleTrace) string {
	t.Helper()
	c := sc.Config
	maxLen := c.MaxModelLen
	if maxLen == 0 {
		maxLen = c.MaxNumBatchedTokens
	}
	policy := c.SchedulingPolicy
	if policy == "" {
		policy = "fcfs"
	}
	src := fmt.Sprintf("block_size: %d\nmax_num_seqs: %d\nmax_num_batched_tokens: %d\nmax_model_len: %d\n"+
		"num_gpu_blocks: %d\nuse_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\nscheduling_policy: %s\n"+
		"enable_prefix_caching: %t\nlong_prefill_token_threshold: %d\nmax_num_active_seqs: %d\n",
		c.BlockSize, c.MaxNumSeqs, c.MaxNumBatchedTokens, maxLen, c.NumBlocks, policy,
		c.EnablePrefixCaching, c.LongPrefillTokenThreshold, c.MaxNumActiveSeqs)
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(src), &node); err != nil {
		t.Fatal(err)
	}
	e, err := Factory{Pricer: fixed}.New(engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *node.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	reqs := append(sc.Requests[:0:0], sc.Requests...)
	sort.SliceStable(reqs, func(i, j int) bool { return reqs[i].ID < reqs[j].ID })
	for _, r := range reqs {
		er := engine.Request{ID: r.ID, Priority: r.Priority, MaxOutputTokens: r.MaxTokens}
		for _, p := range r.Prompt {
			er.Prompt = append(er.Prompt, int32(p))
		}
		for _, o := range r.Output {
			er.Output = append(er.Output, int32(o))
		}
		if err := e.Add(er, 0); err != nil {
			t.Fatal(err)
		}
	}
	finished := map[string]engine.Finished{}
	now := engine.Time(0)
	for i := 0; i < len(want.Steps)+5; i++ {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]int{}
		for _, s := range b.Reqs {
			got[s.ID] = s.Tokens
		}
		var fins []string
		if !b.Empty() {
			now += e.StepTime(b)
			res, err := e.Complete(b.ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range res.Finished {
				fins = append(fins, f.ID)
				finished[f.ID] = f
			}
		}
		if i >= len(want.Steps) {
			if len(got) > 0 {
				return fmt.Sprintf("step %d: the adapter scheduled %v after vLLM's last step", i, got)
			}
			break
		}
		w := want.Steps[i]
		sort.Strings(fins)
		pre := append([]string(nil), b.Preempted...)
		sort.Strings(pre)
		wpre := append([]string(nil), w.Preempted...)
		sort.Strings(wpre)
		wfin := append([]string(nil), w.Finished...)
		sort.Strings(wfin)
		for _, f := range []struct {
			name string
			w, g any
		}{
			{"scheduled", nonNilMap(w.Scheduled), got},
			{"preempted", nonNilSlice(wpre), nonNilSlice(pre)},
			{"finished", nonNilSlice(wfin), nonNilSlice(fins)},
		} {
			if !reflect.DeepEqual(f.w, f.g) {
				return fmt.Sprintf("step %d, field %s: vLLM %v, adapter %v", i, f.name, f.w, f.g)
			}
		}
	}
	for id, w := range want.Finished {
		g, ok := finished[id]
		if !ok || g.OutputTokens != w.OutputTokens || string(g.Reason) != w.Reason {
			return fmt.Sprintf("finish of %s: vLLM %+v, adapter %+v", id, w, g)
		}
	}
	return ""
}

func nonNilMap(m map[string]int) map[string]int {
	if m == nil {
		return map[string]int{}
	}
	return m
}

func nonNilSlice(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
