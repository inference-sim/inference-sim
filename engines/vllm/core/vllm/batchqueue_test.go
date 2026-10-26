package vllm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/engine"
)

// The adapter's batch queue against vLLM v0.32.0's own EngineCore.step_with_batch_queue. Each
// tier-3 scenario with pipeline_parallel_size 1 and async scheduling (max_concurrent_batches 2;
// oracle/scenarios-t3) is driven through engine.Engine in the oracle's loop order
// (oracle/run_enginecore.py): at step i the step's arrivals are added, then one loop iteration
// runs. An iteration is a Schedule; if the batch queue would block, it is the Complete of the
// oldest in-flight batch. The batches it schedules, their tokens and preemptions, and each
// request's finish, are compared with the golden vLLM produced (core/engine/testdata/golden).

type t3Scenario struct {
	Config struct {
		MaxNumSeqs           int    `json:"max_num_seqs"`
		MaxNumBatchedTokens  int    `json:"max_num_batched_tokens"`
		BlockSize            int    `json:"block_size"`
		NumBlocks            int    `json:"num_blocks"`
		MaxModelLen          int    `json:"max_model_len"`
		EnablePrefixCaching  bool   `json:"enable_prefix_caching"`
		SchedulingPolicy     string `json:"scheduling_policy"`
		PipelineParallelSize int    `json:"pipeline_parallel_size"`
		AsyncScheduling      bool   `json:"async_scheduling"`
		UseV2ModelRunner     bool   `json:"use_v2_model_runner"`
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

type t3Golden struct {
	// QueueLen is the batch queue's length after each loop step (the golden's steps[].queue_len).
	QueueLen []int `json:"-"`
	Steps    []struct {
		QueueLen int  `json:"queue_len"`
		Idle     bool `json:"idle"`
	} `json:"steps"`
	Batches []struct {
		Batch     int            `json:"batch"`
		Scheduled map[string]int `json:"scheduled"`
		Preempted []string       `json:"preempted"`
	} `json:"batches"`
	Finished map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
	} `json:"finished"`
}

// batchQueueGoldens are the tier-3 scenarios a pp-1 async engine runs whose KV pool holds one
// max_model_len sequence. t3-async-preempt and t3-async-prefix-priority are not listed: their
// pools are smaller, which vLLM refuses at startup and the factory refuses too, while the tier-3
// oracle builds EngineCore directly and skips that check. Their goldens cannot be run at a
// smaller max_model_len either, because the scheduler reads it in decisions
// (core/sched/scheduler.go:446, 595, 725, 835). Their batch-queue behaviour is diff-tested in
// core/engine, whose Core this adapter drives.
var batchQueueGoldens = []string{"t3-async", "t3-async-maxtokens"}

func TestBatchQueueAgainstVLLM(t *testing.T) {
	for _, name := range batchQueueGoldens {
		t.Run(name, func(t *testing.T) {
			var sc t3Scenario
			readT3(t, filepath.Join("..", "..", "oracle", "scenarios-t3", name+".json"), &sc)
			var want t3Golden
			readT3(t, filepath.Join("..", "engine", "testdata", "golden", name+".json"), &want)
			got := driveBatchQueue(t, sc)
			if d := batchQueueDivergence(want, got); d != "" {
				t.Errorf("earliest divergence from vLLM v0.32.0's step_with_batch_queue: %s", d)
			}
		})
	}
}

func readT3(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

// driveBatchQueue runs sc through the adapter as oracle/run_enginecore.py runs vLLM.
func driveBatchQueue(t *testing.T, sc t3Scenario) t3Golden {
	t.Helper()
	c := sc.Config
	policy := c.SchedulingPolicy
	if policy == "" {
		policy = "fcfs"
	}
	src := fmt.Sprintf("block_size: %d\nmax_num_seqs: %d\nmax_num_batched_tokens: %d\nmax_model_len: %d\n"+
		"num_gpu_blocks: %d\nuse_v2_model_runner: %t\nasync_scheduling: %t\npipeline_parallel_size: %d\n"+
		"scheduling_policy: %s\nenable_prefix_caching: %t\n",
		c.BlockSize, c.MaxNumSeqs, c.MaxNumBatchedTokens, c.MaxModelLen, c.NumBlocks, c.UseV2ModelRunner,
		c.AsyncScheduling, max(c.PipelineParallelSize, 1), policy, c.EnablePrefixCaching)
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(src), &node); err != nil {
		t.Fatal(err)
	}
	e, err := Factory{Pricer: fixed}.New(engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *node.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	reqs := append(sc.Requests[:0:0], sc.Requests...)
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	var got t3Golden
	got.Finished = map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
	}{}
	var inFlight []engine.BatchID
	next, maxSteps := 0, sc.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10000
	}
	for step := 0; step < maxSteps; step++ {
		now := engine.Time(step)
		for next < len(reqs) && reqs[next].ArrivalStep <= step {
			r := reqs[next]
			er := engine.Request{ID: r.ID, Priority: r.Priority, MaxOutputTokens: r.MaxTokens,
				ArrivalTime: engine.Time(r.ArrivalStep * 1e6)}
			for _, p := range r.Prompt {
				er.Prompt = append(er.Prompt, int32(p))
			}
			for _, o := range r.Output {
				er.Output = append(er.Output, int32(o))
			}
			if err := e.Add(er, now); err != nil {
				t.Fatal(err)
			}
			next++
		}
		if len(got.Finished) == len(reqs) && e.(*Engine).core.InFlight() == 0 && next == len(reqs) {
			// has_work is false (vllm/v1/engine/core.py:1534-1540@5df63a72): the oracle stops.
			break
		}
		// One loop step, as step_with_batch_queue runs it (vllm/v1/engine/core.py:698-810@5df63a72):
		// schedule a batch if the scheduler has requests; then, if the step blocks (the queue is
		// full, or nothing more can be scheduled), complete the oldest batch in the same step.
		// Whether it blocks is the adapter's loop state, read here because the golden steps the
		// loop itself; a simulator instead calls Schedule until it schedules nothing, and
		// completes batches when their step time has passed.
		b, err := e.Schedule(now)
		if err != nil && err != engine.ErrNoCapacity {
			t.Fatal(err)
		}
		if err == nil && !b.Empty() {
			inFlight = append(inFlight, b.ID)
			sched := map[string]int{}
			for _, s := range b.Reqs {
				sched[s.ID] = s.Tokens
			}
			pre := append([]string{}, b.Preempted...)
			sort.Strings(pre)
			got.Batches = append(got.Batches, struct {
				Batch     int            `json:"batch"`
				Scheduled map[string]int `json:"scheduled"`
				Preempted []string       `json:"preempted"`
			}{len(got.Batches), sched, pre})
		}
		eng := e.(*Engine)
		if eng.blocked && len(inFlight) > 0 {
			res, err := e.Complete(inFlight[0], engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			inFlight = inFlight[1:]
			for _, f := range res.Finished {
				got.Finished[f.ID] = struct {
					Reason       string `json:"reason"`
					OutputTokens int    `json:"output_tokens"`
				}{map[engine.FinishReason]string{engine.FinishStop: "stop", engine.FinishLength: "length"}[f.Reason], f.OutputTokens}
			}
		}
		// vLLM's queue also holds the schedules that ran no tokens: queued, never in flight
		// through the contract (core/vllm.Engine.queuedEmpty).
		got.QueueLen = append(got.QueueLen, e.(*Engine).core.InFlight())
	}
	return got
}

func batchQueueDivergence(want, got t3Golden) string {
	var wantRun []map[string]int
	var wantPre [][]string
	for _, b := range want.Batches {
		if sumTokens(b.Scheduled) == 0 {
			continue // no forward pass: not in flight through the contract
		}
		wantRun = append(wantRun, b.Scheduled)
		pre := append([]string{}, b.Preempted...)
		sort.Strings(pre)
		wantPre = append(wantPre, pre)
	}
	for i := 0; i < max(len(wantRun), len(got.Batches)); i++ {
		switch {
		case i >= len(wantRun):
			return fmt.Sprintf("batch %d: the adapter ran a batch vLLM did not (%v)", i, got.Batches[i].Scheduled)
		case i >= len(got.Batches):
			return fmt.Sprintf("batch %d: vLLM ran a batch the adapter did not (%v)", i, wantRun[i])
		case !reflect.DeepEqual(wantRun[i], got.Batches[i].Scheduled):
			return fmt.Sprintf("batch %d, scheduled: vLLM %v, adapter %v", i, wantRun[i], got.Batches[i].Scheduled)
		case !reflect.DeepEqual(nonEmpty(wantPre[i]), nonEmpty(got.Batches[i].Preempted)):
			return fmt.Sprintf("batch %d, preempted: vLLM %v, adapter %v", i, wantPre[i], got.Batches[i].Preempted)
		}
	}
	// The batch queue's length after every loop step: a run that kept one batch in flight, or
	// blocked where vLLM does not, makes the same batches here only by luck, and this tells.
	var wantQ []int
	for _, st := range want.Steps {
		if !st.Idle {
			wantQ = append(wantQ, st.QueueLen)
		}
	}
	for i := 0; i < max(len(wantQ), len(got.QueueLen)); i++ {
		if i >= len(wantQ) || i >= len(got.QueueLen) || wantQ[i] != got.QueueLen[i] {
			return fmt.Sprintf("step %d, queue length: vLLM %v, adapter %v", i, wantQ, got.QueueLen)
		}
	}
	for id, w := range want.Finished {
		g, ok := got.Finished[id]
		if !ok {
			return fmt.Sprintf("request %s: finished in vLLM (%s, %d tokens), not by the adapter", id, w.Reason, w.OutputTokens)
		}
		if g.OutputTokens != w.OutputTokens || (w.Reason != "" && "FinishReason."+upper(g.Reason) != w.Reason && g.Reason != w.Reason) {
			return fmt.Sprintf("request %s: vLLM finished it %s with %d tokens, the adapter %s with %d", id, w.Reason, w.OutputTokens, g.Reason, g.OutputTokens)
		}
	}
	return ""
}

func sumTokens(m map[string]int) int {
	n := 0
	for _, v := range m {
		n += v
	}
	return n
}

func nonEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func upper(s string) string {
	out := []byte(s)
	for i, c := range out {
		if c >= 'a' && c <= 'z' {
			out[i] = c - 32
		}
	}
	return string(out)
}
