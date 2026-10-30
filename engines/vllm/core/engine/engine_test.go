package engine

import (
	"testing"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// max_concurrent_batches for every configuration vLLM v0.32.0 starts with
// (vllm/config/vllm.py:577-588@5df63a72).
func TestMaxConcurrentBatches(t *testing.T) {
	for _, c := range []struct {
		pp        int
		async, v2 bool
		want      int
	}{
		{1, false, false, 1}, {1, false, true, 1}, {2, false, false, 2}, {4, false, true, 4},
		{1, true, false, 2}, {1, true, true, 2}, {2, true, true, 3}, {4, true, true, 5},
	} {
		if got := MaxConcurrentBatches(c.pp, c.async, c.v2); got != c.want {
			t.Errorf("pp %d async %v v2 %v: %d, want %d", c.pp, c.async, c.v2, got, c.want)
		}
	}
}

// The V1 model runner refuses async scheduling with PP; everything else starts.
func TestCheckConfigRefusesWhatVLLMRefuses(t *testing.T) {
	if err := CheckConfig(2, true, false); err == nil {
		t.Error("PP 2 with async scheduling on the V1 runner was accepted")
	}
	for _, c := range []struct {
		pp        int
		async, v2 bool
	}{{1, true, false}, {2, true, true}, {2, false, false}, {1, false, false}} {
		if err := CheckConfig(c.pp, c.async, c.v2); err != nil {
			t.Errorf("pp %d async %v v2 %v refused: %v", c.pp, c.async, c.v2, err)
		}
	}
}

// transferGPU samples each request's next token and reports a scripted transfer outcome with
// the batch it is asked for, recording the order in which Core applied the two.
type transferGPU struct {
	outcome sched.TransferOutcome
	asked   int
}

func (g *transferGPU) Output(b Batch) map[string][]int64 {
	out := map[string][]int64{}
	for _, id := range b.Samples {
		out[id] = []int64{eosToken}
	}
	return out
}

func (g *transferGPU) Transfers(Batch) sched.TransferOutcome {
	g.asked++
	o := g.outcome
	g.outcome = sched.TransferOutcome{}
	return o
}

// recvConnector parks every request with a remote prefill in WAITING_FOR_REMOTE_KVS.
type recvConnector struct{}

func (recvConnector) OnNewRequest(*sched.Request) {}
func (recvConnector) MatchedTokens(r *sched.Request, local int) (int, bool) {
	if r.Transfer != nil && r.Transfer.DoRemotePrefill {
		return len(r.Prompt) - local, true
	}
	return 0, false
}
func (recvConnector) AfterAlloc(r *sched.Request, _ sched.AllocatedBlocks, _ int) {
	if r.Transfer != nil {
		r.Transfer.DoRemotePrefill = false
	}
}
func (recvConnector) Finished(*sched.Request, []int) bool { return false }
func (recvConnector) AfterOutput(sched.TransferOutcome)   {}

// A GPU that is a TransferGPU hands the worker's finished transfers to update_from_output with
// the batch's output (ModelRunnerOutput.kv_connector_output, vllm/v1/outputs.py:215-233@5df63a72),
// and Core.Abort's ids are finished before that output is applied, as EngineCore drains its
// aborts queue after the output and before update_from_output (vllm/v1/engine/core.py:780-783,
// 814-822@5df63a72).
func TestCoreAppliesTransfersAndQueuedAborts(t *testing.T) {
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40,
		MaxModelLen: 128, EnableChunkedPrefill: true, Policy: sched.FCFS, EnablePrefixCaching: true,
		ReserveFullSequence: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetConnector(recvConnector{}, sched.ConnectorConfig{}); err != nil {
		t.Fatal(err)
	}
	prompt := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	for _, id := range []string{"pulled", "local"} {
		r := &sched.Request{ID: id, Prompt: prompt, MaxTokens: 4, EOS: eosToken}
		if id == "pulled" {
			r.Transfer = &sched.TransferParams{DoRemotePrefill: true, HasRemote: true, RemoteBlocks: 2}
			r.Prompt = []int64{11, 12, 13, 14, 15, 16, 17, 18}
		}
		if err := s.Add(r); err != nil {
			t.Fatal(err)
		}
	}
	gpu := &transferGPU{}
	c := &Core{Sched: s, QueueSize: 1, GPU: gpu}
	c.Step()
	if got := s.WaitingForRemoteKV(); len(got) != 1 || got[0] != "pulled" {
		t.Fatalf("setup: waiting for remote KV %v, want [pulled]", got)
	}
	gpu.outcome = sched.TransferOutcome{FinishedRecving: []string{"pulled"}}
	c.Step()
	if gpu.asked != 2 {
		t.Errorf("Transfers asked %d times in two steps, want 2", gpu.asked)
	}
	if got := s.RecvLanded(); len(got) != 1 || got[0] != "pulled" {
		t.Errorf("landed pulls %v, want [pulled]: the transfer outcome was not applied", got)
	}
	c.Abort([]string{"pulled"})
	if s.Request("pulled").Status >= sched.FinishedStopped {
		t.Fatal("Abort finished the request at once; EngineCore queues it for the next output")
	}
	c.Step()
	if r := s.Request("pulled"); r != nil && r.Status < sched.FinishedStopped {
		t.Errorf("the queued abort was not applied with the next output: %v", r.Status)
	}
}
