package cmd

import (
	"path/filepath"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

func TestPDPeerResultKeepsTokensServedBeforeHorizon(t *testing.T) {
	common := []string{"--scenario", "pd-p2p.yaml", "--scenarios", "../examples/capabilities",
		"--num-instances", "3", "--prefill-instances", "2", "--decode-instances", "1",
		"--seed", "1"}
	args := append(append([]string{"run"}, common...), "--rate", "2", "--num-requests", "1",
		"--output-tokens", "690", "--output-tokens-stdev", "0", "--output-tokens-min", "690", "--output-tokens-max", "690")
	run := func(horizon string, extra ...string) (*result.Result, map[string]any) {
		t.Helper()
		invocation := append(append([]string{}, args...), "--horizon", horizon)
		out, stderr, err := runKernelCLI(t, append(invocation, extra...)...)
		if err != nil {
			t.Fatalf("P2P run at horizon %s: %v\n%s", horizon, err, lastLines(stderr, 5))
		}
		return parseResult(t, out), resultDoc(t, out)
	}
	tracePath := filepath.Join(t.TempDir(), "control")
	control, _ := run("30000000", "--trace-output", tracePath)
	if len(control.RequestRows) != 1 {
		t.Fatalf("control has %d requests, want one", len(control.RequestRows))
	}
	done := control.RequestRows[0]
	const horizon = 1_000_000
	if done.Fate != result.FateCompleted || done.TTFTUs == nil || done.E2EUs == nil ||
		done.ArrivalUs+*done.TTFTUs >= horizon || done.ArrivalUs+*done.E2EUs <= horizon {
		t.Fatalf("control must start delivering tokens before the cutoff and finish after it: %+v", done)
	}

	cut, doc := run("1000000")
	if len(cut.RequestRows) != 1 {
		t.Fatalf("truncated run has %d requests, want one", len(cut.RequestRows))
	}
	partial := cut.RequestRows[0]
	if partial.Fate != result.FateInFlightAtHorizon || partial.OutputTokens <= 0 || partial.OutputTokens >= done.OutputTokens {
		t.Errorf("want an in-flight request with some of its %d output tokens, got %+v", done.OutputTokens, partial)
	}
	if partial.TTFTUs == nil || *partial.TTFTUs != *done.TTFTUs || partial.E2EUs != nil || partial.TPOTUs != nil {
		t.Errorf("want the same first-token time as the control and no completed-only samples, got %+v", partial)
	}
	if writes, ok := resultValue(doc, "breakdowns.pd.p2p_writes"); !ok || writes != float64(1) {
		t.Errorf("peer write must finish before the run ends, got %v", writes)
	}
	if cancelled, ok := resultValue(doc, "reports.engine.cluster.measured_cancelled_requests"); !ok || cancelled != float64(1) {
		t.Errorf("one incomplete request must count once, got %v", cancelled)
	}

	early, _ := run("550000")
	if len(early.RequestRows) != 1 {
		t.Fatalf("early cutoff has %d requests, want one", len(early.RequestRows))
	}
	if row := early.RequestRows[0]; row.OutputTokens != 0 || row.TTFTUs != nil {
		t.Errorf("a cutoff before the first decoder token must carry no output or token time, got %+v", row)
	}

	// The decode step can run before the host has finished processing its first token.
	firstAt := done.ArrivalUs + *done.TTFTUs
	beforeFirst, _ := run(strconv.FormatInt(firstAt-1, 10))
	if row := beforeFirst.RequestRows[0]; row.OutputTokens != 0 || row.TTFTUs != nil {
		t.Errorf("a cutoff immediately before the first token reported output: %+v", row)
	}
	atFirst, _ := run(strconv.FormatInt(firstAt, 10))
	if row := atFirst.RequestRows[0]; row.OutputTokens != 1 || row.TTFTUs == nil || *row.TTFTUs != *done.TTFTUs {
		t.Errorf("a cutoff at the first token should report that token and its time: %+v", row)
	}

	t.Run("client deadline before first token", func(t *testing.T) {
		input, err := workload.LoadTraceV2(tracePath+".yaml", tracePath+".csv")
		if err != nil {
			t.Fatal(err)
		}
		if len(input.Records) != 1 {
			t.Fatalf("expected one request in the control trace, got %d", len(input.Records))
		}
		input.Records[0].DeadlineUs = firstAt - 1
		deadlinePath := filepath.Join(t.TempDir(), "deadline")
		if err := workload.ExportTraceV2(&input.Header, input.Records, deadlinePath+".yaml", deadlinePath+".csv"); err != nil {
			t.Fatal(err)
		}
		replay := append(append([]string{"replay"}, common...),
			"--trace-header", deadlinePath+".yaml", "--trace-data", deadlinePath+".csv", "--horizon", "30000000")
		out, stderr, err := runKernelCLI(t, replay...)
		if err != nil {
			t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 5))
		}
		rows := parseResult(t, out).RequestRows
		if len(rows) != 1 || rows[0].Fate != result.FateTimedOut || rows[0].OutputTokens != 0 || rows[0].TTFTUs != nil {
			t.Errorf("a deadline before the first token should report a timeout with no output: %+v", rows)
		}
	})
}
