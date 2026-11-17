package cmd

// Workload fidelity of a pooled session replay, checked per request (design §8.5, COV-12).
//
// The recorded run these tests stand for is a benchmark client that holds K sessions in flight.
// Each session is a fixed number of rounds; every round re-sends the whole transcript so far
// (accumulate growth) under one system prompt that all sessions share. The client waits a fixed
// think time after a round completes before it sends the next one, and starts the next session in
// corpus order as soon as one ends. The corpus below has that shape at a small size.
//
// A replay of such a run is faithful only when every request it simulates is the one the client
// sent: the same session and round, the same prompt and output lengths, the same shared prefix,
// sent at the time the client's own rule gives from the simulated completions, inside the window
// the run measured. Aggregates can agree while every one of these is wrong, so each check here is
// per request, read from the Result's request rows.
//
// The checks are: session structure, prompt and output lengths, the shared prefix, the time each
// round is sent, the sessions the client holds in flight and the order it starts them, and the
// measured window (#2130).
//
// NOTE: these tests mutate package-level CLI vars and MUST NOT use t.Parallel().

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// fidelityRun is the recorded run: its client rule and its corpus.
type fidelityRun struct {
	concurrent   int     // sessions the client holds in flight
	thinkUs      int64   // client think time after each completed round
	prefixGroup  string  // the system prompt every session starts with
	prefixLength int     // its length in tokens
	newTokens    [][]int // per session, per round: the tokens the round adds to the transcript
	outputTokens [][]int // per session, per round: the forced output length
}

func (f fidelityRun) sessionID(s int) string { return fmt.Sprintf("sess-%02d", s) }

// prompt is the absolute prompt length of round r of session s: the shared prefix, every earlier
// round's new tokens and output, and this round's new tokens.
func (f fidelityRun) prompt(s, r int) int {
	n := f.prefixLength
	for j := 0; j <= r; j++ {
		n += f.newTokens[s][j]
		if j < r {
			n += f.outputTokens[s][j]
		}
	}
	return n
}

// agenticShapeRun has the shape of the recorded agentic cells: an accumulate transcript under a
// shared system prompt, varied per-round lengths, a fixed think time, and more sessions than the
// client holds in flight, so the pool refills several times.
func agenticShapeRun() fidelityRun {
	f := fidelityRun{concurrent: 2, thinkUs: 2_000_000, prefixGroup: "system", prefixLength: 256}
	for s := 0; s < 5; s++ {
		var in, out []int
		for r := 0; r < 4; r++ {
			in = append(in, 120+37*((s*7+r*3)%5))
			out = append(out, 16+9*((s*5+r)%4))
		}
		f.newTokens = append(f.newTokens, in)
		f.outputTokens = append(f.outputTokens, out)
	}
	return f
}

// writeTrace exports the corpus as the converter writes it: the shared prefix on round 0 only, each
// round's new tokens as its input (the accumulate delta), the client's think time on every round
// after the first, and sessions in the order the client started them. Arrival times are the
// recorded ones; a pooled replay re-derives them from simulated completions.
func (f fidelityRun) writeTrace(t *testing.T) (headerPath, dataPath string) {
	t.Helper()
	dir := t.TempDir()
	headerPath, dataPath = filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
	header := &workload.TraceHeader{Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "generated",
		SessionContextGrowth: "accumulate"}
	var recs []workload.TraceRecord
	id := 0
	for s := range f.newTokens {
		for r := range f.newTokens[s] {
			rec := workload.TraceRecord{RequestID: id, SessionID: f.sessionID(s), RoundIndex: r,
				InputTokens: f.newTokens[s][r], OutputTokens: f.outputTokens[s][r],
				ArrivalTimeUs: int64(s*60_000_000 + r*10_000_000), Status: "ok"}
			if r == 0 {
				rec.PrefixGroup, rec.PrefixLength = f.prefixGroup, f.prefixLength
			} else {
				think := f.thinkUs
				rec.ThinkTimeUs = &think
			}
			recs = append(recs, rec)
			id++
		}
	}
	if err := workload.ExportTraceV2(header, recs, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	return headerPath, dataPath
}

// replayPooled replays the corpus with the client's rule: K sessions in flight, recorded think time.
func (f fidelityRun) replayPooled(t *testing.T) *result.Result {
	t.Helper()
	headerPath, dataPath := f.writeTrace(t)
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)
	out := runReplayCaptureStdout(t, []string{
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t),
		"--registry", kernelRegistryDir(t), "--defaults-filepath", defaultsPath,
		"--concurrent-sessions", fmt.Sprint(f.concurrent),
	})
	return parseResult(t, out)
}

// fidelityRows indexes the completed rows by session and round, failing on a duplicate.
func fidelityRows(t *testing.T, r *result.Result) map[string]map[int]result.RequestRow {
	t.Helper()
	got := map[string]map[int]result.RequestRow{}
	for _, row := range r.RequestRows {
		if row.Fate != result.FateCompleted {
			t.Errorf("request %s (session %q round %d) ended %s; every recorded request completed",
				row.ID, row.SessionID, row.RoundIndex, row.Fate)
			continue
		}
		if row.E2EUs == nil {
			t.Fatalf("completed request %s states no E2E", row.ID)
		}
		if got[row.SessionID] == nil {
			got[row.SessionID] = map[int]result.RequestRow{}
		}
		if _, dup := got[row.SessionID][row.RoundIndex]; dup {
			t.Errorf("session %q round %d was simulated twice", row.SessionID, row.RoundIndex)
		}
		got[row.SessionID][row.RoundIndex] = row
	}
	return got
}

// TestReplayFidelity_EveryRecordedRequestOnce_WithItsLengths checks the session structure and the
// lengths: each recorded (session, round) is simulated exactly once, nothing else is, and each
// carries the absolute prompt the client sent and the output length it forced.
func TestReplayFidelity_EveryRecordedRequestOnce_WithItsLengths(t *testing.T) {
	f := agenticShapeRun()
	got := fidelityRows(t, f.replayPooled(t))
	if len(got) != len(f.newTokens) {
		t.Errorf("replay simulated %d sessions, the run recorded %d", len(got), len(f.newTokens))
	}
	for s := range f.newTokens {
		rows := got[f.sessionID(s)]
		if len(rows) != len(f.newTokens[s]) {
			t.Errorf("session %s: %d rounds simulated, %d recorded", f.sessionID(s), len(rows), len(f.newTokens[s]))
		}
		for r := range f.newTokens[s] {
			row, ok := rows[r]
			if !ok {
				t.Errorf("session %s round %d: recorded, not simulated", f.sessionID(s), r)
				continue
			}
			if want := f.prompt(s, r); row.PromptTokens != want {
				t.Errorf("session %s round %d: prompt %d tokens, the client sent %d", f.sessionID(s), r, row.PromptTokens, want)
			}
			if want := f.outputTokens[s][r]; row.OutputTokens != want {
				t.Errorf("session %s round %d: output %d tokens, the client forced %d", f.sessionID(s), r, row.OutputTokens, want)
			}
		}
	}
}

// TestReplayFidelity_SharedSystemPromptIsReusedAcrossSessions checks the shared prefix: every
// session's first round after the first session's is served with the shared system prompt already
// cached, as the recorded server served it, and the first session's first round is not.
func TestReplayFidelity_SharedSystemPromptIsReusedAcrossSessions(t *testing.T) {
	f := agenticShapeRun()
	got := fidelityRows(t, f.replayPooled(t))
	first := got[f.sessionID(0)][0]
	if first.CachedLocalTokens != 0 {
		t.Errorf("the first request found %d tokens cached in an empty cache", first.CachedLocalTokens)
	}
	for s := f.concurrent; s < len(f.newTokens); s++ {
		row := got[f.sessionID(s)][0]
		// A full block of the shared prefix is the least a hit can be; the block size is the
		// scenario's, so ask only for most of the prefix.
		if row.CachedLocalTokens < f.prefixLength/2 {
			t.Errorf("session %s round 0 found %d tokens cached; the %d-token system prompt every session shares was sent before",
				f.sessionID(s), row.CachedLocalTokens, f.prefixLength)
		}
	}
}

// TestReplayFidelity_RoundsFollowCompletionsByTheThinkTime checks the client's think time: a
// follow-up round is sent exactly the recorded think time after the completion the Result
// states for the round before it (arrival plus E2E).
func TestReplayFidelity_RoundsFollowCompletionsByTheThinkTime(t *testing.T) {
	f := agenticShapeRun()
	got := fidelityRows(t, f.replayPooled(t))
	checked := 0
	for s := range f.newTokens {
		for r := 1; r < len(f.newTokens[s]); r++ {
			prev, row := got[f.sessionID(s)][r-1], got[f.sessionID(s)][r]
			if gap := row.ArrivalUs - (prev.ArrivalUs + *prev.E2EUs); gap != f.thinkUs {
				t.Errorf("session %s round %d is sent %d us after the completion the Result states for round %d; the client thinks %d us",
					f.sessionID(s), r, gap, r-1, f.thinkUs)
			}
			checked++
		}
	}
	if checked != 15 {
		t.Fatalf("non-vacuity: %d follow-up rounds checked, want 15", checked)
	}
}

// TestReplayFidelity_ClientHoldsItsSessionsInFlight_InCorpusOrder checks the pool: the client
// starts its first K sessions together, starts the rest in corpus order, each as an earlier
// session ends, and never has more than K sessions in flight.
func TestReplayFidelity_ClientHoldsItsSessionsInFlight_InCorpusOrder(t *testing.T) {
	f := agenticShapeRun()
	got := fidelityRows(t, f.replayPooled(t))
	start := make([]int64, len(f.newTokens))
	end := make([]int64, len(f.newTokens))
	for s := range f.newTokens {
		last := len(f.newTokens[s]) - 1
		start[s] = got[f.sessionID(s)][0].ArrivalUs
		end[s] = got[f.sessionID(s)][last].ArrivalUs + *got[f.sessionID(s)][last].E2EUs
	}
	for s := 1; s < f.concurrent; s++ {
		if start[s] != start[0] {
			t.Errorf("session %d starts at %d us, session 0 at %d: the client starts its first %d sessions together", s, start[s], start[0], f.concurrent)
		}
	}
	for s := f.concurrent; s < len(start); s++ {
		if start[s] < start[s-1] {
			t.Errorf("session %d starts at %d us, before session %d at %d: sessions start in corpus order", s, start[s], s-1, start[s-1])
		}
		freed := false
		for e := 0; e < s; e++ {
			freed = freed || end[e] == start[s]
		}
		if !freed {
			t.Errorf("session %d starts at %d us, when no earlier session ended", s, start[s])
		}
	}
	for s := range start {
		inFlight := 0
		for o := range start {
			if start[o] <= start[s] && start[s] < end[o] {
				inFlight++
			}
		}
		if inFlight > f.concurrent {
			t.Errorf("%d sessions in flight when session %d starts; the client holds %d", inFlight, s, f.concurrent)
		}
	}
	if start[len(start)-1] == start[0] {
		t.Fatalf("non-vacuity: every session started together")
	}
}

// TestReplayFidelity_WindowIsTheRunsWindow checks the measured window of a replay that measures
// from its first request to its last completion: it opens at the first arrival and closes at the
// latest arrival plus E2E among the rows, as the client reads the run's span from its requests.
func TestReplayFidelity_WindowIsTheRunsWindow(t *testing.T) {
	f := agenticShapeRun()
	res := f.replayPooled(t)
	var first, last int64 = -1, 0
	for _, row := range res.RequestRows {
		if first < 0 || row.ArrivalUs < first {
			first = row.ArrivalUs
		}
		if e := row.ArrivalUs + *row.E2EUs; e > last {
			last = e
		}
	}
	if res.Window.StartUs != first || res.Window.EndUs != last {
		t.Errorf("window [%d, %d] us, the requests span [%d, %d]", res.Window.StartUs, res.Window.EndUs, first, last)
	}
}
