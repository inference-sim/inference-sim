package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// transferLogEnv names the file a child run of the CLI appends every transfer question to.
const transferLogEnv = "BLIS_TEST_TRANSFER_LOG"

// loggingEnvironment answers as the host's Environment and records each move an engine states.
type loggingEnvironment struct {
	env.Environment
	path string
}

type loggedMove struct {
	Kind   string `json:"kind"`
	From   int    `json:"from_node"`
	To     int    `json:"to_node"`
	Tokens int64  `json:"tokens"`
}

func (l loggingEnvironment) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	line, _ := json.Marshal(loggedMove{Kind: string(t.Kind), From: t.From.Node, To: t.To.Node, Tokens: int64(t.Tokens)})
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err == nil {
		_, _ = f.Write(append(line, '\n'))
		_ = f.Close()
	}
	return l.Environment.TransferDemand(p, t)
}

// installTransferLog is called by the CLI child leg (TestKernelCLILeg) when the test asks for the
// log.
func installTransferLog() {
	if path := os.Getenv(transferLogEnv); path != "" {
		wrapEnvironment = func(e env.Environment) env.Environment { return loggingEnvironment{Environment: e, path: path} }
	}
}

// runLogged runs the CLI as runKernelCLI does and returns the moves the engines' hosts were asked
// to price.
func runLogged(t *testing.T, args ...string) (stdout string, moves []loggedMove) {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	log := filepath.Join(t.TempDir(), "moves.jsonl")
	full := append(slices.Clone(args), "--catalog", catalog, "--registry", registry, "--defaults-filepath", "../defaults.yaml")
	if !slices.Contains(args, "--scenarios") {
		full = append(full, "--scenarios", scenarios)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelCLILeg$")
	cmd.Env = append(os.Environ(), kernelCLIArgsEnv+"="+strings.Join(full, "\x1f"), transferLogEnv+"="+log)
	var out, errBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errBuf
	if err := cmd.Run(); err != nil {
		t.Fatalf("blis %s: %v\n%s", strings.Join(args, " "), err, lastLines(errBuf.String(), 5))
	}
	raw, err := os.ReadFile(log)
	if err != nil {
		return out.String(), nil
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var m loggedMove
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("moves log: %v", err)
		}
		moves = append(moves, m)
	}
	return out.String(), moves
}

// A disaggregated run and its replay turn every KV pull into a demand through the host's
// Environment, from the move the decode engine states, and from nothing else (design 6.3): the
// questions the Environment is asked are the run's transfers, one for each disaggregated request,
// from the node of the prefill instance that held the KV to the node of the decode instance, and
// the replay asks the same questions. A simulator that priced a pull itself (a handoff computed
// from the instances, as the retired PDTransferDemand did) asks none; one that reconstructed the
// peer (the prefill instance's placement) in place of the engine's request states one origin for
// every pull, which three prefill instances on three nodes do not.
func TestTransferBoundary_RunAndReplayPriceEveryPullThroughTheEnvironmentWithTheEnginesPeer(t *testing.T) {
	prefix := filepath.Join(t.TempDir(), "trace")
	common := []string{"--scenarios", pdScenarios, "--scenario", "glm-5-h200-3p1d-ib.yaml", "--num-instances", "4",
		"--prefill-instances", "3", "--decode-instances", "1", "--seed", "11", "--horizon", "900000000"}
	load := burst(18, 12, 1500, 32)
	runOut, runMoves := runLogged(t, append(append([]string{"run", "--trace-output", prefix}, load...), common...)...)
	repOut, repMoves := runLogged(t, append([]string{"replay", "--trace-header", prefix + ".yaml", "--trace-data", prefix + ".csv"}, common...)...)

	pulls := func(ms []loggedMove) (out []loggedMove) {
		for _, m := range ms {
			if m.Kind == string(plan.TransferPeer) {
				out = append(out, m)
			}
		}
		return out
	}
	for name, c := range map[string]struct {
		stdout string
		moves  []loggedMove
	}{"run": {runOut, runMoves}, "replay": {repOut, repMoves}} {
		p := pulls(c.moves)
		transfers, _ := resultDoc(t, c.stdout)["breakdowns"].(map[string]any)["pd"].(map[string]any)["kv_transfers"].(float64)
		if transfers < 3 || len(p) != int(transfers) {
			t.Errorf("%s: %d pulls were priced through the Environment, the Result counts %v transfers", name, len(p), transfers)
		}
		origins := map[int]bool{}
		for _, m := range p {
			origins[m.From] = true
			if m.Tokens <= 0 {
				t.Errorf("%s: a pull of %d tokens", name, m.Tokens)
			}
		}
		if len(origins) < 2 {
			t.Errorf("%s: every pull reads from node set %v: the engine's peer is not what places the move", name, origins)
		}
	}
	key := func(ms []loggedMove) []string {
		var ks []string
		for _, m := range pulls(ms) {
			b, _ := json.Marshal(m)
			ks = append(ks, string(b))
		}
		slices.Sort(ks)
		return ks
	}
	if a, b := key(runMoves), key(repMoves); !slices.Equal(a, b) {
		t.Errorf("the replay priced different pulls than the run\nrun:    %v\nreplay: %v", a, b)
	}
}
