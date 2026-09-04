package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The gate's input domains. Tests cross-product these so a new decision row cannot be
// added without a case covering every combination it could capture.
var (
	allCIStatus = []string{"success", "failure", "unknown"}
	allPlanGate = []string{"pass", "absent", "regression", "conflicts", "unverified"}
	allVerdicts = []string{"GREEN", "NOT-GREEN", "MISSING"}
	// `open` means a correction dismissed a finding that the reviewer has not accepted.
	allDismissals = []string{"none", "open", "unknown"}

	// blockingPlanGate are the plan signals that must stop a delivery.
	// `unverified` blocks like a regression: the PR claimed a plan and the check did not run,
	// which is missing evidence rather than a pass.
	blockingPlanGate = []string{"regression", "conflicts", "unverified"}
	// cleanPlanGate are the plan signals that must not stop a delivery. `absent` is the
	// planless case and must behave exactly like `pass` — archon is optional.
	cleanPlanGate = []string{"pass", "absent"}
)

type gateOutcome struct {
	decision string
	reason   string
	exitCode int
	stdout   string
	stderr   string
}

// gateEnv builds a fully valid input set that yields `ready`, with overrides applied. Tests
// state only the variable under test, so a case cannot accidentally depend on a default it
// did not mean to set.
func gateEnv(overrides map[string]string) map[string]string {
	env := map[string]string{
		"CI_STATUS":     "success",
		"PLAN_GATE":     "pass",
		"AGENT_VERDICT": "GREEN",
		"DISMISSALS":    "none",
		"ROUND":         "0",
		"MAX_ROUNDS":    "3",
	}
	for k, v := range overrides {
		env[k] = v
	}
	return env
}

// runGate executes deliver-gate.sh with exactly the supplied environment plus PATH.
//
// The ambient environment is deliberately NOT inherited: the unset-variable contract
// (BC-1) is only meaningful if a stray CI_STATUS in the developer's shell cannot satisfy
// it. A key mapped to the empty string is passed through as an empty value; to test an
// unset variable, delete the key.
func runGate(t *testing.T, env map[string]string) gateOutcome {
	t.Helper()

	cmd := exec.Command(scriptPath(t, "deliver-gate.sh"))
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	var out gateOutcome
	err := cmd.Run()
	out.stdout = stdout.String()
	out.stderr = stderr.String()

	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		out.exitCode = exitErr.ExitCode()
	default:
		t.Fatalf("running deliver-gate.sh: %v", err)
	}

	for _, line := range strings.Split(out.stdout, "\n") {
		switch {
		case strings.HasPrefix(line, "decision="):
			out.decision = strings.TrimPrefix(line, "decision=")
		case strings.HasPrefix(line, "reason="):
			out.reason = strings.TrimPrefix(line, "reason=")
		}
	}
	return out
}

// requireDecision asserts a computed decision: exit 0, the expected value, and a non-empty
// reason. The reason is what lands in the PR comment, so a decision without one would
// leave a human with a stopped delivery and no explanation.
func requireDecision(t *testing.T, out gateOutcome, want string) {
	t.Helper()
	if out.exitCode != 0 {
		t.Errorf("exit code = %d, want 0 (stderr: %s)", out.exitCode, out.stderr)
	}
	if out.decision != want {
		t.Errorf("decision = %q, want %q (stdout: %s)", out.decision, want, out.stdout)
	}
	if strings.TrimSpace(out.reason) == "" {
		t.Errorf("reason is empty; every decision must explain itself (stdout: %s)", out.stdout)
	}
}

// TestDeliverGateWiringErrorsAreLoud covers BC-1's first half: an unset or malformed input
// is a workflow wiring bug and must fail visibly rather than produce a verdict.
func TestDeliverGateWiringErrorsAreLoud(t *testing.T) {
	required := []string{"CI_STATUS", "PLAN_GATE", "AGENT_VERDICT", "DISMISSALS", "ROUND", "MAX_ROUNDS"}

	for _, name := range required {
		t.Run("unset/"+name, func(t *testing.T) {
			env := gateEnv(nil)
			delete(env, name)
			out := runGate(t, env)
			if out.exitCode != 2 {
				t.Errorf("exit code = %d, want 2", out.exitCode)
			}
			if out.decision != "" {
				t.Errorf("decision = %q, want none: an unwired workflow must not receive a verdict", out.decision)
			}
			if !strings.Contains(out.stderr, name) {
				t.Errorf("stderr does not name the missing variable %s: %s", name, out.stderr)
			}
		})

		t.Run("empty/"+name, func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{name: ""}))
			if out.exitCode != 2 {
				t.Errorf("exit code = %d, want 2", out.exitCode)
			}
			if out.decision != "" {
				t.Errorf("decision = %q, want none", out.decision)
			}
		})
	}

	// A non-integer round counter means the label parsing upstream is broken. That is a
	// wiring bug, not an unrecognised signal, so it must be loud rather than needs-human.
	for _, tc := range []struct{ name, round, maxRounds string }{
		{"round-not-a-number", "abc", "3"},
		{"round-fractional", "1.5", "3"},
		{"round-negative", "-1", "3"},
		{"max-not-a-number", "0", "three"},
		{"max-empty-ish-space", "0", " "},
	} {
		t.Run("integer/"+tc.name, func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{"ROUND": tc.round, "MAX_ROUNDS": tc.maxRounds}))
			if out.exitCode != 2 {
				t.Errorf("exit code = %d, want 2 (stdout: %s, stderr: %s)", out.exitCode, out.stdout, out.stderr)
			}
			if out.decision != "" {
				t.Errorf("decision = %q, want none", out.decision)
			}
		})
	}
}

// TestDeliverGateUnrecognisedValuesFailClosed covers BC-1's second half and the catch-all
// row. A value outside the declared domain — a GitHub check conclusion the derivation step
// forgot to map, say — must land on needs-human, never fall through with no decision.
func TestDeliverGateUnrecognisedValuesFailClosed(t *testing.T) {
	cases := []struct{ name, key, value string }{
		// The six check conclusions beyond success/failure. If a derivation step ever
		// passes one through raw, the gate must still stop rather than fall off the end
		// of its decision chain.
		{"ci-cancelled", "CI_STATUS", "cancelled"},
		{"ci-timed-out", "CI_STATUS", "timed_out"},
		{"ci-neutral", "CI_STATUS", "neutral"},
		{"ci-skipped", "CI_STATUS", "skipped"},
		{"ci-action-required", "CI_STATUS", "action_required"},
		{"ci-stale", "CI_STATUS", "stale"},
		{"ci-empty-word", "CI_STATUS", "none"},
		{"plan-gate-bogus", "PLAN_GATE", "bogus"},
		{"plan-gate-verdict-leak", "PLAN_GATE", "REALIZES"},
		{"verdict-lowercase", "AGENT_VERDICT", "green"},
		{"verdict-typo", "AGENT_VERDICT", "NOTGREEN"},
		{"verdict-prose", "AGENT_VERDICT", "looks good to me"},
		{"dismissals-bogus", "DISMISSALS", "maybe"},
		{"dismissals-numeric", "DISMISSALS", "2"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{tc.key: tc.value}))
			requireDecision(t, out, "needs-human")
			if !strings.Contains(out.reason, tc.value) {
				t.Errorf("reason does not name the offending value %q: %s", tc.value, out.reason)
			}
		})
	}
}

// TestDeliverGateClosedByDefault covers BC-2 and BC-3. Undetermined CI and a missing
// verdict marker each stop the delivery regardless of how favourable every other input is.
func TestDeliverGateClosedByDefault(t *testing.T) {
	t.Run("ci-unknown", func(t *testing.T) {
		for _, plan := range allPlanGate {
			for _, verdict := range allVerdicts {
				out := runGate(t, gateEnv(map[string]string{
					"CI_STATUS": "unknown", "PLAN_GATE": plan, "AGENT_VERDICT": verdict,
				}))
				if out.decision != "needs-human" {
					t.Errorf("CI_STATUS=unknown PLAN_GATE=%s AGENT_VERDICT=%s: decision = %q, want needs-human",
						plan, verdict, out.decision)
				}
			}
		}
	})

	t.Run("verdict-missing", func(t *testing.T) {
		for _, ci := range allCIStatus {
			for _, plan := range allPlanGate {
				out := runGate(t, gateEnv(map[string]string{
					"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": "MISSING",
				}))
				if out.decision != "needs-human" {
					t.Errorf("CI_STATUS=%s PLAN_GATE=%s AGENT_VERDICT=MISSING: decision = %q, want needs-human",
						ci, plan, out.decision)
				}
			}
		}
	})
}

// TestDeliverGateDisagreementGoesToAHuman covers BC-4, the guardrail this gate exists for.
// A GREEN review that contradicts an objective signal is not something the loop may
// resolve by itself — not by correcting, and certainly not by marking the PR ready.
func TestDeliverGateDisagreementGoesToAHuman(t *testing.T) {
	type combo struct {
		name string
		env  map[string]string
	}
	var combos []combo

	for _, plan := range cleanPlanGate {
		combos = append(combos, combo{"ci-failure/plan-" + plan,
			map[string]string{"CI_STATUS": "failure", "PLAN_GATE": plan}})
	}
	for _, plan := range blockingPlanGate {
		combos = append(combos, combo{"ci-success/plan-" + plan,
			map[string]string{"CI_STATUS": "success", "PLAN_GATE": plan}})
		combos = append(combos, combo{"ci-failure/plan-" + plan,
			map[string]string{"CI_STATUS": "failure", "PLAN_GATE": plan}})
	}

	// The rule is round-independent: it is not a thing that becomes acceptable early in a
	// delivery, nor a thing the round cap should relabel.
	for _, round := range []string{"0", "1", "3", "9"} {
		for _, c := range combos {
			t.Run(c.name+"/round-"+round, func(t *testing.T) {
				env := gateEnv(c.env)
				env["AGENT_VERDICT"] = "GREEN"
				env["ROUND"] = round
				out := runGate(t, env)
				requireDecision(t, out, "needs-human")
				if out.decision == "ready" {
					t.Fatal("reached ready with a blocking objective signal")
				}
			})
		}
	}
}

// TestDeliverGateUnacceptedDismissalBlocksReady covers BC-12. A correction may dismiss a
// finding rather than fix it, and the reviewer is told to accept or re-raise each one. That
// was prompt adherence the gate could not see. An outstanding dismissal now blocks `ready`
// structurally, and an unreadable dismissal state blocks it too — a reviewer who forgets to
// clear the label costs a human glance rather than passing a waved-away finding.
func TestDeliverGateUnacceptedDismissalBlocksReady(t *testing.T) {
	for _, d := range []string{"open", "unknown"} {
		t.Run("dismissals-"+d, func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{
				"CI_STATUS": "success", "PLAN_GATE": "pass",
				"AGENT_VERDICT": "GREEN", "DISMISSALS": d,
			}))
			requireDecision(t, out, "needs-human")
			if out.decision == "ready" {
				t.Fatal("reached ready with an outstanding dismissal")
			}
		})
	}

	// It must not block a correction round — only the terminal ready verdict.
	t.Run("does-not-block-correction", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"AGENT_VERDICT": "NOT-GREEN", "DISMISSALS": "open", "ROUND": "0",
		}))
		requireDecision(t, out, "correct")
	})

	// And `none` must behave exactly as before, so the new input cannot silently gate
	// deliveries that have no dismissals at all.
	t.Run("none-still-ready", func(t *testing.T) {
		requireDecision(t, runGate(t, gateEnv(map[string]string{"DISMISSALS": "none"})), "ready")
	})
}

// TestDeliverGateReady covers BC-6: ready requires all three signals to agree, and the
// planless case delivers exactly like a satisfied plan.
func TestDeliverGateReady(t *testing.T) {
	for _, plan := range cleanPlanGate {
		t.Run("plan-"+plan, func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{
				"CI_STATUS": "success", "PLAN_GATE": plan, "AGENT_VERDICT": "GREEN",
			}))
			requireDecision(t, out, "ready")
		})
	}

	// Archon is optional: a repo or PR with no plan must not be penalised for it.
	withPlan := runGate(t, gateEnv(map[string]string{"PLAN_GATE": "pass"}))
	withoutPlan := runGate(t, gateEnv(map[string]string{"PLAN_GATE": "absent"}))
	if withPlan.decision != withoutPlan.decision {
		t.Errorf("plan-absent decision %q differs from plan-pass %q; archon must stay optional",
			withoutPlan.decision, withPlan.decision)
	}
}

// TestDeliverGateCorrect covers BC-5: an honest NOT-GREEN routes to correction under every
// objective signal, so long as rounds remain.
func TestDeliverGateCorrect(t *testing.T) {
	for _, ci := range []string{"success", "failure"} {
		for _, plan := range allPlanGate {
			t.Run("ci-"+ci+"/plan-"+plan, func(t *testing.T) {
				out := runGate(t, gateEnv(map[string]string{
					"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": "NOT-GREEN",
					"ROUND": "0", "MAX_ROUNDS": "3",
				}))
				requireDecision(t, out, "correct")
			})
		}
	}
}

// TestDeliverGateRoundCap covers BC-7. The cap converts a correction into a handoff, and
// must not touch a delivery that is genuinely ready.
func TestDeliverGateRoundCap(t *testing.T) {
	notGreen := func(round, maxRounds string) map[string]string {
		return gateEnv(map[string]string{
			"AGENT_VERDICT": "NOT-GREEN", "ROUND": round, "MAX_ROUNDS": maxRounds,
		})
	}

	t.Run("below-cap-corrects", func(t *testing.T) {
		for _, round := range []string{"0", "1", "2"} {
			out := runGate(t, notGreen(round, "3"))
			if out.decision != "correct" {
				t.Errorf("round %s of 3: decision = %q, want correct", round, out.decision)
			}
		}
	})

	t.Run("at-and-past-cap-stops", func(t *testing.T) {
		for _, round := range []string{"3", "4", "99"} {
			out := runGate(t, notGreen(round, "3"))
			requireDecision(t, out, "needs-human")
			if !strings.Contains(out.reason, "3") {
				t.Errorf("round %s: reason should name the cap: %s", round, out.reason)
			}
		}
	})

	// A green delivery that happens to be at the cap is still green. The cap bounds
	// correction attempts, not the delivery itself.
	t.Run("cap-does-not-block-ready", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"AGENT_VERDICT": "GREEN", "ROUND": "3", "MAX_ROUNDS": "3",
		}))
		requireDecision(t, out, "ready")
	})
}

// TestDeliverGateAlwaysDecides is the structural backstop for BC-1: across the entire
// declared input space the gate must always exit 0 with one of exactly three decisions.
// A silent fallthrough is the failure this test exists to make impossible.
func TestDeliverGateAlwaysDecides(t *testing.T) {
	valid := map[string]bool{"ready": true, "correct": true, "needs-human": true}

	for _, ci := range allCIStatus {
		for _, plan := range allPlanGate {
			for _, verdict := range allVerdicts {
				for _, dis := range allDismissals {
					for _, round := range []string{"0", "3"} {
						out := runGate(t, gateEnv(map[string]string{
							"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": verdict,
							"DISMISSALS": dis, "ROUND": round,
						}))
						if out.exitCode != 0 {
							t.Errorf("%s/%s/%s/%s round %s: exit %d, want 0", ci, plan, verdict, dis, round, out.exitCode)
						}
						if !valid[out.decision] {
							t.Errorf("%s/%s/%s/%s round %s: decision = %q, want ready/correct/needs-human",
								ci, plan, verdict, dis, round, out.decision)
						}
					}
				}
			}
		}
	}
}
