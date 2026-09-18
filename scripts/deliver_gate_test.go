package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// The gate's input domains. Tests cross-product these so a new decision row cannot be
// added without a case covering every combination it could capture.
var (
	allCIStatus = []string{"success", "failure", "unknown"}
	allPlanGate = []string{"pass", "absent", "regression", "conflicts", "unverified"}
	allVerdicts = []string{"GREEN", "NOT-GREEN", "MISSING"}
	// The cross-vendor qa-review signal (#1715). A second reviewer from a different model
	// family than the implementer and the AGENT_VERDICT reviewer, gated in parallel with them.
	allQAVerdict = []string{"PASS", "BLOCK", "MISSING"}
	// `open` means a correction dismissed a finding that the reviewer has not accepted.
	allDismissals = []string{"none", "open", "unknown"}
	// The branch's mergeability against main. `conflicting` (REST mergeable_state "dirty")
	// must never be marked ready; `unknown` means the state could not be read.
	allMergeState = []string{"mergeable", "conflicting", "unknown"}

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
		"CI_STATUS":       "success",
		"PLAN_GATE":       "pass",
		"AGENT_VERDICT":   "GREEN",
		"QA_VERDICT":      "PASS",
		"DISMISSALS":      "none",
		"MERGE_STATE":     "mergeable",
		"REVIEWS_SKIPPED": "false",
		"ROUND":           "0",
		"MAX_ROUNDS":      "3",
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
	// QA_VERDICT and MERGE_STATE are guarded exactly like the other six: fail-closed is the whole
	// contract of a blocking review signal (and of the mergeability signal), so a workflow that
	// forgets to wire either must exit 2 rather than silently decide without it.
	required := []string{"CI_STATUS", "PLAN_GATE", "AGENT_VERDICT", "QA_VERDICT", "DISMISSALS", "MERGE_STATE", "REVIEWS_SKIPPED", "ROUND", "MAX_ROUNDS"}

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
		// The qa-review marker is derived from the answerer's JSON in the workflow, so a
		// derivation that leaks a status name or the render_report emoji must not fall through.
		{"qa-verdict-lowercase", "QA_VERDICT", "pass"},
		{"qa-verdict-typo", "QA_VERDICT", "BLOCKED"},
		{"qa-verdict-status-leak", "QA_VERDICT", "FLAW_FOUND"},
		{"qa-verdict-green", "QA_VERDICT", "GREEN"},
		// The gate's merge domain is the mapped value, not GitHub's raw mergeable_state.
		// The derivation step maps "dirty"→conflicting and "behind"→mergeable; a raw GitHub
		// value reaching the gate means that mapping was skipped, so it must stop rather than
		// fall through the merge branch with no decision.
		{"merge-state-raw-dirty", "MERGE_STATE", "dirty"},
		{"merge-state-raw-behind", "MERGE_STATE", "behind"},
		{"merge-state-bogus", "MERGE_STATE", "sideways"},
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

	// A qa-review that produced no marker is MISSING EVIDENCE, exactly like an unread
	// DELIVER-VERDICT. This is the case that matters most for a review pass that can crash,
	// exhaust its tool budget or lose its model: the gate must never read "no marker" as a pass.
	t.Run("qa-verdict-missing", func(t *testing.T) {
		for _, ci := range allCIStatus {
			for _, plan := range allPlanGate {
				for _, verdict := range allVerdicts {
					out := runGate(t, gateEnv(map[string]string{
						"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": verdict,
						"QA_VERDICT": "MISSING",
					}))
					if out.decision != "needs-human" {
						t.Errorf("CI_STATUS=%s PLAN_GATE=%s AGENT_VERDICT=%s QA_VERDICT=MISSING: decision = %q, want needs-human",
							ci, plan, verdict, out.decision)
					}
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
	//
	// QA_VERDICT is stated rather than inherited from the default: since #1715 the rule is
	// "NEITHER reviewer is asking for a correction", so the premise of a disagreement is that
	// both came back clean.
	for _, round := range []string{"0", "1", "3", "9"} {
		for _, c := range combos {
			t.Run(c.name+"/round-"+round, func(t *testing.T) {
				env := gateEnv(c.env)
				env["AGENT_VERDICT"] = "GREEN"
				env["QA_VERDICT"] = "PASS"
				env["ROUND"] = round
				out := runGate(t, env)
				requireDecision(t, out, "needs-human")
				if out.decision == "ready" {
					t.Fatal("reached ready with a blocking objective signal")
				}
			})
		}
	}

	// The generalisation (#1715, AC-2). An objective blocker only reaches a human when NEITHER
	// reviewer named something to fix. If EITHER did, there are findings a correction round can
	// act on, and routing that to a human instead would waste the loop's whole reason for
	// having correction rounds.
	t.Run("either-review-non-green-corrects-instead", func(t *testing.T) {
		for _, review := range []struct{ agent, qa string }{
			{"NOT-GREEN", "PASS"},
			{"GREEN", "BLOCK"},
			{"NOT-GREEN", "BLOCK"},
		} {
			for _, c := range combos {
				name := c.name + "/" + review.agent + "+" + review.qa
				t.Run(name, func(t *testing.T) {
					env := gateEnv(c.env)
					env["AGENT_VERDICT"] = review.agent
					env["QA_VERDICT"] = review.qa
					env["ROUND"] = "0"
					out := runGate(t, env)
					requireDecision(t, out, "correct")
					// The reason has to carry the objective blocker too, or the correction agent
					// is told about the findings and not about the failure it must also fix.
					if !strings.Contains(out.reason, "CI is failing") &&
						!strings.Contains(out.reason, "archon plan") &&
						!strings.Contains(out.reason, "dist ratchet") {
						t.Errorf("reason does not name the objective blocker: %s", out.reason)
					}
				})
			}
		}
	})
}

// TestDeliverGateQAVerdictBlocks covers #1715's AC-1: qa-review is a blocking signal in its own
// right. A BLOCK routes to a correction round exactly like an Anthropic-side NOT-GREEN, is
// subject to the same round cap, and — the load-bearing half — makes `ready` unreachable.
func TestDeliverGateQAVerdictBlocks(t *testing.T) {
	// A clean objective picture and a GREEN Anthropic review: qa-review alone must divert the
	// delivery. If this passed `ready`, the whole cross-vendor gate would be advisory.
	t.Run("block-alone-corrects", func(t *testing.T) {
		for _, plan := range cleanPlanGate {
			out := runGate(t, gateEnv(map[string]string{
				"CI_STATUS": "success", "PLAN_GATE": plan,
				"AGENT_VERDICT": "GREEN", "QA_VERDICT": "BLOCK", "ROUND": "0",
			}))
			requireDecision(t, out, "correct")
			if !strings.Contains(out.reason, "qa-review") {
				t.Errorf("plan %s: reason does not name qa-review as the blocker: %s", plan, out.reason)
			}
		}
	})

	// The cap bounds correction attempts whatever produced them, so a qa-BLOCK that will not
	// converge must stop rather than loop.
	t.Run("block-at-cap-stops", func(t *testing.T) {
		for _, round := range []string{"3", "4", "99"} {
			out := runGate(t, gateEnv(map[string]string{
				"AGENT_VERDICT": "GREEN", "QA_VERDICT": "BLOCK",
				"ROUND": round, "MAX_ROUNDS": "3",
			}))
			requireDecision(t, out, "needs-human")
			if !strings.Contains(out.reason, "3") {
				t.Errorf("round %s: reason should name the cap: %s", round, out.reason)
			}
		}
	})

	// `ready` is unreachable without PASS, across every other input that could otherwise carry
	// it there. This is the assertion that would fail if a future edit dropped QA_VERDICT from
	// the ready conjunction.
	t.Run("ready-requires-pass", func(t *testing.T) {
		for _, qa := range allQAVerdict {
			for _, plan := range cleanPlanGate {
				for _, round := range []string{"0", "3"} {
					out := runGate(t, gateEnv(map[string]string{
						"CI_STATUS": "success", "PLAN_GATE": plan,
						"AGENT_VERDICT": "GREEN", "DISMISSALS": "none",
						"QA_VERDICT": qa, "ROUND": round,
					}))
					if qa == "PASS" {
						requireDecision(t, out, "ready")
						continue
					}
					if out.decision == "ready" {
						t.Errorf("QA_VERDICT=%s plan=%s round=%s reached ready; only PASS may",
							qa, plan, round)
					}
				}
			}
		}
	})

	// Both reviews non-green is ONE correction round naming both, not a stop: the correction
	// agent reads both review comments.
	t.Run("both-non-green-is-one-correction", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"AGENT_VERDICT": "NOT-GREEN", "QA_VERDICT": "BLOCK", "ROUND": "0",
		}))
		requireDecision(t, out, "correct")
		for _, needle := range []string{"NOT-GREEN", "qa-review"} {
			if !strings.Contains(out.reason, needle) {
				t.Errorf("reason does not name %q, so the correction agent is not told which "+
					"reviews blocked: %s", needle, out.reason)
			}
		}
	})
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

// TestDeliverGateMergeConflictNeverReady covers #1758. A branch that conflicts with main
// must never be marked ready — GitHub cannot compute its merge ref, so a `ready-for-merge`
// on it is a stale label a human cannot act on. A conflict is not a review disagreement
// (the reviewer approved the code, not the mergeability), so it routes to the correction
// agent to merge main + resolve, subject to the ordinary round cap.
func TestDeliverGateMergeConflictNeverReady(t *testing.T) {
	// The would-be-ready inputs: every quality signal green. Only the merge state blocks.
	t.Run("green-but-conflicting-corrects", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{"MERGE_STATE": "conflicting", "ROUND": "0"}))
		requireDecision(t, out, "correct")
		if out.decision == "ready" {
			t.Fatal("reached ready on a conflicting branch")
		}
	})

	// A conflict is not the GREEN-vs-blocking disagreement: it must correct, not stop, even
	// though the review is GREEN. This is what distinguishes it from a failing objective signal.
	t.Run("green-conflict-is-not-a-disagreement", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"AGENT_VERDICT": "GREEN", "MERGE_STATE": "conflicting", "ROUND": "0",
		}))
		requireDecision(t, out, "correct")
	})

	// But a real failing signal (CI) with a GREEN review still wins as a disagreement: a
	// human is needed regardless of the branch being dirty too.
	t.Run("ci-failure-disagreement-wins-over-conflict", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"CI_STATUS": "failure", "AGENT_VERDICT": "GREEN", "MERGE_STATE": "conflicting",
		}))
		requireDecision(t, out, "needs-human")
	})

	// An honest NOT-GREEN on a conflicting branch still corrects (the agent merges main AND
	// fixes findings in the same round).
	t.Run("not-green-conflict-corrects", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"AGENT_VERDICT": "NOT-GREEN", "MERGE_STATE": "conflicting", "ROUND": "0",
		}))
		requireDecision(t, out, "correct")
	})

	// The round cap bounds conflict resolution exactly as it bounds any other correction: a
	// branch that cannot be auto-resolved within the cap stops loudly at needs-human.
	t.Run("conflict-at-cap-stops", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"MERGE_STATE": "conflicting", "ROUND": "3", "MAX_ROUNDS": "3",
		}))
		requireDecision(t, out, "needs-human")
		if !strings.Contains(out.reason, "3") {
			t.Errorf("reason should name the cap: %s", out.reason)
		}
	})

	// An undeterminable merge state withholds the terminal ready verdict — but it must NOT be a
	// terminal needs-human either. Mergeability that could not be read (a transient API blip on
	// an otherwise-green PR) is re-checkable, not a reason to stop the delivery for a human. The
	// gate returns the non-terminal `recheck`, on which verify re-verifies on the next event
	// rather than announcing the loop stopped (#1758 G1).
	t.Run("unknown-is-recheck-not-terminal", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{"MERGE_STATE": "unknown"}))
		requireDecision(t, out, "recheck")
		if out.decision == "ready" {
			t.Fatal("reached ready with an undeterminable merge state")
		}
		if out.decision == "needs-human" {
			t.Fatal("an undeterminable merge state must not terminally stop the delivery; it is re-checkable")
		}
	})

	// Regression guard: a mergeable branch with all signals green still reaches ready — the
	// new input cannot silently gate deliveries that are genuinely mergeable.
	t.Run("mergeable-still-ready", func(t *testing.T) {
		requireDecision(t, runGate(t, gateEnv(map[string]string{"MERGE_STATE": "mergeable"})), "ready")
	})

	// Precedence: the conflict path is nested UNDER `DISMISSALS=none`, so an outstanding
	// dismissal is decided before the merge state is even read — a human is needed to accept
	// the dismissal regardless of the branch being dirty too. It must NOT be quietly downgraded
	// to a correction round by the conflict.
	t.Run("open-dismissal-beats-conflict", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{"DISMISSALS": "open", "MERGE_STATE": "conflicting"}))
		requireDecision(t, out, "needs-human")
	})

	// Precedence: a plan-blocking signal collects into `blocking`, and a GREEN review against a
	// blocking signal is the disagreement guardrail — needs-human — reached before the merge
	// branch. A conflict does not turn that disagreement into a correction round.
	t.Run("plan-regression-disagreement-beats-conflict", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"PLAN_GATE": "regression", "AGENT_VERDICT": "GREEN", "MERGE_STATE": "conflicting",
		}))
		requireDecision(t, out, "needs-human")
	})

	// But an HONEST NOT-GREEN on a conflicting branch with a plan regression still corrects
	// (rounds permitting): the agent merges main AND addresses the findings in one round.
	t.Run("plan-regression-not-green-conflict-corrects", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"PLAN_GATE": "regression", "AGENT_VERDICT": "NOT-GREEN", "MERGE_STATE": "conflicting", "ROUND": "0",
		}))
		requireDecision(t, out, "correct")
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
// declared input space the gate must always exit 0 with one of exactly four decisions
// (ready, correct, needs-human, recheck). A silent fallthrough is the failure this test
// exists to make impossible.
func TestDeliverGateAlwaysDecides(t *testing.T) {
	valid := map[string]bool{"ready": true, "correct": true, "needs-human": true, "recheck": true}

	for _, ci := range allCIStatus {
		for _, plan := range allPlanGate {
			for _, verdict := range allVerdicts {
				for _, qa := range allQAVerdict {
					for _, dis := range allDismissals {
						for _, merge := range allMergeState {
							for _, round := range []string{"0", "3"} {
								out := runGate(t, gateEnv(map[string]string{
									"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": verdict,
									"QA_VERDICT": qa, "DISMISSALS": dis, "MERGE_STATE": merge, "ROUND": round,
								}))
								where := ci + "/" + plan + "/" + verdict + "/qa=" + qa + "/dis=" + dis + "/merge=" + merge + " round " + round
								if out.exitCode != 0 {
									t.Errorf("%s: exit %d, want 0", where, out.exitCode)
								}
								if !valid[out.decision] {
									t.Errorf("%s: decision = %q, want ready/correct/needs-human/recheck", where, out.decision)
								}
								if strings.TrimSpace(out.reason) == "" {
									t.Errorf("%s: empty reason; every decision must explain itself", where)
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestDeliverGateReviewsSkippedIsNonTerminalWhenNotConflicting covers #1781 G1. When verify
// skipped both agent reviews on a stale `conflicting` hint (so the markers are MISSING) but the
// branch is NOT actually conflicting by the time mergeability is read, the gate must return the
// NON-TERMINAL `recheck` — never a terminal needs-human, and never a forced conflict that could
// dead-end at the round cap. The prior workflow forced MERGE_STATE=conflicting in exactly this
// case, which became a terminal `needs-human` naming a conflict that no longer existed at
// ROUND==MAX_ROUNDS. Asserted at both round 0 and the cap, and across the dismissal/marker axes,
// because none of them may turn a skipped-review non-conflicting round terminal.
func TestDeliverGateReviewsSkippedIsNonTerminalWhenNotConflicting(t *testing.T) {
	for _, merge := range []string{"mergeable", "unknown"} {
		for _, dis := range allDismissals {
			for _, round := range []string{"0", "3"} {
				// Markers are MISSING by construction when reviews were skipped; assert the gate
				// does not read them as a stop.
				out := runGate(t, gateEnv(map[string]string{
					"MERGE_STATE": merge, "REVIEWS_SKIPPED": "true",
					"AGENT_VERDICT": "MISSING", "QA_VERDICT": "MISSING",
					"DISMISSALS": dis, "ROUND": round, "MAX_ROUNDS": "3",
				}))
				where := "merge=" + merge + "/dis=" + dis + " round " + round
				if out.exitCode != 0 {
					t.Fatalf("%s: exit %d, want 0", where, out.exitCode)
				}
				if out.decision != "recheck" {
					t.Errorf("%s: decision = %q, want recheck — a skipped-review round on a non-conflicting branch must re-verify, not decide terminally", where, out.decision)
				}
			}
		}
	}
}

// TestDeliverGateReviewsSkippedStillNamesARealConflict is the companion: when reviews were skipped
// AND the branch really is conflicting, the round is NOT diverted to recheck — it takes the
// conflict path (correct while rounds remain, a conflict-naming needs-human at the cap), so a
// genuine conflict is still resolved or named (#1758(a)/(b)).
func TestDeliverGateReviewsSkippedStillNamesARealConflict(t *testing.T) {
	for _, round := range []string{"0", "3"} {
		out := runGate(t, gateEnv(map[string]string{
			"MERGE_STATE": "conflicting", "REVIEWS_SKIPPED": "true",
			"AGENT_VERDICT": "MISSING", "QA_VERDICT": "MISSING",
			"ROUND": round, "MAX_ROUNDS": "3", "CONFLICT_FILES": "CLAUDE.md",
		}))
		if out.exitCode != 0 || out.decision == "recheck" || out.decision == "ready" {
			t.Fatalf("round %s: exit %d decision %q, want a non-recheck non-ready decision", round, out.exitCode, out.decision)
		}
		if !strings.Contains(out.reason, conflictClause) || !strings.Contains(out.reason, "CLAUDE.md") {
			t.Errorf("round %s: reason must name the conflict and the file: %q", round, out.reason)
		}
	}
}

// conflictClause is the phrase every decision on a conflicting branch must carry (#1781). Matched
// as a substring rather than the whole reason so the clause can be composed with whatever else
// that row had to say, which is exactly the property under test.
const conflictClause = "merge conflicts with main"

// TestDeliverGateConflictIsAlwaysNamed covers C1 of #1781 across the WHOLE declared input space:
// whenever the branch conflicts with main, the reason a human reads names the conflict — on every
// decision row, not just the one row that used to mention it.
//
// This is the regression that dead-ended PR #1778. The branch was `dirty`, the review marker was
// unreadable, the missing-marker row fired first, and the delivery stopped for a human with the
// reason "the verify phase posted no DELIVER-VERDICT marker" — the symptom. #1758's acceptance
// criterion is that such a PR is "explicitly flagged needs-human NAMING THE CONFLICT", so the
// naming is the contract, and it has to hold whichever row happens to decide.
//
// Asserted as a cross-product rather than on the handful of rows that exist today: a new row added
// later inherits the requirement instead of quietly reintroducing an unnamed conflict.
func TestDeliverGateConflictIsAlwaysNamed(t *testing.T) {
	for _, ci := range allCIStatus {
		for _, plan := range allPlanGate {
			for _, verdict := range allVerdicts {
				for _, qa := range allQAVerdict {
					for _, dis := range allDismissals {
						for _, round := range []string{"0", "3"} {
							out := runGate(t, gateEnv(map[string]string{
								"CI_STATUS": ci, "PLAN_GATE": plan, "AGENT_VERDICT": verdict,
								"QA_VERDICT": qa, "DISMISSALS": dis, "MERGE_STATE": "conflicting",
								"ROUND": round,
							}))
							where := ci + "/" + plan + "/" + verdict + "/qa=" + qa + "/dis=" + dis + " round " + round
							if !strings.Contains(out.reason, conflictClause) {
								t.Errorf("%s: decision %q reason does not name the conflict: %q",
									where, out.decision, out.reason)
							}
							// And it can never be ready — the pre-existing #1758 guarantee, re-asserted
							// here because C1's decoration must not have widened the ready path.
							if out.decision == "ready" {
								t.Errorf("%s: reached ready on a conflicting branch", where)
							}
						}
					}
				}
			}
		}
	}
}

// The clause must appear ONLY when the branch actually conflicts. A reason that mentions a
// conflict on a mergeable branch would send a human looking for one that does not exist, and on
// `unknown` it would misreport a transient API blip as a real conflict.
func TestDeliverGateConflictClauseAbsentWhenNotConflicting(t *testing.T) {
	for _, merge := range []string{"mergeable", "unknown"} {
		for _, verdict := range allVerdicts {
			out := runGate(t, gateEnv(map[string]string{
				"MERGE_STATE": merge, "AGENT_VERDICT": verdict,
				// Supplied deliberately: a stale file list must not leak into a non-conflicting reason.
				"CONFLICT_FILES": "CLAUDE.md",
			}))
			if strings.Contains(out.reason, conflictClause) {
				t.Errorf("merge=%s verdict=%s: reason claims a conflict on a non-conflicting branch: %q",
					merge, verdict, out.reason)
			}
			if strings.Contains(out.reason, "CLAUDE.md") {
				t.Errorf("merge=%s verdict=%s: CONFLICT_FILES leaked into a non-conflicting reason: %q",
					merge, verdict, out.reason)
			}
		}
	}
}

// CONFLICT_FILES names the conflicting paths in the reason (#1781, C4/C6). This is the half of
// #1758(b) that PR #1778 was missing entirely: a human was told to take over without being told
// what conflicted.
func TestDeliverGateConflictNamesTheFiles(t *testing.T) {
	// Newline-separated, as scripts/conflicting-files.sh prints it.
	t.Run("newline-separated", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"MERGE_STATE": "conflicting", "CONFLICT_FILES": "CLAUDE.md\ndocs/guide/a.md",
		}))
		for _, want := range []string{"CLAUDE.md", "docs/guide/a.md"} {
			if !strings.Contains(out.reason, want) {
				t.Errorf("reason does not name %q: %q", want, out.reason)
			}
		}
		// One line, always: the reason is pasted into a PR comment and into the correct phase's
		// gate_reason input, so an embedded newline would truncate or corrupt both.
		if strings.Contains(out.reason, "\n") {
			t.Errorf("reason contains a newline: %q", out.reason)
		}
	})

	// #1781 G4 — a git path may legally contain spaces AND commas (git forbids only NUL), so the
	// reason must reproduce each path VERBATIM and split only on the newline that separates
	// entries. An earlier version split on commas and all whitespace and mangled real filenames:
	// "docs/My File.md" became two entries and "a,b.go" became "a, b.go".
	t.Run("filename-safe-spaces-and-commas", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"MERGE_STATE":    "conflicting",
			"CONFLICT_FILES": "docs/My File.md\nsrc/a,b.go",
		}))
		for _, want := range []string{"docs/My File.md", "src/a,b.go"} {
			if !strings.Contains(out.reason, want) {
				t.Errorf("reason does not reproduce %q verbatim (path split on a space or comma?): %q", want, out.reason)
			}
		}
		if strings.Contains(out.reason, "\n") {
			t.Errorf("reason contains a newline: %q", out.reason)
		}
	})

	// Absent or blank is the degraded case: scripts/conflicting-files.sh is best-effort, so the
	// conflict must still be NAMED even when the paths could not be computed. A gate that required
	// the list would turn a diagnostic failure into an unexplained stop.
	for i, files := range []string{"", "   ", "\n"} {
		t.Run("blank-"+strconv.Itoa(i)+"-still-names-the-conflict", func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{
				"MERGE_STATE": "conflicting", "CONFLICT_FILES": files,
			}))
			if !strings.Contains(out.reason, conflictClause) {
				t.Errorf("reason does not name the conflict: %q", out.reason)
			}
			if strings.Contains(out.reason, "conflicting files:") {
				t.Errorf("reason advertises an empty file list: %q", out.reason)
			}
		})
	}

	// CONFLICT_FILES is OPTIONAL: unset must not be a wiring error, or every existing caller
	// (and every other test in this file) would exit 2.
	t.Run("unset-is-not-a-wiring-error", func(t *testing.T) {
		env := gateEnv(map[string]string{"MERGE_STATE": "conflicting"})
		delete(env, "CONFLICT_FILES")
		out := runGate(t, env)
		if out.exitCode != 0 {
			t.Errorf("exit %d with CONFLICT_FILES unset, want 0 — the input is optional", out.exitCode)
		}
		if !strings.Contains(out.reason, conflictClause) {
			t.Errorf("reason does not name the conflict: %q", out.reason)
		}
	})
}

// TestDeliverGateConflictOutranksAMissingMarker covers C2 of #1781 — the ordering that is the
// actual fix for PR #1778's dead-end.
//
// A `dirty` branch has no merge ref, so verify now skips both agent reviews on one
// (deliver-verify.yml) and MISSING is the EXPECTED reading rather than an anomaly. The conflict is
// also the only thing a correction round can act on, so it must route to `correct` while rounds
// remain (#1758(a)) and, at the cap, to a `needs-human` that names the conflict (#1758(b)) —
// never to the bare "posted no DELIVER-VERDICT marker" stop.
func TestDeliverGateConflictOutranksAMissingMarker(t *testing.T) {
	// Every way a marker can be missing, alone or together.
	missing := []struct {
		name, agent, qa string
	}{
		{"review-missing", "MISSING", "PASS"},
		{"qa-missing", "GREEN", "MISSING"},
		{"both-missing", "MISSING", "MISSING"},
		// A NOT-GREEN/BLOCK alongside a missing sibling marker is still missing evidence.
		{"review-missing-qa-block", "MISSING", "BLOCK"},
		{"qa-missing-review-not-green", "NOT-GREEN", "MISSING"},
	}

	for _, m := range missing {
		t.Run(m.name+"/rounds-remain-corrects", func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{
				"MERGE_STATE": "conflicting", "AGENT_VERDICT": m.agent, "QA_VERDICT": m.qa,
				"ROUND": "0", "CONFLICT_FILES": "CLAUDE.md",
			}))
			requireDecision(t, out, "correct")
			if !strings.Contains(out.reason, conflictClause) {
				t.Errorf("reason does not name the conflict: %q", out.reason)
			}
			if !strings.Contains(out.reason, "CLAUDE.md") {
				t.Errorf("reason does not name the conflicting file: %q", out.reason)
			}
			// The observed #1778 message must NOT be the reason a human is given.
			if strings.Contains(out.reason, "posted no DELIVER-VERDICT marker") ||
				strings.Contains(out.reason, "posted no QA-VERDICT marker") {
				t.Errorf("reason reports the missing marker instead of the conflict: %q", out.reason)
			}
		})

		t.Run(m.name+"/at-cap-stops-naming-the-conflict", func(t *testing.T) {
			out := runGate(t, gateEnv(map[string]string{
				"MERGE_STATE": "conflicting", "AGENT_VERDICT": m.agent, "QA_VERDICT": m.qa,
				"ROUND": "3", "MAX_ROUNDS": "3", "CONFLICT_FILES": "CLAUDE.md",
			}))
			requireDecision(t, out, "needs-human")
			if !strings.Contains(out.reason, conflictClause) {
				t.Errorf("#1758(b) requires the stop to name the conflict; reason: %q", out.reason)
			}
			if !strings.Contains(out.reason, "CLAUDE.md") {
				t.Errorf("the stop does not name the conflicting file, so a human must go and find it: %q", out.reason)
			}
		})
	}

	// The unchanged half: on a MERGEABLE branch a missing marker is still real missing evidence and
	// still stops for a human with its own reason. C2 must not have turned an unreadable review
	// into a correction round in general — only on a branch whose markers cannot mean anything.
	t.Run("mergeable-missing-review-still-stops", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{"AGENT_VERDICT": "MISSING"}))
		requireDecision(t, out, "needs-human")
		if !strings.Contains(out.reason, "DELIVER-VERDICT") {
			t.Errorf("reason should name the missing marker on a mergeable branch: %q", out.reason)
		}
	})
	t.Run("mergeable-missing-qa-still-stops", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{"QA_VERDICT": "MISSING"}))
		requireDecision(t, out, "needs-human")
		if !strings.Contains(out.reason, "QA-VERDICT") {
			t.Errorf("reason should name the missing marker on a mergeable branch: %q", out.reason)
		}
	})
	// `unknown` mergeability with a missing marker is NOT the conflict case: nothing is known to
	// conflict, so the missing evidence governs and the delivery stops rather than correcting.
	t.Run("unknown-mergeability-missing-review-still-stops", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"MERGE_STATE": "unknown", "AGENT_VERDICT": "MISSING",
		}))
		requireDecision(t, out, "needs-human")
	})

	// An unreadable CI status still outranks everything, conflict included: without CI nothing can
	// be trusted, and a correction round has no failing signal to work from. The conflict is still
	// NAMED, which is the C1 guarantee.
	t.Run("ci-unknown-outranks-the-conflict", func(t *testing.T) {
		out := runGate(t, gateEnv(map[string]string{
			"CI_STATUS": "unknown", "MERGE_STATE": "conflicting", "AGENT_VERDICT": "MISSING",
		}))
		requireDecision(t, out, "needs-human")
		if !strings.Contains(out.reason, conflictClause) {
			t.Errorf("reason does not name the conflict: %q", out.reason)
		}
		if !strings.Contains(out.reason, "CI status could not be determined") {
			t.Errorf("reason does not name the unreadable CI status: %q", out.reason)
		}
	})
}
