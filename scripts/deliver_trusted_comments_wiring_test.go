package scripts_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// ── The wiring: the filter must be a STRUCTURAL boundary, not a prompt instruction (#1806) ──────
//
// deliver_trusted_comments_test.go pins the selection LAW. This file pins that the flows USE it, and
// use it in the one way that actually closes the hole: a WORKFLOW STEP assembles the filtered digest
// from trusted code BEFORE the agent runs, and the agent reads the resulting FILE. An earlier draft
// had the agent run the helper itself from its prompt — but that makes the filtering contingent on
// the model obeying prose, which is the exact behavioural-not-structural weakness this hardening
// replaces (raised in review). So the assertions below check producer + consumer + ordering, not
// merely that a prompt names the script.
//
// For a workflow file the declared structure IS the behaviour — GitHub reads nothing else — so these
// assertions are behavioural, in the same sense as those in claude_workflow_test.go.

type tcWiringStep struct {
	Name string `yaml:"name"`
	ID   string `yaml:"id"`
	Uses string `yaml:"uses"`
	Run  string `yaml:"run"`
	With struct {
		Prompt     string `yaml:"prompt"`
		ClaudeArgs string `yaml:"claude_args"`
		Ref        string `yaml:"ref"`
	} `yaml:"with"`
}

func wiringJobSteps(t *testing.T, workflow, job string) []tcWiringStep {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", workflow)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []tcWiringStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	j, ok := wf.Jobs[job]
	if !ok {
		t.Fatalf("%s has no %q job", workflow, job)
	}
	if len(j.Steps) == 0 {
		t.Fatalf("%s job %q has no steps", workflow, job)
	}
	return j.Steps
}

func indexOf(steps []tcWiringStep, match func(tcWiringStep) bool) int {
	for i, s := range steps {
		if match(s) {
			return i
		}
	}
	return -1
}

// The two dispatched delivery phases must assemble the digest in a workflow step (PRODUCER) that runs
// the filter, BEFORE the agent, and the agent must read that file (CONSUMER). Ordering matters: a
// producer placed after the agent would satisfy a presence check while handing the agent nothing.
func TestTrustedComments_VerifyAndCorrectAssembleDigestBeforeTheAgent(t *testing.T) {
	for _, tc := range []struct{ workflow, job string }{
		{"deliver-verify.yml", "verify"},
		{"deliver-correct.yml", "correct"},
	} {
		t.Run(tc.workflow, func(t *testing.T) {
			steps := wiringJobSteps(t, tc.workflow, tc.job)

			producer := indexOf(steps, func(s tcWiringStep) bool {
				return strings.Contains(s.Run, "deliver-trusted-comments.sh")
			})
			if producer < 0 {
				t.Fatalf("%s job %q has no workflow step running deliver-trusted-comments.sh — the "+
					"filter must be assembled by a step from trusted code, not left to the agent (#1806)",
					tc.workflow, tc.job)
			}
			// The producer must write the digest to a file the agent then reads.
			if !strings.Contains(steps[producer].Run, "trusted-comments.md") {
				t.Errorf("%s producer step does not write trusted-comments.md; the agent reads a file, "+
					"so the step must produce one", tc.workflow)
			}

			agent := indexOf(steps, func(s tcWiringStep) bool {
				return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
			})
			if agent < 0 {
				t.Fatalf("%s job %q no longer runs anthropics/claude-code-action", tc.workflow, tc.job)
			}
			if producer >= agent {
				t.Errorf("%s: the digest-producer step is at index %d, not BEFORE the agent at %d — a "+
					"digest assembled after the agent runs is never read, and the agent would fall back "+
					"to unfiltered comments (#1806)", tc.workflow, producer, agent)
			}

			// CONSUMER: the agent prompt must read the digest file...
			if !strings.Contains(steps[agent].With.Prompt, "trusted-comments.md") {
				t.Errorf("%s: the agent prompt does not read the trusted-comments.md digest the step "+
					"produced, so the structural filter does not reach the agent (#1806)", tc.workflow)
			}
		})
	}
}

// The correction phase's digest MUST be assembled from the TRUSTED default-branch checkout — after
// it, and BEFORE the delivery-branch checkout replaces the workspace. Otherwise the producer would
// run a PR-CONTROLLED copy of the filter, and this phase holds `contents: write`. `producer < agent`
// alone does not pin this (the producer could sit between the delivery checkout and the agent), so
// this asserts the tighter window explicitly: it fails if the producer is moved across the delivery
// checkout (raised in review).
func TestTrustedComments_CorrectAssemblesDigestFromTheTrustedCheckout(t *testing.T) {
	steps := wiringJobSteps(t, "deliver-correct.yml", "correct")

	isCheckout := func(s tcWiringStep) bool { return strings.HasPrefix(s.Uses, "actions/checkout") }

	// The trusted checkout pins the default branch; the delivery checkout is the later one with no
	// explicit ref (it takes the dispatched `deliver/issue-N` ref implicitly).
	trusted := indexOf(steps, func(s tcWiringStep) bool {
		return isCheckout(s) && strings.Contains(s.With.Ref, "default_branch")
	})
	if trusted < 0 {
		t.Fatal("deliver-correct.yml has no trusted default-branch checkout (ref: default_branch) — the " +
			"digest must be assembled from it, so its absence is itself the regression (#1806)")
	}
	delivery := -1
	for i, s := range steps {
		if i > trusted && isCheckout(s) && !strings.Contains(s.With.Ref, "default_branch") {
			delivery = i
			break
		}
	}
	if delivery < 0 {
		t.Fatal("deliver-correct.yml has no delivery-branch checkout after the trusted one")
	}
	producer := indexOf(steps, func(s tcWiringStep) bool {
		return strings.Contains(s.Run, "deliver-trusted-comments.sh")
	})
	if producer < 0 {
		t.Fatal("deliver-correct.yml has no digest-producer step running deliver-trusted-comments.sh")
	}
	if !(trusted < producer && producer < delivery) {
		t.Errorf("the digest producer (step %d) must run AFTER the trusted default-branch checkout "+
			"(step %d) and BEFORE the delivery-branch checkout (step %d). Outside that window the "+
			"filter runs from PR-controlled code with `contents: write` (#1806).", producer, trusted, delivery)
	}
}

// No agent prompt in the dispatched phases may instruct a raw, unfiltered comment read — leaving one
// beside the filtered digest re-opens the hole, because the agent would run the call it was shown.
// Matched on the raw API/CLI shapes rather than any mention of "comment", so the prompts can discuss
// comments at length (they must) while being unable to instruct an unfiltered fetch.
func TestTrustedComments_NoAgentPromptInstructsAnUnfilteredCommentRead(t *testing.T) {
	forbidden := []string{"--comments", "/comments?", "/comments\"", "/comments'"}
	for _, tc := range []struct{ workflow, job string }{
		{"deliver-verify.yml", "verify"},
		{"deliver-correct.yml", "correct"},
	} {
		t.Run(tc.workflow, func(t *testing.T) {
			for _, s := range wiringJobSteps(t, tc.workflow, tc.job) {
				if strings.TrimSpace(s.With.Prompt) == "" {
					continue
				}
				for _, shape := range forbidden {
					if strings.Contains(s.With.Prompt, shape) {
						t.Errorf("an agent prompt in %s still contains %q. A raw comment read beside the "+
							"filtered digest re-opens the injection surface (#1806).\nprompt:\n%s",
							tc.workflow, shape, s.With.Prompt)
					}
				}
			}
		})
	}
}

// claude.yml is OUT OF SCOPE for #1806 (tag mode assembles comment context itself; documented in
// agent-trust.md). Pin that it stays unwired, so a future edit cannot silently reintroduce the
// out-of-scope tag-mode mitigation the review asked to remove without also updating the docs/decision.
func TestTrustedComments_ClaudeYmlStaysOutOfScope(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "claude.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if strings.Contains(string(raw), "deliver-trusted-comments") {
		t.Errorf("claude.yml references deliver-trusted-comments, but #1806 records it as OUT OF SCOPE " +
			"(tag mode leaves no seam to filter). If this is intentional new scope, update the issue and " +
			"the agent-trust.md decision rather than wiring it silently.")
	}
}
