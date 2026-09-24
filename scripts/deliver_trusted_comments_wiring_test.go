package scripts_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// ── The wiring: the three flows must actually USE the filter ───────────────────────────────────
//
// deliver_trusted_comments_test.go pins the selection LAW. But a helper that nothing calls is inert,
// and a prompt that ALSO hands the agent a raw `gh api …/comments` call re-opens the hole beside the
// filter rather than instead of it. Those two regressions are invisible in the logic tests and are
// exactly what a later edit to a 1500-line workflow does by accident. For a workflow file the
// declared structure IS the behaviour — GitHub reads nothing else — so these assertions are
// behavioural, in the same sense as those in claude_workflow_test.go.

// agentInstructions returns every piece of text in a workflow that INSTRUCTS an agent: each step's
// `with.prompt` (the dispatched delivery phases) and `with.claude_args` (tag mode's only injection
// point). Workflow `run:` scripts are deliberately excluded — the verdict-marker readers legitimately
// query the raw comments API and filter on `user.type == "Bot"` themselves; they hand nothing to a
// model.
func agentInstructions(t *testing.T, workflow string) []string {
	t.Helper()

	path := filepath.Join("..", ".github", "workflows", workflow)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				With struct {
					Prompt     string `yaml:"prompt"`
					ClaudeArgs string `yaml:"claude_args"`
				} `yaml:"with"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}

	var out []string
	for _, job := range wf.Jobs {
		for _, step := range job.Steps {
			for _, text := range []string{step.With.Prompt, step.With.ClaudeArgs} {
				if strings.TrimSpace(text) != "" {
					out = append(out, text)
				}
			}
		}
	}
	if len(out) == 0 {
		t.Fatalf("%s carries no agent prompt or claude_args at all — the workflow was "+
			"restructured, so re-derive these assertions rather than deleting them", workflow)
	}
	return out
}

// Every flow that puts comment text in front of an agent must name the filter. Without this the
// helper can be added, tested, and then silently never called — which is indistinguishable from not
// having written it.
func TestTrustedComments_EveryAIFlowNamesTheFilter(t *testing.T) {
	for workflow, want := range map[string]string{
		// The two dispatched phases pass their own `prompt:`, so they invoke the script directly.
		"deliver-verify.yml":  "deliver-trusted-comments.sh",
		"deliver-correct.yml": "deliver-trusted-comments.sh",
		// claude.yml runs in TAG mode: a workflow step writes the digest and the appended system
		// prompt names the file, because the action assembles the thread itself.
		"claude.yml": "trusted-comments.md",
	} {
		t.Run(workflow, func(t *testing.T) {
			found := false
			for _, text := range agentInstructions(t, workflow) {
				if strings.Contains(text, want) {
					found = true
				}
			}
			if !found {
				t.Errorf("no agent instruction in %s names %q, so the comment filter is not "+
					"reaching this flow's agent — any GitHub user's comment text can still steer "+
					"it (#1806)", workflow, want)
			}
		})
	}
}

// The other half: no agent prompt may ALSO hand the agent a raw, unfiltered comment read. Leaving one
// in place beside the filter re-opens the hole — the agent would simply run the call it was shown.
// Deliberately matched on the raw API/CLI shapes rather than on any mention of "comment", so the
// prompts can go on discussing comments at length (they must) while still being unable to instruct an
// unfiltered fetch.
func TestTrustedComments_NoAgentPromptInstructsAnUnfilteredCommentRead(t *testing.T) {
	// Each entry is a shape that, appearing in an agent's instructions, tells it to read the raw
	// thread. `gh pr view … --comments` and a `gh api` path ending in `/comments` are the two the
	// prompts used before #1806.
	forbidden := []string{
		"--comments",
		"/comments?",
		"/comments\"",
		"/comments'",
	}
	for _, workflow := range []string{"deliver-verify.yml", "deliver-correct.yml", "claude.yml"} {
		t.Run(workflow, func(t *testing.T) {
			for _, text := range agentInstructions(t, workflow) {
				for _, shape := range forbidden {
					if strings.Contains(text, shape) {
						t.Errorf("an agent instruction in %s still contains %q. A raw comment read "+
							"beside the filter re-opens the injection surface, because the agent "+
							"will run the call it was shown (#1806).\ninstruction:\n%s",
							workflow, shape, text)
					}
				}
			}
		})
	}
}

// claude.yml's two agent jobs are kept step-for-step identical by
// TestClaudeWorkflow_AgentJobsStayInSync; this pins that the filter step is one of those steps in
// the first place. The review path is the one that matters most to get right here — it is the path a
// `/blis-pr-review` on a public PR takes — and it is also the easier one to forget, since it is the
// duplicate.
func TestTrustedComments_ClaudeWorkflowRunsTheFilterInBothAgentJobs(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "claude.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Name string `yaml:"name"`
				Run  string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}

	for _, job := range []string{"claude", "claude-review"} {
		steps, ok := wf.Jobs[job]
		if !ok {
			t.Fatalf("job %q missing from %s", job, path)
		}
		found := false
		for _, step := range steps.Steps {
			if strings.Contains(step.Run, "deliver-trusted-comments.sh") {
				found = true
			}
		}
		if !found {
			t.Errorf("job %q in %s has no step running deliver-trusted-comments.sh, so nothing "+
				"records which comment authors were excluded for that trigger (#1806)", job, path)
		}
	}
}
