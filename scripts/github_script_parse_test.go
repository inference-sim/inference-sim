package scripts_test

// Every `actions/github-script` step carries its JavaScript as an inline YAML string, and the
// Go tests that decode the workflows assert on that text as text: a script that does not parse
// matches them as well as one that does, and the first run of it is on a real trigger (#1811).
// This test parses each body with Node, wrapped as actions/github-script wraps it, an
// AsyncFunction of (github, context, core, glob, io, exec, require, __original_require__,
// __dirname...), so top-level await is legal.
//
// It needs a Node that parses current syntax (optional chaining): the runner's own can be too
// old, so CI pins one with actions/setup-node and sets REQUIRE_NODE=1, under which a missing
// Node fails the test instead of skipping it (X16).

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type scriptStep struct {
	Name string         `yaml:"name"`
	Uses string         `yaml:"uses"`
	With map[string]any `yaml:"with"`
}

// githubScriptBodies returns "file:job/step" and the body of every github-script step.
func githubScriptBodies(t *testing.T) map[string]string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no workflow files found: %v", err)
	}
	out := map[string]string{}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var wf struct {
			Jobs map[string]struct {
				Steps []scriptStep `yaml:"steps"`
			} `yaml:"jobs"`
		}
		if err := yaml.Unmarshal(raw, &wf); err != nil {
			t.Fatalf("parse %s: %v", f, err)
		}
		for job, j := range wf.Jobs {
			for i, s := range j.Steps {
				if !strings.HasPrefix(s.Uses, "actions/github-script@") {
					continue
				}
				body, _ := s.With["script"].(string)
				out[filepath.Base(f)+":"+job+"/"+strings.Join(strings.Fields(s.Name), " ")+" #"+strconv.Itoa(i)] = body
			}
		}
	}
	return out
}

// wrapper makes node --check read the body as an AsyncFunction body.
const asyncWrapper = `const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('github', 'context', 'core', 'glob', 'io', 'exec', 'require', '__original_require__', %s);
`

func TestGithubScriptBodies_ParseUnderNode(t *testing.T) {
	bodies := githubScriptBodies(t)
	if len(bodies) < 10 {
		t.Fatalf("non-vacuity: found %d github-script steps, expected about 16", len(bodies))
	}
	node, err := exec.LookPath("node")
	if err != nil {
		if os.Getenv("REQUIRE_NODE") != "" {
			t.Fatalf("REQUIRE_NODE is set and node is not installed: %v", err)
		}
		t.Skip("node is not installed; CI runs this with a pinned Node (REQUIRE_NODE=1)")
	}
	dir := t.TempDir()
	for name, body := range bodies {
		quoted, err := jsString(body)
		if err != nil {
			t.Fatal(err)
		}
		src := filepath.Join(dir, "check.js")
		if err := os.WriteFile(src, []byte(strings.Replace(asyncWrapper, "%s", quoted, 1)), 0o644); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command(node, src).CombinedOutput(); err != nil {
			t.Errorf("%s: the inline script does not parse as an async function body:\n%s", name, out)
		}
	}
}

// A body that does not parse is caught; one that uses top-level await and optional chaining is
// accepted. Without this the test could pass against a Node that checks nothing.
func TestGithubScriptBodies_NodeRejectsASyntaxError(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		if os.Getenv("REQUIRE_NODE") != "" {
			t.Fatalf("REQUIRE_NODE is set and node is not installed: %v", err)
		}
		t.Skip("node is not installed")
	}
	dir := t.TempDir()
	run := func(body string) error {
		quoted, err := jsString(body)
		if err != nil {
			t.Fatal(err)
		}
		src := filepath.Join(dir, "c.js")
		if err := os.WriteFile(src, []byte(strings.Replace(asyncWrapper, "%s", quoted, 1)), 0o644); err != nil {
			t.Fatal(err)
		}
		return exec.Command(node, src).Run()
	}
	if err := run("const x = context.payload.comment?.body ?? '';\nawait github.rest.issues.get({});"); err != nil {
		t.Errorf("a valid body (top-level await, optional chaining) was rejected: %v", err)
	}
	if err := run("const x = ;"); err == nil {
		t.Error("a body with a syntax error was accepted")
	}
}

// jsString is s as a JavaScript string literal (a JSON string is one).
func jsString(s string) (string, error) {
	b, err := json.Marshal(s)
	return string(b), err
}
