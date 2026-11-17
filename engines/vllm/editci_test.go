// Original to blis-vllm.

package blisvllm

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// An edit of a pull request's title or body (not of its base) starts the pull-request workflows
// too, because a retarget is also an edit. That run must do nothing and must not cancel the long
// run in progress: it has its own concurrency group and every job is skipped. A retarget runs the
// full set. The checks that read the text (blis.yml, private-names.yml) still run on an edit.
// These tests evaluate the workflows' own expressions for three events, so they check what a run
// does and not how the files are written.

type ghEvent struct{ action, changesBase string }

var (
	eventPush     = ghEvent{action: "synchronize"}
	eventEdit     = ghEvent{action: "edited"}
	eventRetarget = ghEvent{action: "edited", changesBase: "main"}
)

type wfJob struct {
	Name  string `yaml:"name"`
	If    string `yaml:"if"`
	Needs any    `yaml:"needs"`
}

type wfFile struct {
	On          map[string]any         `yaml:"on"`
	Concurrency struct{ Group string } `yaml:"concurrency"`
	Jobs        map[string]wfJob       `yaml:"jobs"`
}

func readWf(t *testing.T, text string) wfFile {
	t.Helper()
	var w wfFile
	if err := yaml.Unmarshal([]byte(text), &w); err != nil {
		t.Fatal(err)
	}
	return w
}

func (j wfJob) needs() []string {
	switch n := j.Needs.(type) {
	case string:
		return []string{n}
	case []any:
		var out []string
		for _, x := range n {
			out = append(out, fmt.Sprint(x))
		}
		return out
	}
	return nil
}

// evalExpr evaluates the ${{ }} expressions of s for the event, over the few names the workflows
// use here: github.event.action, github.event.changes.base, github.event.pull_request.number,
// github.ref, always(), strings, ==, !=, &&, ||, ! and parentheses.
func evalExpr(t *testing.T, s string, ev ghEvent) string {
	t.Helper()
	re := regexp.MustCompile(`\$\{\{(.*?)\}\}`)
	return re.ReplaceAllStringFunc(s, func(m string) string {
		p := &exprParser{t: t, ev: ev, src: strings.TrimSpace(m[3 : len(m)-2])}
		v := p.or()
		p.skip()
		if p.i != len(p.src) {
			t.Fatalf("cannot evaluate %q (stopped at %d)", m, p.i)
		}
		return v
	})
}

type exprParser struct {
	t   *testing.T
	ev  ghEvent
	src string
	i   int
}

func truthy(v string) bool { return v != "" && v != "false" && v != "0" }
func (p *exprParser) skip() {
	for p.i < len(p.src) && p.src[p.i] == ' ' {
		p.i++
	}
}
func (p *exprParser) eat(tok string) bool {
	p.skip()
	if strings.HasPrefix(p.src[p.i:], tok) {
		p.i += len(tok)
		return true
	}
	return false
}
func (p *exprParser) or() string {
	v := p.and()
	for p.eat("||") {
		r := p.and()
		if !truthy(v) {
			v = r
		}
	}
	return v
}
func (p *exprParser) and() string {
	v := p.cmp()
	for p.eat("&&") {
		r := p.cmp()
		if truthy(v) {
			v = r
		}
	}
	return v
}
func (p *exprParser) cmp() string {
	v := p.unary()
	for {
		switch {
		case p.eat("!="):
			v = fmt.Sprint(v != p.unary())
		case p.eat("=="):
			v = fmt.Sprint(v == p.unary())
		default:
			return v
		}
	}
}
func (p *exprParser) unary() string {
	p.skip()
	if p.i < len(p.src) && p.src[p.i] == '!' && !strings.HasPrefix(p.src[p.i:], "!=") {
		p.i++
		return fmt.Sprint(!truthy(p.unary()))
	}
	if p.eat("(") {
		v := p.or()
		if !p.eat(")") {
			p.t.Fatalf("missing ) in %q", p.src)
		}
		return v
	}
	if p.i < len(p.src) && p.src[p.i] == '\'' {
		j := strings.IndexByte(p.src[p.i+1:], '\'')
		v := p.src[p.i+1 : p.i+1+j]
		p.i += j + 2
		return v
	}
	rest := p.src[p.i:]
	n := 0
	for n < len(rest) && (rest[n] == '.' || rest[n] == '_' || rest[n] == '(' || rest[n] == ')' || rest[n] >= 'a' && rest[n] <= 'z' || rest[n] >= 'A' && rest[n] <= 'Z' || rest[n] >= '0' && rest[n] <= '9') {
		if rest[n] == ')' && !strings.HasSuffix(rest[:n], "(") {
			break
		}
		n++
	}
	name := rest[:n]
	p.i += n
	switch name {
	case "github.event.action":
		return p.ev.action
	case "github.event.changes.base":
		return p.ev.changesBase
	case "github.event.pull_request.number":
		return "7"
	case "github.ref":
		return "refs/pull/7/merge"
	case "github.event_name":
		return "pull_request"
	case "always()":
		return "true"
	case "cancelled()":
		return "false"
	}
	p.t.Fatalf("unknown name %q in %q", name, p.src)
	return ""
}

// A condition that names a status function drops the implicit success() of a job, so the job
// runs although a job it needs was skipped.
var statusFunc = regexp.MustCompile(`\b(always|cancelled|failure|success)\(\)`)

// jobsThatRun returns the jobs a run for the event starts: a job runs when its condition holds
// and each job it needs ran, unless its condition names always().
func jobsThatRun(t *testing.T, w wfFile, ev ghEvent) map[string]bool {
	t.Helper()
	ran := map[string]bool{}
	for changed := true; changed; {
		changed = false
		ids := make([]string, 0, len(w.Jobs))
		for id := range w.Jobs {
			ids = append(ids, id)
		}
		slices.Sort(ids)
		for _, id := range ids {
			if ran[id] {
				continue
			}
			j := w.Jobs[id]
			cond := truthy(evalExpr(t, j.If, ev)) || j.If == ""
			if !statusFunc.MatchString(j.If) {
				for _, n := range j.needs() {
					cond = cond && ran[n]
				}
			}
			if cond {
				ran[id] = true
				changed = true
			}
		}
	}
	return ran
}

// The jobs branch protection reads. On an edit their names change, so a skipped check run of the
// edit is not named like one: the newest check run of a name is the one read.
var requiredNamed = map[string][]string{
	"ci.yml":       {"go", "lint", "oracle-tier2", "metrics-manifest", "goldens-reviewed"},
	"ci-light.yml": nil,
}

func editCIFindings(t *testing.T, files map[string]string) []string {
	t.Helper()
	var bad []string
	add := func(f, s string) { bad = append(bad, f+": "+s) }
	for _, f := range []string{"ci.yml", "ci-light.yml"} {
		w := readWf(t, files[f])
		push, edit, retarget := evalExpr(t, w.Concurrency.Group, eventPush), evalExpr(t, w.Concurrency.Group, eventEdit), evalExpr(t, w.Concurrency.Group, eventRetarget)
		if edit == push {
			add(f, "an edit of the title or body shares the concurrency group of the run in progress, so it cancels it")
		}
		if retarget != push {
			add(f, "a retarget has a group of its own, so it can run beside the run it replaces")
		}
		if ran := jobsThatRun(t, w, eventEdit); len(ran) > 0 {
			var ids []string
			for id := range ran {
				ids = append(ids, id)
			}
			slices.Sort(ids)
			add(f, "an edit of the title or body runs jobs: "+strings.Join(ids, ", "))
		}
		ranR, ranP := jobsThatRun(t, w, eventRetarget), jobsThatRun(t, w, eventPush)
		for id := range ranP {
			if !ranR[id] {
				add(f, "a retarget does not run job "+id+", which a push runs")
			}
		}
		for _, id := range requiredNamed[f] {
			j, ok := w.Jobs[id]
			if !ok {
				add(f, "has no job "+id)
				continue
			}
			name := j.Name
			if name == "" {
				name = id
			}
			if evalExpr(t, name, eventEdit) == evalExpr(t, name, eventPush) {
				add(f, "job "+id+" has the same name on an edit, so its skipped check run stands in for the run in progress")
			}
		}
	}
	for _, f := range []string{"blis.yml", "private-names.yml"} {
		w := readWf(t, files[f])
		pr, _ := w.On["pull_request"].(map[string]any)
		types, _ := pr["types"].([]any)
		if !slices.Contains(types, any("edited")) {
			add(f, "does not run for an edited pull request, but its check reads the title and body")
		}
		for id, j := range w.Jobs {
			if j.If != "" && !truthy(evalExpr(t, j.If, eventEdit)) && j.If != "" && strings.Contains(j.If, "edited") {
				add(f, "job "+id+" is skipped on an edit")
			}
		}
	}
	return bad
}

func editCIFiles(t *testing.T) map[string]string {
	t.Helper()
	m := map[string]string{}
	for _, f := range []string{"ci.yml", "ci-light.yml", "blis.yml", "private-names.yml"} {
		b, err := os.ReadFile(filepath.Join(".github", "workflows", f))
		if err != nil {
			t.Fatal(err)
		}
		m[f] = string(b)
	}
	return m
}

func TestAnEditOfTheTitleOrBodyLeavesTheRunningCIAlone(t *testing.T) {
	files := editCIFiles(t)
	if bad := editCIFindings(t, files); len(bad) > 0 {
		t.Fatalf("findings:\n%s", strings.Join(bad, "\n"))
	}
	mutate := func(file, old, new string) map[string]string {
		m := map[string]string{}
		for k, v := range files {
			m[k] = v
		}
		if !strings.Contains(m[file], old) {
			t.Fatalf("%s: %q not found", file, old)
		}
		m[file] = strings.Replace(m[file], old, new, 1)
		return m
	}
	has := func(t *testing.T, m map[string]string, want string) {
		t.Helper()
		if got := strings.Join(editCIFindings(t, m), "\n"); !strings.Contains(got, want) {
			t.Errorf("no finding contains %q; got:\n%s", want, got)
		}
	}
	t.Run("an edit that shares the group cancels the run in progress", func(t *testing.T) {
		has(t, mutate("ci.yml", "${{ github.event.action == 'edited' && !github.event.changes.base && '-edit' || '' }}\n  cancel-in-progress", "\n  cancel-in-progress"), "shares the concurrency group")
	})
	t.Run("a job that runs on an edit", func(t *testing.T) {
		has(t, mutate("ci-light.yml", "light-lint:\n    name: \"light - golangci-lint${{ github.event.action == 'edited' && !github.event.changes.base && ' (edit)' || '' }}\"\n    if: ${{ github.event.action != 'edited' || github.event.changes.base }}\n", "light-lint:\n    name: \"light - golangci-lint${{ github.event.action == 'edited' && !github.event.changes.base && ' (edit)' || '' }}\"\n"), "runs jobs: light-lint")
	})
	t.Run("a skipped required check keeps its name on an edit", func(t *testing.T) {
		has(t, mutate("ci.yml", "name: \"lint${{ github.event.action == 'edited' && !github.event.changes.base && ' (edit)' || '' }}\"", "name: lint"), "job lint has the same name on an edit")
	})
	t.Run("a retarget that starts nothing", func(t *testing.T) {
		has(t, mutate("ci.yml", "if: ${{ github.event.action != 'edited' || github.event.changes.base }}\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    outputs:", "if: ${{ github.event.action != 'edited' }}\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    outputs:"), "retarget")
	})
	t.Run("the text checks must run on an edit", func(t *testing.T) {
		has(t, mutate("private-names.yml", "[opened, edited, synchronize, reopened]", "[opened, synchronize, reopened]"), "does not run for an edited pull request")
	})
}
