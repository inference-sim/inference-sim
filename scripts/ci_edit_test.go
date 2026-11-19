package scripts_test

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

// An edit of a pull request's title or body starts no CI run. The ruleset reads the newest CI run
// on the head commit, so a run that skips every job leaves the required checks "expected" and
// blocks the merge (#2261). A retarget is also an edit; a push or a close and reopen runs CI.
// The checks that read the text (blis.yml, private-names.yml, golden-change.yml) still run on
// an edit. The text checks' conditions are evaluated for an edit, so the tests check what a run
// does and not how the files are written.

type ghEvent struct{ action string }

var (
	eventPush = ghEvent{action: "synchronize"}
	eventEdit = ghEvent{action: "edited"}
)

type wfJob struct {
	Name string `yaml:"name"`
	If   string `yaml:"if"`
}

type wfFile struct {
	On   map[string]any   `yaml:"on"`
	Jobs map[string]wfJob `yaml:"jobs"`
}

func readWf(t *testing.T, text string) wfFile {
	t.Helper()
	var w wfFile
	if err := yaml.Unmarshal([]byte(text), &w); err != nil {
		t.Fatal(err)
	}
	return w
}

// evalExpr evaluates the ${{ }} expressions of s for the event, over the few names the workflows
// use here: github.event.action, github.event.pull_request.number,
// github.ref, always(), strings, ==, !=, &&, ||, ! and parentheses.
func evalExpr(t *testing.T, s string, ev ghEvent) string {
	t.Helper()
	re := regexp.MustCompile(`\$\{\{(.*?)\}\}`)
	return re.ReplaceAllStringFunc(s, func(m string) string {
		p := &editExprParser{t: t, ev: ev, src: strings.TrimSpace(m[3 : len(m)-2])}
		v := p.or()
		p.skip()
		if p.i != len(p.src) {
			t.Fatalf("cannot evaluate %q (stopped at %d)", m, p.i)
		}
		return v
	})
}

type editExprParser struct {
	t   *testing.T
	ev  ghEvent
	src string
	i   int
}

func editTruthy(v string) bool { return v != "" && v != "false" && v != "0" }
func (p *editExprParser) skip() {
	for p.i < len(p.src) && p.src[p.i] == ' ' {
		p.i++
	}
}
func (p *editExprParser) eat(tok string) bool {
	p.skip()
	if strings.HasPrefix(p.src[p.i:], tok) {
		p.i += len(tok)
		return true
	}
	return false
}
func (p *editExprParser) or() string {
	v := p.and()
	for p.eat("||") {
		r := p.and()
		if !editTruthy(v) {
			v = r
		}
	}
	return v
}
func (p *editExprParser) and() string {
	v := p.cmp()
	for p.eat("&&") {
		r := p.cmp()
		if editTruthy(v) {
			v = r
		}
	}
	return v
}
func (p *editExprParser) cmp() string {
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
func (p *editExprParser) unary() string {
	p.skip()
	if p.i < len(p.src) && p.src[p.i] == '!' && !strings.HasPrefix(p.src[p.i:], "!=") {
		p.i++
		return fmt.Sprint(!editTruthy(p.unary()))
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
	case "github.run_id":
		return "100"
	case "inputs.save_cache", "github.event.inputs.save_cache":
		return ""
	}
	if strings.HasPrefix(name, "needs.") && strings.Contains(name, ".outputs.") {
		return "true" // a selection output of a job that ran: every group on
	}
	p.t.Fatalf("unknown name %q in %q", name, p.src)
	return ""
}

// The jobs branch protection reads. Each reports one fixed name on every event.
var requiredNamed = map[string][]string{
	"ci.yml": {"build", "lint", "test-workload-cmd"},
}

func editCIFindings(t *testing.T, files map[string]string) []string {
	t.Helper()
	var bad []string
	add := func(f, s string) { bad = append(bad, f+": "+s) }
	for _, f := range []string{"ci.yml"} {
		w := readWf(t, files[f])
		pr, _ := w.On["pull_request"].(map[string]any)
		types, _ := pr["types"].([]any)
		if slices.Contains(types, any("edited")) {
			add(f, "runs for an edited pull request; that run is the newest on the head commit and reports no required check (#2261)")
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
			if evalExpr(t, name, eventEdit) != evalExpr(t, name, eventPush) {
				add(f, "job "+id+" has a name that depends on the event, so the ruleset may not find the required name")
			}
		}
	}
	for _, f := range []string{"blis.yml", "private-names.yml", "golden-change.yml"} {
		w := readWf(t, files[f])
		pr, _ := w.On["pull_request"].(map[string]any)
		types, _ := pr["types"].([]any)
		if !slices.Contains(types, any("edited")) {
			add(f, "does not run for an edited pull request, but its check reads the title and body")
		}
		for id, j := range w.Jobs {
			if j.If != "" && !editTruthy(evalExpr(t, j.If, eventEdit)) && strings.Contains(j.If, "edited") {
				add(f, "job "+id+" is skipped on an edit")
			}
		}
	}
	return bad
}

func editCIFiles(t *testing.T) map[string]string {
	t.Helper()
	m := map[string]string{}
	for _, f := range []string{"ci.yml", "blis.yml", "private-names.yml", "golden-change.yml"} {
		b, err := os.ReadFile(filepath.Join("..", ".github", "workflows", f))
		if err != nil {
			t.Fatal(err)
		}
		m[f] = string(b)
	}
	return m
}

func TestAnEditOfThePullRequestTextStartsNoCIRun(t *testing.T) {
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
	t.Run("ci.yml runs on an edit", func(t *testing.T) {
		has(t, mutate("ci.yml", "types: [opened, synchronize, reopened, ready_for_review]", "types: [opened, edited, synchronize, reopened, ready_for_review]"), "runs for an edited pull request")
	})
	t.Run("a required check changes its name on an edit", func(t *testing.T) {
		has(t, mutate("ci.yml", `name: "lint"`, `name: "lint${{ github.event.action == 'edited' && ' (edit)' || '' }}"`), "job lint has a name that depends on the event")
	})
	t.Run("the text checks must run on an edit", func(t *testing.T) {
		has(t, mutate("private-names.yml", "[opened, edited, synchronize, reopened]", "[opened, synchronize, reopened]"), "does not run for an edited pull request")
	})
}
