package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// repo is a local git repository for a test.
type repo struct {
	t   *testing.T
	dir string
}

func newRepo(t *testing.T, gomod string) *repo {
	t.Helper()
	r := &repo{t, t.TempDir()}
	r.git("init", "-q", "-b", "main")
	if gomod != "" {
		r.write("go.mod", gomod)
	}
	r.write("README.md", "x\n")
	r.commit("Initial commit")
	return r
}

func (r *repo) git(args ...string) string {
	r.t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = r.dir
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=Ada Person", "GIT_AUTHOR_EMAIL=ada@example.com",
		"GIT_COMMITTER_NAME=Ada Person", "GIT_COMMITTER_EMAIL=ada@example.com",
		"GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_SYSTEM=/dev/null")
	out, err := cmd.CombinedOutput()
	if err != nil {
		r.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (r *repo) write(name, content string) {
	r.t.Helper()
	p := filepath.Join(r.dir, name)
	os.MkdirAll(filepath.Dir(p), 0o755)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		r.t.Fatal(err)
	}
}

func (r *repo) commit(msg string) string {
	r.git("add", "-A")
	r.git("commit", "-q", "-m", msg)
	return r.git("rev-parse", "HEAD")
}

// branch makes a branch with one change and returns its head; main is left where it was.
func (r *repo) branch(name, file, content, msg string) string {
	r.git("checkout", "-q", "-b", name)
	r.write(file, content)
	h := r.commit(msg)
	r.git("checkout", "-q", "main")
	return h
}

func registry(t *testing.T) *invariants.Registry {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

// allPassed returns a result for every ci check the registry lists for a repository, all
// passing.
func allPassed(reg *invariants.Registry, repo string) map[string]bool {
	m := map[string]bool{}
	for _, inv := range reg.Invariants {
		for _, c := range inv.Checks {
			if c.Repo == repo && c.Status == invariants.StatusCI {
				m[c.Test] = true
			}
		}
	}
	return m
}

func env(t *testing.T, bugs map[string][]Bug) Env {
	reg := registry(t)
	return Env{
		Registry: reg,
		Passed:   func(repo, _ string) (map[string]bool, error) { return allPassed(reg, repo), nil },
		Bugs:     func(repo string) ([]Bug, error) { return bugs[repo], nil },
	}
}

const subMod = "module github.com/inference-sim/blis-substrate\n\ngo 1.24\n"
const vllmMod = "module github.com/inference-sim/blis-vllm\n\ngo 1.27.2\n\nrequire github.com/inference-sim/blis-substrate v0.11.0\n"
const simMod = "module github.com/inference-sim/inference-sim\n\ngo 1.27.2\n\nrequire (\n\tgithub.com/inference-sim/blis-substrate v0.11.0\n\tgithub.com/inference-sim/blis-vllm v0.32.0-blis.2\n)\n"

// A review set across three repositories is planned bottom-up, each tag follows its scheme,
// each upper repository pins the tags the set gives the lower ones, and a dry run changes
// nothing.
func TestReviewSetIsPlannedBottomUp(t *testing.T) {
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	subHead := sub.branch("m2/units", "contracts/units.go", "package contracts\n", "Add unit types\n\nSUB-16.")
	vllm := newRepo(t, vllmMod)
	vllm.git("tag", "v0.32.0-blis.1")
	vllm.git("tag", "v0.32.0-blis.2")
	vllmHead := vllm.branch("m2/units", "core/a.go", "package core\n", "Use unit types")
	sim := newRepo(t, simMod)
	sim.git("tag", "v0.16.0")
	simHead := sim.branch("m2/units", "sim/a.go", "package sim\n", "Use unit types")

	lock := `set: units-as-types
milestone: 1
repos:
  - {repo: inference-sim, branch: m2/units, head: ` + simHead + `, bump: minor}
  - {repo: blis-vllm, branch: m2/units, head: ` + vllmHead + `, upstream: v0.32.0}
  - {repo: blis-substrate, branch: m2/units, head: ` + subHead + `, bump: minor}
`
	l, err := ParseStackLock([]byte(lock))
	if err != nil {
		t.Fatal(err)
	}
	dirs := map[string]string{"blis-substrate": sub.dir, "blis-vllm": vllm.dir, "inference-sim": sim.dir}
	steps, err := Plan(env(t, nil), l, dirs, "main")
	if err != nil {
		t.Fatal(err)
	}
	order := []string{steps[0].Repo, steps[1].Repo, steps[2].Repo}
	if strings.Join(order, ",") != "blis-substrate,blis-vllm,inference-sim" {
		t.Fatalf("order %v; the set is released bottom-up (X10)", order)
	}
	want := []string{"v0.12.0", "v0.32.0-blis.3", "v0.17.0"}
	for i, st := range steps {
		if st.Tag != want[i] {
			t.Errorf("%s: tag %s, want %s", st.Repo, st.Tag, want[i])
		}
		if len(st.Refusals) > 0 {
			t.Errorf("%s: refused: %v", st.Repo, st.Refusals)
		}
	}
	if steps[1].Pins["github.com/inference-sim/blis-substrate"] != "v0.12.0" {
		t.Errorf("blis-vllm pins %v; it must pin the set's substrate tag", steps[1].Pins)
	}
	if steps[2].Pins["github.com/inference-sim/blis-vllm"] != "v0.32.0-blis.3" || steps[2].Pins["github.com/inference-sim/blis-substrate"] != "v0.12.0" {
		t.Errorf("inference-sim pins %v", steps[2].Pins)
	}
	if !strings.Contains(steps[0].Notes, "SUB-16") || !strings.Contains(steps[0].Notes, "Changed contracts\n\n- contracts/units.go") {
		t.Errorf("substrate notes do not list the cited invariant and the changed contract:\n%s", steps[0].Notes)
	}
	// The dry run left every repository as it was.
	for _, r := range []*repo{sub, vllm, sim} {
		if strings.Contains(r.git("tag", "--list"), "v0.12.0") || strings.Contains(r.git("tag", "--list"), "blis.3") || strings.Contains(r.git("tag", "--list"), "v0.17.0") {
			t.Errorf("%s: a dry run made a tag", r.dir)
		}
		if r.git("rev-parse", "main") == r.git("rev-parse", "m2/units") {
			t.Errorf("%s: a dry run merged", r.dir)
		}
	}

	// Applying merges, pins once and tags, locally, in order; nothing is pushed (the test
	// repositories have no remote).
	if err := Apply(steps, false); err != nil {
		t.Fatal(err)
	}
	if got := sim.git("show", "v0.17.0:go.mod"); !strings.Contains(got, "github.com/inference-sim/blis-vllm v0.32.0-blis.3") || !strings.Contains(got, "github.com/inference-sim/blis-substrate v0.12.0") {
		t.Errorf("inference-sim's tagged go.mod:\n%s", got)
	}
	if msg := sub.git("tag", "-l", "--format=%(contents)", "v0.12.0"); !strings.Contains(msg, "blis-substrate v0.12.0") {
		t.Errorf("the tag message is not the release notes:\n%s", msg)
	}
	if a := sim.git("log", "-1", "--format=%an <%ae>", "v0.17.0"); !strings.Contains(a, "blis-release-bot") {
		t.Errorf("the pin commit is not the bot's: %s", a)
	}
	if strings.Contains(sim.git("log", "-1", "--format=%B", "v0.17.0"), "Signed-off-by") {
		t.Error("a release-bot commit carries no sign-off (X20)")
	}
}

// Each refusal has a known-bad input (X18).
func TestRefusals(t *testing.T) {
	t.Run("a ci check failed at the head (LIF-3)", func(t *testing.T) {
		sub := newRepo(t, subMod)
		sub.git("tag", "v0.11.0")
		h := sub.branch("b", "a.go", "package a\n", "Change")
		e := env(t, nil)
		e.Passed = func(repo, _ string) (map[string]bool, error) {
			m := allPassed(e.Registry, repo)
			m["TestNoNestedModule"] = false
			return m, nil
		}
		st := planOne(t, e, "blis-substrate", sub.dir, h, "bump: patch", 1)
		has(t, st.Refusals, `check "TestNoNestedModule" failed at this commit`)
	})
	t.Run("a ci check has no result (LIF-3)", func(t *testing.T) {
		sub := newRepo(t, subMod)
		h := sub.branch("b", "a.go", "package a\n", "Change")
		e := env(t, nil)
		e.Passed = func(string, string) (map[string]bool, error) { return map[string]bool{}, nil }
		st := planOne(t, e, "blis-substrate", sub.dir, h, "bump: patch", 1)
		has(t, st.Refusals, "has no recorded result")
	})
	t.Run("a reached target milestone is not met (design §13.3)", func(t *testing.T) {
		sub := newRepo(t, subMod)
		h := sub.branch("b", "a.go", "package a\n", "Change")
		// Within M1, M1's targets do not hold a release back.
		st := planOne(t, env(t, nil), "blis-substrate", sub.dir, h, "bump: patch", 1)
		for _, r := range st.Refusals {
			if strings.Contains(r, "is reached") {
				t.Errorf("a release within M1 is held to M1's targets: %s", r)
			}
		}
		// In M2, M1 has been reached.
		st = planOne(t, env(t, nil), "blis-substrate", sub.dir, h, "bump: patch", 2)
		has(t, st.Refusals, "milestone M1 is reached")
	})
	t.Run("an engine tag that skips N (LIF-5)", func(t *testing.T) {
		has(t, CheckEngineTag("v0.32.0-blis.4", []string{"v0.32.0-blis.1", "v0.32.0-blis.2"}), "N must be 3")
		has(t, CheckEngineTag("v0.32.0", nil), "is a plain tag")
		has(t, CheckEngineTag("v0.32.0-blis.0", nil), "is not v<upstream>-blis.N")
		has(t, CheckEngineTag("v0.32.0-blis.2", []string{"v0.32.0-blis.2"}), "already exists")
		if bad := CheckEngineTag("v0.33.0-blis.1", []string{"v0.32.0-blis.5"}); len(bad) > 0 {
			t.Errorf("a new upstream release starts at blis.1: %v", bad)
		}
		if bad := CheckEngineTag("v0.5.6-post.1.blis.1", []string{"v0.5.6-blis.3"}); len(bad) > 0 {
			t.Errorf("a post release counts N on its own: %v", bad)
		}
	})
	t.Run("an engine requires a newer substrate than inference-sim (design §3.2)", func(t *testing.T) {
		vllm := newRepo(t, strings.Replace(vllmMod, "v0.11.0", "v0.12.0", 1))
		vllm.git("tag", "v0.32.0-blis.2")
		h := vllm.branch("b", "core/a.go", "package core\n", "Change")
		sim := newRepo(t, simMod)
		lock := "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-vllm, branch: b, head: " + h + ", upstream: v0.32.0}\n"
		l, err := ParseStackLock([]byte(lock))
		if err != nil {
			t.Fatal(err)
		}
		steps, err := Plan(env(t, nil), l, map[string]string{"blis-vllm": vllm.dir, "inference-sim": sim.dir}, "main")
		if err != nil {
			t.Fatal(err)
		}
		has(t, steps[0].Refusals, "requires blis-substrate v0.12.0, newer than inference-sim's v0.11.0")
	})
	t.Run("a MINOR release at v0 with a deprecated feature (LIF-7)", func(t *testing.T) {
		sim := newRepo(t, simMod)
		sim.git("tag", "v0.15.0")
		h := sim.branch("b", "cmd/observe.go", "package cmd\n\nvar observeCmd = struct{ Deprecated string }{\n\tDeprecated: \"scheduled for removal (#1901)\",\n}\n", "Keep observe")
		st := planOne(t, env(t, nil), "inference-sim", sim.dir, h, "bump: minor", 1)
		has(t, st.Refusals, "cmd/observe.go:4: a deprecated feature remains")
		// A PATCH release may keep it.
		st = planOne(t, env(t, nil), "inference-sim", sim.dir, h, "bump: patch", 1)
		for _, r := range st.Refusals {
			if strings.Contains(r, "a deprecated feature remains") {
				t.Errorf("a PATCH release is refused: %s", r)
			}
		}
	})
	t.Run("a relaxed deprecated feature stays until M4 (plan §3.3)", func(t *testing.T) {
		sim := newRepo(t, simMod)
		sim.git("tag", "v0.16.0")
		h := sim.branch("b", "cmd/root.go", "package cmd\n\n// Deprecated: --scheduler is stated in the engine block (#2037).\nvar scheduler string\n", "Deprecate --scheduler")
		st := planOne(t, env(t, nil), "inference-sim", sim.dir, h, "bump: minor", 2)
		for _, r := range st.Refusals {
			if strings.Contains(r, "a deprecated feature remains") {
				t.Errorf("a relaxed feature is refused before M4: %s", r)
			}
		}
		st = planOne(t, env(t, nil), "inference-sim", sim.dir, h, "bump: minor", 4)
		has(t, st.Refusals, "cmd/root.go:3: a deprecated feature remains")
	})
	t.Run("an open invariant bug (X27)", func(t *testing.T) {
		sub := newRepo(t, subMod)
		h := sub.branch("b", "a.go", "package a\n", "Change")
		bugs := map[string][]Bug{"blis-substrate": {{Repo: "inference-sim/blis-substrate", Number: 7, Title: "SUB-5 broken"}}}
		st := planOne(t, env(t, bugs), "blis-substrate", sub.dir, h, "bump: patch", 1)
		has(t, st.Refusals, "open invariant bug inference-sim/blis-substrate#7")
		st = planOne(t, env(t, bugs), "blis-substrate", sub.dir, h, "bump: patch\n    fixes: [7]", 1)
		for _, r := range st.Refusals {
			if strings.Contains(r, "X27") {
				t.Errorf("a release that fixes only the bug is refused: %s", r)
			}
		}
		st = planOne(t, env(t, bugs), "blis-substrate", sub.dir, h, "bump: patch\n    waivers: [7]", 1)
		for _, r := range st.Refusals {
			if strings.Contains(r, "X27") {
				t.Errorf("a waived bug blocks: %s", r)
			}
		}
	})
	t.Run("apply refuses when anything is refused", func(t *testing.T) {
		if err := Apply([]Step{{Repo: "x", Refusals: []string{"no"}}}, false); err == nil {
			t.Fatal("applied a refused release")
		}
	})
}

// planOne plans a one-repository set and returns its step. extra is the YAML after head
// (bump or upstream, and optional fixes or waivers).
func planOne(t *testing.T, e Env, name, dir, head, extra string, milestone int) Step {
	t.Helper()
	top := ""
	if strings.Contains(extra, "milestone_release: true") {
		extra = strings.Replace(extra, "\nmilestone_release: true", "", 1)
		top = "milestone_release: true\n"
	}
	lock := "set: s\nmilestone: " + itoa(milestone) + "\n" + top + "repos:\n  - repo: " + name + "\n    branch: b\n    head: " + head + "\n    " + extra + "\n"
	l, err := ParseStackLock([]byte(lock))
	if err != nil {
		t.Fatal(err)
	}
	steps, err := Plan(e, l, map[string]string{name: dir}, "main")
	if err != nil {
		t.Fatal(err)
	}
	return steps[0]
}

func itoa(n int) string { return strconv.Itoa(n) }

func has(t *testing.T, got []string, want string) {
	t.Helper()
	if !strings.Contains(strings.Join(got, "\n"), want) {
		t.Errorf("no refusal contains %q; got:\n%s", want, strings.Join(got, "\n"))
	}
}

func TestStackLockIsStrict(t *testing.T) {
	h := strings.Repeat("a", 40)
	for name, doc := range map[string]string{
		"unknown repo":       "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-x, branch: b, head: " + h + ", bump: minor}\n",
		"short head":         "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: abc, bump: minor}\n",
		"engine with bump":   "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-vllm, branch: b, head: " + h + ", bump: minor}\n",
		"substrate upstream": "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", upstream: v1.0.0, bump: minor}\n",
		"twice":              "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: minor}\n  - {repo: blis-substrate, branch: c, head: " + h + ", bump: minor}\n",
		"unknown key":        "set: s\nmilestone: 1\nowner: me\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: minor}\n",
		"no milestone":       "set: s\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: minor}\n",
		"bad bump":           "set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: huge}\n",
	} {
		if _, err := ParseStackLock([]byte(doc)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The relaxation list is plan §3.3's: it ends at M4 and names exactly these features.
func TestRelaxationsMatchThePlan(t *testing.T) {
	want := []string{"engine_version", "compat/", "rules/v0_29/", "rules/v0_31/", "contracts/rules/",
		"contracts/kernel/enginesettings/", "scheduler", "preemption-policy", "dp-load-balancing",
		"swa-bounded-replay", "kv-offload-config", "lazy-generation", "eager"}
	var got []string
	for _, r := range relaxations {
		if r.Milestone != 4 {
			t.Errorf("%s ends at M%d; plan §3.3 relaxes until M4", r.Reason, r.Milestone)
		}
		k := r.Match
		if k == "" {
			k = r.Path
		}
		got = append(got, k)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("relaxations %v, want %v", got, want)
	}
}

func TestNextSemver(t *testing.T) {
	for _, c := range []struct {
		tags       []string
		bump, want string
	}{
		{nil, "minor", "v0.1.0"},
		{[]string{"v0.10.0", "v0.9.3", "v0.11.0-rc.1"}, "minor", "v0.11.0"},
		{[]string{"v0.11.0"}, "patch", "v0.11.1"},
		{[]string{"v0.11.4"}, "major", "v1.0.0"},
	} {
		got, err := NextSemver(c.tags, c.bump)
		if err != nil || got != c.want {
			t.Errorf("NextSemver(%v, %s) = %s, %v; want %s", c.tags, c.bump, got, err, c.want)
		}
	}
	if !breaksCompatibility("v0.15.0", "v0.16.0") || breaksCompatibility("v0.15.0", "v0.15.1") || !breaksCompatibility("v1.2.0", "v2.0.0") || breaksCompatibility("v1.2.0", "v1.3.0") {
		t.Error("breaksCompatibility disagrees with LIF-7")
	}
}

// The command's dry run against a local repository prints the plan and changes nothing.
func TestCommandDryRun(t *testing.T) {
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	h := sub.branch("b", "a.go", "package a\n", "Change")
	dir := t.TempDir()
	lock := filepath.Join(dir, "stack.lock")
	os.WriteFile(lock, []byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: "+h+", bump: patch}\n"), 0o644)
	reg := registry(t)
	res := `{"blis-substrate": {"` + h + `": {`
	var parts []string
	for k := range allPassed(reg, "blis-substrate") {
		parts = append(parts, `"`+k+`": true`)
	}
	res += strings.Join(parts, ",") + "}}}"
	resPath := filepath.Join(dir, "results.json")
	os.WriteFile(resPath, []byte(res), 0o644)
	var out strings.Builder
	code, err := run([]string{"-lock", lock, "-results", resPath, "-dir", "blis-substrate=" + sub.dir}, &out)
	if err != nil {
		t.Fatal(err)
	}
	if code != 0 || !strings.Contains(out.String(), "== blis-substrate: v0.11.1") || !strings.Contains(out.String(), "dry run: nothing changed") {
		t.Fatalf("code %d\n%s", code, out.String())
	}
	if strings.Contains(sub.git("tag", "--list"), "v0.11.1") {
		t.Error("a dry run made a tag")
	}
}

// The milestone release reaches its milestone: every invariant targeted at it must have its
// target status (design §13.3). Known-bad input: the M1 milestone release today.
func TestMilestoneReleaseIsHeldToItsTargets(t *testing.T) {
	sub := newRepo(t, subMod)
	h := sub.branch("b", "a.go", "package a\n", "Change")
	st := planOne(t, env(t, nil), "blis-substrate", sub.dir, h, "bump: minor\nmilestone_release: true", 1)
	has(t, st.Refusals, "X14: milestone M1 is reached and the invariant is new")
}
