package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/blis-substrate/ledger"
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
		Lint:     func(string) ([]string, error) { return nil, nil },
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

	// Applying works one layer at a time, locally, in order; nothing is pushed (the test
	// repositories have no remote). Each layer is planned again from the repositories as
	// they stand, as each command run does, and a layer's tag is "published" by the test.
	e := env(t, nil)
	published := map[string]bool{}
	e.TagPublished = func(_, tag string) bool { return published[tag] }
	for _, st := range steps {
		cur, err := Plan(e, l, dirs, "main")
		if err != nil {
			t.Fatal(err)
		}
		if err := Stage(e, cur, "", false); err != nil {
			t.Fatalf("stage %s: %v", st.Repo, err)
		}
		if strings.Contains(sub.git("tag", "--list")+sim.git("tag", "--list"), "v0.17.0") {
			t.Fatal("Stage made a tag of the top layer")
		}
		if err := Tag(e, l, cur, "", false); err != nil {
			t.Fatalf("tag %s: %v", st.Repo, err)
		}
		published[st.Tag] = true
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
		if err := Stage(Env{}, []Step{{Repo: "x", Refusals: []string{"no"}}}, "", false); err == nil {
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
	old := fetchCheckRuns
	t.Cleanup(func() { fetchCheckRuns = old })
	fetchCheckRuns = func(repo, _ string) (map[string]string, error) { return everyRunSucceeded(t, repo), nil }
	var out strings.Builder
	code, err := run([]string{"-lock", lock, "-check-runs", filepath.Join("..", "..", ".github", "check-runs.yaml"), "-dir", "blis-substrate=" + sub.dir}, &out)
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
	has(t, st.Refusals, "X14: milestone M1 is reached and its automated checks are new, not the target ci")
}

// A maintainer practice has no check run to wait for, so it neither blocks a milestone release
// nor counts toward an invariant's status there (design §13.1): an invariant whose only checks
// are practices is not held to a target by the gate, and one with automated checks is held by
// those alone.
func TestAPracticeIsNotHeldByTheMilestoneGate(t *testing.T) {
	one := 1
	mk := func(checks ...invariants.Check) *invariants.Registry {
		return &invariants.Registry{Invariants: []invariants.Invariant{{
			ID: "T-1", Checks: checks, Status: invariants.StatusNew,
			Target: invariants.Target{Status: invariants.StatusCI, Milestone: &one}}}}
	}
	practice := invariants.Check{Repo: "blis-substrate", Runs: invariants.RunsPractice, Status: invariants.StatusNew, Check: "branch protection"}
	autoNew := invariants.Check{Repo: "blis-substrate", Runs: invariants.RunsPullRequest, Status: invariants.StatusNew, Check: "a check", Test: "t"}
	autoCI := invariants.Check{Repo: "blis-substrate", Runs: invariants.RunsPullRequest, Status: invariants.StatusCI, Check: "a check", Test: "t"}
	passed := map[string]bool{"t": true}
	// A practice alone: nothing for the gate to hold, though its status is new.
	if bad := CheckInvariants(mk(practice), "blis-substrate", passed, 1, true); len(bad) != 0 {
		t.Errorf("a practice alone was held by the gate: %v", bad)
	}
	// A practice beside an automated check at ci: held by the automated status, which is met.
	if bad := CheckInvariants(mk(practice, autoCI), "blis-substrate", passed, 1, true); len(bad) != 0 {
		t.Errorf("a practice made an invariant whose automated checks are ci fail the gate: %v", bad)
	}
	// A practice beside an automated check still new: refused, naming the automated status.
	has(t, CheckInvariants(mk(practice, autoNew), "blis-substrate", passed, 1, true),
		"T-1: milestone M1 is reached and its automated checks are new, not the target ci")
	// A practice at status ci names no test and needs no recorded result.
	done := practice
	done.Status = invariants.StatusCI
	if bad := CheckInvariants(mk(done), "blis-substrate", map[string]bool{}, 1, false); len(bad) != 0 {
		t.Errorf("a practice at ci needed a recorded result: %v", bad)
	}
}

// An invariant whose target is after M4 is held by no milestone release from M1 to M4: its
// checks may be new at every milestone and the gate refuses nothing for it.
func TestATargetAfterM4IsNotHeldByAnyMilestoneRelease(t *testing.T) {
	after := invariants.PostM4
	reg := &invariants.Registry{Invariants: []invariants.Invariant{{
		ID: "T-1", Status: invariants.StatusNew,
		Checks: []invariants.Check{{Repo: "blis-substrate", Runs: invariants.RunsPullRequest, Status: invariants.StatusNew, Check: "a check", Test: "t"}},
		Target: invariants.Target{Status: invariants.StatusCI, Milestone: &after}}}}
	for m := 1; m <= 4; m++ {
		for _, release := range []bool{false, true} {
			if bad := CheckInvariants(reg, "blis-substrate", map[string]bool{"t": true}, m, release); len(bad) != 0 {
				t.Errorf("milestone %d (release %v) held an invariant targeted after M4: %v", m, release, bad)
			}
		}
	}
	// The same invariant targeted at M4 is held by the M4 release, so the exemption is the target's.
	four := 4
	reg.Invariants[0].Target.Milestone = &four
	has(t, CheckInvariants(reg, "blis-substrate", map[string]bool{"t": true}, 4, true), "milestone M4 is reached")
}

// stagedSet plans and stages a one-repository set and returns what Tag needs.
func stagedSet(t *testing.T, e Env) (*repo, *StackLock, []Step) {
	t.Helper()
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	h := sub.branch("b", "a.go", "package a\n", "Change")
	l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: patch}\n"))
	if err != nil {
		t.Fatal(err)
	}
	steps, err := Plan(e, l, map[string]string{"blis-substrate": sub.dir}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	return sub, l, steps
}

// LIF-3 regression: the tag lands on the commit whose results were checked, and a tip that
// moved after the check is refused. The results are looked up by the tip's own SHA.
func TestTagTargetIsTheCheckedCommit(t *testing.T) {
	var asked []string
	e := env(t, nil)
	reg := e.Registry
	e.Passed = func(repo, commit string) (map[string]bool, error) {
		asked = append(asked, commit)
		return allPassed(reg, repo), nil
	}
	sub, l, steps := stagedSet(t, e)
	tip := sub.git("rev-parse", "release/v0.11.1")
	if err := Tag(e, l, steps, "", false); err != nil {
		t.Fatal(err)
	}
	if got := sub.git("rev-list", "-n1", "v0.11.1"); got != tip {
		t.Errorf("tag v0.11.1 is on %s, the staged tip is %s", got, tip)
	}
	if len(asked) == 0 || asked[len(asked)-1] != tip {
		t.Errorf("results were asked for %v, want the tip %s last", asked, tip)
	}
}

func TestTagRefusesATipThatMovedAfterTheCheck(t *testing.T) {
	e := env(t, nil)
	reg := e.Registry
	sub, l, steps := stagedSet(t, e)
	checked := sub.git("rev-parse", "release/v0.11.1")
	// Results exist only for the commit that was checked.
	e.Passed = func(repo, commit string) (map[string]bool, error) {
		if commit != checked {
			return nil, nil
		}
		return allPassed(reg, repo), nil
	}
	sub.git("checkout", "-q", "release/v0.11.1")
	sub.write("late.go", "package late\n")
	sub.commit("A change after the check")
	if err := Tag(e, l, steps, "", false); err == nil {
		t.Fatal("tagged a tip with no results")
	}
	if strings.Contains(sub.git("tag", "--list"), "v0.11.1") {
		t.Error("a tag exists after a refusal")
	}
}

// Re-planning a head that is already released is refused (LIF-1).
func TestPlanRefusesAReleasedHead(t *testing.T) {
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	h := sub.branch("b", "a.go", "package a\n", "Change")
	sub.git("tag", "v0.12.0", h)
	st := planOne(t, env(t, nil), "blis-substrate", sub.dir, h, "bump: minor", 1)
	has(t, st.Refusals, "already carries the release tag v0.12.0")
}

// A requirement on a BLIS module at a pseudo-version, even one the set does not re-pin, is
// refused (X11, LIF-2).
func TestPlanRefusesAPseudoVersionRequirement(t *testing.T) {
	mod := "module github.com/inference-sim/inference-sim\n\ngo 1.27.2\n\nrequire github.com/inference-sim/blis-substrate v0.0.0-20261105193227-4a3becf2d09a\n"
	sim := newRepo(t, mod)
	sim.git("tag", "v0.16.0")
	h := sim.branch("b", "sim/a.go", "package sim\n", "Change")
	st := planOne(t, env(t, nil), "inference-sim", sim.dir, h, "bump: minor", 1)
	has(t, st.Refusals, "pseudo-version v0.0.0-20261105193227-4a3becf2d09a")
}

// The router fork's bare v1.0.0-blis tag counts as N=0, and an earlier upstream's tags are
// the previous release of a new upstream, so the notes do not say "First release".
func TestPreviousTagSeesTheBareAndEarlierUpstreamTags(t *testing.T) {
	if got := previousTag([]string{"v1.0.0-blis"}, KindRouter, "v1.0.0"); got != "v1.0.0-blis" {
		t.Errorf("previous of a bare-tag router = %q", got)
	}
	if got := previousTag([]string{"v1.0.0-blis", "v1.0.0-blis.1", "v1.0.0-blis.2"}, KindRouter, "v1.0.0"); got != "v1.0.0-blis.2" {
		t.Errorf("previous = %q", got)
	}
	if got := previousTag([]string{"v0.31.0-blis.4", "v0.29.0-blis.9"}, KindEngine, "v0.32.0"); got != "v0.31.0-blis.4" {
		t.Errorf("previous on a new upstream = %q", got)
	}
	// The first -blis.1 of a new router upstream: the baseline is the last release of the old
	// one, whether or not the retained bare tag is among the tags.
	old := []string{"v1.0.0-blis", "v1.0.0-blis.1", "v1.0.0-blis.2"}
	if got := previousTag(old, KindRouter, "v1.1.0"); got != "v1.0.0-blis.2" {
		t.Errorf("previous on a new router upstream = %q", got)
	}
	if got := previousTag([]string{"v1.0.0-blis"}, KindRouter, "v1.1.0"); got != "v1.0.0-blis" {
		t.Errorf("previous of a bare tag on a new router upstream = %q", got)
	}
}

// Retry identity (LIF-1): a head Stage merged is the same release when it is planned again,
// whether the run stopped after Stage (branch, no tag) or after Tag (tag on the staged tip).
// The version does not advance; a different head gets the next one.
func TestPlanKeepsTheTagOfAStagedOrReleasedHead(t *testing.T) {
	e := env(t, nil)
	sub, l, steps := stagedSet(t, e)
	head := l.Repos[0].Head
	replan := func(h string) Step {
		t.Helper()
		return planOne(t, e, "blis-substrate", sub.dir, h, "bump: patch", 1)
	}
	if steps[0].Tag != "v0.11.1" {
		t.Fatalf("first plan gave %s", steps[0].Tag)
	}
	t.Run("after Stage", func(t *testing.T) {
		st := replan(head)
		if st.Tag != "v0.11.1" || !st.Staged || len(st.Refusals) != 0 {
			t.Errorf("tag %s staged %v refusals %v; want the staged v0.11.1, unrefused", st.Tag, st.Staged, st.Refusals)
		}
	})
	if err := Tag(e, l, steps, "", false); err != nil {
		t.Fatal(err)
	}
	t.Run("after Tag", func(t *testing.T) {
		st := replan(head)
		if st.Tag != "v0.11.1" {
			t.Errorf("tag %s advanced; want v0.11.1", st.Tag)
		}
		has(t, st.Refusals, "already released at v0.11.1")
	})
	t.Run("a different head", func(t *testing.T) {
		sub.git("checkout", "-q", "main")
		h2 := sub.branch("c", "b.go", "package b\n", "Another change")
		st := replan(h2)
		if st.Tag != "v0.11.2" || st.Staged || len(st.Refusals) != 0 {
			t.Errorf("tag %s staged %v refusals %v; want v0.11.2", st.Tag, st.Staged, st.Refusals)
		}
	})
}

// A staged tip is validated on its selected module graph, with no deferral (X11, LIF-2). A
// finding the direct requirements of the review head cannot show, such as a transitive BLIS
// pseudo-version, stops Tag, and nothing is tagged.
func TestTagRefusesAStagedTipThatFailsTheModuleLint(t *testing.T) {
	e := env(t, nil)
	sub, l, steps := stagedSet(t, e)
	var linted string
	e.Lint = func(dir string) ([]string, error) {
		raw, err := os.ReadFile(filepath.Join(dir, "README.md"))
		linted = string(raw)
		if err != nil {
			return nil, err
		}
		return []string{"github.com/inference-sim/blis-vllm resolves to v0.32.1-0.20261105193227-4a3becf2d09a, which is not a release tag (X11, LIF-2)"}, nil
	}
	err := Tag(e, l, steps, "", false)
	if err == nil || !strings.Contains(err.Error(), "fails the module lint") {
		t.Fatalf("Tag = %v; want a module lint refusal", err)
	}
	if linted == "" {
		t.Error("the lint was not run on a checkout of the staged tip")
	}
	if strings.Contains(sub.git("tag", "--list"), "v0.11.1") {
		t.Error("a tag exists after a refusal")
	}
	e.Lint = nil
	if err := Tag(e, l, steps, "", false); err == nil {
		t.Error("Tag ran without a module lint")
	}
}

// The deferral file is removed in the change that re-pins; a staged tip that still holds it
// is refused.
func TestTagRefusesAStagedTipThatHoldsTheDeferral(t *testing.T) {
	e := env(t, nil)
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	sub.write(DeferralFile, "issue: x/y#1\n")
	sub.commit("Add the deferral")
	h := sub.branch("b", "a.go", "package a\n", "Change")
	l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: patch}\n"))
	if err != nil {
		t.Fatal(err)
	}
	steps, err := Plan(e, l, map[string]string{"blis-substrate": sub.dir}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	err = Tag(e, l, steps, "", false)
	if err == nil || !strings.Contains(err.Error(), "still holds "+DeferralFile) {
		t.Fatalf("Tag = %v; want a refusal for the deferral file", err)
	}
}

// With the real lint binary: a BLIS replace in a staged tip is found from the module graph.
func TestRealModlintRefusesABLISReplaceInAStagedTip(t *testing.T) {
	if testing.Short() {
		t.Skip("builds modlint")
	}
	bin := filepath.Join(t.TempDir(), "modlint")
	if out, err := exec.Command("go", "build", "-o", bin, "github.com/inference-sim/blis-substrate/tools/modlint").CombinedOutput(); err != nil {
		t.Fatalf("build modlint: %v\n%s", err, out)
	}
	lower := t.TempDir()
	for name, body := range map[string]string{
		"go.mod":              "module github.com/inference-sim/blis-substrate\n\ngo 1.24\n",
		"contracts/toolchain": "go1.27.2\n",
	} {
		p := filepath.Join(lower, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mod := subMod + "\nrequire github.com/inference-sim/blis-sub v0.32.0-blis.2\n\nreplace github.com/inference-sim/blis-sub => " + lower + "\n"
	e := env(t, nil)
	e.Lint = func(dir string) ([]string, error) { return runModlint(bin, dir) }
	sub := newRepo(t, mod)
	sub.write("contracts/toolchain", "go1.27.2\n")
	sub.commit("Add the toolchain file")
	sub.git("tag", "v0.11.0")
	h := sub.branch("b", "core/a.go", "package core\n", "Change")
	l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: patch}\n"))
	if err != nil {
		t.Fatal(err)
	}
	steps, err := Plan(e, l, map[string]string{"blis-substrate": sub.dir}, "main")
	if err != nil {
		t.Fatal(err)
	}
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	err = Tag(e, l, steps, "", false)
	if err == nil || !strings.Contains(err.Error(), "is replaced by") {
		t.Fatalf("Tag = %v; want the replace finding", err)
	}
}

// layered builds the three-layer set of TestReviewSetIsPlannedBottomUp.
func layered(t *testing.T) (*StackLock, map[string]string, map[string]*repo) {
	t.Helper()
	sub := newRepo(t, subMod)
	sub.git("tag", "v0.11.0")
	subHead := sub.branch("m2/units", "contracts/units.go", "package contracts\n", "Add unit types")
	vllm := newRepo(t, vllmMod)
	vllm.git("tag", "v0.32.0-blis.2")
	vllmHead := vllm.branch("m2/units", "core/a.go", "package core\n", "Use unit types")
	sim := newRepo(t, simMod)
	sim.git("tag", "v0.16.0")
	simHead := sim.branch("m2/units", "sim/a.go", "package sim\n", "Use unit types")
	l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: inference-sim, branch: m2/units, head: " + simHead + ", bump: minor}\n  - {repo: blis-vllm, branch: m2/units, head: " + vllmHead + ", upstream: v0.32.0}\n  - {repo: blis-substrate, branch: m2/units, head: " + subHead + ", bump: minor}\n"))
	if err != nil {
		t.Fatal(err)
	}
	dirs := map[string]string{"blis-substrate": sub.dir, "blis-vllm": vllm.dir, "inference-sim": sim.dir}
	return l, dirs, map[string]*repo{"blis-substrate": sub, "blis-vllm": vllm, "inference-sim": sim}
}

// A release interrupted at any point resumes: the plan reports a tagged layer as done, not
// as a refusal, keeps every version number, and the commands work on exactly the layers that
// remain. A layer is staged only once the tags below it are published.
func TestLayeredReleaseResumesAfterEveryInterruption(t *testing.T) {
	l, dirs, repos := layered(t)
	e := env(t, nil)
	published := map[string]bool{}
	e.TagPublished = func(_, tag string) bool { return published[tag] }
	want := map[string]string{"blis-substrate": "v0.12.0", "blis-vllm": "v0.32.0-blis.3", "inference-sim": "v0.17.0"}
	order := []string{"blis-substrate", "blis-vllm", "inference-sim"}
	// check re-plans and compares each layer's tag, and its done and staged state.
	check := func(when string, done, staged []string) []Step {
		t.Helper()
		steps, err := Plan(e, l, dirs, "main")
		if err != nil {
			t.Fatalf("%s: %v", when, err)
		}
		for _, st := range steps {
			if st.Tag != want[st.Repo] {
				t.Errorf("%s: %s is planned at %s, want %s; versions must not advance", when, st.Repo, st.Tag, want[st.Repo])
			}
			if len(st.Refusals) > 0 {
				t.Errorf("%s: %s refused: %v", when, st.Repo, st.Refusals)
			}
			if st.Done != slices.Contains(done, st.Repo) || st.Staged != slices.Contains(staged, st.Repo) {
				t.Errorf("%s: %s done %v staged %v; want done %v staged %v", when, st.Repo, st.Done, st.Staged, done, staged)
			}
		}
		return steps
	}
	steps := check("fresh", nil, nil)
	if err := Stage(e, steps, "inference-sim", false); err == nil {
		t.Error("staged the top layer while the layers below are untagged")
	}
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	steps = check("after staging layer 1", nil, []string{"blis-substrate"})
	if err := Tag(e, l, steps, "blis-vllm", false); err == nil {
		t.Error("tagged layer 2 before layer 1")
	}
	if err := Tag(e, l, steps, "", false); err != nil {
		t.Fatal(err)
	}
	if got := repos["blis-vllm"].git("tag", "--list"); strings.Contains(got, "blis.3") {
		t.Error("Tag made a tag outside its layer")
	}
	steps = check("after tagging layer 1", []string{"blis-substrate"}, nil)
	if err := Stage(e, steps, "", false); err == nil || !strings.Contains(err.Error(), "not published") {
		t.Fatalf("Stage(layer 2) = %v; want a refusal until layer 1's tag is published", err)
	}
	published["v0.12.0"] = true
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	if got := repos["blis-vllm"].git("show", "release/v0.32.0-blis.3:go.mod"); !strings.Contains(got, "blis-substrate v0.12.0") {
		t.Errorf("layer 2 is not pinned to layer 1's real tag:\n%s", got)
	}
	steps = check("after tagging layer 1 and staging layer 2", []string{"blis-substrate"}, []string{"blis-vllm"})
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatalf("staging layer 2 again: %v", err)
	}
	tip := repos["blis-vllm"].git("rev-parse", "release/v0.32.0-blis.3")
	if err := Tag(e, l, steps, "", false); err != nil {
		t.Fatal(err)
	}
	if got := repos["blis-vllm"].git("rev-list", "-n1", "v0.32.0-blis.3"); got != tip {
		t.Errorf("restaging moved the tip: tag on %s, tip was %s", got, tip)
	}
	published["v0.32.0-blis.3"] = true
	steps = check("after tagging layer 2", order[:2], nil)
	if err := Stage(e, steps, "", false); err != nil {
		t.Fatal(err)
	}
	if err := Tag(e, l, steps, "", false); err != nil {
		t.Fatal(err)
	}
	steps, err := Plan(e, l, dirs, "main")
	if err != nil {
		t.Fatal(err)
	}
	for _, st := range steps {
		if !st.Done || st.Tag != want[st.Repo] {
			t.Errorf("after the last layer: %s done %v tag %s", st.Repo, st.Done, st.Tag)
		}
		has(t, st.Refusals, "no layer of the set remains")
	}
	if err := Tag(e, l, steps, "", false); err == nil {
		t.Error("Tag found a layer to do after the last")
	}
	// A different head of the lowest repository is a new release and gets the next version.
	sub := repos["blis-substrate"]
	h2 := sub.branch("m2/more", "contracts/more.go", "package contracts\n", "More")
	st := planOne(t, e, "blis-substrate", sub.dir, h2, "bump: minor", 1)
	if st.Tag != "v0.13.0" || st.Done || st.Staged {
		t.Errorf("a different head: tag %s done %v staged %v; want a new v0.13.0", st.Tag, st.Done, st.Staged)
	}
}

// The bound of design §3.2 reads inference-sim's go.mod, so any set with an engine needs its
// checkout, even when inference-sim is not released.
func TestEngineSetNeedsTheInferenceSimCheckout(t *testing.T) {
	plan := func(t *testing.T, vllmGomod string, dirs func(vllm, sim string) map[string]string, simGomod string) ([]Step, error) {
		t.Helper()
		vllm := newRepo(t, vllmGomod)
		vllm.git("tag", "v0.32.0-blis.2")
		h := vllm.branch("b", "core/a.go", "package core\n", "Change")
		sim := newRepo(t, simGomod)
		l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-vllm, branch: b, head: " + h + ", upstream: v0.32.0}\n"))
		if err != nil {
			t.Fatal(err)
		}
		return Plan(env(t, nil), l, dirs(vllm.dir, sim.dir), "main")
	}
	both := func(v, s string) map[string]string { return map[string]string{"blis-vllm": v, "inference-sim": s} }
	t.Run("engine only, bound holds", func(t *testing.T) {
		steps, err := plan(t, vllmMod, both, simMod)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range steps[0].Refusals {
			if strings.Contains(r, "substrate") {
				t.Errorf("the bound holds but the release is refused: %s", r)
			}
		}
	})
	t.Run("engine only, bound broken", func(t *testing.T) {
		steps, err := plan(t, strings.Replace(vllmMod, "v0.11.0", "v0.12.0", 1), both, simMod)
		if err != nil {
			t.Fatal(err)
		}
		has(t, steps[0].Refusals, "newer than inference-sim's v0.11.0")
	})
	t.Run("engine only, no checkout is refused up front", func(t *testing.T) {
		_, err := plan(t, vllmMod, func(v, _ string) map[string]string { return map[string]string{"blis-vllm": v} }, simMod)
		if err == nil || !strings.Contains(err.Error(), "-dir inference-sim=<checkout>") {
			t.Fatalf("want a refusal naming -dir inference-sim, got %v", err)
		}
	})
	t.Run("substrate and engine", func(t *testing.T) {
		sub := newRepo(t, subMod)
		sub.git("tag", "v0.11.0")
		sh := sub.branch("s", "a.go", "package a\n", "Change")
		vllm := newRepo(t, vllmMod)
		vllm.git("tag", "v0.32.0-blis.2")
		vh := vllm.branch("b", "core/a.go", "package core\n", "Change")
		sim := newRepo(t, simMod)
		l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: s, head: " + sh + ", bump: minor}\n  - {repo: blis-vllm, branch: b, head: " + vh + ", upstream: v0.32.0}\n"))
		if err != nil {
			t.Fatal(err)
		}
		d := map[string]string{"blis-substrate": sub.dir, "blis-vllm": vllm.dir}
		if _, err := Plan(env(t, nil), l, d, "main"); err == nil || !strings.Contains(err.Error(), "-dir inference-sim=<checkout>") {
			t.Fatalf("want a refusal naming -dir inference-sim, got %v", err)
		}
		d["inference-sim"] = sim.dir
		steps, err := Plan(env(t, nil), l, d, "main")
		if err != nil {
			t.Fatal(err)
		}
		// The set moves substrate to v0.12.0, which inference-sim (v0.11.0) does not yet require.
		has(t, steps[1].Refusals, "newer than inference-sim's v0.11.0")
	})
	t.Run("a checkout of the wrong module", func(t *testing.T) {
		_, err := plan(t, vllmMod, both, vllmMod)
		if err == nil || !strings.Contains(err.Error(), "not github.com/inference-sim/inference-sim") {
			t.Fatalf("want a wrong-module refusal, got %v", err)
		}
	})
	t.Run("a checkout without exactly one direct substrate requirement", func(t *testing.T) {
		none := "module github.com/inference-sim/inference-sim\n\ngo 1.27.2\n"
		_, err := plan(t, vllmMod, both, none)
		if err == nil || !strings.Contains(err.Error(), "exactly one direct requirement") {
			t.Fatalf("want a refusal, got %v", err)
		}
	})
}

// ledgerFiles writes a small data ledger into a staged tip: one source with three units and,
// unless fit and anchor are both nil, the three published lists. fit and anchor name units by
// the index digit, "0" to "2"; the report list is empty.
func ledgerFiles(t *testing.T, r *repo, fit, anchor []string) {
	t.Helper()
	seed := strings.Repeat("ab", 32)
	r.write("catalog/ledger/ledger.yaml", `kind: Ledger
partition_version: 1
seed:
  sha256: `+seed+`
  generated_by: owner
  generated_on: 2026-11-04
  rotation_owner: owner
rules:
  s:
    cut_points:
      - {role: fit, from: 0, to: 29}
      - {role: select, from: 30, to: 39}
      - {role: validate, from: 40, to: 49}
      - {role: held, from: 50, to: 79}
      - {role: anchor, from: 80, to: 99}
`)
	r.write("catalog/ledger/sources.yaml", `kind: LedgerSources
sources:
  - id: s
    title: t
    revision:
      kind: git_commit
      url: https://github.com/o/r/tree/2f0f4cd8606d580567b253d02f3d699fe36db865
      ref: 2f0f4cd8606d580567b253d02f3d699fe36db865
    units: units/s.jsonl.gz
    licence:
      spdx: Apache-2.0
      evidence:
        - {url: "https://github.com/o/r/blob/2f0f4cd8606d580567b253d02f3d699fe36db865/LICENSE", sha256: "`+strings.Repeat("a", 64)+`", lines: "1", quote: Apache}
      sign_off: {by: "", on: ""}
`)
	var ids []string
	var units []ledger.Unit
	for i := 0; i < 3; i++ {
		id := map[string]any{"n": i}
		uid, _ := ledger.ID(id)
		rk, _ := ledger.ID(map[string]any{"k": i})
		units = append(units, ledger.Unit{UnitID: uid, RoleKey: rk, Identity: id})
	}
	sort.Slice(units, func(i, j int) bool { return units[i].UnitID < units[j].UnitID })
	for _, u := range units {
		ids = append(ids, u.UnitID)
	}
	var gz bytes.Buffer
	if err := ledger.EncodeUnits(&gz, units); err != nil {
		t.Fatal(err)
	}
	r.write("catalog/ledger/units/s.jsonl.gz", gz.String())
	list := func(role string, units []string) string {
		s := "kind: PublishedUnits\nrole: " + role + "\npartition_version: 1\nseed_sha256: " + seed + "\nunits:\n"
		if len(units) == 0 {
			return strings.Replace(s, "units:\n", "units: []\n", 1)
		}
		for _, u := range units {
			s += "  - " + u + "\n"
		}
		return s
	}
	if fit != nil || anchor != nil {
		resolve := func(idx []string) []string {
			var out []string
			for _, i := range idx {
				n := int(i[0] - '0')
				out = append(out, ids[n])
			}
			sort.Strings(out)
			return out
		}
		r.write("catalog/ledger/published/fit.yaml", list("fit", resolve(fit)))
		r.write("catalog/ledger/published/anchor.yaml", list("anchor", resolve(anchor)))
		r.write("catalog/ledger/published/report.yaml", list("report", nil))
	}
}

// A staged tip that carries the data ledger must carry its published lists, fit and anchor
// nonempty (D7, COV-2): the tag is refused for a tree with no lists and for one whose anchor
// list is empty, and given for a tree whose lists hold units. A tip with no ledger is not asked.
func TestTagRefusesAStagedTipWithoutReleasableLedgerLists(t *testing.T) {
	cases := []struct {
		name        string
		fit, anchor []string
		publishNone bool
		want        string
	}{
		{name: "no lists", publishNone: true, want: "fit list is not published"},
		{name: "an empty anchor list", fit: []string{"0"}, anchor: []string{}, want: "anchor list holds no unit"},
		{name: "an empty fit list", fit: []string{}, anchor: []string{"1"}, want: "fit list holds no unit"},
		{name: "nonempty fit and anchor lists", fit: []string{"0"}, anchor: []string{"1"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e := env(t, nil)
			sub := newRepo(t, subMod)
			sub.git("tag", "v0.11.0")
			sub.git("checkout", "-q", "-b", "b")
			if c.publishNone {
				ledgerFiles(t, sub, nil, nil)
			} else {
				ledgerFiles(t, sub, c.fit, c.anchor)
			}
			h := sub.commit("Add the ledger")
			sub.git("checkout", "-q", "main")
			l, err := ParseStackLock([]byte("set: s\nmilestone: 1\nrepos:\n  - {repo: blis-substrate, branch: b, head: " + h + ", bump: patch}\n"))
			if err != nil {
				t.Fatal(err)
			}
			steps, err := Plan(e, l, map[string]string{"blis-substrate": sub.dir}, "main")
			if err != nil {
				t.Fatal(err)
			}
			if err := Stage(e, steps, "", false); err != nil {
				t.Fatal(err)
			}
			err = Tag(e, l, steps, "", false)
			if c.want == "" {
				if err != nil {
					t.Fatalf("Tag = %v; want a tag for releasable lists", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "releasable ledger") || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Tag = %v; want a refusal containing %q", err, c.want)
			}
			if strings.Contains(sub.git("tag", "--list"), "v0.11.1") {
				t.Error("a tag exists after a refusal")
			}
		})
	}
}
