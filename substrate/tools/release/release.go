package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/blis-substrate/tools/internal/bot"
)

// Env is what the bot needs that is not in git: the registry, the invariant check results
// at each commit, the open invariant bugs, and the plan's current milestone. A test supplies
// it directly; the command reads it from files and the GitHub API.
type Env struct {
	Registry *invariants.Registry
	// Passed returns, for a repository at a commit, each ci check's test name and whether
	// it passed there.
	Passed func(repo, commit string) (map[string]bool, error)
	// Bugs returns the open issues labelled invariant-bug in a repository and in each
	// repository whose code it ships.
	Bugs func(repo string) ([]Bug, error)
	// Lint runs the selected-module-graph lint (tools/modlint) on a checkout of a staged tip,
	// with no deferral, and returns its findings. It sees the whole graph, so a transitive
	// BLIS pseudo-version or a BLIS replace is found as a direct one is. Tag refuses to run
	// without it.
	Lint func(dir string) ([]string, error)
	// TagPublished reports whether a release tag of a repository is on its remote. Stage
	// requires it for every layer below before it pins to that layer's tag.
	TagPublished func(repo, tag string) bool
}

// DeferralFile is the pin-by-commit deferral. It is removed in the change that re-pins to
// tags, so a staged tip never carries it.
const DeferralFile = ".github/modlint-deferral.yaml"

// Step is one repository's planned release.
type Step struct {
	Repo     string
	Dir      string // the local checkout
	Base     string // the branch the release merges into
	Head     string // the review set's head commit
	Prev     string // the previous release tag, "" for none
	Tag      string
	Pins     map[string]string // module path -> the tag this repository must pin, written once
	Notes    string
	Refusals []string
	// Staged is true when a release/<Tag> branch already holds this head's merge (an
	// interrupted release); Tag is then that branch's tag, not a new one.
	Staged bool
	// Done is true when Tag is already made on the staged tip of this head. The step is
	// reported and skipped; it is never a refusal, or a release interrupted after its first
	// layer could not resume.
	Done bool
}

// Plan works out each repository's release from a review set, bottom-up, without changing
// anything. dirs maps each repository to its local checkout. A review set of one repository
// is a single-repository release.
func Plan(env Env, lock *StackLock, dirs map[string]string, base string) ([]Step, error) {
	var steps []Step
	tags := map[string]string{} // repo -> the tag this set gives it
	for _, r := range lock.Repos {
		dir, ok := dirs[r.Repo]
		if !ok {
			return nil, fmt.Errorf("%s: no local checkout given", r.Repo)
		}
		info := Repos[r.Repo]
		st := Step{Repo: r.Repo, Dir: dir, Base: base, Head: r.Head, Pins: map[string]string{}}
		if _, err := gitIn(dir, "cat-file", "-e", r.Head+"^{commit}"); err != nil {
			return nil, fmt.Errorf("%s: head %s is not in %s", r.Repo, r.Head, dir)
		}
		existing, err := tagList(dir)
		if err != nil {
			return nil, err
		}
		// Retry identity: a head that Stage merged is already staged (branch release/<tag>) or
		// tagged (that tag on the staged tip). The plan then keeps that tag and does not
		// advance to the next version. A tagged step is done, not refused, so the layers
		// above it can resume; the refusal below is only for a head with nothing left to do.
		prior, tagged, err := stagedAs(dir, r.Head)
		if err != nil {
			return nil, err
		}
		if prior != "" {
			existing = without(existing, prior)
		}
		st.Prev = previousTag(existing, info.Kind, r.Upstream)
		// A head that already carries a release tag was released; planning it again would give
		// the same commit a second release tag.
		if at, err := gitIn(dir, "tag", "--points-at", r.Head); err == nil {
			for _, t := range strings.Fields(at) {
				if plainTag.MatchString(t) || engineTag.MatchString(t) {
					st.Refusals = append(st.Refusals, fmt.Sprintf("the head %s already carries the release tag %s; a commit is released once (LIF-1)", short(r.Head), t))
				}
			}
		}
		switch {
		case prior != "":
			st.Tag, st.Staged, st.Done = prior, !tagged, tagged
		case info.Kind == KindEngine || info.Kind == KindRouter:
			_, n := highestN(existing, r.Upstream)
			st.Tag = engineTagFor(r.Upstream, n+1)
			st.Refusals = append(st.Refusals, CheckEngineTag(st.Tag, existing)...)
		default:
			if st.Tag, err = NextSemver(existing, r.Bump); err != nil {
				return nil, err
			}
		}
		if info.Kind != KindEngine && info.Kind != KindRouter {
			if st.Prev != "" && breaksCompatibility(st.Prev, st.Tag) {
				found, err := deprecationsAt(dir, r.Head)
				if err != nil {
					return nil, err
				}
				st.Refusals = append(st.Refusals, CheckDeprecations(r.Repo, found, relaxations, lock.Milestone)...)
			}
		}
		// LIF-3 and design §13.3: every ci check passed at the head.
		passed, err := env.Passed(r.Repo, r.Head)
		if err != nil {
			return nil, err
		}
		st.Refusals = append(st.Refusals, CheckInvariants(env.Registry, r.Repo, passed, lock.Milestone, lock.MilestoneRelease)...)
		// X27.
		bugs, err := env.Bugs(r.Repo)
		if err != nil {
			return nil, err
		}
		st.Refusals = append(st.Refusals, CheckBugs(bugs, r.Fixes, r.Waivers, r.Security)...)
		// X10: pin each lower repository in this set at the tag the set gives it. Each pin
		// is written once, by the bot, after the lower tag exists.
		gomod, err := gitIn(dir, "show", r.Head+":go.mod")
		if err == nil {
			mf, err := modfile.Parse("go.mod", []byte(gomod), nil)
			if err != nil {
				return nil, fmt.Errorf("%s: go.mod: %w", r.Repo, err)
			}
			for _, req := range mf.Require {
				for lower, tag := range tags {
					if Repos[lower].Module == req.Mod.Path {
						st.Pins[req.Mod.Path] = tag
					}
				}
			}
			// Every BLIS module a release requires is at a release tag. A pin the bot writes
			// replaces a requirement in the set; any other requirement at a pseudo-version
			// would ship in a permanent tag (X11, LIF-2).
			for _, req := range mf.Require {
				if _, pinned := st.Pins[req.Mod.Path]; !pinned && strings.HasPrefix(req.Mod.Path, "github.com/inference-sim/") && module.IsPseudoVersion(req.Mod.Version) {
					st.Refusals = append(st.Refusals, fmt.Sprintf("go.mod requires %s at the pseudo-version %s; a release requires BLIS modules at release tags (X11, LIF-2)", req.Mod.Path, req.Mod.Version))
				}
			}
			for _, rep := range mf.Replace {
				if rep.Old.Path == Repos["llm-d-router"].Module {
					if tag, ok := tags["llm-d-router"]; ok {
						st.Pins["replace:"+rep.Old.Path] = tag
					}
				}
			}
			// Design §3.2: an engine or router never requires a newer substrate than
			// inference-sim does.
			if info.Kind == KindEngine || info.Kind == KindRouter {
				st.Refusals = append(st.Refusals, CheckSubstrateBound(requirement(mf, Repos["blis-substrate"].Module, tags["blis-substrate"]), simSubstrate(lock, dirs, tags))...)
			}
		}
		notes, err := Notes(env.Registry, st, dir)
		if err != nil {
			return nil, err
		}
		st.Notes = notes
		if st.Done {
			// Its checks were made when it was tagged; a bug opened since must not stop the
			// layers above it from resuming.
			st.Refusals = nil
		}
		tags[r.Repo] = st.Tag
		steps = append(steps, st)
	}
	remaining := false
	for _, st := range steps {
		remaining = remaining || !st.Done
	}
	for i := range steps {
		st := &steps[i]
		if !st.Done {
			continue
		}
		if !remaining {
			st.Refusals = append(st.Refusals, fmt.Sprintf("the head %s is already released at %s (its staged tip carries the tag) and no layer of the set remains to do; a commit is released once (LIF-1)", short(st.Head), st.Tag))
		}
		for _, lo := range steps {
			if !lo.Done && Repos[lo.Repo].Layer < Repos[st.Repo].Layer {
				st.Refusals = append(st.Refusals, fmt.Sprintf("%s is tagged but the layer below it, %s, is not; the release is inconsistent (X10)", st.Tag, lo.Repo))
			}
		}
	}
	return steps, nil
}

// pickLayer returns the index of the step the next command works on: the named repository,
// or, when layer is empty, the first step in lock order that is not yet tagged. A step that
// is already tagged is refused.
func pickLayer(steps []Step, layer string) (int, error) {
	for i, st := range steps {
		switch {
		case layer == "" && !st.Done:
			return i, nil
		case layer == st.Repo && st.Done:
			return 0, fmt.Errorf("%s: %s is already tagged for this head; nothing to do for this layer", st.Repo, st.Tag)
		case layer == st.Repo:
			return i, nil
		}
	}
	if layer == "" {
		return 0, fmt.Errorf("every layer of the set is already tagged")
	}
	return 0, fmt.Errorf("-layer %s: not a repository of the set", layer)
}

// lowerLayers returns the steps in layers strictly below step i.
func lowerLayers(steps []Step, i int) []Step {
	var out []Step
	for _, lo := range steps {
		if Repos[lo.Repo].Layer < Repos[steps[i].Repo].Layer {
			out = append(out, lo)
		}
	}
	return out
}

// requirement returns the version a go.mod requires of a module, or the tag this set gives
// it when the module is in the set (the pin the bot will write).
func requirement(mf *modfile.File, path, setTag string) string {
	for _, r := range mf.Require {
		if r.Mod.Path == path {
			if setTag != "" {
				return setTag
			}
			return r.Mod.Version
		}
	}
	return ""
}

// simSubstrate returns the substrate version inference-sim will require: the set's
// substrate tag if inference-sim is in the set, otherwise its current requirement.
func simSubstrate(lock *StackLock, dirs map[string]string, tags map[string]string) string {
	sub := Repos["blis-substrate"].Module
	for _, r := range lock.Repos {
		if r.Repo == "inference-sim" {
			if t, ok := tags["blis-substrate"]; ok {
				return t
			}
			gm, err := gitIn(dirs["inference-sim"], "show", r.Head+":go.mod")
			if err != nil {
				return ""
			}
			mf, err := modfile.Parse("go.mod", []byte(gm), nil)
			if err != nil {
				return ""
			}
			return requirement(mf, sub, "")
		}
	}
	dir, ok := dirs["inference-sim"]
	if !ok {
		return ""
	}
	gm, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		return ""
	}
	mf, err := modfile.Parse("go.mod", gm, nil)
	if err != nil {
		return ""
	}
	return requirement(mf, sub, "")
}

// without returns tags minus one.
func without(tags []string, drop string) []string {
	var out []string
	for _, t := range tags {
		if t != drop {
			out = append(out, t)
		}
	}
	return out
}

// stagedAs finds the release tag Stage gave a head. Stage leaves a merge commit whose second
// parent is the head, and, when pins are written, one pin commit on top of it; the branch
// release/<tag> or the tag points at that tip. It returns the tag and whether the tag
// itself exists ("" when the head was never staged). A tag wins over a branch.
func stagedAs(dir, head string) (tag string, tagged bool, err error) {
	out, err := gitIn(dir, "for-each-ref", "--format=%(refname)", "refs/tags", "refs/heads/release")
	if err != nil {
		return "", false, err
	}
	refs := strings.Fields(out)
	sort.Strings(refs)
	for _, ref := range refs {
		isTag := strings.HasPrefix(ref, "refs/tags/")
		t := strings.TrimPrefix(strings.TrimPrefix(ref, "refs/tags/"), "refs/heads/release/")
		if !isTag && ref == t {
			continue
		}
		if !plainTag.MatchString(t) && !engineTag.MatchString(t) {
			continue
		}
		ok, err := mergesHead(dir, ref+"^{commit}", head)
		if err != nil {
			return "", false, err
		}
		if !ok {
			continue
		}
		if isTag {
			return t, true, nil
		}
		tag = t
	}
	return tag, false, nil
}

// mergesHead reports whether the commit is Stage's merge of head, or the pin commit on it.
func mergesHead(dir, rev, head string) (bool, error) {
	for i := 0; i < 2; i++ {
		out, err := gitIn(dir, "rev-list", "--parents", "-n1", rev)
		if err != nil {
			return false, err
		}
		f := strings.Fields(out)
		if len(f) >= 3 && f[2] == head {
			return true, nil
		}
		if len(f) != 2 {
			return false, nil
		}
		rev = f[1]
	}
	return false, nil
}

func tagList(dir string) ([]string, error) {
	out, err := gitIn(dir, "tag", "--list")
	if err != nil {
		return nil, err
	}
	return strings.Fields(out), nil
}

func highestN(tags []string, upstream string) (string, int) {
	best, high := "", 0
	for _, t := range tags {
		if u, n, ok := upstreamOf(t); ok && u == upstream {
			if best == "" || n > high {
				best, high = t, n
			}
		}
	}
	return best, high
}

// latestEngineTag is the newest engine or router tag of any upstream release: the highest
// upstream by semver, then the highest N. It is the "previous release" of a first release on
// a new upstream release.
func latestEngineTag(tags []string) string {
	best, bestUp, bestN := "", "", -1
	for _, t := range tags {
		u, n, ok := upstreamOf(t)
		if !ok {
			continue
		}
		if c := semver.Compare(u, bestUp); best == "" || c > 0 || (c == 0 && n > bestN) {
			best, bestUp, bestN = t, u, n
		}
	}
	return best
}

func previousTag(tags []string, kind Kind, upstream string) string {
	if kind == KindEngine || kind == KindRouter {
		if t, _ := highestN(tags, upstream); t != "" {
			return t
		}
		return latestEngineTag(tags)
	}
	var rel []string
	for _, t := range tags {
		if plainTag.MatchString(t) {
			rel = append(rel, t)
		}
	}
	if len(rel) == 0 {
		return ""
	}
	sort.Slice(rel, func(i, j int) bool { return semver.Compare(rel[i], rel[j]) < 0 })
	return rel[len(rel)-1]
}

func deprecationsAt(dir, commit string) ([]Deprecation, error) {
	// One git grep over the commit's tree finds every candidate line; FindDeprecations
	// applies the exact rule to each. git grep exits 1 when nothing matches.
	cmd := exec.Command("git", "grep", "-n", "-I", "-E", `Deprecated:|MarkDeprecated\(|DEPRECATED`, commit, "--", "*.go", ":!*_test.go")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 {
			return nil, nil
		}
		return nil, fmt.Errorf("git grep: %w", err)
	}
	var all []Deprecation
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		// <commit>:<file>:<line>:<text>
		parts := strings.SplitN(strings.TrimPrefix(l, commit+":"), ":", 3)
		if len(parts) != 3 {
			continue
		}
		for _, d := range FindDeprecations(parts[0], []byte(parts[2])) {
			fmt.Sscan(parts[1], &d.Line)
			all = append(all, d)
		}
	}
	return all, nil
}

// Branch is the branch Stage builds a repository's release commit on: the base branch plus
// the head's merge and the pin commit, with no tag (LIF-3).
func Branch(st Step) string { return "release/" + st.Tag }

// Stage builds one layer's final release commit, in that repository's local checkout, and
// tags nothing. layer names the repository; empty means the first layer in lock order that is
// not yet tagged. It first requires, for every layer below, that its release tag exists and
// is published (env.TagPublished), because this layer's go.mod is pinned to those real tags.
// It then merges the head into the base with a merge commit and writes the pins the plan
// names in one commit, both on the branch Branch names; the base branch does not move. A
// branch left by an interrupted run is continued, not rebuilt, so the commit that was or will
// be checked keeps its SHA; a missing pin commit is added to it. The commit is not the head
// Plan checked, so it has no recorded results yet. Tag, a later step, tags the branch tip
// only after every required result is recorded for that exact commit (LIF-3). Stage refuses
// to start if any step has a refusal. It never pushes: pushing the branch, running the layer's
// checks on it, pushing the tag and running the next layer's checks against it are manual
// until the release job exists (design §13.6).
func Stage(env Env, steps []Step, layer string, sign bool) error {
	for _, st := range steps {
		if len(st.Refusals) > 0 {
			return fmt.Errorf("%s: %d refusal(s); nothing was changed", st.Repo, len(st.Refusals))
		}
	}
	i, err := pickLayer(steps, layer)
	if err != nil {
		return err
	}
	st := steps[i]
	for _, lo := range lowerLayers(steps, i) {
		if !lo.Done {
			return fmt.Errorf("%s: the layer below, %s, is not tagged yet; tag %s first", st.Repo, lo.Repo, lo.Repo)
		}
		if env.TagPublished == nil {
			return fmt.Errorf("%s: cannot tell whether %s is published", st.Repo, lo.Tag)
		}
		if !env.TagPublished(lo.Repo, lo.Tag) {
			return fmt.Errorf("%s: %s of %s is not published; push it and run %s's checks against it first", st.Repo, lo.Tag, lo.Repo, st.Repo)
		}
	}
	ident := []string{"-c", "user.name=" + bot.Name, "-c", "user.email=" + bot.Email}
	if st.Staged {
		if _, err := gitIn(st.Dir, "checkout", "-q", Branch(st)); err != nil {
			return err
		}
	} else {
		if _, err := gitIn(st.Dir, "checkout", "-q", "-B", Branch(st), st.Base); err != nil {
			return err
		}
		if _, err := gitIn(st.Dir, append(ident, "merge", "--no-ff", "-q", "-m", fmt.Sprintf("Merge %s for %s", short(st.Head), st.Tag), st.Head)...); err != nil {
			return fmt.Errorf("%s: merge: %w", st.Repo, err)
		}
	}
	if len(st.Pins) > 0 {
		if err := writePins(st.Dir, st.Pins); err != nil {
			return fmt.Errorf("%s: %w", st.Repo, err)
		}
		if _, err := gitIn(st.Dir, "add", "go.mod"); err != nil {
			return err
		}
		// Nothing to commit when an earlier run already wrote these pins.
		if _, err := gitIn(st.Dir, "diff", "--cached", "--quiet"); err != nil {
			args := append(ident, "commit", "-q", "-m", "Pin "+pinSummary(st.Pins)+" for "+st.Tag)
			if sign {
				args = append(args, "-S")
			}
			if _, err := gitIn(st.Dir, args...); err != nil {
				return fmt.Errorf("%s: pin commit: %w", st.Repo, err)
			}
		}
	}
	return nil
}

// Tag makes the annotated tag, whose message is the release notes, on the tip of one layer's
// staged branch, and on nothing else. layer names the repository; empty means the first layer
// in lock order that is not yet tagged. The layers below must already be tagged. Before it
// tags, it checks that the tip still contains the planned head and that the environment
// records a passing result for every ci check at the tip's own SHA (LIF-3, design §13.3). A
// tip that moved since it was checked has no results and is refused.
func Tag(env Env, lock *StackLock, steps []Step, layer string, sign bool) error {
	i, err := pickLayer(steps, layer)
	if err != nil {
		return err
	}
	st := steps[i]
	for _, lo := range lowerLayers(steps, i) {
		if !lo.Done {
			return fmt.Errorf("%s: the layer below, %s, is not tagged yet; tag %s first", st.Repo, lo.Repo, lo.Repo)
		}
	}
	tip, err := gitIn(st.Dir, "rev-parse", "--verify", "refs/heads/"+Branch(st)+"^{commit}")
	if err != nil {
		return fmt.Errorf("%s: no staged release branch %s: %w", st.Repo, Branch(st), err)
	}
	if _, err := gitIn(st.Dir, "merge-base", "--is-ancestor", st.Head, tip); err != nil {
		return fmt.Errorf("%s: staged tip %s does not contain the planned head %s", st.Repo, short(tip), short(st.Head))
	}
	passed, err := env.Passed(st.Repo, tip)
	if err != nil {
		return err
	}
	if bad := CheckInvariants(env.Registry, st.Repo, passed, lock.Milestone, lock.MilestoneRelease); len(bad) > 0 {
		return fmt.Errorf("%s: staged tip %s is not checked: %s", st.Repo, short(tip), strings.Join(bad, "; "))
	}
	if err := checkStagedTip(env, st, tip); err != nil {
		return err
	}
	ident := []string{"-c", "user.name=" + bot.Name, "-c", "user.email=" + bot.Email}
	args := append(ident, "tag", "-a", st.Tag, "-F", "-", tip)
	if sign {
		args = append(ident, "tag", "-s", st.Tag, "-F", "-", tip)
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = st.Dir
	cmd.Stdin = strings.NewReader(st.Notes)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%s: tag %s: %v: %s", st.Repo, st.Tag, err, out)
	}
	return nil
}

// checkStagedTip validates the commit that will carry the tag: the deferral file is gone
// from it, and the selected module graph of its checkout passes the lint with no deferral
// (X11, LIF-2). The direct requirements in the review head do not stand in for this: a BLIS
// pseudo-version can arrive through another module's requirements.
func checkStagedTip(env Env, st Step, tip string) error {
	if _, err := gitIn(st.Dir, "cat-file", "-e", tip+":"+DeferralFile); err == nil {
		return fmt.Errorf("%s: staged tip %s still holds %s; remove it in the change that re-pins to tags", st.Repo, short(tip), DeferralFile)
	}
	if env.Lint == nil {
		return fmt.Errorf("%s: no module lint is configured, so the staged tip's module graph cannot be checked", st.Repo)
	}
	tmp, err := os.MkdirTemp("", "release-lint-")
	if err != nil {
		return err
	}
	wt := filepath.Join(tmp, "tip")
	if _, err := gitIn(st.Dir, "worktree", "add", "--detach", "-q", wt, tip); err != nil {
		return err
	}
	bad, lerr := env.Lint(wt)
	if _, err := gitIn(st.Dir, "worktree", "remove", wt); err != nil {
		return err
	}
	if err := os.Remove(tmp); err != nil {
		return err
	}
	if lerr != nil {
		return fmt.Errorf("%s: module lint of staged tip %s: %w", st.Repo, short(tip), lerr)
	}
	if len(bad) > 0 {
		return fmt.Errorf("%s: staged tip %s fails the module lint: %s", st.Repo, short(tip), strings.Join(bad, "; "))
	}
	return nil
}

func pinSummary(p map[string]string) string {
	var s []string
	for k, v := range p {
		s = append(s, strings.TrimPrefix(k, "replace:")+" "+v)
	}
	sort.Strings(s)
	return strings.Join(s, ", ")
}

// writePins rewrites go.mod's requirements (and the router replace) to the set's tags. The
// go.sum entries are added by go mod tidy in the next layer's checks, which run against the
// new tags once they are pushed.
func writePins(dir string, pins map[string]string) error {
	path := filepath.Join(dir, "go.mod")
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	mf, err := modfile.Parse(path, raw, nil)
	if err != nil {
		return err
	}
	for k, v := range pins {
		if old, ok := strings.CutPrefix(k, "replace:"); ok {
			if err := mf.AddReplace(old, "", Repos["llm-d-router"].Module, ""); err != nil {
				return err
			}
			if err := mf.AddReplace(old, "", "github.com/inference-sim/llm-d-router", v); err != nil {
				return err
			}
			continue
		}
		if err := mf.AddRequire(k, v); err != nil {
			return err
		}
	}
	mf.Cleanup()
	out, err := mf.Format()
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0o644)
}

func gitIn(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(string(out)), nil
}

func short(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}
